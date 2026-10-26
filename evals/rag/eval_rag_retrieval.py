"""RAG 檢索面評估：同一份 golden set，比較不同檢索設定的 Hit@k / MRR。

回答的問題是「使用者問一句話，該撈到的那篇文章有沒有被撈出來、排第幾」——只評估檢索，
不評估最後 LLM 生成的回答（生成面要另外做 faithfulness 評估，不在這支腳本的範圍）。

評分單位是「文章」不是 chunk：golden set 每題標的是一篇 expected_article_id，
把檢索結果的 chunk 依序去重成文章清單後，看 expected 那篇排在第幾名。

可比較的設定（--configs 指定，預設全跑）：
  production           走正式程式碼：graph.filter_define_llm 拆解作者/時間 filter → tool.retrieve_nodes
                       （會真的打 Anthropic API，每題一次）。最多只回 3 個 chunk，所以只看 Hit@1/Hit@3。
  vec_top2_rerank_en   不帶 filter，複製 retrieve_nodes 主要路徑的實際行為：
                       as_retriever() 沒指定 similarity_top_k → 預設只撈 2 筆，再丟給英文 reranker
  vec_top10            純向量檢索 10 筆，不 rerank
  vec_top10_rerank_en  向量 10 筆 → cross-encoder/ms-marco-MiniLM-L-6-v2（英文模型）重排
  vec_top10_rerank_zh  向量 10 筆 → BAAI/bge-reranker-base（支援中文）重排；第一次跑會下載約 1GB 模型
  bm25_top10           只用 BM25 關鍵字檢索 10 筆（jieba 斷詞），看關鍵字那一路單獨有多少能力
  hybrid_top10         向量 10 筆 + BM25 10 筆，RRF 合併後留 10 筆（跟正式程式碼同一個 hybrid_search）
  hybrid_top10_rerank_zh  hybrid 10 筆 → 中文 reranker：修完之後的正式流程（不含 LLM filter 拆解）

用法：
    cd Substack_post_download
    uv run python evals/rag/eval_rag_retrieval.py
    uv run python evals/rag/eval_rag_retrieval.py --configs vec_top10 vec_top10_rerank_en
    uv run python evals/rag/eval_rag_retrieval.py --threshold 0.7   # 最佳設定 Hit@3 低於門檻時 exit code 非 0

跟 eval_router.py 一樣不會被 `pytest tests/` 收集到，需要手動執行。
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

from llama_index.core import QueryBundle, VectorStoreIndex  # noqa: E402
from llama_index.core.postprocessor import SentenceTransformerRerank  # noqa: E402
import chromadb  # noqa: E402
import torch  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402
from vector_store import COLLECTION_NAME, VectorStoreService  # noqa: E402
from bm25_index import get_bm25_index  # noqa: E402
from tool import hybrid_search  # noqa: E402
from authors import AUTHORS  # noqa: E402

ALL_CONFIGS = [
    "production",
    "vec_top2_rerank_en", "vec_top10", "vec_top10_rerank_en", "vec_top10_rerank_zh",
    "bm25_top10", "hybrid_top10", "hybrid_top10_rerank_zh",
]
RERANK_MODELS = {
    "en": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "zh": "BAAI/bge-reranker-base",
}
# 題目裡有沒有提到作者（名稱或別名），來自 authors.json
AUTHOR_HINT = re.compile(
    "|".join(re.escape(n) for a in AUTHORS for n in [a["name"], *a.get("aliases", [])]), re.IGNORECASE
)


def load_golden_set(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dedupe_articles(article_ids: list) -> list:
    seen, ordered = set(), []
    for a in article_ids:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    return ordered


class ComponentRetriever:
    """不經過 LLM 的檢索元件組合：向量 top_k 粗篩 →（選擇性）cross-encoder 重排。"""

    def __init__(self, mode: str, top_k: int, rerank: str | None, db_path: Path | None = None,
                 embed_model_name: str | None = None, bf16: bool = False):
        if db_path is None:
            v_service = VectorStoreService()
            vector_store, embed_model = v_service.vector_store, v_service.embed_model
        else:
            # 實驗用資料庫（build_experiment_index.py 建的）：查詢時要用跟建庫時同一個 embedding 模型
            collection = chromadb.PersistentClient(path=str(db_path)).get_collection(COLLECTION_NAME)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            model_kwargs = {"torch_dtype": torch.bfloat16} if bf16 else {}
            embed_model = HuggingFaceEmbedding(model_name=embed_model_name, model_kwargs=model_kwargs)
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        self.vector_store = vector_store
        self.mode = mode
        self.top_k = top_k
        # top_n 設成跟 top_k 一樣：重排全部候選、保留完整順序，才算得出 MRR
        self.reranker = (
            SentenceTransformerRerank(model=RERANK_MODELS[rerank], top_n=top_k) if rerank else None
        )

    def __call__(self, query: str) -> list:
        bundle = QueryBundle(query_str=query)
        if self.mode == "vec":
            nodes = self.index.as_retriever(similarity_top_k=self.top_k).retrieve(bundle)
        elif self.mode == "bm25":
            nodes = get_bm25_index(self.vector_store).search(query, top_k=self.top_k)
        else:  # hybrid：跟正式程式碼走同一個函式
            nodes = hybrid_search(self.index, self.vector_store, query, None)
        if self.reranker:
            nodes = self.reranker.postprocess_nodes(nodes=nodes, query_bundle=bundle)
        return dedupe_articles([n.node.metadata.get("article_id") for n in nodes])


class ProductionRetriever:
    """走跟 graph.rag_search_node 一樣的前兩步（filter 拆解 + retrieve_nodes），跳過 judge。
    retrieve_nodes 回傳的 chunk 沒有 article_id，用 (title, post_date) 對回去。"""

    def __init__(self):
        import graph as graph_module  # 延遲 import：只跑 component 設定時不需要載入整張 graph
        import inspect
        self.g = graph_module
        # 直接從 rag_search_node 原始碼取出 system prompt，避免評估腳本複製一份之後跟正式邏輯跑偏
        src = inspect.getsource(graph_module.rag_search_node)
        self.system_instruction = re.search(r'system_instruction = """(.*?)"""', src, re.S).group(1)
        self.title_date_to_article = self._build_lookup()
        self.last_meta: dict = {}

    @staticmethod
    def _build_lookup() -> dict:
        collection = VectorStoreService().vector_store._collection
        metas = collection.get(include=["metadatas"])["metadatas"]
        return {(m.get("title"), m.get("post_date")): m.get("article_id") for m in metas}

    def __call__(self, query: str) -> list:
        intent = self.g.filter_define_llm.invoke([("system", self.system_instruction), ("human", query)])
        filters = intent.to_llama_index_filters
        retrieval = self.g.retrieve_nodes.invoke(
            {"user_query": query, "filtered_query": intent.query, "final_filters": filters}
        )
        self.last_meta = {
            "filtered_query": intent.query,
            "author": intent.author,
            "filters": [f.model_dump() for f in intent.filters],
        }
        chunks = retrieval["data"] or []
        return dedupe_articles(
            [self.title_date_to_article.get((c["title"], c["post_date"])) for c in chunks]
        )


# --db-path / --embed-model 指定的實驗資料庫；None 代表用正式資料庫
EXPERIMENT = {"db_path": None, "embed_model_name": None, "bf16": False}


def build_retriever(name: str):
    if name == "production":
        return ProductionRetriever()
    m = re.fullmatch(r"(vec|bm25|hybrid)_top(\d+)(?:_rerank_(en|zh))?", name)
    return ComponentRetriever(mode=m.group(1), top_k=int(m.group(2)), rerank=m.group(3), **EXPERIMENT)


def score(ranked: list, expected) -> dict:
    rank = ranked.index(expected) + 1 if expected in ranked else None
    return {
        "rank": rank,
        "hit@1": rank == 1,
        "hit@3": rank is not None and rank <= 3,
        "hit@10": rank is not None and rank <= 10,
        "rr": 1 / rank if rank else 0.0,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    if not n:
        return {"n": 0}
    return {
        "n": n,
        "hit@1": sum(r["hit@1"] for r in rows) / n,
        "hit@3": sum(r["hit@3"] for r in rows) / n,
        "hit@10": sum(r.get("hit@10", r["hit@3"]) for r in rows) / n,
        "mrr": sum(r["rr"] for r in rows) / n,
    }


def run_config(name: str, cases: list[dict]) -> list[dict]:
    print(f"\n--- {name} ---")
    retriever = build_retriever(name)
    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        ranked = retriever(case["query"])
        latency = time.perf_counter() - t0
        s = score(ranked, case["expected_article_id"])
        row = {
            "id": case["id"],
            "query": case["query"],
            "expected_author": case["expected_author"],
            "has_author_hint": bool(AUTHOR_HINT.search(case["query"])),
            "ambiguous": case.get("ambiguous", False),
            "ranked_article_ids": ranked,
            "latency_s": round(latency, 2),
            **s,
        }
        if isinstance(retriever, ProductionRetriever):
            row["production_meta"] = retriever.last_meta
        rows.append(row)
        status = f"rank={s['rank']}" if s["rank"] else "MISS  "
        print(f"[{i}/{len(cases)}] {status:7s} {case['id']:7s} | {case['query']}")
    return rows


def summarize(rows: list[dict]) -> dict:
    # 主要指標只算核心題；ambiguous 題（每週週報這類多篇都答得出來的主題）只標一篇當答案不公平，另外列參考
    core = [r for r in rows if not r["ambiguous"]]
    return {
        "overall": aggregate(core),
        "ambiguous": aggregate([r for r in rows if r["ambiguous"]]),
        "by_author": {
            a: aggregate([r for r in core if r["expected_author"] == a])
            for a in sorted({r["expected_author"] for r in core})
        },
        "with_author_hint": aggregate([r for r in core if r["has_author_hint"]]),
        "without_author_hint": aggregate([r for r in core if not r["has_author_hint"]]),
        "avg_latency_s": round(sum(r["latency_s"] for r in rows) / len(rows), 2),
    }


def fmt(agg: dict) -> str:
    if not agg.get("n"):
        return "n/a"
    return f"H@1 {agg['hit@1']:.0%}  H@3 {agg['hit@3']:.0%}  MRR {agg['mrr']:.2f}  (n={agg['n']})"


def print_report(summaries: dict[str, dict]) -> None:
    print("\n" + "=" * 78)
    print("RAG 檢索評估報告（文章層級，只算核心題）")
    print("=" * 78)
    print(f"{'config':22s} {'Hit@1':>6s} {'Hit@3':>6s} {'Hit@10':>7s} {'MRR':>6s} {'延遲(s)':>8s}")
    for name, s in summaries.items():
        o = s["overall"]
        print(f"{name:22s} {o['hit@1']:6.0%} {o['hit@3']:6.0%} {o['hit@10']:7.0%} {o['mrr']:6.2f} {s['avg_latency_s']:8.2f}")
    print("\n細分（題目裡有沒有提到作者 / 依作者）：")
    for name, s in summaries.items():
        print(f"  {name}")
        print(f"    邊界題    : {fmt(s['ambiguous'])}（僅供參考）")
        print(f"    有提作者  : {fmt(s['with_author_hint'])}")
        print(f"    沒提作者  : {fmt(s['without_author_hint'])}")
        for author, agg in s["by_author"].items():
            print(f"    {author:10s}: {fmt(agg)}")
    print("\n注意：production 最多只回 3 個 chunk，MRR 只算前 3 名，跟其他設定（前 10 名）不完全可比。")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", type=Path, default=Path(__file__).parent / "golden_set.jsonl")
    parser.add_argument("--configs", nargs="+", default=ALL_CONFIGS, choices=ALL_CONFIGS)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "last_run_report.json")
    parser.add_argument("--threshold", type=float, default=None,
                        help="最佳設定的 Hit@3 低於這個值時以 exit code 1 結束")
    parser.add_argument("--db-path", type=Path, default=None,
                        help="評估 build_experiment_index.py 建的實驗資料庫，而不是正式資料庫（不能搭配 production）")
    parser.add_argument("--embed-model", default="BAAI/bge-base-zh-v1.5", help="搭配 --db-path：建庫時用的模型")
    parser.add_argument("--bf16", action="store_true")
    args = parser.parse_args()

    if args.db_path is not None:
        if "production" in args.configs:
            parser.error("--db-path 只能搭配 vec_* 設定；production 永遠走正式資料庫")
        EXPERIMENT.update(db_path=args.db_path, embed_model_name=args.embed_model, bf16=args.bf16)

    cases = load_golden_set(args.set)
    results, summaries = {}, {}
    for name in args.configs:
        results[name] = run_config(name, cases)
        summaries[name] = summarize(results[name])
        # 每跑完一個設定就存一次：production 會打幾十次 API，中途斷掉不用全部重跑
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "golden_set": str(args.set),
            "db_path": str(args.db_path) if args.db_path else "production",
            "embed_model": args.embed_model if args.db_path else "BAAI/bge-base-zh-v1.5",
            "summaries": summaries,
            "results": results,
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print_report(summaries)
    print(f"\n完整結果已存到: {args.out}")

    if args.threshold is not None:
        best = max(s["overall"]["hit@3"] for s in summaries.values())
        if best < args.threshold:
            print(f"最佳 Hit@3 {best:.1%} 低於門檻 {args.threshold:.1%}")
            sys.exit(1)


if __name__ == "__main__":
    main()
