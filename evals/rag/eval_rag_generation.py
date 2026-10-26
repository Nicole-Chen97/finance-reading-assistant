"""RAG 生成面評估：用 LlamaIndex 內建評估器，替「最終回答」打分數。

跟 eval_rag_retrieval.py 的分工：
  檢索評估 —— 該撈到的文章有沒有撈到（用公式算，不用 LLM）
  生成評估 —— 拿著撈到的資料，LLM 最後寫出來的回答好不好（用另一個 LLM 當裁判）

流程：每一題都走正式程式碼 graph.rag_search_node → graph.answer_node 產生回答，再交給三個評估器：
  FaithfulnessEvaluator     回答裡的說法，在撈到的資料裡找得到根據嗎？（抓幻覺）       → 通過 / 不通過
  AnswerRelevancyEvaluator  回答有沒有回答到問題？（抓答非所問）                      → 0~1 分
  ContextRelevancyEvaluator 撈到的資料跟問題相關嗎？（從生成的角度再看一次檢索品質）   → 0~1 分
CorrectnessEvaluator 需要每題都有「標準答案」，目前 golden set 只標了應該撈到哪篇文章，所以沒用。

另外記錄每題「檢索有沒有撈到正確文章」，可以交叉看：撈錯資料的時候，LLM 是老實說不知道，還是硬掰？

要注意：
  - 每題大約 6~8 次 API 呼叫（filter 拆解、runtime judge、回答、三個評估器），會花錢。
  - 裁判本身是 LLM，分數會抖動、也會判錯；看趨勢跟個別失敗案例，不要把小數點當精確測量。
  - Faithfulness 的預設 prompt 偏寬鬆（只要資料裡「有任何部分」支持就判 YES）。

用法：
    cd Substack_post_download
    uv run python evals/rag/eval_rag_generation.py               # 核心題全部
    uv run python evals/rag/eval_rag_generation.py --limit 5     # 先試跑 5 題
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

from langchain_core.messages import HumanMessage  # noqa: E402
from llama_index.core.evaluation import (  # noqa: E402
    AnswerRelevancyEvaluator,
    ContextRelevancyEvaluator,
    FaithfulnessEvaluator,
)
from llama_index.llms.anthropic import Anthropic  # noqa: E402
import graph as graph_module  # noqa: E402  (正式的 rag_search_node / answer_node，不 mock)

JUDGE_MODEL = "claude-sonnet-4-6"


def load_golden_set(path: Path, include_ambiguous: bool) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    return cases if include_ambiguous else [c for c in cases if not c.get("ambiguous")]


def run_pipeline(query: str) -> tuple[str, list[dict]]:
    """跟正式環境一樣：rag_search_node 檢索 → answer_node 生成。回傳 (回答, 撈到的 chunk)。"""
    state = {"messages": [HumanMessage(content=query)], "router": "rag", "tool_result": None}
    state = graph_module.rag_search_node(state)
    chunks = state["tool_result"]["raw"] or []
    state = graph_module.answer_node(state)
    return state["messages"][-1].content, chunks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", type=Path, default=Path(__file__).parent / "golden_set.jsonl")
    parser.add_argument("--include-ambiguous", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "last_generation_report.json")
    args = parser.parse_args()

    cases = load_golden_set(args.set, args.include_ambiguous)[: args.limit]
    judge = Anthropic(model=JUDGE_MODEL, temperature=0, api_key=os.environ["ANTHROPIC_API_KEY"].strip())
    faithfulness = FaithfulnessEvaluator(llm=judge)
    answer_relevancy = AnswerRelevancyEvaluator(llm=judge)
    context_relevancy = ContextRelevancyEvaluator(llm=judge)

    rows = []
    for i, case in enumerate(cases, 1):
        t0 = time.perf_counter()
        answer, chunks = run_pipeline(case["query"])
        contexts = [c["text"] for c in chunks]
        retrieval_hit = any(
            c["title"] == case["expected_title"] and c["post_date"] == case["expected_post_date"] for c in chunks
        )

        f = faithfulness.evaluate(query=case["query"], response=answer, contexts=contexts)
        a = answer_relevancy.evaluate(query=case["query"], response=answer)
        c = context_relevancy.evaluate(query=case["query"], contexts=contexts)

        row = {
            "id": case["id"],
            "query": case["query"],
            "expected_title": case["expected_title"],
            "retrieved_titles": [ch["title"] for ch in chunks],
            "retrieval_hit": retrieval_hit,
            "answer": answer,
            "faithful": bool(f.passing),
            "faithfulness_feedback": f.feedback,
            "answer_relevancy": a.score,
            "answer_relevancy_feedback": a.feedback,
            "context_relevancy": c.score,
            "context_relevancy_feedback": c.feedback,
            "seconds": round(time.perf_counter() - t0, 1),
        }
        rows.append(row)
        print(f"[{i}/{len(cases)}] {'撈對' if retrieval_hit else '撈錯'} "
              f"faithful={'Y' if row['faithful'] else 'N'} "
              f"ans_rel={row['answer_relevancy'] or 0:.2f} ctx_rel={row['context_relevancy'] or 0:.2f} "
              f"| {case['id']} {case['query']}", flush=True)
        args.out.write_text(json.dumps({"results": rows}, ensure_ascii=False, indent=2), encoding="utf-8")

    def agg(rs: list[dict]) -> dict:
        if not rs:
            return {"n": 0}
        return {
            "n": len(rs),
            "faithfulness": sum(r["faithful"] for r in rs) / len(rs),
            "answer_relevancy": sum(r["answer_relevancy"] or 0 for r in rs) / len(rs),
            "context_relevancy": sum(r["context_relevancy"] or 0 for r in rs) / len(rs),
        }

    summary = {
        "overall": agg(rows),
        "retrieval_hit": agg([r for r in rows if r["retrieval_hit"]]),
        "retrieval_miss": agg([r for r in rows if not r["retrieval_hit"]]),
    }
    args.out.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "judge_model": JUDGE_MODEL,
        "summary": summary,
        "results": rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print("RAG 生成評估報告")
    print("=" * 70)
    for label, key in [("全部", "overall"), ("檢索撈對時", "retrieval_hit"), ("檢索撈錯時", "retrieval_miss")]:
        s = summary[key]
        if s["n"]:
            print(f"{label:8s} (n={s['n']:2d})  Faithfulness {s['faithfulness']:.0%}  "
                  f"Answer relevancy {s['answer_relevancy']:.2f}  Context relevancy {s['context_relevancy']:.2f}")
    print(f"\n完整結果（含每題回答與裁判理由）: {args.out}")


if __name__ == "__main__":
    main()
