"""建一份「實驗用」的向量資料庫：換 chunk 切法 / metadata / embedding 模型，不動正式資料庫。

來源是正式 Chroma 裡現有的 chunk（MarkdownNodeParser 按標題切好的結果），所以不用重新下載文章。
在這之上可以：
  --split tokens     再用 SentenceSplitter 把超過 --chunk-size 的段落切小（以該 embedding 模型自己的 tokenizer 計算長度），
                     保留原本的 header_path 等 metadata —— 也就是「先按標題切，太長的再按句子切」
  --metadata slim    計算 embedding 時只帶 title / author / header_path，不帶 article_id、網址、時間戳這類對語意沒幫助的欄位
                     （metadata 本身仍完整存進資料庫，filter 照樣能用）
  --embed-model      換 embedding 模型

輸出：data/experiments/<name>/ 底下一份獨立的 Chroma，外加 build_info.json（chunk 數、建庫耗時、截斷統計）。
之後用 eval_rag_retrieval.py --db-path data/experiments/<name> --embed-model <同一個模型> 評估。

用法：
    cd Substack_post_download
    uv run python evals/rag/build_experiment_index.py --name base_slim --metadata slim
    uv run python evals/rag/build_experiment_index.py --name base_slim_split --metadata slim --split tokens
    uv run python evals/rag/build_experiment_index.py --name m3_slim_split --metadata slim --split tokens --embed-model BAAI/bge-m3
"""
import argparse
import json
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

import chromadb  # noqa: E402
import torch  # noqa: E402
from llama_index.core.node_parser import SentenceSplitter  # noqa: E402
from llama_index.core.schema import MetadataMode, TextNode  # noqa: E402
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # noqa: E402
from llama_index.vector_stores.chroma import ChromaVectorStore  # noqa: E402
from vector_store import COLLECTION_NAME, DB_PATH  # noqa: E402

EXPERIMENTS_DIR = APP_DIR.parent / "data" / "experiments"
# 正式資料庫裡每個 chunk 帶的 11 個 metadata 欄位
METADATA_KEYS = ["article_id", "publication_id", "author", "title", "type", "post_date",
                 "post_date_timestamp", "audience", "canonical_url", "slug", "header_path"]
SLIM_EMBED_KEYS = {"title", "author", "header_path"}


def load_source_chunks() -> list[TextNode]:
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    nodes = []
    for node_id, text, meta in zip(data["ids"], data["documents"], data["metadatas"]):
        nodes.append(TextNode(id_=node_id, text=text, metadata={k: meta[k] for k in METADATA_KEYS if k in meta}))
    return nodes


def apply_metadata_mode(nodes: list[TextNode], mode: str) -> None:
    for n in nodes:
        n.excluded_embed_metadata_keys = (
            [k for k in n.metadata if k not in SLIM_EMBED_KEYS] if mode == "slim" else []
        )


def split_long_nodes(nodes: list[TextNode], tokenizer, max_tokens: int, overlap: int) -> list[TextNode]:
    """每個 chunk 的 token 預算 = max_tokens - 這個 chunk 自己的 metadata 前綴長度，超過的才切。"""
    out = []
    for n in nodes:
        meta_tokens = len(tokenizer(n.get_metadata_str(mode=MetadataMode.EMBED)))
        budget = max(128, max_tokens - meta_tokens - 8)  # 8：特殊 token 與換行的緩衝
        if len(tokenizer(n.text)) <= budget:
            out.append(n)
            continue
        splitter = SentenceSplitter(chunk_size=budget, chunk_overlap=min(overlap, budget // 4), tokenizer=tokenizer)
        for i, piece in enumerate(splitter.split_text(n.text)):
            out.append(TextNode(
                id_=f"{n.id_}::part-{i + 1}",
                text=piece,
                metadata=dict(n.metadata),
                excluded_embed_metadata_keys=list(n.excluded_embed_metadata_keys),
            ))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", required=True)
    parser.add_argument("--embed-model", default="BAAI/bge-base-zh-v1.5")
    parser.add_argument("--metadata", choices=["full", "slim"], default="full")
    parser.add_argument("--split", choices=["none", "tokens"], default="none")
    parser.add_argument("--max-tokens", type=int, default=512, help="embedding 時整段（metadata + 內文）的 token 上限")
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--bf16", action="store_true", help="用 bfloat16 載入模型，記憶體減半（大模型在小機器上用）")
    args = parser.parse_args()

    out_dir = EXPERIMENTS_DIR / args.name
    if out_dir.exists():
        sys.exit(f"{out_dir} 已存在，換一個 --name 或先刪掉")

    model_kwargs = {"torch_dtype": torch.bfloat16} if args.bf16 else {}
    embed_model = HuggingFaceEmbedding(model_name=args.embed_model, max_length=args.max_tokens,
                                       embed_batch_size=args.batch_size, model_kwargs=model_kwargs)
    hf_tokenizer = embed_model._model.tokenizer

    def tokenize(text: str) -> list[int]:
        return hf_tokenizer(text, add_special_tokens=False)["input_ids"]

    nodes = load_source_chunks()
    n_source = len(nodes)
    apply_metadata_mode(nodes, args.metadata)
    if args.split == "tokens":
        nodes = split_long_nodes(nodes, tokenize, args.max_tokens, args.overlap)

    embed_texts = [n.get_content(metadata_mode=MetadataMode.EMBED) for n in nodes]
    lengths = [len(tokenize(t)) for t in embed_texts]
    truncated = sum(length > args.max_tokens - 2 for length in lengths)
    lost = sum(max(0, length - (args.max_tokens - 2)) for length in lengths) / sum(lengths)
    print(f"{n_source} 個來源 chunk -> {len(nodes)} 個 chunk；超過 {args.max_tokens} token 會被截斷的: "
          f"{truncated} ({truncated / len(nodes):.0%})，被截掉的 token 佔 {lost:.0%}")

    # 依長度排序再分批，同一批長度接近，padding 少、CPU 上快很多
    order = sorted(range(len(nodes)), key=lambda i: lengths[i])
    t0 = time.perf_counter()
    for start in range(0, len(order), args.batch_size):
        batch = order[start:start + args.batch_size]
        vectors = embed_model.get_text_embedding_batch([embed_texts[i] for i in batch])
        for i, v in zip(batch, vectors):
            nodes[i].embedding = v
        done = start + len(batch)
        if done % (args.batch_size * 25) < args.batch_size or done == len(order):
            print(f"  embedding {done}/{len(order)}  ({time.perf_counter() - t0:.0f}s)", flush=True)
    embed_seconds = time.perf_counter() - t0

    out_dir.mkdir(parents=True)
    collection = chromadb.PersistentClient(path=str(out_dir)).get_or_create_collection(COLLECTION_NAME)
    store = ChromaVectorStore(chroma_collection=collection)
    for start in range(0, len(nodes), 500):
        store.add(nodes[start:start + 500])

    info = {
        "name": args.name,
        "embed_model": args.embed_model,
        "metadata": args.metadata,
        "split": args.split,
        "max_tokens": args.max_tokens,
        "bf16": args.bf16,
        "n_source_chunks": n_source,
        "n_chunks": len(nodes),
        "truncated_chunks": truncated,
        "truncated_token_ratio": round(lost, 3),
        "embed_seconds": round(embed_seconds, 1),
    }
    (out_dir / "build_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
