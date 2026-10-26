"""半自動生成 RAG 檢索面 golden set 候選題目。

做法：從已經 index 進 Chroma 的真實 chunk 裡抽樣（分作者、避免抽到太短的雜訊 chunk），
拿抽到的內容請 LLM 反推「使用者可能會怎麼問，才會需要用到這段內容來回答」，
產生 (query, 應該撈到哪篇文章) 的候選配對。

⚠️ 這份輸出是「候選」，不是最終 golden set——LLM 反推出來的問題品質不保證，
需要人工看過、修掉不自然或太簡單的題目，再存成 golden_set.jsonl 給 eval_rag_retrieval.py 用。

用法：
    cd Substack_post_download
    uv run python evals/rag/generate_golden_set_candidates.py --n-per-author 15
"""
import argparse
import json
import random
import sqlite3
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

import os  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from authors import AUTHOR_NAMES  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

from langchain_anthropic import ChatAnthropic  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", str(APP_DIR.parent / "data" / "chroma_db"))
CHROMA_SQLITE_FILE = str(Path(CHROMA_DB_PATH) / "chroma.sqlite3")
MIN_CHUNK_CHARS = 150  # 避免抽到太短、資訊量不足以反推出好問題的 chunk


class GeneratedQuery(BaseModel):
    query: str = Field(description="一個真實使用者可能會問的自然問題，繁體中文，語氣像日常提問，不要照抄原文用詞")


gen_llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.7,
    api_key=os.environ["ANTHROPIC_API_KEY"].strip(),
).with_structured_output(GeneratedQuery)

SYSTEM_PROMPT = """你是在幫忙製作 RAG 系統的評估題庫。
給你一段財經 newsletter 的內容片段，請反推「使用者可能會問什麼問題，才會需要檢索到這段內容來回答」。

規則：
1. 問題要像真實使用者會打的話，不要照抄原文的句子或用詞。
2. 大約一半的問題可以自然地帶入作者名字或時間線索（例如「某作者對...的看法」「最近...」），
   另一半就是單純問主題本身，不提作者或時間——兩種情境系統都要測到。
3. 只問這段內容本身能回答的具體問題，不要問到原文沒提到的東西。
4. 問題長度跟真人打字差不多，一兩句話，不要寫成論文標題。"""


def fetch_chunks() -> list[dict]:
    conn = sqlite3.connect(CHROMA_SQLITE_FILE)
    cur = conn.cursor()
    cur.execute("""
        SELECT e.id, m1.string_value as title, m2.string_value as author,
               m3.string_value as post_date, m4.int_value as article_id,
               m5.string_value as doc
        FROM embeddings e
        JOIN embedding_metadata m1 ON e.id=m1.id AND m1.key='title'
        JOIN embedding_metadata m2 ON e.id=m2.id AND m2.key='author'
        JOIN embedding_metadata m3 ON e.id=m3.id AND m3.key='post_date'
        JOIN embedding_metadata m4 ON e.id=m4.id AND m4.key='article_id'
        JOIN embedding_metadata m5 ON e.id=m5.id AND m5.key='chroma:document'
    """)
    rows = cur.fetchall()
    conn.close()
    return [
        {"chunk_id": r[0], "title": r[1], "author": r[2], "post_date": r[3],
         "article_id": r[4], "text": r[5]}
        for r in rows
    ]


def pick_best_chunk_per_article(chunks: list[dict]) -> dict[int, dict]:
    """每篇文章挑一個「最長」的 chunk 代表——內容量夠、比較不會是雜訊/免責聲明片段。"""
    best: dict[int, dict] = {}
    for c in chunks:
        if len(c["text"]) < MIN_CHUNK_CHARS:
            continue
        cur_best = best.get(c["article_id"])
        if cur_best is None or len(c["text"]) > len(cur_best["text"]):
            best[c["article_id"]] = c
    return best


def generate_query(chunk: dict) -> str:
    human = f"標題：{chunk['title']}\n\n內容片段：\n{chunk['text'][:1200]}"
    result = gen_llm.invoke([("system", SYSTEM_PROMPT), ("human", human)])
    return result.query


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-per-author", type=int, default=15, help="每位作者抽幾篇文章出題")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path(__file__).parent / "golden_set_candidates.jsonl")
    args = parser.parse_args()

    random.seed(args.seed)

    chunks = fetch_chunks()
    best_by_article = pick_best_chunk_per_article(chunks)

    by_author: dict[str, list[dict]] = {}
    for c in best_by_article.values():
        by_author.setdefault(c["author"], []).append(c)

    sampled = [
        chunk
        for author in AUTHOR_NAMES
        for chunk in random.sample(by_author.get(author, []), min(args.n_per_author, len(by_author.get(author, []))))
    ]

    candidates = []
    for i, chunk in enumerate(sampled, 1):
        query = generate_query(chunk)
        candidate = {
            "id": f"rag-{i:02d}",
            "query": query,
            "expected_author": chunk["author"],
            "expected_title": chunk["title"],
            "expected_post_date": chunk["post_date"],
            "expected_article_id": chunk["article_id"],
            "source_chunk_preview": chunk["text"][:200].replace("\n", " "),
        }
        candidates.append(candidate)
        print(f"[{i}/{len(sampled)}] {chunk['author']:12s} | {query}")

    with open(args.out, "w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"\n已寫出 {len(candidates)} 題候選到 {args.out}")
    print("下一步：人工看過每一題，刪掉不自然/太簡單/答非所問的，再存成 golden_set.jsonl")


if __name__ == "__main__":
    main()
