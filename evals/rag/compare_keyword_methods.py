"""比較三種「關鍵字 / 詞彙層級」的檢索方法：BM25、TF-IDF、word2vec。

三種方法都用同一套斷詞（app/bm25_index.tokenize：繁轉簡 → 小寫 → jieba → 去停用詞），
同一份題庫、同樣以「文章」計分，所以差別只來自方法本身：

  BM25      詞有沒有出現、出現幾次（次數多到一定程度就不再加分）、這個詞稀不稀有、文章長短做校正
            → 正式流程用的方法（app/bm25_index.py）
  TF-IDF    詞出現次數 × 稀有程度，轉成向量後算 cosine 相似度。跟 BM25 很像，但次數不會飽和、長度校正不同
  word2vec  先用這批文章訓練詞向量（意思相近的詞向量也相近），文章向量 = 文章裡所有詞向量的平均，
            問題向量也一樣，再算 cosine 相似度。不需要同一個詞出現，「意思接近」就能撈到

輸出（evals/rag/experiments/）：
  keyword_summary.txt    三種方法的分數總表
  keyword_bm25.txt       每一題：斷詞結果、正確文章排第幾、前 5 名撈到什麼
  keyword_tfidf.txt      同上，另外列出問題裡每個詞的 IDF（越高代表越稀有、越有鑑別力）
  keyword_word2vec.txt   同上，另外列出問題裡每個詞「最相近的詞」，以及模型沒見過的詞

用法：
    cd Substack_post_download
    uv run python evals/rag/compare_keyword_methods.py
"""
import json
import sys
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

import chromadb  # noqa: E402
import numpy as np  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from bm25_index import tokenize  # noqa: E402
from vector_store import COLLECTION_NAME, DB_PATH  # noqa: E402

OUT_DIR = Path(__file__).parent / "experiments"
GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
SHOW_TOP = 5
CHUNK_POOL = 100  # 先取前 100 個 chunk，再去重成文章排名


def load_corpus():
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    metas = data["metadatas"]
    tokens = [
        tokenize(f"{m.get('title', '')} {m.get('header_path', '')} {text}")
        for text, m in zip(data["documents"], metas)
    ]
    return metas, tokens


def to_article_ranking(chunk_scores: np.ndarray, metas: list[dict]) -> list[tuple]:
    """chunk 分數 → 文章排名（同一篇文章取最高分的 chunk），回傳 [(article_id, title, score), ...]"""
    order = np.argsort(-chunk_scores)[:CHUNK_POOL]
    seen, ranking = set(), []
    for i in order:
        if chunk_scores[i] <= 0:
            break
        aid = metas[i]["article_id"]
        if aid in seen:
            continue
        seen.add(aid)
        ranking.append((aid, metas[i].get("title", ""), float(chunk_scores[i])))
    return ranking


class BM25Method:
    name = "bm25"

    def __init__(self, tokens):
        self.bm25 = BM25Okapi(tokens)

    def scores(self, query_tokens):
        return self.bm25.get_scores(query_tokens)

    def explain(self, query_tokens):
        return []


class TfidfMethod:
    name = "tfidf"

    def __init__(self, tokens):
        # 已經斷好詞了，所以 analyzer 直接回傳 token 清單，不讓 sklearn 再切一次
        self.vectorizer = TfidfVectorizer(analyzer=lambda toks: toks, sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(tokens)  # 每列已經 L2 正規化
        self.idf = dict(zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_))

    def scores(self, query_tokens):
        q = self.vectorizer.transform([query_tokens])
        return (self.matrix @ q.T).toarray().ravel()

    def explain(self, query_tokens):
        parts = [f"{t}({self.idf[t]:.1f})" if t in self.idf else f"{t}(沒出現過)" for t in query_tokens]
        return ["詞的 IDF（越高越稀有）：" + "  ".join(parts)]


class Word2VecMethod:
    name = "word2vec"

    def __init__(self, tokens):
        t0 = time.perf_counter()
        # skip-gram（sg=1）對小語料、罕見詞比 CBOW 好；min_count=2 去掉只出現一次的雜訊詞
        self.model = Word2Vec(tokens, vector_size=100, window=5, min_count=2, sg=1, epochs=10, workers=2, seed=42)
        self.train_seconds = time.perf_counter() - t0
        doc_vectors = np.vstack([self._mean_vector(t) for t in tokens])
        norms = np.linalg.norm(doc_vectors, axis=1, keepdims=True)
        self.doc_vectors = doc_vectors / np.where(norms == 0, 1, norms)

    def _mean_vector(self, toks):
        vecs = [self.model.wv[t] for t in toks if t in self.model.wv]
        return np.mean(vecs, axis=0) if vecs else np.zeros(self.model.vector_size)

    def scores(self, query_tokens):
        q = self._mean_vector(query_tokens)
        n = np.linalg.norm(q)
        if n == 0:
            return np.zeros(len(self.doc_vectors))
        return self.doc_vectors @ (q / n)

    def explain(self, query_tokens):
        lines, oov = [], []
        for t in query_tokens:
            if t in self.model.wv:
                similar = ", ".join(w for w, _ in self.model.wv.most_similar(t, topn=4))
                lines.append(f"  {t} → {similar}")
            else:
                oov.append(t)
        out = ["每個詞最相近的詞："] + lines
        if oov:
            out.append(f"模型沒見過的詞（忽略）：{', '.join(oov)}")
        return out


def evaluate(method, cases, metas):
    rows = []
    for case in cases:
        q_tokens = tokenize(case["query"])
        ranking = to_article_ranking(method.scores(q_tokens), metas)
        ids = [a for a, _, _ in ranking]
        rank = ids.index(case["expected_article_id"]) + 1 if case["expected_article_id"] in ids else None
        rows.append({"case": case, "tokens": q_tokens, "ranking": ranking, "rank": rank,
                     "explain": method.explain(q_tokens)})
    return rows


def metrics(rows):
    core = [r for r in rows if not r["case"]["ambiguous"]]
    n = len(core)

    def hit(k):
        return sum(r["rank"] is not None and r["rank"] <= k for r in core) / n

    mrr = sum(1 / r["rank"] for r in core if r["rank"]) / n
    return {"n": n, "hit@1": hit(1), "hit@3": hit(3), "hit@10": hit(10), "mrr": mrr}


def write_detail(method, rows, path):
    m = metrics(rows)
    lines = [
        f"方法：{method.name}",
        f"核心題 {m['n']} 題：Hit@1 {m['hit@1']:.0%}  Hit@3 {m['hit@3']:.0%}  Hit@10 {m['hit@10']:.0%}  MRR {m['mrr']:.2f}",
        "標記：✓ = 正確文章　[邊界題] = 多篇都答得出來，不計分",
        "=" * 80,
    ]
    for r in rows:
        c = r["case"]
        tag = " [邊界題]" if c["ambiguous"] else ""
        status = f"第 {r['rank']} 名" if r["rank"] else "前 10 名沒撈到"
        lines += [
            "",
            f"[{c['id']}]{tag} {c['query']}",
            f"斷詞：{' / '.join(r['tokens'])}",
            *r["explain"],
            f"正確文章：{c['expected_title']}　→ {status}",
            f"前 {SHOW_TOP} 名：",
        ]
        for i, (aid, title, score) in enumerate(r["ranking"][:SHOW_TOP], 1):
            mark = "✓" if aid == c["expected_article_id"] else " "
            lines.append(f"  {mark} {i}. ({score:.3f}) {title}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    OUT_DIR.mkdir(exist_ok=True)
    cases = [json.loads(line) for line in GOLDEN_SET.open(encoding="utf-8") if line.strip()]
    print("讀取並斷詞全部 chunk…", flush=True)
    metas, tokens = load_corpus()

    methods = []
    for cls in (BM25Method, TfidfMethod, Word2VecMethod):
        t0 = time.perf_counter()
        methods.append(cls(tokens))
        print(f"  {cls.name} 建好索引 ({time.perf_counter() - t0:.1f}s)", flush=True)

    summary = [
        "三種關鍵字 / 詞彙層級檢索方法比較",
        f"語料：{len(tokens)} 個 chunk（正式資料庫）；斷詞：app/bm25_index.tokenize",
        f"題庫：{GOLDEN_SET.name}，只算核心題；以「文章」計分；都不 rerank",
        "",
        f"{'方法':10s} {'Hit@1':>6s} {'Hit@3':>6s} {'Hit@10':>7s} {'MRR':>6s}",
    ]
    per_method_rows = {}
    for method in methods:
        rows = evaluate(method, cases, metas)
        per_method_rows[method.name] = rows
        write_detail(method, rows, OUT_DIR / f"keyword_{method.name}.txt")
        m = metrics(rows)
        summary.append(f"{method.name:10s} {m['hit@1']:6.0%} {m['hit@3']:6.0%} {m['hit@10']:7.0%} {m['mrr']:6.2f}")

    summary += ["", "每一題的名次（核心題；- = 前 10 名沒撈到）：", f"{'題目':8s} {'bm25':>5s} {'tfidf':>6s} {'w2v':>5s}  問題"]
    for i, case in enumerate(cases):
        if case["ambiguous"]:
            continue
        ranks = [per_method_rows[m.name][i]["rank"] for m in methods]
        cells = [f"{r if r and r <= 10 else '-':>5}" for r in ranks]
        summary.append(f"{case['id']:8s} {cells[0]} {cells[1]:>6s} {cells[2]}  {case['query']}")

    (OUT_DIR / "keyword_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print("\n".join(summary[:10]))
    print(f"\n結果檔在 {OUT_DIR}/keyword_*.txt")


if __name__ == "__main__":
    main()
