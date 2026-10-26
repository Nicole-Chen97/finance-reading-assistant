"""直接用整篇文章的文字斷詞，比較四種關鍵字抽取方法。

斷詞都用正式程式的繁體斷詞器（app/bm25_index.py：jieba 繁體大詞典 + 自訂詞典 + 台灣用字），
同一篇文章的 chunk 依原本順序接回整篇文字再處理。

四種方法對「什麼詞重要」的定義不同：
  TF-IDF（自己的語料）  分數 = 這個詞在這篇出現的次數（取 log）× 這個詞在 273 篇文章裡有多稀有（IDF）。
                        IDF 是從你自己的文章算出來的，所以「這個作者常用的詞」會被正確地視為不稀有
  word2vec              先用這批文章訓練詞向量，算出整篇文章的平均向量（代表整篇的「意思」），
                        再挑出向量最接近這個平均的詞 —— 最能代表整篇意思的詞，不管它稀不稀有
  jieba TF-IDF          jieba.analyse.extract_tags：同樣是 TF-IDF，但 IDF 用 jieba 內建的 IDF 表。
                        那份表是用大量一般簡體文本統計的：不認得的詞（大部分繁體詞）一律給中位數 IDF，
                        所以排名主要由出現次數決定。用來看「jieba 開箱即用」的效果
  jieba TextRank        jieba.analyse.textrank：不看稀有程度，把詞當成節點、在同一個視窗內一起出現就連一條邊，
                        像 Google PageRank 一樣算出哪些詞跟最多重要的詞連在一起。預設只保留名詞、動詞類

輸出（evals/rag/experiments/）：
  keywords_side_by_side.txt   每篇文章四種方法的前 10 個關鍵字並排（最適合比較）
  keywords_tfidf.txt / keywords_word2vec.txt / keywords_jieba_tfidf.txt / keywords_jieba_textrank.txt
                              各方法單獨，附分數

用法：
    cd Substack_post_download
    uv run python evals/rag/extract_keywords.py
"""
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

import chromadb  # noqa: E402
import jieba.analyse  # noqa: E402
import jieba.posseg  # noqa: E402
import numpy as np  # noqa: E402
from gensim.models import Word2Vec  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from bm25_index import STOPWORDS, _tokenizer, tokenize  # noqa: E402
from vector_store import COLLECTION_NAME, DB_PATH  # noqa: E402

OUT_DIR = Path(__file__).parent / "experiments"
TOP_N = 10
MIN_TOKEN_LEN = 2  # 單一字（「年」「股」）幾乎沒有關鍵字的意義，不列入候選


def load_articles():
    """回傳 [(author, post_date, title, full_text)]，同一篇文章的 chunk 依原本順序接回整篇。"""
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    chunks, info = defaultdict(list), {}
    for node_id, text, m in zip(data["ids"], data["documents"], data["metadatas"]):
        aid = m["article_id"]
        order = int(node_id.rsplit("chunk-", 1)[-1]) if "chunk-" in node_id else 0
        chunks[aid].append((order, text))
        info[aid] = (m.get("author", ""), str(m.get("post_date", ""))[:10], m.get("title", ""))
    articles = []
    for aid, parts in chunks.items():
        author, date, title = info[aid]
        articles.append((author, date, title, title + "\n" + "\n".join(t for _, t in sorted(parts))))
    articles.sort(key=lambda a: a[1], reverse=True)  # 新的文章在前面
    return articles


def is_candidate(token: str) -> bool:
    if len(token) < MIN_TOKEN_LEN or not any(ch.isalnum() for ch in token):  # 擋掉 ###、** 這類 Markdown 符號
        return False
    try:
        float(token)  # 純數字、價位（0.193、6.72）不當關鍵字
        return False
    except ValueError:
        return True


def tfidf_keywords(all_tokens):
    vectorizer = TfidfVectorizer(analyzer=lambda toks: [t for t in toks if is_candidate(t)], sublinear_tf=True)
    matrix = vectorizer.fit_transform(all_tokens)
    vocab = vectorizer.get_feature_names_out()
    result = []
    for row in matrix:
        row = row.tocoo()
        top = sorted(zip(row.col, row.data), key=lambda x: -x[1])[:TOP_N]
        result.append([(vocab[i], float(s)) for i, s in top])
    return result


def word2vec_keywords(all_tokens):
    model = Word2Vec(all_tokens, vector_size=100, window=5, min_count=2, sg=1, epochs=10, workers=2, seed=42)
    wv = model.wv
    result = []
    for toks in all_tokens:
        counts = Counter(t for t in toks if is_candidate(t) and t in wv)
        if not counts:
            result.append([])
            continue
        centroid = np.mean([wv[t] for t in toks if t in wv], axis=0)
        centroid /= np.linalg.norm(centroid)
        # 只看在這篇出現至少 2 次的詞，避免挑到只是剛好跟平均向量很像、但跟這篇無關的詞
        candidates = [t for t, c in counts.items() if c >= 2] or list(counts)
        scores = {t: float(np.dot(wv[t], centroid) / np.linalg.norm(wv[t])) for t in candidates}
        result.append(sorted(scores.items(), key=lambda x: -x[1])[:TOP_N])
    return result


def make_jieba_extractors():
    """讓 jieba.analyse 用跟正式程式同一個繁體斷詞器、同一份停用詞，差別只在打分數的方法。"""
    stop_file = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    stop_file.write("\n".join(STOPWORDS))
    stop_file.close()

    tfidf = jieba.analyse.TFIDF()
    tfidf.tokenizer = _tokenizer
    tfidf.set_stop_words(stop_file.name)

    textrank = jieba.analyse.TextRank()
    textrank.tokenizer = textrank.postokenizer = jieba.posseg.POSTokenizer(_tokenizer)
    textrank.set_stop_words(stop_file.name)
    return tfidf, textrank


def jieba_keywords(extractor, texts, **kwargs):
    result = []
    for text in texts:
        pairs = extractor.extract_tags(text.lower(), topK=TOP_N * 3, withWeight=True, **kwargs)
        result.append([(w, float(s)) for w, s in pairs if is_candidate(w)][:TOP_N])
    return result


def fmt_words(pairs, with_score=False):
    if with_score:
        return "  ".join(f"{w}({s:.2f})" for w, s in pairs)
    return "、".join(w for w, _ in pairs)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    print("讀取文章、斷詞…", flush=True)
    articles = load_articles()
    texts = [a[3] for a in articles]
    all_tokens = [tokenize(t) for t in texts]
    print(f"{len(articles)} 篇文章；計算四種方法…", flush=True)

    jieba_tfidf, jieba_textrank = make_jieba_extractors()
    methods = {
        "tfidf": ("TF-IDF（自己的語料）", tfidf_keywords(all_tokens)),
        "word2vec": ("word2vec", word2vec_keywords(all_tokens)),
        "jieba_tfidf": ("jieba TF-IDF（內建 IDF）", jieba_keywords(jieba_tfidf, texts)),
        "jieba_textrank": ("jieba TextRank", jieba_keywords(jieba_textrank, texts)),
    }

    header = [
        f"每篇文章的前 {TOP_N} 個關鍵字（{len(articles)} 篇，新的在前）",
        "斷詞：jieba 繁體大詞典 + 自訂詞典 + 台灣用字；單一字和純數字不列入。",
        "TF-IDF：這篇常出現、其他篇少出現的詞（IDF 用自己的文章算）",
        "word2vec：最接近整篇平均意思的詞",
        "jieba TF-IDF：jieba 內建的 IDF 表（一般簡體文本統計），繁體詞大多查不到，排名主要看出現次數",
        "jieba TextRank：跟最多重要的詞一起出現的詞（只看名詞、動詞類）",
        "=" * 80,
    ]

    side = list(header)
    for i, (author, date, title, _) in enumerate(articles):
        side += ["", f"【{date}｜{author}】{title}"]
        for label, keywords in methods.values():
            side.append(f"  {label:18s}：{fmt_words(keywords[i])}")
    (OUT_DIR / "keywords_side_by_side.txt").write_text("\n".join(side) + "\n", encoding="utf-8")

    for name, (label, keywords) in methods.items():
        lines = [f"方法：{label}"] + header[1:]
        for i, (author, date, title, _) in enumerate(articles):
            lines += ["", f"【{date}｜{author}】{title}", f"  {fmt_words(keywords[i], with_score=True)}"]
        (OUT_DIR / f"keywords_{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 上一版（簡體斷詞）留下的 BM25 關鍵字檔已經過時，刪掉避免混淆
    (OUT_DIR / "keywords_bm25.txt").unlink(missing_ok=True)
    print(f"完成，結果在 {OUT_DIR}/keywords_*.txt")


if __name__ == "__main__":
    main()
