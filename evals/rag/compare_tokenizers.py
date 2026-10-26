"""比較不同的中文斷詞設定（詞典）對 BM25 檢索的影響。

詞典不是加進 BM25，而是加進斷詞器：詞典決定「詞怎麼切」，BM25 只拿切好的詞去算分數。
切錯（例如「創世使命」被切成「創世 / 使命」），BM25 就比對不到。

比較的設定（都用 jieba、轉小寫、同一份停用詞，只改「要不要繁轉簡」和「用哪些詞典」）：
  A_t2s_default      現況：繁轉簡 → jieba 預設詞典（簡體為主）+ 自訂詞典 app/bm25_userdict.txt
  B_t2s_thuocl       A + 清華 THUOCL 財經詞庫（3,830 詞，簡體，MIT 授權）
  C_trad_big         不轉簡體，直接用 jieba 官方繁體詞典 dict.txt.big + 自訂詞典（轉成繁體）
  D_trad_big_thuocl  C + THUOCL 財經詞庫（轉成繁體）

需要先把詞典放在 data/dicts/：
  THUOCL_caijing.txt  https://github.com/thunlp/THUOCL/blob/master/data/THUOCL_caijing.txt
  dict.txt.big        https://github.com/fxsjy/jieba/blob/master/extra_dict/dict.txt.big

輸出：evals/rag/experiments/tokenizer_compare.txt（分數 + 幾個難切的詞實際被切成什麼）

用法：
    cd Substack_post_download
    uv run python evals/rag/compare_tokenizers.py
"""
import json
import sys
import tempfile
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

from dotenv import load_dotenv  # noqa: E402
load_dotenv(APP_DIR.parent / ".env")

import chromadb  # noqa: E402
import jieba  # noqa: E402
import numpy as np  # noqa: E402
import opencc  # noqa: E402
from rank_bm25 import BM25Okapi  # noqa: E402

from bm25_index import STOPWORDS, _PUNCT  # noqa: E402
from vector_store import COLLECTION_NAME, DB_PATH  # noqa: E402

DICT_DIR = APP_DIR.parent / "data" / "dicts"
USER_DICT = APP_DIR / "bm25_userdict.txt"
GOLDEN_SET = Path(__file__).parent / "golden_set.jsonl"
OUT = Path(__file__).parent / "experiments" / "tokenizer_compare.txt"

# 看看這些詞在各設定下被切成什麼
PROBE_SENTENCES = [
    "怎麼看台灣的國防產業化？",
    "創世使命系列文章總共有幾篇？",
    "用斐波那契回撤找支撐，形成雙底與圓弧底",
    "柯立芝時代的股市表現怎麼樣？",
    "AXTI 的籌碼面與基本面全解析",
    "低軌衛星公司的估值與護城河",
    "穩定幣被禁止生息，機構卻維持看漲比特幣",
    "降息循環下的山寨幣輪動",
]

t2s = opencc.OpenCC("t2s")
s2t = opencc.OpenCC("s2tw")  # 台灣標準：台（不是臺）、裡（不是裏）


def write_word_list(words, convert=None) -> str:
    """寫成 jieba 自訂詞典格式（只放詞、不放詞頻，讓 jieba 自己決定一個能被切出來的頻率）"""
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    for w in words:
        f.write((convert(w) if convert else w) + "\n")
    f.close()
    return f.name


def read_userdict():
    return [line.split()[0] for line in USER_DICT.read_text(encoding="utf-8").splitlines() if line.strip()]


def read_thuocl():
    return [line.split()[0] for line in (DICT_DIR / "THUOCL_caijing.txt").read_text(encoding="utf-8").splitlines() if line.strip()]


def make_tokenizer(trad: bool, thuocl: bool, tw_words: bool = False):
    tk = jieba.Tokenizer(dictionary=str(DICT_DIR / "dict.txt.big")) if trad else jieba.Tokenizer()
    # 自訂詞典和停用詞現在是繁體（跟正式程式一致）；簡體設定要先轉成簡體，THUOCL 是簡體、繁體設定要轉成繁體
    tk.load_userdict(write_word_list(read_userdict(), None if trad else t2s.convert))
    if thuocl:
        tk.load_userdict(write_word_list(read_thuocl(), s2t.convert if trad else None))
    if tw_words:
        # 只加進詞典不夠：「看台」在大詞典裡詞頻比較高，「怎麼看台灣」還是會被切成「看台 / 灣」。
        # suggest_freq(tune=True) 實測沒用（還是被切成「看台 / 灣的」），
        # 改成直接沿用大詞典裡「臺」字版本的詞頻：台灣 ← 臺灣（8,683）
        tk.initialize()
        for w in TW_WORDS:
            tk.add_word(w, freq=tk.FREQ.get(w.replace("台", "臺")) or 10000)
    stop = STOPWORDS if trad else {t2s.convert(w) for w in STOPWORDS}

    def tokenize(text: str) -> list[str]:
        text = text.lower() if trad else t2s.convert(text).lower()
        return [t for t in tk.lcut(text) if t.strip() and t not in stop and not _PUNCT.match(t)]

    return tokenize


CONFIGS = {
    "A_t2s_default": dict(trad=False, thuocl=False),
    "B_t2s_thuocl": dict(trad=False, thuocl=True),
    "C_trad_big": dict(trad=True, thuocl=False),
    "D_trad_big_thuocl": dict(trad=True, thuocl=True),
    "E_trad_big_tw": dict(trad=True, thuocl=False, tw_words=True),
}

# dict.txt.big 用的是「臺灣」這類標準繁體，台灣常用的寫法要另外補進去
TW_WORDS = ["台灣", "台北", "台積電", "台股", "台幣", "台海"]


def evaluate(tokenize, texts, metas, cases):
    t0 = time.perf_counter()
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    build = time.perf_counter() - t0
    ranks = []
    for case in cases:
        scores = bm25.get_scores(tokenize(case["query"]))
        seen, ranking = set(), []
        for i in np.argsort(-scores)[:100]:
            aid = metas[i]["article_id"]
            if aid not in seen:
                seen.add(aid)
                ranking.append(aid)
        exp = case["expected_article_id"]
        ranks.append(ranking.index(exp) + 1 if exp in ranking[:10] else None)
    n = len(ranks)
    return {
        "hit@1": sum(r == 1 for r in ranks) / n,
        "hit@3": sum(r is not None and r <= 3 for r in ranks) / n,
        "hit@10": sum(r is not None for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks if r) / n,
        "build_s": build,
        "ranks": ranks,
    }


def main():
    jieba.setLogLevel(60)
    collection = chromadb.PersistentClient(path=DB_PATH).get_collection(COLLECTION_NAME)
    data = collection.get(include=["documents", "metadatas"])
    metas = data["metadatas"]
    texts = [f"{m.get('title', '')} {m.get('header_path', '')} {d}" for d, m in zip(data["documents"], metas)]
    cases = [c for c in map(json.loads, GOLDEN_SET.open(encoding="utf-8")) if not c["ambiguous"]]

    results, tokenizers = {}, {}
    for name, cfg in CONFIGS.items():
        print(f"{name} …", flush=True)
        tokenizers[name] = make_tokenizer(**cfg)
        results[name] = evaluate(tokenizers[name], texts, metas, cases)

    lines = [
        "中文斷詞設定對 BM25 檢索的影響（只算核心題，以文章計分，不 rerank）",
        f"語料 {len(texts)} 個 chunk；核心題 {len(cases)} 題",
        "",
        f"{'設定':20s} {'Hit@1':>6s} {'Hit@3':>6s} {'Hit@10':>7s} {'MRR':>6s} {'建索引(s)':>10s}",
    ]
    for name, r in results.items():
        lines.append(f"{name:20s} {r['hit@1']:6.0%} {r['hit@3']:6.0%} {r['hit@10']:7.0%} {r['mrr']:6.2f} {r['build_s']:10.1f}")

    lines += ["", "名次有變化的題目（- = 前 10 名沒撈到）：", f"{'題目':8s} " + " ".join(f"{n[:1]:>3s}" for n in CONFIGS) + "  問題"]
    for i, case in enumerate(cases):
        ranks = [results[n]["ranks"][i] for n in CONFIGS]
        if len(set(ranks)) > 1:
            cells = " ".join(f"{r if r else '-':>3}" for r in ranks)
            lines.append(f"{case['id']:8s} {cells}  {case['query']}")

    lines += ["", "難切的詞實際被切成什麼："]
    for sentence in PROBE_SENTENCES:
        lines.append(f"\n  {sentence}")
        for name, tokenize in tokenizers.items():
            lines.append(f"    {name:20s} {' / '.join(tokenize(sentence))}")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
