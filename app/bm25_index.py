"""BM25 關鍵字檢索：hybrid 檢索的「關鍵字」那一半。

為什麼需要：向量檢索比的是「意思像不像」，擅長換句話說，但容易漏掉專有名詞和代號
（柯立芝、ONDO、USDJPY）。BM25 比的是「字有沒有出現、這個字稀不稀有」，剛好互補。
Chroma 本身不支援關鍵字檢索（傳 HYBRID 模式給它會被默默當成一般向量檢索），所以在 Chroma 外面自己建。

中文要先斷詞才能做 BM25，這裡的處理順序：
  1. 轉小寫：讓 ETH / eth 是同一個詞
  2. jieba 斷詞，用官方的繁體大詞典 jieba_dict.txt.big（文章幾乎全是繁體，不需要轉簡體；
     比較過「轉簡體 + 預設詞典」，分數不輸，見 evals/rag/compare_tokenizers.py），再載入兩份自訂詞典：
       bm25_userdict.txt     財經專有名詞（創世使命、斐波那契……），避免被切成「創世 / 使命」。
                             只寫詞、不寫詞頻，讓 jieba 自己給一個保證切得出來的詞頻。
                             同一個詞有兩種寫法的都要放（數據中心 / 資料中心、斐波那契 / 費波那契）
       bm25_userdict_tw.txt  台灣用字。大詞典只收「臺灣」，沒有「台灣」，「怎麼看台灣」會被切成「看台 / 灣的」；
                             要給跟「臺灣」一樣的詞頻（8683）才贏得過「看台」
  3. 去掉停用詞和標點：「的」「是」「什麼」每篇都有，對檢索沒幫助
建索引和查詢都走同一個 tokenize()，改了詞典或停用詞之後索引會在下次查詢時重建（見 get_bm25_index）。

索引放在記憶體裡，從 Chroma 讀出全部 chunk 建立（約 3,500 個 chunk 要 10 秒）。
用 collection.count() 偵測資料庫有沒有新增文章，有的話下次查詢時自動重建，
避免 download_node 下載新文章之後，新文章只有向量檢索找得到。
"""
import logging
import re
from pathlib import Path
from typing import Optional

import jieba
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import FilterCondition, FilterOperator, MetadataFilters
from llama_index.vector_stores.chroma.base import ChromaVectorStore
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

jieba.setLogLevel(logging.WARNING)
_HERE = Path(__file__).parent
_tokenizer = jieba.Tokenizer(dictionary=str(_HERE / "jieba_dict.txt.big"))
_tokenizer.load_userdict(str(_HERE / "bm25_userdict.txt"))
_tokenizer.load_userdict(str(_HERE / "bm25_userdict_tw.txt"))

STOPWORDS = set(
    "的 了 是 在 和 與 及 或 也 都 就 而 但 對 把 被 讓 從 到 為 以 於 之 其 這 那 這個 那個 這些 那些 "
    "什麼 怎麼 怎麼樣 為什麼 如何 哪些 哪個 嗎 呢 吧 啊 有 沒有 會 能 可以 要 我 你 他 她 它 我們 你們 他們 "
    "一個 一些 現在 目前 最近 之前 關於 請 幫 幫我 一下 看法 覺得 認為 說".split()
)
_PUNCT = re.compile(r"^[\W_]+$", re.UNICODE)


def tokenize(text: str) -> list[str]:
    tokens = _tokenizer.lcut(text.lower())
    return [t for t in tokens if t.strip() and t not in STOPWORDS and not _PUNCT.match(t)]


def _match(meta: dict, f) -> bool:
    """在 Python 這邊重現 Chroma 的 metadata filter，讓 BM25 和向量檢索套用同樣的作者/日期條件。"""
    value = meta.get(f.key)
    if value is None:
        return False
    op = f.operator
    if op == FilterOperator.EQ:
        return value == f.value
    if op == FilterOperator.NE:
        return value != f.value
    if op == FilterOperator.IN:
        return value in f.value
    if op == FilterOperator.NIN:
        return value not in f.value
    comparisons = {
        FilterOperator.GT: lambda a, b: a > b,
        FilterOperator.GTE: lambda a, b: a >= b,
        FilterOperator.LT: lambda a, b: a < b,
        FilterOperator.LTE: lambda a, b: a <= b,
    }
    if op in comparisons:
        try:
            return comparisons[op](float(value), float(f.value))
        except (TypeError, ValueError):
            return False
    logger.warning("[bm25] 不支援的 filter 運算子 %s，忽略這條條件", op)
    return True


def matches_filters(meta: dict, filters: Optional[MetadataFilters]) -> bool:
    if not filters or not filters.filters:
        return True
    results = [_match(meta, f) for f in filters.filters]
    return any(results) if filters.condition == FilterCondition.OR else all(results)


class BM25Index:
    def __init__(self, vector_store: ChromaVectorStore):
        collection = vector_store._collection
        data = collection.get(include=["documents", "metadatas"])
        self.doc_count = collection.count()
        self.nodes = [
            metadata_dict_to_node(meta, text=text)
            for text, meta in zip(data["documents"], data["metadatas"])
        ]
        self.metadatas = data["metadatas"]
        # 標題和章節路徑也一起建索引：專有名詞常常出現在標題（例如「AI 賣水人 AXTI 基本面與籌碼面全解析」）
        corpus = [
            tokenize(f"{m.get('title', '')} {m.get('header_path', '')} {text}")
            for text, m in zip(data["documents"], data["metadatas"])
        ]
        self.bm25 = BM25Okapi(corpus)
        logger.info("[bm25] 已建立索引：%d 個 chunk", len(corpus))

    def search(self, query: str, top_k: int = 10, filters: Optional[MetadataFilters] = None) -> list[NodeWithScore]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        candidates = [
            i for i in range(len(self.nodes))
            if scores[i] > 0 and matches_filters(self.metadatas[i], filters)
        ]
        candidates.sort(key=lambda i: scores[i], reverse=True)
        return [NodeWithScore(node=self.nodes[i], score=float(scores[i])) for i in candidates[:top_k]]


_index_cache: dict[str, BM25Index] = {}


def get_bm25_index(vector_store: ChromaVectorStore) -> BM25Index:
    """同一個資料庫只建一次索引；資料庫的 chunk 數變了（有新文章）就重建。
    用 collection 的 UUID 當 key：正式資料庫和實驗資料庫的 collection 名稱一樣，但 UUID 不同。"""
    collection = vector_store._collection
    key = str(collection.id)
    cached = _index_cache.get(key)
    if cached is None or cached.doc_count != collection.count():
        cached = BM25Index(vector_store)
        _index_cache[key] = cached
    return cached
