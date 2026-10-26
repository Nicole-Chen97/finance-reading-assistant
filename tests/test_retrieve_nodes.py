"""
測試 tool.py 裡的 retrieve_nodes（RAG 檢索工具）。

retrieve_nodes 內部會做三件「很貴」的事：
  1. 連線真的 Chroma 向量資料庫 (v_service.vector_store)
  2. 用 HuggingFace embedding 模型算向量 (v_service.embed_model)
  3. 用 cross-encoder reranker 模型重新排序 (SentenceTransformerRerank)

我們不想在測試裡真的跑這三件事（會很慢、還會動到正式資料庫的資料）。
所以策略是：把這三個「外部依賴」換成假的（monkeypatch），
只測 retrieve_nodes 自己的『邏輯』——例如：
  - 到底是用 filtered_query 還是 user_query 去搜尋？
  - 撈到空結果時，有沒有正確觸發 fallback（拿掉 filters 重查一次）？
  - reranker 排完是空的時候，有沒有正確退回原始結果？

retrieve_nodes 現在保證回傳固定信封 {"success", "data", "error"}，
data 裡是單純的 dict（不是 llama_index 的 NodeWithScore 物件），而且永遠不會拋例外。
"""

import tool as tool_module
from tool import retrieve_nodes
from llama_index.core.schema import NodeWithScore, TextNode


def make_node(node_id: str, text: str = "content", score: float = 0.9) -> NodeWithScore:
    return NodeWithScore(node=TextNode(id_=node_id, text=text), score=score)


def expected_chunk(node_with_score: NodeWithScore) -> dict:
    """retrieve_nodes 現在會把每個 NodeWithScore 攤平成這種單純的 dict"""
    node = node_with_score.node
    return {
        "author": node.metadata.get("author", "Unknown"),
        "title": node.metadata.get("title", "No title"),
        "post_date": node.metadata.get("post_date", "Unknown date"),
        "header_path": node.metadata.get("header_path", "No path"),
        "text": node.text,
        "score": node_with_score.score,
    }


class FakeRetriever:
    """假裝是 index.as_retriever(...) 吐出來的檢索器"""

    def __init__(self, results):
        self.results = results
        self.received_query_bundle = None

    def retrieve(self, query_bundle):
        self.received_query_bundle = query_bundle
        return self.results


class FakeIndex:
    """假裝是 VectorStoreIndex.from_vector_store(...) 吐出來的 index"""

    def __init__(self, primary_results, fallback_results=None):
        self.primary_results = primary_results
        self.fallback_results = fallback_results or []
        self.retrievers = []  # 依呼叫順序記錄每一個被建立出來的 FakeRetriever

    def as_retriever(self, vector_store_query_mode=None, filters=None, similarity_top_k=None):
        # 第一次呼叫是第一階段檢索，之後的呼叫只會來自 fallback 分支
        is_fallback = len(self.retrievers) > 0
        retriever = FakeRetriever(self.fallback_results if is_fallback else self.primary_results)
        retriever.filters = filters
        retriever.similarity_top_k = similarity_top_k
        self.retrievers.append(retriever)
        return retriever


class FakeBM25:
    """假裝是 bm25_index.get_bm25_index(...) 吐出來的 BM25 索引"""

    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k=10, filters=None):
        self.calls.append({"query": query, "filters": filters})
        return self.results


def make_fake_index_class(fake_index):
    class FakeVectorStoreIndex:
        @staticmethod
        def from_vector_store(vector_store=None, embed_model=None):
            return fake_index

    return FakeVectorStoreIndex


def make_fake_reranker_class(result):
    class FakeReranker:
        last_instance = None

        def __init__(self, model=None, top_n=None):
            self.model = model
            self.top_n = top_n
            self.received_nodes = None
            self.received_query_bundle = None
            FakeReranker.last_instance = self

        def postprocess_nodes(self, nodes, query_bundle=None):
            self.received_nodes = nodes
            self.received_query_bundle = query_bundle
            return result

    return FakeReranker


class FakeVService:
    vector_store = "fake-vector-store"
    embed_model = "fake-embed-model"


def setup_fakes(monkeypatch, fake_index, reranker_result):
    monkeypatch.setattr(tool_module, "v_service", FakeVService())
    monkeypatch.setattr(tool_module, "VectorStoreIndex", make_fake_index_class(fake_index))
    fake_reranker_cls = make_fake_reranker_class(reranker_result)
    monkeypatch.setattr(tool_module, "SentenceTransformerRerank", fake_reranker_cls)
    # reranker 會被快取在模組裡，每個測試都要清掉，才會用到這次的假 reranker
    monkeypatch.setattr(tool_module, "_reranker", None)
    # BM25 那一路預設回傳空的，這樣既有測試只看向量那一路；hybrid 的合併另外測
    monkeypatch.setattr(tool_module, "get_bm25_index", lambda vector_store: FakeBM25([]))
    return fake_reranker_cls


# ---------- 第一階段檢索：filtered_query vs user_query ----------

def test_uses_filtered_query_when_provided(monkeypatch):
    node = make_node("doc1")
    fake_index = FakeIndex(primary_results=[node])
    setup_fakes(monkeypatch, fake_index, reranker_result=[node])

    retrieve_nodes.func(
        user_query="Author B 2026年對以太坊的看法",
        filtered_query="以太坊",
        final_filters=None,
    )

    assert fake_index.retrievers[0].received_query_bundle.query_str == "以太坊"


def test_first_stage_retrieves_10_candidates(monkeypatch):
    # 沒傳 similarity_top_k 時 LlamaIndex 預設只撈 2 筆，reranker 會沒東西可挑
    node = make_node("doc1")
    fake_index = FakeIndex(primary_results=[node])
    setup_fakes(monkeypatch, fake_index, reranker_result=[node])

    retrieve_nodes.func(user_query="以太坊", filtered_query=None, final_filters=None)

    assert fake_index.retrievers[0].similarity_top_k == 10


def test_falls_back_to_user_query_when_no_filtered_query(monkeypatch):
    node = make_node("doc1")
    fake_index = FakeIndex(primary_results=[node])
    setup_fakes(monkeypatch, fake_index, reranker_result=[node])

    retrieve_nodes.func(
        user_query="以太坊趨勢",
        filtered_query=None,
        final_filters=None,
    )

    assert fake_index.retrievers[0].received_query_bundle.query_str == "以太坊趨勢"


# ---------- Fallback（filters 太嚴格撈到 0 筆）----------

def test_fallback_triggered_when_filtered_search_is_empty(monkeypatch):
    fallback_node = make_node("doc-fallback")
    fake_index = FakeIndex(primary_results=[], fallback_results=[fallback_node])
    setup_fakes(monkeypatch, fake_index, reranker_result=[fallback_node])

    result = retrieve_nodes.func(
        user_query="法拉利趨勢",
        filtered_query="法拉利",
        final_filters="some-filters-object",  # 只要不是 None 就會觸發 fallback 判斷
    )

    # 應該呼叫了兩次 as_retriever：第一次帶 filters，第二次 fallback 不帶 filters
    assert len(fake_index.retrievers) == 2
    assert fake_index.retrievers[1].filters is None
    assert fake_index.retrievers[1].received_query_bundle.query_str == "法拉利趨勢"
    assert result == {"success": True, "data": [expected_chunk(fallback_node)], "error": None}


def test_no_fallback_when_no_filters_and_primary_search_is_empty(monkeypatch):
    fake_index = FakeIndex(primary_results=[], fallback_results=[make_node("should-not-be-used")])
    setup_fakes(monkeypatch, fake_index, reranker_result=[])

    result = retrieve_nodes.func(
        user_query="完全查無資料的問題",
        filtered_query=None,
        final_filters=None,  # 沒有 filters，代表本來就是全庫搜尋，撈不到就是真的沒有
    )

    # 不該觸發 fallback，只呼叫過一次 as_retriever
    assert len(fake_index.retrievers) == 1
    assert result == {"success": False, "data": [], "error": "查無相關文件"}


def test_returns_failure_envelope_when_both_primary_and_fallback_are_empty(monkeypatch):
    fake_index = FakeIndex(primary_results=[], fallback_results=[])
    fake_reranker_cls = setup_fakes(monkeypatch, fake_index, reranker_result=[])

    result = retrieve_nodes.func(
        user_query="完全查無資料的問題",
        filtered_query=None,
        final_filters="some-filters-object",
    )

    assert result == {"success": False, "data": [], "error": "查無相關文件"}
    # 撈到 0 筆就該直接收工：SentenceTransformerRerank(...) 這個建構子雖然在函式一開始
    # 就會被呼叫（不管後面撈不撈得到都會先建好），但重點是 .postprocess_nodes() 不該被執行
    assert fake_reranker_cls.last_instance.received_nodes is None


# ---------- Reranker 階段 ----------

def test_reranker_is_called_with_raw_user_query_not_filtered_query(monkeypatch):
    node = make_node("doc1")
    fake_index = FakeIndex(primary_results=[node])
    fake_reranker_cls = setup_fakes(monkeypatch, fake_index, reranker_result=[node])

    retrieve_nodes.func(
        user_query="Author B 2026年對以太坊的看法",
        filtered_query="以太坊",   # 檢索階段用這個（乾淨關鍵字）
        final_filters=None,
    )

    # 但 rerank 階段，程式碼故意用完整的原始問題，不是拔過雜訊的 filtered_query
    assert fake_reranker_cls.last_instance.received_query_bundle.query_str == "Author B 2026年對以太坊的看法"


def test_falls_back_to_raw_top2_when_reranker_returns_nothing(monkeypatch):
    node1 = make_node("doc1")
    node2 = make_node("doc2")
    node3 = make_node("doc3")
    fake_index = FakeIndex(primary_results=[node1, node2, node3])
    setup_fakes(monkeypatch, fake_index, reranker_result=[])  # reranker 覺得全部不相關

    result = retrieve_nodes.func(
        user_query="模糊的問題",
        filtered_query=None,
        final_filters=None,
    )

    # reranker 開天窗時，退回第一階段撈到的前 2 筆，而不是真的回傳空的
    assert result == {
        "success": True,
        "data": [expected_chunk(node1), expected_chunk(node2)],
        "error": None,
    }


def test_returns_reranked_nodes_when_reranker_succeeds(monkeypatch):
    node1 = make_node("doc1")
    node2 = make_node("doc2")
    fake_index = FakeIndex(primary_results=[node1, node2])
    setup_fakes(monkeypatch, fake_index, reranker_result=[node2])

    result = retrieve_nodes.func(
        user_query="問題",
        filtered_query=None,
        final_filters=None,
    )

    assert result == {"success": True, "data": [expected_chunk(node2)], "error": None}


# ---------- 例外處理：整個函式現在保證不拋例外 ----------

def test_never_raises_when_vector_store_connection_crashes(monkeypatch):
    class BrokenVService:
        @property
        def vector_store(self):
            raise ConnectionError("Chroma 連不上")

        embed_model = "fake-embed-model"

    monkeypatch.setattr(tool_module, "v_service", BrokenVService())

    result = retrieve_nodes.func(
        user_query="任何問題",
        filtered_query=None,
        final_filters=None,
    )

    assert result["success"] is False
    assert result["data"] == []
    assert "Chroma 連不上" in result["error"]


# ---------- hybrid：向量 + BM25 用 RRF 合併 ----------

def test_fuse_rrf_ranks_items_found_by_both_lists_first():
    a, b, c = make_node("a"), make_node("b"), make_node("c")
    # b 在兩份排名裡都出現，應該勝過只在一份裡排第 1 的 a 和 c
    fused = tool_module.fuse_rrf([[a, b], [c, b]], top_k=10)
    assert [n.node.node_id for n in fused] == ["b", "a", "c"]


def test_fuse_rrf_respects_top_k():
    nodes = [make_node(f"n{i}") for i in range(15)]
    assert len(tool_module.fuse_rrf([nodes], top_k=10)) == 10


def test_keyword_only_hit_reaches_reranker(monkeypatch):
    # 向量檢索漏掉、只有 BM25 撈到的 chunk（例如專有名詞），也要交給 reranker
    vector_node, keyword_node = make_node("vec"), make_node("kw")
    fake_index = FakeIndex(primary_results=[vector_node])
    fake_reranker_cls = setup_fakes(monkeypatch, fake_index, reranker_result=[keyword_node])
    monkeypatch.setattr(tool_module, "get_bm25_index", lambda vector_store: FakeBM25([keyword_node]))

    retrieve_nodes.func(user_query="柯立芝", filtered_query=None, final_filters=None)

    reranked_ids = {n.node.node_id for n in fake_reranker_cls.last_instance.received_nodes}
    assert reranked_ids == {"vec", "kw"}


def test_bm25_receives_same_query_and_filters_as_vector_search(monkeypatch):
    node = make_node("doc1")
    fake_index = FakeIndex(primary_results=[node])
    setup_fakes(monkeypatch, fake_index, reranker_result=[node])
    fake_bm25 = FakeBM25([])
    monkeypatch.setattr(tool_module, "get_bm25_index", lambda vector_store: fake_bm25)

    retrieve_nodes.func(user_query="Author B 對 ONDO 的看法", filtered_query="ONDO", final_filters="author-filter")

    assert fake_bm25.calls[0] == {"query": "ONDO", "filters": "author-filter"}
