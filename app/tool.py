# tool.py
import logging
from langchain_core.tools import tool
from vector_store import VectorStoreService
from article_ingestion import ArticleIngestionPipeline

from typing import List, Union,Literal, Optional

logger = logging.getLogger(__name__)

# 假設你有一個宣告好的向量服務
# my_vector_service = VectorStoreService(db=chroma_db)
v_service = VectorStoreService()

@tool
def download_substack_article(author_name: str) -> dict:
    """
    下載指定 Substack 作者的最新文章，自動進行 HTML 清洗、轉 Markdown、
    切分文本節點 (Nodes)，並將計算完 Embedding 的向量節點永久存入 Chroma 資料庫中。

    當使用者要求「下載某人的文章」、「把某作者的最新貼文存入資料庫」或「更新知識庫」時，
    請呼叫此工具。

    Args:
        author_name (str): 作者的姓名（例如：小明、Coco、劉德華）。必須是系統中存在的作者。

    Returns:
        dict: 固定形狀 {"success": bool, "data": dict | None, "error": str | None}，
        不管內部哪一步出錯，這個工具永遠不會把例外丟給呼叫端，一定用這個信封回報。
    """
    logger.info("[download_substack_article] Agent 指派任務：下載並儲存「%s」的文章", author_name)

    try:
        # 3. 實例化下載流水線，並將全域的 v_service 注入進去（相依性注入）
        pipeline = ArticleIngestionPipeline(
            author_name=author_name,
            vector_service=v_service
        )

        # 4. 按下總開關，一氣呵成執行 ETL（download_and_process 現在保證永遠回傳同一種 dict 形狀）
        result = pipeline.download_and_process()

        return {
            "success": bool(result.get("success")),
            "data": result,
            "error": None if result.get("success") else result.get("message"),
        }

    except Exception as e:
        # 防禦性程式設計：萬一中間有任何打錯字或沒想到的崩潰，確保 Agent 能量化抓到錯誤
        error_msg = f"工具在調度流水線時發生未預期的程式錯誤: {str(e)}"
        logger.error("[download_substack_article] %s", error_msg)
        return {"success": False, "data": None, "error": error_msg}
    
    
from typing import List, Optional
from llama_index.core import VectorStoreIndex, QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores.types import MetadataFilters
from llama_index.core.postprocessor import SentenceTransformerRerank
from bm25_index import get_bm25_index
# from llama_index.core.tools import tool

FIRST_STAGE_TOP_K = 10  # 向量、BM25 各撈 10 筆，合併後留 10 筆給 reranker
RRF_K = 60  # RRF 的平滑常數，業界慣用 60：讓第 1 名和第 2 名的差距不會大到壓過另一份排名

# 中英雙語的 cross-encoder。原本的 cross-encoder/ms-marco-MiniLM-L-6-v2 只用英文資料訓練，
# 在中文文章上反而讓排名變差（Hit@1 60% → 49%）；換成這個之後 Hit@1 83%（見 README 第 7 節）。
RERANK_MODEL = "BAAI/bge-reranker-base"
RERANK_TOP_N = 3
_reranker = None


def _get_reranker() -> SentenceTransformerRerank:
    """reranker 模型約 1GB，每次查詢都重新載入會多花好幾秒，所以只載入一次、之後重複使用。"""
    global _reranker
    if _reranker is None:
        _reranker = SentenceTransformerRerank(model=RERANK_MODEL, top_n=RERANK_TOP_N)
    return _reranker


def fuse_rrf(result_lists: list[list[NodeWithScore]], top_k: int) -> list[NodeWithScore]:
    """Reciprocal Rank Fusion：每個 chunk 的分數 = Σ 1 / (RRF_K + 它在各份排名的名次)。
    只看名次、不看原始分數，所以向量的相似度（0~1）和 BM25 分數（沒有上限）尺度不同也能直接合併；
    兩邊都排前面的 chunk 分數最高。回傳原本的 NodeWithScore 物件，分數不改（之後會被 reranker 重新打分）。"""
    fused: dict[str, float] = {}
    first_seen: dict[str, NodeWithScore] = {}
    for results in result_lists:
        for rank, item in enumerate(results, start=1):
            node_id = item.node.node_id
            fused[node_id] = fused.get(node_id, 0.0) + 1.0 / (RRF_K + rank)
            first_seen.setdefault(node_id, item)
    ordered = sorted(fused, key=lambda node_id: fused[node_id], reverse=True)
    return [first_seen[node_id] for node_id in ordered[:top_k]]


def hybrid_search(index, vector_store, query: str, filters: Optional[MetadataFilters]) -> list[NodeWithScore]:
    """真正的 hybrid 檢索：向量（語意）和 BM25（關鍵字）各撈一份，再用 RRF 合併。
    Chroma 不支援 VectorStoreQueryMode.HYBRID（傳了會被默默當成一般向量檢索），所以 BM25 在 Chroma 外面自己做。"""
    query_bundle = QueryBundle(query_str=query)
    # 一定要指定 similarity_top_k：沒指定時 LlamaIndex 預設只撈 2 筆，reranker 等於沒東西可挑
    vector_hits = index.as_retriever(similarity_top_k=FIRST_STAGE_TOP_K, filters=filters).retrieve(query_bundle)
    keyword_hits = get_bm25_index(vector_store).search(query, top_k=FIRST_STAGE_TOP_K, filters=filters)
    logger.debug("[hybrid_search] 向量 %d 筆、BM25 %d 筆", len(vector_hits), len(keyword_hits))
    return fuse_rrf([vector_hits, keyword_hits], top_k=FIRST_STAGE_TOP_K)


@tool
def retrieve_nodes(
    user_query: str,
    filtered_query: Optional[str] = None,# 📥 拔除雜訊後的乾淨關鍵字 (選填，例如: "法拉利")
    final_filters: Optional[MetadataFilters] = None,
    ) -> dict:
    """【高階 RAG 模式】混合檢索 (撈出10筆) + Reranker (精選出 top_n 筆)。

    Returns:
        dict: 固定形狀 {"success": bool, "data": list[dict], "error": str | None}。
        data 裡每筆都是單純的 dict（author/title/post_date/header_path/text/score），
        llama_index 的內部物件不會外流到呼叫端；這個工具永遠不會把例外丟出去。
    """
    try:
        # 1. 呼叫剛剛寫好的連線工具，直接拿到通道
        vector_store = v_service.vector_store

        # 2. 開啟「讀取模式」：
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,  # 1. 告訴大腦：舊資料在位這位經理管的硬碟公寓裡
            embed_model=v_service.embed_model,  # 2. 告訴大腦：等等使用者提問時，要用哪一個模型來幫問題算向量
            )

        # 3. 取得 Reranker 處理器（只在第一次呼叫時載入模型）
        reranker = _get_reranker()

        # 第一階段：hybrid 檢索（向量 + BM25，各 10 筆，RRF 合併後留 10 筆）
        # 💡 向量負責「換句話說」，BM25 負責專有名詞（柯立芝、ONDO、USDJPY），兩者互補
        chosen_query = filtered_query if filtered_query else user_query
        logger.debug("[retrieve_nodes] 第一階段檢索將使用關鍵字: '%s'", chosen_query)
        retriever_output = hybrid_search(index, vector_store, chosen_query, final_filters)

        # 🎯 3. 【防禦核心】檢查是不是因為 filter 設太死，導致撈到空信箱？
        if not retriever_output and final_filters is not None:
            logger.warning(
                "[retrieve_nodes] 帶有條件的檢索結果為 0 筆，懷疑條件記錯，自動解開限制切換為全庫混合檢索"
            )

            # 拿掉 filters（👈 徹底摘除過濾鎖鏈），改用完整的原始問題對全庫重查
            retriever_output = hybrid_search(index, vector_store, user_query, None)

        logger.debug("[retrieve_nodes] 第一階段粗篩最終：已撈出 %d 筆 Node", len(retriever_output))

        # 4. 如果連全庫搜尋都撈不到（可能真的沒這關鍵字），就直接收工
        if not retriever_output:
            logger.info("[retrieve_nodes] 全庫亦查無此資料")
            return {"success": False, "data": [], "error": "查無相關文件"}

        # 5. 🎯 第二階段：用 Reranker 精選出 top_n 筆最相關的 Node
        ranked_nodes = reranker.postprocess_nodes(
            nodes=retriever_output,
            query_bundle=QueryBundle(query_str=user_query)
        )

        # 🎯 7. 【第三道防線】萬一被 Reranker 審查完，悲劇地變成空的呢？
        if not ranked_nodes:
            logger.warning(
                "[retrieve_nodes] 經過重排後發現資料全都不相關，盲選並保留第一階段的前 2 筆原始資料"
            )
            ranked_nodes = retriever_output[:2]

        logger.debug("[retrieve_nodes] 第二階段 Rerank 最終輸出：已精選出 %d 筆最相關 Node", len(ranked_nodes))

        chunks = [
            {
                "author": item.node.metadata.get("author", "Unknown"),
                "title": item.node.metadata.get("title", "No title"),
                "post_date": item.node.metadata.get("post_date", "Unknown date"),
                "header_path": item.node.metadata.get("header_path", "No path"),
                "text": item.node.text,
                "score": item.score,
            }
            for item in ranked_nodes
        ]

        return {"success": True, "data": chunks, "error": None}

    except Exception as e:
        error_msg = f"檢索時發生錯誤: {str(e)}"
        logger.error("[retrieve_nodes] %s", error_msg)
        return {"success": False, "data": [], "error": error_msg}