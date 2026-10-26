# ----------------Vectore Store ----------------------

import os
import logging
from pathlib import Path
from dotenv import load_dotenv
from llama_index.core import Document, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

load_dotenv()

logger = logging.getLogger(__name__)

# 設定全域變數，方便統一管理路徑。優先讀 .env 的 CHROMA_DB_PATH；
# 如果沒設定，退回到「這個專案自己的 data/ 資料夾」，不寫死成特定機器的絕對路徑，
# 保證換一台機器、換一個帳號、clone 到別的地方都能正常運作。
_DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "chroma_db"
DB_PATH = os.getenv("CHROMA_DB_PATH", str(_DEFAULT_DB_PATH))
COLLECTION_NAME = "substack_articles"

EMBED_MODEL = HuggingFaceEmbedding(model_name="BAAI/bge-base-zh-v1.5")

class VectorStoreService:
    """【服務類別】專門負責向量資料庫的連線與操作，讓其他部分專注在業務邏輯上"""
    def __init__(self):
        """
        初始化向量資料庫服務，並準備好 Embedding 模型與本地路徑
        """
        # 1. 設定 Embedding 模型
        self.db_path = DB_PATH
        self.collection_name = COLLECTION_NAME
        self.embed_model = EMBED_MODEL

    # 底線開頭：Private（私有／內部專用），
    def _connect_to_vector_store(self):
        """[內部工具] 轉開水龍頭，建立與 ChromaDB 的連線通道"""
        # 1. 建立實體連線
        db = chromadb.PersistentClient(path=self.db_path)
        chroma_collection = db.get_or_create_collection(self.collection_name)

        # 2. 包裝成 LlamaIndex 看得懂的組件
            # ChromaVectorStore:經過這個包裝，LlamaIndex 就能透過這個「經理」對資料庫進行讀寫了
            # StorageContext:繼件托運單
                # 告訴 LlamaIndex 的大腦：「聽好囉，我等一下算完 Embedding 的向量檔案，
                # 通通都要對準上面這位物業經理（vector_store）管理的公寓倒進去，別送錯地方了！」
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # 把這兩個核心通道吐給後面的函式用
        return storage_context, vector_store
    
    # 🌟 這是新增的快捷通道！專門給外部的檢索器（Retriever）直接拿取物業經理
    @property
    def vector_store(self):
        """【路徑二：讀取快捷口】只拿取 vector_store 經理，供檢索讀取使用"""
        _, vector_store = self._connect_to_vector_store()
        return vector_store

    def save_nodes_to_vector(self,nodes: list, article_info: dict) -> dict:
        """【路徑一：儲存】接收 nodes 並永久寫入硬碟資料庫"""
        if not nodes:
            logger.warning("沒有傳入任何 nodes，取消儲存。")
            return {
            "status": "empty",
            "success": False,
            "message": "沒有傳入任何 nodes，取消儲存。"
        }


        # 1. 從傳進來的字典裡，優雅地拿取需要的欄位
        article_id = article_info.get("article_id")
        post_date = article_info.get("post_date")
        article_title = article_info.get("title")
        
        # 2. 呼叫剛剛寫好的連線工具，直接拿到通道
        storage_context, vector_store = self._connect_to_vector_store()

        # 3. 防禦機制：利用原生 chroma 檢查這篇文章有沒有存過
        result = vector_store._collection.get(
            where={"$and": [{"article_id": article_id}, {"post_date": post_date}]}
        )

        if result["ids"]:
            msg = f"文章 {article_id} (日期: {post_date}) {article_title}已經存在資料庫中，跳過不重複寫入。"
            logger.info(msg)

            return{
                 "status": "dupliate",
            "success": True,
            "message": msg
            }

        # 3. 確定沒存過，開始計算 Embedding 並順著通道倒進硬碟
        logger.info("正在計算 %d 個節點的向量並寫入 ChromaDB...", len(nodes))
        try:
            index = VectorStoreIndex(
                nodes=nodes,
                storage_context=storage_context,
                embed_model=self.embed_model
                )
            msg = "🚀 寫入成功！"
            logger.info(msg)

            return{
                 "status": "inserted",
                "success": True,
                "message": msg
                }



        except Exception as e:

            msg = f"❌ 寫入失敗：{e}"
            logger.error(msg)
            return{
                 "status": "write_failed",
                "success": False,
                "message": msg
                }