import os
import json
import logging
from langchain_core.tools import tool
import requests
from bs4 import BeautifulSoup
import html2text
import re
from urllib.parse import unquote
from pydantic import BaseModel
from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser
from typing import Optional, Any
from datetime import datetime
from dotenv import load_dotenv
from authors import AUTHORS

load_dotenv()

logger = logging.getLogger(__name__)

# 作者清單來自 authors.json（見 authors.py）。Substack 的登入 cookie 是機密，
# 放在 .env 裡，用設定檔的 sid_env 指定的環境變數名稱讀取，不要寫死在原始碼裡。
AUTHOR_MAP = {
    a["name"]: {
        "profile_user_id": a["profile_user_id"],
        "substack.sid": os.getenv(a["sid_env"]),
    }
    for a in AUTHORS
}

def convert_to_timestamp(raw_date_str: str) -> int:
    """
    將 Substack 的 ISO 時間字串轉為 ChromaDB 接受的整數時間戳記 (UTC)
    例如: "2026-05-31T01:04:59.977Z" -> 1777597499
    """
    try:
        # 1. 為了相容性，把 Z 換成標準時區標示 +00:00
        clean_date_str = raw_date_str.replace('Z', '+00:00')
        
        # 2. 將字串解析為帶有時區資訊的 Python datetime 物件
        dt_object = datetime.fromisoformat(clean_date_str)
        
        # 3. 關鍵一步：轉換成從 1970 年累計至今的「總秒數」（整數 int）
        timestamp_int = int(dt_object.timestamp())
        
        return timestamp_int
    except Exception as e:
        logger.warning("時間欄位解析失敗 [%s]: %s", raw_date_str, e)
        # 如果真的遇到髒資料解析失敗，回傳 0 作為保險（代表 1970 元旦）
        return 0

class ArticleIngestionPipeline(BaseModel):
    author_name: str                 # 宣告：我需要一個字串
    vector_service: Optional[Any] = None  # 宣告：我需要一個選填的服務
    
    def download_and_process(self) -> dict:
        """ 🚀 外部呼叫的主入口：一氣呵成跑完 ETL。無論哪個分支，永遠回傳同一種形狀：
        {"status": str, "success": bool, "message": str}，呼叫端不用再猜這次拿到的是字串還是 dict。 """
        logger.info("開始處理作者 %s 的文章流水線...", self.author_name)

        # 步驟 1: 下載
        html_raw, article_info = self.down_load_article()
        if not html_raw:
            return {"status": "download_failed", "success": False, "message": "下載失敗，中止後續流程。"}

        # 步驟 2: 轉成 Markdown
        clean_markdown = self.markdown_content(html_raw)

        # 步驟 3: 解析成 Nodes
        nodes = self.article_parser(clean_markdown, article_info)

        # 步驟 4: 彈性儲存（如果外面有傳 vector_service 進來就存，沒有就跳過）

        if self.vector_service:
            # 假設你的 vector_service 有一個 save_nodes 方法
            save_result = self.vector_service.save_nodes_to_vector(nodes, article_info)

            return save_result

        return {"status": "no_service", "success": False, "message": "未設定 Vector Service"}
        # return f"✨ 成功完成 {self.author_name} 的文章下載與清洗，但未寫入 Vector Store。"
            
            # # 總管根據回報，對 LangGraph / Agent 講不同格式的話
            # if success:
            #     return f"🎉 成功完成 {self.author_name} 的文章下載、清洗並存入 Vector Store！"
            # else:
            #     # 👈 如果回傳 False，這裡就能精準抓到錯誤！
            #     return "⚠️ 文章下載與處理都成功，但在最後『寫入向量資料庫』時失敗了，請檢查資料庫連線。"
            # # return f"🎉 成功完成 {self.author_name} 的文章下載、清洗並存入 Vector Store！"
        
        # return f"✨ 成功完成 {self.author_name} 的文章下載與清洗（未存入資料庫，已返回 {len(nodes)} 個 Nodes）。"
    
    
    
    
       
    def down_load_article(self) -> tuple[Optional[str], Optional[dict]]:
        """回傳 (html_body, article_info)。任何一步失敗都明確回傳 (None, None)，
        不讓例外中途逃出、也不留下沒被賦值的變數——download_and_process 會用
        `if not html_raw:` 判斷這裡是不是失敗了。"""

        if self.author_name not in AUTHOR_MAP:
            logger.warning("不支援的作者: '%s'，AUTHOR_MAP 裡沒有這個人的登入資訊。", self.author_name)
            return None, None

        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Safari/605.1.15",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                }

        author_id = AUTHOR_MAP[self.author_name]['profile_user_id']
        substack_sid = AUTHOR_MAP[self.author_name]['substack.sid']
        cookies ={
            "substack.sid":substack_sid
        }

        publication_url = "https://substack.com/api/v1/profile/posts"

        params = {
            "profile_user_id": author_id,
            "limit": 12,
        }
        try:
            articles_info = requests.get(publication_url, params=params,headers=headers, cookies=cookies)
            if articles_info.status_code != 200:
                logger.warning("無法下載文章列表，狀態碼: %s", articles_info.status_code)
                if articles_info.status_code in [401, 403]:
                    logger.warning("看起來是 substack.sid Cookie 過期或權限不足了")
                return None, None

            post = articles_info.json()['posts'][0]
            bylines = post.get("publishedBylines", [])
            author_name = "未知作者"
            if bylines and isinstance(bylines, list):
                author_name = bylines[0].get("name", "未知作者")

            article_info = {
                'article_id': post.get('id'),
                "publication_id": post.get("publication_id"),
                "author": author_name,
                'title': post.get('title'),
                'type': post.get('type'),
                'post_date': post.get('post_date'),
                "post_date_timestamp":convert_to_timestamp(post.get('post_date')),
                'audience': post.get('audience'),
                'canonical_url': post.get('canonical_url'),
                'slug': post.get('slug'),
            }
        except Exception as e:
            # 🎯 這裡會直接捕捉所有程式出錯（例如：網路完全斷線、JSON 解析失敗、或是打錯字）
            logger.error("下載文章資訊時發生程式錯誤，原因: %s", e)
            return None, None

        try:
            article_id = article_info['article_id']
            article_url = f"https://substack.com/api/v1/posts/by-id/{article_id}"
            article_r = requests.get(article_url, headers=headers, cookies=cookies)
            article_body = article_r.json()['post']['body_html']
            logger.info("下載文章: %s", article_info['title'])
            return article_body, article_info

        except Exception as e:
            # 🎯 這裡會直接捕捉所有程式出錯（例如：網路完全斷線、JSON 解析失敗、或是打錯字）
            logger.error("無法下載文章 '%s'，發生程式錯誤，原因: %s", article_info.get('title'), e)
            return None, None


    def  markdown_content(self,html_raw:str) ->str:
        # 3. 設定 Markdown 轉換器
        converter = html2text.HTML2Text()
        converter.body_width = 0           # 禁用自動換行，保持結構完整
        converter.ignore_links = False     # 保留網址連結
        converter.ignore_images = False    # 保留圖片連結
        converter.protect_links = True     # 保護連結不被意外斷開

        # 4. 執行轉換
        markdown_result = converter.handle(html_raw)
        pattern= r"\[\!\[\]\(https:.*?((\.png)|(\.webp))\)\]"
        ctxt = re.sub(pattern,"",markdown_result)
        # matches = re.findall(pattern_1, markdown_result)
        # print(f"共找到 {len(matches)} 處符合的圖片網址尾巴：")
        # for m sin matches:
        #     print(f"找到片段: {m}")

        pattern_1 = r"https://substackcdn\.com/image/fetch/[^/]+/"
        ctxt_2= re.sub(pattern_1,"",ctxt)


        p = r"\(\<https.*?(?:\.webp|\.png)"
        clean_ctxt= re.sub(p,lambda url : unquote(url.group(0)) ,ctxt_2)
        
        return clean_ctxt
     




    def article_parser(self,clean_markdown:str,article_info:dict):
        doc = Document(
                text=clean_markdown,
                metadata=article_info,
                id_ = str(article_info['article_id'])
                ) 
            
        
        parser = MarkdownNodeParser()
        nodes = parser.get_nodes_from_documents([doc],include_metadata=True,include_prev_next_rel=True) # 注意這裡傳入的是我們清洗過的 [doc]
        for indx_node , node in enumerate(nodes):
            p_url_remove = r"\(\<https://.*?\.(?:png|webp|jpg|jpeg|gif)\>\)"
            content = re.sub(p_url_remove,'',node.text)
            # 2. 💡 完美結合：先還原 \\n 成為真換行，再收縮多餘的 \n，最後 strip()
            content = content.replace('\\n', '\n')
            content = re.sub(r'\n{3,}', '\n\n', content).strip()
            node.text = content 

            # --- node 編號 ---
            node.node_id = f"{doc.doc_id}::chunk-{indx_node+1}"
        
        return nodes