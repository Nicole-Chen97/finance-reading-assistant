# 做rag filter 方便查詢
from llama_index.core.vector_stores.types import (
    MetadataFilters, 
    MetadataFilter, 
    FilterOperator, 
    FilterCondition
)
from typing import List, Union,Literal, Optional
from pydantic import BaseModel, Field # 拿來寫說明書的工具
from datetime import datetime
from authors import AuthorName, author_prompt_hint

# 讓 LLM 知道怎麼填寫單一的 Filter
class Metadata_Filter(BaseModel):
    # key 是表格的「欄位名稱」
    
    # 作者不在這裡：作者有自己的 author 欄位（見 SearchIntentPayload），用 Literal 限定只能填資料庫裡真的有的名字
    key: Literal["post_date_timestamp", "type", "audience"] = Field(
        description="""【嚴格限定】只能填寫資料庫欄位名稱（作者請填在 author 欄位，不要放在這裡）：
        - 如果提及年份、月份、具體日期、時間 ➔ 'post_date_timestamp'
        - 如果提及文章類型（如 post） ➔ 'type'
        - 如果提及受眾（如 public） ➔ 'audience'"""
    )
    
    # value 是表格內容的「填入值」
    value: str = Field(
        description="""過濾欄位要對應的具體數值：請根據query判斷。
        - 如果 key 是 'post_date_timestamp'：必須轉換為標準 ISO 8601 時間字串（例如 '2026-01-01T00:00:00Z'）。"""
    )
    
    # 限制 LLM 只能填寫 LlamaIndex 支援的標準運算子字串
    operator: Literal["EQ", "GT", "LT", "GTE", "LTE", "NE", "IN"] = Field(
        description="""比較運算子。
        - 文字比對（如 type、audience）一律用 'EQ'。
        - 時間比對：若是特定年份（如2026），必須拆成兩條：一條是 >= 2026-01-01 (GTE)，另一條是 <= 2026-12-31 (LTE)。"""
    )
    
    

  
    
# 2️⃣ 總機輸出規格書> 這是大腦最終要打包交給你的「完整搜尋意圖包裹」
class SearchIntentPayload(BaseModel):
    query: str = Field(
        description="""純粹的『語意搜尋關鍵字』。
        如果有找到filter，請按照以下步驟，如果沒有按照原來的query 。
        請拔掉原句中已經被放進 filters 的『作者名稱』、『時間詞彙（例如 2026年、最近）』。
        只留下人類真正想問的『主體事件關鍵字』。

        範例對照（A 作者、B 作者代表任一位作者）：
        - 輸入：'A 作者講法拉利趨勢' ➔ 拔除作者後填寫：'法拉利趨勢'
        - 輸入：'幫我查 B 作者 2026年關於以太坊的看法' ➔ 拔除作者和 2026年後填寫：'以太坊的看法'
        - 輸入：'A 作者最近分析的台積電走勢' ➔ 拔除作者和最近後填寫：'台積電走勢'"""


        """剝離了明確的時間限制、作者名字後的『純粹語意搜尋關鍵字』。
        例如：'B 作者 2026年對以太坊看法' ➔ 剝離後變成 '以太坊 看法 趨勢分析'。
        請盡可能保留核心財經、加密貨幣概念，去掉贅詞。"""
    )
    
    # 用 Literal 而不是 str（AuthorName，見 authors.py）：LLM 收到的 JSON schema 裡只有設定檔裡的作者，
    # 從格式層面杜絕拼法跟資料庫對不上的值（之前作者 filter 從沒生效過就是因為這個）
    author: Optional[AuthorName] = Field(
        default=None,
        description="問題裡提到的作者。只能是以下其中一個值，沒提到作者就填 null：\n" + author_prompt_hint()
    )

    filters: List[Metadata_Filter] = Field(
        default=[],
        description="作者以外的過濾條件清單（時間、類型、受眾）。沒有提到這些條件時，請保持空清單 []。"
    )
    
    
    condition: Optional[Literal["AND", "OR"]] = Field(
        default=None,  # 預設直接給 None
        description="""所有過濾條件之間的邏輯運算。
       - 如果 filters 清單是空的 [] ➔ 這裡請直接填寫 null (或不填)。
        - 如果 filters 清單裡有 2 條以上的規則：除非使用者要求『或者(OR)』，否則預設填寫 'AND'。"""
    )
    
    @property
    def to_llama_index_filters(self) -> Optional[MetadataFilters]:
        if self.filters or self.author:
            print(f"📌 偵測到 {len(self.filters) + bool(self.author)} 條過濾規則，開始封裝 LlamaIndex 官方過濾器...")

            processed_filters = []
            if self.author:
                processed_filters.append(
                    MetadataFilter(key="author", value=self.author, operator=FilterOperator.EQ)
                )
            for f in self.filters:
                current_value = f.value
                
                # 🌟 關鍵修正：如果是日期欄位，且拿到的值是字串，強轉成數字時間戳記
                if 'date' in f.key and isinstance(current_value, str):
                    
                    try:
                        # 把 ISO 字串（例如 '2026-05-01T00:00:00Z'）轉成 timestamp 數字
                        clean_date_str = current_value.replace('Z', '')
                        current_value = datetime.fromisoformat(clean_date_str).timestamp()
                        print(f"   ⏰ 已自動將 {f.key} 的字串值轉換為數字時間戳記: {current_value}")
                    except Exception as e:
                        print(f"   ⚠️ 日期轉換失敗，維持原樣: {e}")
                        
                # 封裝進 LlamaIndex 篩選器
                processed_filters.append(
                    MetadataFilter(
                        key=f.key, 
                        value=current_value,  # 👈 這裡用我們處理過後的 current_value
                        operator=getattr(FilterOperator, f.operator)
                    )
                )
            
            final_filters = MetadataFilters(
                filters=processed_filters,
                condition=getattr(FilterCondition, self.condition or "AND")
            )
            
            
            return final_filters
            
        else:
            print("ℹ️ 沒有偵測到任何過濾規則，本次檢索將退化為『全庫純語意搜尋』。")
            final_filters = None
            # filter_query = None
            
            return final_filters