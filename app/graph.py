import os
import logging
from dotenv import load_dotenv
load_dotenv()

from datetime import datetime
from typing import TypedDict, Annotated, Optional, List, Literal, Any
from typing import Annotated, TypedDict

logger = logging.getLogger(__name__)


def configure_logging(level: int = logging.INFO) -> None:
    """給各個入口點（chainlit_app.py、run_chat_loop）呼叫，統一設定 log 格式。
    只在真正的應用程式入口呼叫，不要在這個檔案被 import 時就自動執行——
    否則跑測試時也會被動觸發，不乾淨。"""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    )


from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from authors import AUTHOR_NAMES, author_prompt_hint
from langchain_core.tools import tool


from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(
    model="claude-sonnet-4-5-20250929",
    temperature=0.7,
    api_key=os.environ["ANTHROPIC_API_KEY"].strip()
)



class ToolResult(TypedDict):
    source: str               # 沿用 router 的值："download" / "rag" / "market_price"
    success: bool             # 這一步做得順不順利
    summary: str              # 給 answer_node 組 context 用的文字
    raw: Optional[Any]        # 原始結構化資料（rag 的 nodes、之後查價的 fig...），給非文字用途保留
    error: Optional[str]      # 失敗原因，成功時是 None
    meta: Optional[dict]      # 次要信號（RAG 品質回饋、之後的信心分數/重試次數...都放這裡）


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]   # 唯一跨輪保留的東西，session 記憶就是它
    router: Optional[str]           # 記錄當前要走的路由: "download", "chat", "rag"
    tool_result: Optional[ToolResult]   # 取代掉原本 download_status/download_result/error_log/rag_doc/verify_result_feedback 這幾個各自為政的欄位



#=============Router===============
class RouteDecision(BaseModel):
    route: Literal["rag", "download", "market_price", "chat"] = Field(description="根據使用者的意圖選擇最適合的下一步路由")

router_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7,api_key = os.environ["ANTHROPIC_API_KEY"].strip()).with_structured_output(RouteDecision)

# def  函數名稱( 參數 : 規定傳進來的型別 ) -> 規定吐出來的型別 :
def router_node(state: AgentState) -> AgentState:
    # 1. 讀取人類問的最新問題 q
    query = state["messages"][-1].content
    system_prompt = """你是一個高效率的路由分流總機（Router）。請分析使用者最新輸入的這句話，並從以下四個選項中精確選擇一條路線：

        【分流判斷準則】
        1. 'download'：當使用者明確要求「下載」、「同步」、「抓取」、「更新」特定作者的文章至資料庫時。
        2. 'rag'：當使用者詢問關於知識庫內文章的「具體內容」、「主題分析」、「精華總結」或「跨文章對比」時。
        3. 'market_price'：當使用者詢問美股、美股ETF、加密貨幣、指數的「價格」、「走勢」、「K線」、「圖表」時。
        4. 'chat'：當使用者只是在進行日常問候（如：哈囉、你好）、閒聊、或是詢問與上述三者完全無關的一般性問題時。

        請保持絕對的客觀與理性，僅根據上述準則進行單選。
            """

    # 2. 把問題丟給總機 AI，強迫它填寫剛剛的「三選一假單」——router 只決定路線，
    # 不再順便判斷作者是誰，那件事現在是 download_node 自己的責任。
    decision = router_llm.invoke([
        ("system", system_prompt),
        ("user",query)
    ])

    new_state = {**state,
                 "router":decision.route,
                 # 每輪對話重新分流時，先清空上一輪殘留的一次性結果，
                 # 避免 checkpointer 打開後，舊的 tool_result 混進這一輪的回答
                 "tool_result": None,
                 }
    return new_state




#=============Download Node===============
class DownLoad_Author(BaseModel):
    # 利用 Literal 鎖死選項，AI 絕對無法自創其他作者名字。
    # 選項和 AUTHOR_MAP 都來自同一份 authors.json，新增作者時兩邊自動同步，
    # 不會出現 LLM 選中「合法但沒設定登入資訊」的作者。
    author_name: Literal[(*AUTHOR_NAMES, "None")]


author_decide_llm  = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7,api_key = os.environ["ANTHROPIC_API_KEY"].strip()).with_structured_output(DownLoad_Author)

from tool import download_substack_article


def download_node(state: AgentState) -> AgentState:

    # 1. 自己判斷要下載哪位作者——不再依賴 router 事先塞好的值，直接從原始使用者問題
    # 判斷，判斷完馬上用掉，不需要進 state（只有這個 node 需要知道這件事）。
    user_query = state["messages"][-1].content

    author_decision = author_decide_llm.invoke([
        ("system", "請從使用者的問題中，判斷他想下載哪一位作者的文章，精確對應到系統支援的作者名稱。如果無法判斷，回傳 'None'。\n"
                   + author_prompt_hint()),
        ("user", user_query)
    ])
    chosen_author = author_decision.author_name

    logger.info("[download_node] 後台管線啟動，從使用者問題判斷出目標作者: '%s'", chosen_author)

    if chosen_author == "None":
        logger.warning("[download_node] 無法判斷作者，跳過下載。")
        return {**state, "tool_result": {
            "source": state["router"],
            "success": False,
            "summary": "無法從問題中判斷出要下載哪位作者的文章，請明確指定作者名稱。",
            "raw": None,
            "error": "無法判斷作者",
            "meta": None,
        }}

    # 2. 執行下載：download_substack_article 保證永遠回傳 {"success","data","error"}
    # 這個固定信封，而且永遠不會把例外丟出來，這裡不需要再包 try/except。
    result = download_substack_article.invoke({"author_name": chosen_author})
    data = result["data"] or {}

    new_state = {**state, "tool_result": {
        "source": state["router"],
        "success": result["success"],
        "summary": data.get("message") or result["error"] or "無回傳訊息。",
        "raw": data,
        "error": result["error"],
        "meta": {"author": chosen_author, "status": data.get("status", "unknown")},
    }}
    logger.debug("[download_node] result: %s", new_state)
    logger.info("[download_node] '%s' 處理完成，success=%s", chosen_author, result['success'])
    return new_state
    

# 需要重新考量下有關下載文章之後，是否作出文章摘要給我？



#===============Retriever Node ===============
from schema import  SearchIntentPayload
from tool import retrieve_nodes

class Judge_RAG_answer(BaseModel):
    is_good_rag: bool
    feedback: str | None = None

rag_judge_llm  = ChatAnthropic(
    model="claude-sonnet-4-6", 
    temperature=0,
    api_key = os.environ["ANTHROPIC_API_KEY"].strip()).with_structured_output(Judge_RAG_answer)


filter_define_llm = ChatAnthropic(
        model="claude-sonnet-4-5-20250929",
        temperature=0,
        api_key=os.environ["ANTHROPIC_API_KEY"].strip()
            ).with_structured_output(SearchIntentPayload) 
 
# 要想執行任務，統一格式全部都叫做 .invoke()（意思是「執行/呼叫」）。
def rag_search_node(state: AgentState) -> AgentState:
    human_query = state["messages"][-1].content

    system_instruction = """你是一個精準的搜尋意圖拆解專家。請從問題中，抽離出結構化搜尋意圖。
        【執行規則】：
        1. 提取規則：
        - 如果提到作者，填在 author 欄位（只能填 author 欄位列出的作者名稱），不要放進 filters。
        - 如果提到年份（如 2026年），必須在 filters 清單裡建立【兩條】規則進行前後包夾：
            #  * 第一條：key='post_date_timestamp', operator='GTE', value='2026-01-01T00:00:00Z'
            #  * 第二條：key='post_date_timestamp', operator='LTE', value='2026-12-31T23:59:59Z'
        2. 提取乾淨關鍵字（query）：
        - 徹底拔除問題中的作者名稱、年份（如 2026），只留下與主題相關的核心事件。"""

    # 步驟一：拆解 query——把作者/時間這類過濾條件抽出來，剩下乾淨的語意關鍵字，
    # 這是為了縮小搜尋範圍、提高精準度。
    output = filter_define_llm.invoke([
        ("system", system_instruction),
        ("human", human_query)]
        )

    filters = output.to_llama_index_filters

    # 步驟二：帶著拆解出來的過濾條件去搜尋。retrieve_nodes 保證回傳固定信封
    # {"success", "data", "error"}，data 是單純的 dict list，不會外流 llama_index 物件。
    retrieval = retrieve_nodes.invoke({
            "user_query": human_query,
            "filtered_query": output.query,
            "final_filters": filters
        })
    chunks = retrieval["data"] or []

    # 步驟三：品質驗證——這是輔助信號，不是主要內容，放進 meta。
    varrify_result = rag_judge_llm.invoke([
        ("system","Judge the quality of RAG result, just answer yes or no. If no, give me some feedback to improve."),
        ("user",f"Human query : {human_query} \n RAG search result : {chunks}")
    ])

    # 步驟四：組出給 answer_node 看的文字——誰產生資料，誰負責組成文字，
    # answer_node 只需要讀 tool_result["summary"]。
    parts = [
        f"[文件資訊]\n作者: {c['author']}\n標題: {c['title']}\n日期: {c['post_date']}\n章節: {c['header_path']}\n[內容]\n{c['text']}\n"
        for c in chunks
    ]
    summary = "\n\n".join(parts) if parts else (retrieval["error"] or "查無相關文件。")

    new_state = {**state, "tool_result": {
        "source": state["router"],
        "success": retrieval["success"],
        "summary": summary,
        "raw": chunks,
        "error": retrieval["error"],
        "meta": {
            "judge_feedback": varrify_result.feedback,
            "search_query": output.query,
            "filters_applied": filters is not None,
        },
    }}
    logger.debug("[rag_search_node] result: %s", new_state)
    return new_state



#=============Market Price Node===============
from market_tool import market_price_check

def market_price_node(state: AgentState) -> AgentState:
    user_query = state["messages"][-1].content

    logger.info("[market_price_node] 查詢市場價格: '%s'", user_query)

    # market_price_check 保證永遠回傳 {"success","data","error"} 這個固定信封，
    # 而且永遠不會把例外丟出來，這裡不需要再包 try/except。
    result = market_price_check.invoke({"query": user_query})
    data = result["data"] or {}

    new_state = {**state, "tool_result": {
        "source": state["router"],
        "success": result["success"],
        "summary": data.get("report") or result["error"] or "查價失敗，無回傳訊息。",
        "raw": {"fig": data.get("fig"), "symbols": data.get("symbols"), "mode": data.get("mode")},
        "error": result["error"],
        "meta": None,
    }}
    logger.debug("[market_price_node] result: %s", new_state)
    return new_state


#=============Answer Node  ===============

# 寫法由來：
# 因為對話歷史（state["messages"]）裡面可能已經混雜了 AIMessage 或是 ToolMessage。為了不被干擾，這行程式碼做了一次倒序搜尋：
# reversed(...)：從最新（最後面）的訊息往回（最前面）看。
# if isinstance(m, HumanMessage)：一邊倒著看，一邊找「哪一條是人類（User）講的話」。
# next(...)：一抓到最近的那一句，立刻停下來，把文字存進 user_q（例如："幫我抓 A 作者的文章"）。
answer_llm = ChatAnthropic(model="claude-sonnet-4-6", temperature=0.7,api_key =os.environ["ANTHROPIC_API_KEY"].strip() )
def answer_node(state: AgentState) -> AgentState:
    user_q = state["messages"][-1].content

    tool_result = state.get("tool_result")
    if tool_result:
        context = tool_result["summary"]
        feedback = (tool_result.get("meta") or {}).get("judge_feedback")
    else:
        context = "No external context available."
        feedback = None

    feedback_section = f"\nValidation feedback: {feedback}" if feedback else "No feedback."

    system_prompt = f"""Please answer the user's question using the provided context.

                Question: {user_q}

                Validation feadback: {feedback_section}

                    Context:
                    {context}

                    Provide a helpful, accurate, and concise response based on the available information."""

    ans = answer_llm.invoke([
            ("system", system_prompt),
            ("user", user_q)
        ]).content

    logger.debug("[answer_node] response: %s", ans)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }

 
#==============Chat Node==================

def chat_node (state):
    user_query = state["messages"][-1].content
    final_response = llm.invoke(state["messages"])
    ans = final_response.content
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=ans)]
    }   

# ── Routing helpers ─────────────────────────────────────────────────
def from_router(st: AgentState) -> Literal["download","rag","market_price","chat"]:
    return st["router"]





#==================Agent=======================
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
config = {"configurable": {"thread_id": "user_session_001"}}




from langgraph.graph import StateGraph, START, END
agent_graph = StateGraph(AgentState)
agent_graph.add_node("router",      router_node)
agent_graph.add_node("retrieve",  rag_search_node)
agent_graph.add_node("download article",  download_node)
agent_graph.add_node("market price", market_price_node)
agent_graph.add_node("chat", chat_node)
agent_graph.add_node("answer",  answer_node)

agent_graph.set_entry_point("router")

# 從 'router' 出發，根據 from_router 函數的結果，決定跳到哪一個節點
# 如果 from_router 回傳 "rag" -> 去 "retrieve"
agent_graph.add_conditional_edges(
    "router",
    from_router,
        {"download":"download article",
        "rag": "retrieve",
        "market_price": "market price",
         "chat":"chat"
         })

agent_graph.add_edge("retrieve", "answer")
agent_graph.add_edge("download article", "answer")
agent_graph.add_edge("market price", "answer")
agent_graph.add_edge("answer", END)
agent_graph.add_edge("chat", END)

agent = agent_graph.compile(checkpointer=memory)



def run_chat_loop():
    """執行對話迴圈的函式"""
    print("--- 對話開始！(輸入 'exit' 結束) ---")
    
    while True:
        query = input("\n👤 您: ")
        
        if query.lower() in ["exit", "quit", "掰掰"]:
            print("🤖 Assistant: 再見！")
            break
            
        try:
            print("🤖 Assistant: 思考中...")
            # checkpointer 已經打開，config 裡固定的 thread_id 會讓這個 CLI 對話有記憶
            initial_state = {"messages": [HumanMessage(content=query)]}
            result = agent.invoke(initial_state, config=config)

            ai_response = result["messages"][-1].content
            print(f"🤖 Assistant: {ai_response}")
            
        except Exception as e:
            print(f"❌ 發生錯誤: {e}")

    print("\nGoodbye!")

# 這是關鍵的入口點
if __name__ == "__main__":
    configure_logging()
    run_chat_loop()