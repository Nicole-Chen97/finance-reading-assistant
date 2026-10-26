import chainlit as cl
from langchain_core.messages import HumanMessage
from graph import agent, configure_logging

# chainlit 是實際跑起來的應用程式入口，所以在這裡設定 logging——
# 個別模組只宣告 logger，不自己設定 handler/格式，避免跟其他入口點打架。
configure_logging()

# 在這裡放入你的 agent 初始化程式碼
# agent = ...
# config = ...


def extract_chart_figure(result: dict):
    """從 agent 的輸出結果裡，安全地取出查價才會有的圖表物件（沒有就回傳 None）。

    圖表不能經過 LLM（LLM 只能處理文字），所以走「附件」路線。
    只有 market_price 這個來源的 raw 裡才會有 fig，其他來源（download 的 raw 是 dict、
    rag 的 raw 是 list）形狀不一樣，所以先確認 source 再去拿 fig，避免對不是 dict 的
    raw 呼叫 .get() 而炸掉。
    """
    tool_result = result.get("tool_result") or {}
    if tool_result.get("source") == "market_price":
        return (tool_result.get("raw") or {}).get("fig")
    return None


@cl.on_chat_start
async def start():
    # 這裡只會在第一次開啟視窗時執行
    await cl.Message(content="你好！我是你的 AI 助手，請開始對話。").send()

@cl.on_message
async def main(message: cl.Message):
    # 用 chainlit 自己的 session.thread_id 當 LangGraph 的 thread_id，
    # 讓每個瀏覽器 session 有自己獨立的對話記憶，不會跟別人的對話混在一起。
    config = {"configurable": {"thread_id": cl.context.session.thread_id}}
    result = agent.invoke({"messages": [HumanMessage(content=message.content)]}, config=config)

    ai_response = result["messages"][-1].content

    fig = extract_chart_figure(result)
    elements = [cl.Plotly(name="market_chart", figure=fig)] if fig is not None else []
    await cl.Message(content=ai_response, elements=elements).send()