"""
測試 graph.py 打開 checkpointer 之後，session 記憶真的有作用：
同一個 thread_id 跨輪對話能累積 messages；不同 thread_id 之間互不影響。

這裡不是測單一 node，而是直接呼叫 graph_module.agent.invoke(...)（真正編譯好的
graph，走真正的 conditional edges + checkpointer），只把會打 API 的 LLM 換成假的。
"""

import graph as graph_module
from langchain_core.messages import HumanMessage


class FakeChatResponse:
    def __init__(self, content):
        self.content = content


class FakeChatLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages):
        return FakeChatResponse(self.content)


class RouteDecision:
    def __init__(self, route):
        self.route = route


class FakeRouterLLM:
    def invoke(self, messages):
        return RouteDecision("chat")


def test_agent_accumulates_messages_across_turns_with_same_thread_id(monkeypatch):
    monkeypatch.setattr(graph_module, "router_llm", FakeRouterLLM())
    monkeypatch.setattr(graph_module, "llm", FakeChatLLM("哈囉！"))

    config = {"configurable": {"thread_id": "test-thread-same"}}

    graph_module.agent.invoke({"messages": [HumanMessage(content="你好")]}, config=config)
    result2 = graph_module.agent.invoke({"messages": [HumanMessage(content="我剛剛說了什麼？")]}, config=config)

    # 第二輪應該看得到第一輪的 human+ai，加上這一輪自己的 human+ai，總共 4 則
    assert len(result2["messages"]) == 4
    assert result2["messages"][0].content == "你好"
    assert result2["messages"][2].content == "我剛剛說了什麼？"


def test_agent_keeps_different_thread_ids_isolated(monkeypatch):
    monkeypatch.setattr(graph_module, "router_llm", FakeRouterLLM())
    monkeypatch.setattr(graph_module, "llm", FakeChatLLM("哈囉！"))

    graph_module.agent.invoke(
        {"messages": [HumanMessage(content="thread A 的訊息")]},
        config={"configurable": {"thread_id": "thread-A"}},
    )
    result_b = graph_module.agent.invoke(
        {"messages": [HumanMessage(content="thread B 的訊息")]},
        config={"configurable": {"thread_id": "thread-B"}},
    )

    # thread B 不該看到 thread A 的任何訊息
    assert len(result_b["messages"]) == 2
    assert result_b["messages"][0].content == "thread B 的訊息"


def test_tool_result_does_not_leak_across_turns_even_with_checkpointer_on(monkeypatch):
    # 這是我們之前在 router_node 加的清空邏輯要保護的情境：
    # 第一輪走 download（有 tool_result），第二輪換成 chat，
    # 不該讓第一輪的 tool_result 殘留到第二輪的 answer。
    monkeypatch.setattr(graph_module, "download_substack_article", type(
        "FakeTool", (), {"invoke": staticmethod(lambda args: {
            "success": True, "data": {"status": "inserted", "message": "寫入成功"}, "error": None,
        })}
    )())
    monkeypatch.setattr(graph_module, "author_decide_llm", type(
        "FakeAuthorLLM", (), {"invoke": staticmethod(lambda msgs: type("D", (), {"author_name": "Author A"})())}
    )())
    monkeypatch.setattr(graph_module, "llm", FakeChatLLM("哈囉！"))
    # download 路線最後會進 answer_node；沒換掉的話這個測試會真的打 Anthropic API
    monkeypatch.setattr(graph_module, "answer_llm", FakeChatLLM("已下載完成"))

    config = {"configurable": {"thread_id": "test-thread-no-leak"}}

    class RouterSequence:
        def __init__(self, routes):
            self.routes = list(routes)

        def invoke(self, messages):
            return RouteDecision(self.routes.pop(0))

    monkeypatch.setattr(graph_module, "router_llm", RouterSequence(["download", "chat"]))

    graph_module.agent.invoke({"messages": [HumanMessage(content="幫我下載 Author A 的文章")]}, config=config)
    result2 = graph_module.agent.invoke({"messages": [HumanMessage(content="哈囉")]}, config=config)

    assert result2["tool_result"] is None
