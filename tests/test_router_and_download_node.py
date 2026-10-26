"""
測試 graph.py 裡的 router_node 和 download_node。

策略：把 graph 模組裡的 LLM/工具「整個名字」換成假的（monkeypatch 模組屬性），
不去 mock StructuredTool 實例本身的方法（那是 pydantic 物件，直接改屬性容易踩雷）。
這樣可以只測「node 收到 LLM/工具的結果之後，state 有沒有被正確組裝」，
不用真的打 API、不用真的下載文章。
"""

import graph as graph_module
from langchain_core.messages import HumanMessage


class FakeLLM:
    """假裝是 router_llm / author_decide_llm，.invoke() 永遠回傳你指定好的物件"""

    def __init__(self, result):
        self.result = result

    def invoke(self, messages):
        return self.result


class FakeTool:
    """假裝是 download_substack_article 這個 @tool，.invoke() 回傳你指定好的 dict"""

    def __init__(self, result):
        self.result = result
        self.received_args = None
        self.call_count = 0

    def invoke(self, args):
        self.call_count += 1
        self.received_args = args
        return self.result


class RouteDecision:
    def __init__(self, route):
        self.route = route


class AuthorDecision:
    def __init__(self, author_name):
        self.author_name = author_name


def make_state(query, router=None, tool_result="stale-from-last-turn"):
    return {
        "messages": [HumanMessage(content=query)],
        "router": router,
        "tool_result": tool_result,
    }


# ---------- router_node ----------

def test_router_node_routes_to_download(monkeypatch):
    monkeypatch.setattr(graph_module, "router_llm", FakeLLM(RouteDecision("download")))
    result = graph_module.router_node(make_state("幫我下載 Author A 的文章"))
    assert result["router"] == "download"
    assert result["tool_result"] is None  # 清空上一輪殘留，避免 checkpointer 打開後污染這一輪


def test_router_node_routes_to_rag(monkeypatch):
    monkeypatch.setattr(graph_module, "router_llm", FakeLLM(RouteDecision("rag")))
    result = graph_module.router_node(make_state("Author B 對以太坊的看法是什麼？"))
    assert result["router"] == "rag"
    assert result["tool_result"] is None


def test_router_node_routes_smalltalk_to_chat(monkeypatch):
    monkeypatch.setattr(graph_module, "router_llm", FakeLLM(RouteDecision("chat")))
    result = graph_module.router_node(make_state("哈囉，你好嗎？"))
    assert result["router"] == "chat"
    assert result["tool_result"] is None


def test_router_node_routes_price_questions_to_market_price(monkeypatch):
    # market_price 現在是獨立路線，不再借用 chat 分類再二次判斷。
    monkeypatch.setattr(graph_module, "router_llm", FakeLLM(RouteDecision("market_price")))
    result = graph_module.router_node(make_state("特斯拉現在股價多少？"))
    assert result["router"] == "market_price"


# ---------- download_node ----------

def test_download_node_skips_tool_call_when_author_not_detected(monkeypatch):
    monkeypatch.setattr(graph_module, "author_decide_llm", FakeLLM(AuthorDecision("None")))
    fake_tool = FakeTool({"success": True, "data": {}, "error": None})
    monkeypatch.setattr(graph_module, "download_substack_article", fake_tool)

    result = graph_module.download_node(make_state("幫我下載文章", router="download"))

    assert fake_tool.call_count == 0  # 判斷不出作者，不該真的去呼叫下載工具
    assert result["tool_result"]["success"] is False
    assert result["tool_result"]["error"] == "無法判斷作者"


def test_download_node_writes_tool_result_on_success(monkeypatch):
    monkeypatch.setattr(graph_module, "author_decide_llm", FakeLLM(AuthorDecision("Author A")))
    fake_tool = FakeTool({
        "success": True,
        "data": {"status": "inserted", "success": True, "message": "🚀 寫入成功！"},
        "error": None,
    })
    monkeypatch.setattr(graph_module, "download_substack_article", fake_tool)

    result = graph_module.download_node(make_state("幫我下載 Author A 的文章", router="download"))

    tr = result["tool_result"]
    assert fake_tool.received_args == {"author_name": "Author A"}
    assert tr["source"] == "download"
    assert tr["success"] is True
    assert tr["summary"] == "🚀 寫入成功！"
    assert tr["error"] is None
    assert tr["meta"] == {"author": "Author A", "status": "inserted"}


def test_download_node_writes_tool_result_on_failure(monkeypatch):
    monkeypatch.setattr(graph_module, "author_decide_llm", FakeLLM(AuthorDecision("Author A")))
    fake_tool = FakeTool({
        "success": False,
        "data": None,
        "error": "工具在調度流水線時發生未預期的程式錯誤: boom",
    })
    monkeypatch.setattr(graph_module, "download_substack_article", fake_tool)

    result = graph_module.download_node(make_state("幫我下載 Author A 的文章", router="download"))

    tr = result["tool_result"]
    assert tr["success"] is False
    assert tr["error"] == "工具在調度流水線時發生未預期的程式錯誤: boom"
    assert tr["raw"] == {}
