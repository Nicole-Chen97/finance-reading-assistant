"""
測試 graph.py 裡新增的 market_price_node——查價現在是獨立路線，
不再借用 chat_node_2 的內部二次判斷。
"""

import graph as graph_module
from langchain_core.messages import HumanMessage


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.received_args = None

    def invoke(self, args):
        self.received_args = args
        return self.result


def make_state(query, router="market_price", tool_result="stale-from-last-turn"):
    return {
        "messages": [HumanMessage(content=query)],
        "router": router,
        "tool_result": tool_result,
    }


def test_market_price_node_writes_tool_result_on_success(monkeypatch):
    fake_fig = object()  # 隨便一個 sentinel，代表蠟燭圖物件
    fake_tool = FakeTool({
        "success": True,
        "data": {"report": "[BTC/USD 最新報價]\n收盤價：64382.355", "fig": fake_fig, "symbols": "BTC/USD", "mode": "latest"},
        "error": None,
    })
    monkeypatch.setattr(graph_module, "market_price_check", fake_tool)

    result = graph_module.market_price_node(make_state("比特幣現在多少錢？"))

    tr = result["tool_result"]
    assert fake_tool.received_args == {"query": "比特幣現在多少錢？"}
    assert tr["source"] == "market_price"
    assert tr["success"] is True
    assert tr["summary"] == "[BTC/USD 最新報價]\n收盤價：64382.355"
    assert tr["raw"] == {"fig": fake_fig, "symbols": "BTC/USD", "mode": "latest"}
    assert tr["error"] is None


def test_market_price_node_writes_tool_result_on_failure(monkeypatch):
    fake_tool = FakeTool({
        "success": False,
        "data": None,
        "error": "查詢市場價格時發生錯誤: 找不到代號 XYZ",
    })
    monkeypatch.setattr(graph_module, "market_price_check", fake_tool)

    result = graph_module.market_price_node(make_state("XYZ 現在多少錢？"))

    tr = result["tool_result"]
    assert tr["success"] is False
    assert tr["error"] == "查詢市場價格時發生錯誤: 找不到代號 XYZ"
    assert tr["summary"] == "查詢市場價格時發生錯誤: 找不到代號 XYZ"
    assert tr["raw"] == {"fig": None, "symbols": None, "mode": None}
