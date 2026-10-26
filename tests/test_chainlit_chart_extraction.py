"""
測試 chainlit_app.py 的 extract_chart_figure——從 agent 的輸出結果裡，
安全地取出查價才會有的圖表物件。

@cl.on_message 那個 handler 本身需要一個活著的 chainlit session 才能呼叫
cl.Message(...).send()，沒辦法直接在 pytest 裡測，所以把「該不該顯示圖表」
這個純邏輯抽成一個獨立函式來測。
"""

from chainlit_app import extract_chart_figure


def test_returns_none_when_no_tool_result():
    assert extract_chart_figure({"messages": []}) is None


def test_returns_none_when_tool_result_is_none():
    assert extract_chart_figure({"messages": [], "tool_result": None}) is None


def test_returns_none_for_download_source_even_though_raw_is_a_dict():
    result = {
        "tool_result": {
            "source": "download",
            "raw": {"status": "inserted", "message": "寫入成功"},  # 沒有 fig 這個 key
        }
    }
    assert extract_chart_figure(result) is None


def test_returns_none_for_rag_source_without_crashing_on_list_raw():
    # rag 的 raw 是 list，不是 dict——這裡驗證不會因為對 list 呼叫 .get() 而炸掉
    result = {
        "tool_result": {
            "source": "rag",
            "raw": [{"author": "Author_B", "text": "..."}],
        }
    }
    assert extract_chart_figure(result) is None


def test_returns_none_when_market_price_did_not_request_a_chart():
    result = {
        "tool_result": {
            "source": "market_price",
            "raw": {"fig": None, "symbols": "BTC/USD", "mode": "latest"},
        }
    }
    assert extract_chart_figure(result) is None


def test_returns_figure_when_market_price_has_one():
    fake_fig = object()
    result = {
        "tool_result": {
            "source": "market_price",
            "raw": {"fig": fake_fig, "symbols": "AAPL", "mode": "history"},
        }
    }
    assert extract_chart_figure(result) is fake_fig
