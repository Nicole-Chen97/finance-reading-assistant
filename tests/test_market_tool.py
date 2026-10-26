import pytest

import market_tool as market_tool_module
from market_tool import AlpacaQueryParams, generate_market_report, plot_candlestick, market_price_check

# assert 是 Python 用來「檢查條件是否成立」的關鍵字。如果條件成立，測試就通過（綠燈）；如果條件不成立，測試就會失敗報錯（紅燈）

# ---------- AlpacaQueryParams.validate_logic ----------

def test_stock_without_feed_defaults_to_iex():
    p = AlpacaQueryParams(category="stock", symbols="AAPL")
    assert p.feed == "iex"


def test_crypto_without_feed_stays_none():
    p = AlpacaQueryParams(category="crypto", symbols="BTC/USD")
    assert p.feed is None


def test_latest_mode_clears_timeframe_and_fig():
    p = AlpacaQueryParams(category="stock", symbols="AAPL", mode="latest", fig=True)
    assert p.timeframe is None
    assert p.fig is False


def test_history_mode_defaults_fig_to_false_when_unset():
    p = AlpacaQueryParams(category="stock", symbols="AAPL", mode="history")
    assert p.fig is False


def test_history_mode_parses_string_start_and_end():
    p = AlpacaQueryParams(
        category="stock",
        symbols="AAPL",
        mode="history",
        start="2026-01-01",
        end="2026-01-15",
    )
    assert p.start == "2026-01-01T00:00:00Z"
    assert p.end == "2026-01-15T00:00:00Z"


def test_history_mode_defaults_start_to_90_days_before_end():
    p = AlpacaQueryParams(
        category="stock", symbols="AAPL", mode="history", end="2026-01-15"
    )
    assert p.start == "2025-10-17T00:00:00Z"


# ---------- generate_market_report ----------

def test_latest_report_with_dict_price_info():
    result = {
        "bars": {
            "AAPL": {"t": "2026-01-15T20:00:00Z", "vw": 150.25, "c": 150.5, "v": 1000000}
        }
    }
    report = generate_market_report(result, "AAPL", "latest")
    assert "AAPL 最新報價" in report
    assert "150.5" in report


def test_latest_report_with_list_wrapped_price_info():
    # Alpaca 有時候會把 latest 包成一個只有 1 筆的 list，不是單一 dict
    result = {
        "bars": {
            "AAPL": [{"t": "2026-01-15T20:00:00Z", "vw": 150.25, "c": 150.5, "v": 1000000}]
        }
    }
    report = generate_market_report(result, "AAPL", "latest")
    assert "AAPL 最新報價" in report
    assert "150.5" in report


def test_history_report_describes_the_whole_range_not_just_latest_bar():
    # 刻意打亂順序，驗證 sort_values(by='t', ascending=False) 真的有把最新的排到第一筆。
    # history 模式代表一段區間，摘要要能看出起訖日期、區間最高/最低（不是只有最新一天的），
    # 還有整體漲跌幅——不然 LLM 會誤以為只查得到一天的資料。
    result = {
        "bars": {
            "AAPL": [
                {"t": "2026-01-13T00:00:00Z", "o": 100, "h": 105, "l": 99, "c": 102, "v": 1000},
                {"t": "2026-01-15T00:00:00Z", "o": 102, "h": 110, "l": 101, "c": 108, "v": 1200},
                {"t": "2026-01-14T00:00:00Z", "o": 101, "h": 107, "l": 100, "c": 105, "v": 1100},
            ]
        }
    }
    report = generate_market_report(result, "AAPL", "history")
    assert "2026年01月13日" in report  # 區間起始日
    assert "2026年01月15日" in report  # 區間結束日（也是最新收盤）
    assert "共 3 筆" in report
    assert "$108.00" in report  # 最新收盤價
    assert "$110.00" in report  # 區間最高（三天裡的最高，不是只看最新一天）
    assert "$99.00" in report   # 區間最低（三天裡的最低，不是只看最新一天）
    assert "+5.88%" in report  # (108-102)/102*100


def test_history_report_with_single_dict_not_wrapped_in_list():
    result = {
        "bars": {
            "AAPL": {"t": "2026-01-15T00:00:00Z", "o": 102, "h": 110, "l": 101, "c": 108, "v": 1200}
        }
    }
    report = generate_market_report(result, "AAPL", "history")
    assert "2026年01月15日" in report
    assert "$108.00" in report
    assert "+0.00%" in report  # 只有一筆資料，起訖同一天，漲跌幅為 0


# ---------- plot_candlestick ----------

def test_plot_candlestick_raises_when_symbol_missing():
    with pytest.raises(ValueError):
        plot_candlestick({"bars": {}}, "AAPL", "1Day")


def test_plot_candlestick_returns_figure_for_valid_bars():
    json_data = {
        "bars": {
            "AAPL": [
                {"t": "2026-01-13T00:00:00Z", "o": 100, "h": 105, "l": 99, "c": 102, "v": 1000},
                {"t": "2026-01-14T00:00:00Z", "o": 101, "h": 107, "l": 100, "c": 105, "v": 1100},
            ]
        }
    }
    fig = plot_candlestick(json_data, "AAPL", "1Day")
    assert fig is not None


# ---------- market_price_check ----------
# checkprice_llm 是在函式內部才建立的（不是模組層級變數），
# 所以要 mock 整個 ChatAnthropic 類別本身，讓它建構出來的物件符合我們要的假回應。

def make_fake_chat_anthropic_class(decision):
    class FakeStructuredLLM:
        def invoke(self, messages):
            return decision

    class FakeChatAnthropic:
        def __init__(self, model=None, temperature=None, api_key=None):
            pass

        def with_structured_output(self, schema):
            return FakeStructuredLLM()

    return FakeChatAnthropic


def test_market_price_check_returns_success_envelope(monkeypatch):
    decision = AlpacaQueryParams(category="crypto", symbols="BTC/USD", mode="latest")
    monkeypatch.setattr(market_tool_module, "ChatAnthropic", make_fake_chat_anthropic_class(decision))
    monkeypatch.setattr(market_tool_module, "fetch_alpaca_data", lambda params: {
        "bars": {"BTC/USD": {"t": "2026-01-15T20:00:00Z", "vw": 64364.479, "c": 64382.355, "v": 3.2e-05}}
    })

    result = market_price_check.func(query="比特幣現在多少錢？")

    assert result["success"] is True
    assert result["error"] is None
    assert "BTC/USD 最新報價" in result["data"]["report"]
    assert result["data"]["fig"] is None  # mode=latest 時 fig 一律是 False
    assert result["data"]["symbols"] == "BTC/USD"


def test_market_price_check_surfaces_fetch_error_without_crashing(monkeypatch):
    decision = AlpacaQueryParams(category="stock", symbols="ZZZZ", mode="latest")
    monkeypatch.setattr(market_tool_module, "ChatAnthropic", make_fake_chat_anthropic_class(decision))
    monkeypatch.setattr(market_tool_module, "fetch_alpaca_data", lambda params: {"error": "HTTP 錯誤: 404 找不到代號"})

    result = market_price_check.func(query="ZZZZ 現在多少錢？")

    assert result["success"] is False
    assert result["data"] is None
    assert result["error"] == "HTTP 錯誤: 404 找不到代號"


def test_market_price_check_never_raises_on_unexpected_crash(monkeypatch):
    class CrashingStructuredLLM:
        def invoke(self, messages):
            raise RuntimeError("Claude 連不上")

    class CrashingChatAnthropic:
        def __init__(self, model=None, temperature=None, api_key=None):
            pass

        def with_structured_output(self, schema):
            return CrashingStructuredLLM()

    monkeypatch.setattr(market_tool_module, "ChatAnthropic", CrashingChatAnthropic)

    result = market_price_check.func(query="隨便問問")

    assert result["success"] is False
    assert result["data"] is None
    assert "Claude 連不上" in result["error"]
