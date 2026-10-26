"""
測試 graph.py 裡的 rag_search_node 和 answer_node，驗證兩邊有沒有真的
對齊同一種 tool_result 信封（這是這次重構的核心：RAG 產生資料的格式，
要跟 answer_node 讀取資料的格式一致）。
"""

import graph as graph_module
from langchain_core.messages import HumanMessage


class FakeLLM:
    def __init__(self, result):
        self.result = result
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return self.result


class FakeChatResponse:
    def __init__(self, content):
        self.content = content


class FakeChatLLM:
    def __init__(self, content):
        self.content = content
        self.received_messages = None

    def invoke(self, messages):
        self.received_messages = messages
        return FakeChatResponse(self.content)


class FakeTool:
    def __init__(self, result):
        self.result = result
        self.received_args = None

    def invoke(self, args):
        self.received_args = args
        return self.result


class FakeSearchIntent:
    def __init__(self, query, filters=None):
        self.query = query
        self._filters = filters

    @property
    def to_llama_index_filters(self):
        return self._filters


class FakeJudge:
    def __init__(self, feedback):
        self.feedback = feedback


def make_state(query, router="rag", tool_result="stale-from-last-turn"):
    return {
        "messages": [HumanMessage(content=query)],
        "router": router,
        "tool_result": tool_result,
    }


# ---------- rag_search_node ----------

def test_rag_search_node_builds_tool_result_on_success(monkeypatch):
    monkeypatch.setattr(graph_module, "filter_define_llm", FakeLLM(FakeSearchIntent(query="以太坊", filters=None)))
    fake_retrieve = FakeTool({
        "success": True,
        "data": [{
            "author": "Author_B", "title": "以太坊分析", "post_date": "2026-01-01",
            "header_path": "第一章", "text": "以太坊近期看多", "score": 0.9,
        }],
        "error": None,
    })
    monkeypatch.setattr(graph_module, "retrieve_nodes", fake_retrieve)
    monkeypatch.setattr(graph_module, "rag_judge_llm", FakeLLM(FakeJudge(feedback="資料充足")))

    result = graph_module.rag_search_node(make_state("Author B 對以太坊的看法"))

    tr = result["tool_result"]
    assert tr["source"] == "rag"
    assert tr["success"] is True
    assert "Author_B" in tr["summary"]
    assert "以太坊近期看多" in tr["summary"]
    assert tr["raw"] == fake_retrieve.result["data"]
    assert tr["error"] is None
    assert tr["meta"] == {
        "judge_feedback": "資料充足",
        "search_query": "以太坊",
        "filters_applied": False,
    }


def test_rag_search_node_handles_no_results_gracefully(monkeypatch):
    monkeypatch.setattr(graph_module, "filter_define_llm", FakeLLM(FakeSearchIntent(query="不存在的主題")))
    monkeypatch.setattr(graph_module, "retrieve_nodes", FakeTool({"success": False, "data": [], "error": "查無相關文件"}))
    monkeypatch.setattr(graph_module, "rag_judge_llm", FakeLLM(FakeJudge(feedback=None)))

    result = graph_module.rag_search_node(make_state("問一個不存在的主題"))

    tr = result["tool_result"]
    assert tr["success"] is False
    assert tr["summary"] == "查無相關文件"
    assert tr["error"] == "查無相關文件"


def test_rag_search_node_records_whether_filters_were_applied(monkeypatch):
    monkeypatch.setattr(graph_module, "filter_define_llm", FakeLLM(FakeSearchIntent(query="法拉利", filters="some-filters-object")))
    monkeypatch.setattr(graph_module, "retrieve_nodes", FakeTool({"success": True, "data": [], "error": None}))
    monkeypatch.setattr(graph_module, "rag_judge_llm", FakeLLM(FakeJudge(feedback=None)))

    result = graph_module.rag_search_node(make_state("Author A 講法拉利的文章"))

    assert result["tool_result"]["meta"]["filters_applied"] is True


# ---------- answer_node ----------

def test_answer_node_uses_tool_result_summary_and_feedback(monkeypatch):
    fake_llm = FakeChatLLM("以太坊近期看多。")
    monkeypatch.setattr(graph_module, "answer_llm", fake_llm)

    state = make_state(
        "Author B 對以太坊的看法",
        tool_result={
            "source": "rag",
            "success": True,
            "summary": "[文件資訊]...以太坊近期看多...",
            "raw": [],
            "error": None,
            "meta": {"judge_feedback": "資料稍嫌不足"},
        },
    )

    result = graph_module.answer_node(state)

    assert result["messages"][-1].content == "以太坊近期看多。"
    system_prompt = fake_llm.received_messages[0][1]
    assert "以太坊近期看多" in system_prompt
    assert "資料稍嫌不足" in system_prompt


def test_answer_node_falls_back_when_no_tool_result(monkeypatch):
    fake_llm = FakeChatLLM("哈囉！")
    monkeypatch.setattr(graph_module, "answer_llm", fake_llm)

    state = make_state("哈囉", tool_result=None)
    result = graph_module.answer_node(state)

    system_prompt = fake_llm.received_messages[0][1]
    assert "No external context available." in system_prompt
    assert result["messages"][-1].content == "哈囉！"


# ---------- graph wiring ----------

def test_agent_graph_compiles():
    # 如果 wiring 有問題（例如某個 conditional edge 指到不存在的 node），
    # graph_module 在 import 階段就會直接炸掉，所以能執行到這裡就代表 compile 成功。
    assert graph_module.agent is not None
