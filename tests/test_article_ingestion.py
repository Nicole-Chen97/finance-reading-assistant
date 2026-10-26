"""
測試 article_ingestion.py 的 down_load_article。

這個函式改過之前，只要中間任何一步失敗，就會讓變數在沒被賦值的狀態下
被存取，導致例外從函式裡「意外」飛出去，而不是乾淨地回報失敗。
這裡驗證修好之後：任何失敗路徑都明確回傳 (None, None)，成功路徑維持原本行為。
"""

import article_ingestion as ai_module
from article_ingestion import ArticleIngestionPipeline


class FakeResponse:
    def __init__(self, status_code=200, json_data=None):
        self.status_code = status_code
        self._json_data = json_data or {}

    def json(self):
        return self._json_data


def make_pipeline(author_name="Author A"):
    return ArticleIngestionPipeline(author_name=author_name, vector_service=None)


def make_list_response(bylines=None):
    return FakeResponse(status_code=200, json_data={
        "posts": [{
            "id": "123",
            "publication_id": "pub1",
            "publishedBylines": bylines if bylines is not None else [{"name": "Test Author"}],
            "title": "Test Title",
            "type": "post",
            "post_date": "2026-01-01T00:00:00.000Z",
            "audience": "public",
            "canonical_url": "https://x.com",
            "slug": "test",
        }]
    })


def test_returns_none_none_for_unsupported_author():
    # Uncle_Stock_Note 在 DownLoad_Author 的 Literal 裡合法，但 AUTHOR_MAP 根本沒設定它的 cookie
    pipeline = make_pipeline(author_name="Uncle_Stock_Note")
    html, info = pipeline.down_load_article()
    assert (html, info) == (None, None)


def test_returns_none_none_when_list_request_is_not_200(monkeypatch):
    monkeypatch.setattr(ai_module.requests, "get", lambda *a, **k: FakeResponse(status_code=403))
    pipeline = make_pipeline()
    html, info = pipeline.down_load_article()
    assert (html, info) == (None, None)


def test_returns_none_none_when_list_request_raises(monkeypatch):
    def fake_get(*args, **kwargs):
        raise ConnectionError("network down")

    monkeypatch.setattr(ai_module.requests, "get", fake_get)
    pipeline = make_pipeline()
    html, info = pipeline.down_load_article()
    assert (html, info) == (None, None)


def test_returns_none_none_when_article_body_request_fails_after_list_succeeds(monkeypatch):
    calls = {"count": 0}

    def fake_get(url, params=None, headers=None, cookies=None):
        calls["count"] += 1
        if calls["count"] == 1:
            return make_list_response()
        raise ConnectionError("network down on 2nd call")

    monkeypatch.setattr(ai_module.requests, "get", fake_get)
    pipeline = make_pipeline()
    html, info = pipeline.down_load_article()
    assert (html, info) == (None, None)


def test_success_path_returns_html_and_article_info(monkeypatch):
    def fake_get(url, params=None, headers=None, cookies=None):
        if "profile/posts" in url:
            return make_list_response()
        return FakeResponse(status_code=200, json_data={"post": {"body_html": "<p>hello</p>"}})

    monkeypatch.setattr(ai_module.requests, "get", fake_get)
    pipeline = make_pipeline()
    html, info = pipeline.down_load_article()
    assert html == "<p>hello</p>"
    assert info["article_id"] == "123"
    assert info["author"] == "Test Author"


def test_defaults_to_unknown_author_when_no_bylines(monkeypatch):
    def fake_get(url, params=None, headers=None, cookies=None):
        if "profile/posts" in url:
            return make_list_response(bylines=[])
        return FakeResponse(status_code=200, json_data={"post": {"body_html": "<p>hello</p>"}})

    monkeypatch.setattr(ai_module.requests, "get", fake_get)
    pipeline = make_pipeline()
    html, info = pipeline.down_load_article()
    assert info["author"] == "未知作者"
