import pytest
from pydantic import ValidationError

from schema import SearchIntentPayload, Metadata_Filter


def test_no_filters_returns_none():
    payload = SearchIntentPayload(query="法拉利趨勢", filters=[])
    assert payload.to_llama_index_filters is None


def test_single_filter_with_condition_none_defaults_to_and():
    # 這是之前會炸掉的情境：只有 1 條 filter，LLM 沒有填 condition。
    payload = SearchIntentPayload(
        query="法拉利趨勢",
        filters=[Metadata_Filter(key="type", value="newsletter", operator="EQ")],
        condition=None,
    )
    result = payload.to_llama_index_filters
    assert result is not None
    assert result.condition.value == "and"


def test_author_becomes_eq_filter_with_database_value():
    payload = SearchIntentPayload(query="以太坊", author="Author B")
    result = payload.to_llama_index_filters
    assert len(result.filters) == 1
    assert result.filters[0].key == "author"
    assert result.filters[0].value == "Author B"
    assert result.filters[0].operator.value == "=="


def test_author_rejects_names_not_in_database():
    # 之前 prompt 叫 LLM 填 Author_B，資料庫存的是 Author B，filter 從沒生效過；
    # 現在 author 是 Literal，拼錯的值在格式驗證這層就會被擋下來
    with pytest.raises(ValidationError):
        SearchIntentPayload(query="以太坊", author="Author_B")


def test_author_and_year_filters_are_combined():
    payload = SearchIntentPayload(
        query="以太坊",
        author="Author A",
        filters=[
            Metadata_Filter(key="post_date_timestamp", value="2026-01-01T00:00:00Z", operator="GTE"),
            Metadata_Filter(key="post_date_timestamp", value="2026-12-31T23:59:59Z", operator="LTE"),
        ],
    )
    result = payload.to_llama_index_filters
    assert [f.key for f in result.filters] == ["author", "post_date_timestamp", "post_date_timestamp"]
    assert result.condition.value == "and"


def test_date_filter_value_converted_to_timestamp():
    payload = SearchIntentPayload(
        query="以太坊",
        filters=[
            Metadata_Filter(
                key="post_date_timestamp",
                value="2026-01-01T00:00:00Z",
                operator="GTE",
            )
        ],
        condition="AND",
    )
    result = payload.to_llama_index_filters
    assert isinstance(result.filters[0].value, float)
