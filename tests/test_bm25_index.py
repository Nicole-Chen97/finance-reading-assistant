from llama_index.core.vector_stores.types import (
    FilterCondition,
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

from bm25_index import matches_filters, tokenize


# ---------- 斷詞 ----------

def test_tokenize_keeps_traditional_characters():
    # 文章幾乎全是繁體，所以不轉簡體，斷出來的詞也是繁體
    assert "比特幣" in tokenize("比特幣的走勢")


def test_tokenize_handles_taiwan_variant_of_taiwan():
    # 繁體大詞典只收「臺灣」；沒補台灣用字時，「怎麼看台灣」會被切成「看台 / 灣的」
    assert "台灣" in tokenize("Author A 怎麼看台灣的國防產業化？")


def test_tokenize_lowercases_english_terms():
    assert "ondo" in tokenize("Author B 對 ONDO 的看法")


def test_tokenize_drops_stopwords_and_punctuation():
    tokens = tokenize("柯立芝時代的股市表現怎麼樣？")
    assert "柯立芝" in tokens
    assert "的" not in tokens
    assert "？" not in tokens


def test_tokenize_keeps_user_dict_terms_whole():
    # 沒有自訂詞典時，jieba 會把「創世使命」切成「創世 / 使命」
    assert "創世使命" in tokenize("創世使命系列文章總共有幾篇？")


def test_tokenize_keeps_both_spellings_of_fibonacci():
    # 文章裡兩種寫法都有（斐波那契 42 次、費波那契 11 次），兩種都要能完整切出來
    assert "斐波那契" in tokenize("用斐波那契回撤")
    assert "費波那契" in tokenize("用費波那契回撤")


# ---------- metadata filter（要跟向量檢索套用同樣的條件） ----------

def _filters(*filters, condition=FilterCondition.AND):
    return MetadataFilters(filters=list(filters), condition=condition)


def test_no_filters_matches_everything():
    assert matches_filters({"author": "Author A"}, None)


def test_author_eq_filter():
    f = _filters(MetadataFilter(key="author", value="Author B", operator=FilterOperator.EQ))
    assert matches_filters({"author": "Author B"}, f)
    assert not matches_filters({"author": "Author A"}, f)


def test_year_range_filter_on_timestamp():
    f = _filters(
        MetadataFilter(key="post_date_timestamp", value=1767225600.0, operator=FilterOperator.GTE),  # 2026-01-01
        MetadataFilter(key="post_date_timestamp", value=1798761599.0, operator=FilterOperator.LTE),  # 2026-12-31
    )
    assert matches_filters({"post_date_timestamp": 1783145862}, f)  # 2026-07
    assert not matches_filters({"post_date_timestamp": 1735689600}, f)  # 2025-01


def test_or_condition():
    f = _filters(
        MetadataFilter(key="author", value="Author A", operator=FilterOperator.EQ),
        MetadataFilter(key="author", value="Author B", operator=FilterOperator.EQ),
        condition=FilterCondition.OR,
    )
    assert matches_filters({"author": "Author B"}, f)


def test_missing_metadata_key_does_not_match():
    f = _filters(MetadataFilter(key="author", value="Author A", operator=FilterOperator.EQ))
    assert not matches_filters({}, f)
