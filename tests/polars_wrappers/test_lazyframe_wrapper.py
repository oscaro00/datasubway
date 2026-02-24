import datetime

import polars as pl

from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
from datasubway.polars_wrappers.lazygroupby_wrapper import LazyGroupByWrapper

SAMPLE_LF = pl.LazyFrame(
    {
        "country": ["US", "CA", "UK", "CA"],
        "revenue": [500, 1500, 200, 800],
        "score": [None, 1.0, None, 2.0],
    }
)

# LazyFrame with pre-aggregated columns for from_pre_agg tests
PRE_AGG_LF = pl.LazyFrame(
    {
        "country": ["US", "CA"],
        "revenue-sum": [500, 2300],
        "revenue-count": [1, 2],
    }
)


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_stores_lazyframe():
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    wrapper = LazyFrameWrapper(lf)
    assert wrapper.lf is lf


def test_init_from_pre_agg_defaults_false():
    wrapper = LazyFrameWrapper(pl.LazyFrame({"a": [1]}))
    assert wrapper.from_pre_agg is False


def test_init_from_pre_agg_can_be_true():
    wrapper = LazyFrameWrapper(pl.LazyFrame({"a": [1]}), from_pre_agg=True)
    assert wrapper.from_pre_agg is True


# ---------------------------------------------------------------------------
# __getattr__ delegation
# ---------------------------------------------------------------------------


def test_getattr_delegates_to_underlying_lazyframe():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.select(pl.col("country"))
    rows = result.collect().to_dicts()
    assert rows == [
        {"country": "US"},
        {"country": "CA"},
        {"country": "UK"},
        {"country": "CA"},
    ]


def test_getattr_wraps_lazyframe_result_as_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.with_columns(pl.lit(1).alias("x"))
    assert isinstance(result, LazyFrameWrapper)


def test_getattr_non_lazyframe_result_returned_directly():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    schema = wrapper.collect_schema()
    assert not isinstance(schema, LazyFrameWrapper)


# ---------------------------------------------------------------------------
# filter
# ---------------------------------------------------------------------------


def test_filter_no_predicates_returns_self():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert wrapper.filter() is wrapper


def test_filter_expression_predicate():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.filter(pl.col("country") == "US")
    rows = result.collect().to_dicts()
    assert rows == [{"country": "US", "revenue": 500, "score": None}]


def test_filter_keyword_constraint():
    lf = pl.LazyFrame({"country": ["US", "CA"], "revenue": [100, 200]})
    wrapper = LazyFrameWrapper(lf)
    result = wrapper.filter(country="US")
    rows = result.collect().to_dicts()
    assert rows == [{"country": "US", "revenue": 100}]


def test_filter_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.filter(pl.col("revenue") > 1000), LazyFrameWrapper)


def test_filter_preserves_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    result = wrapper.filter(pl.col("revenue") > 500)
    assert result.from_pre_agg is True


# ---------------------------------------------------------------------------
# group_by
# ---------------------------------------------------------------------------


def test_group_by_no_args_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.group_by(), LazyFrameWrapper)


def test_group_by_no_args_preserves_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    assert wrapper.group_by().from_pre_agg is True


def test_group_by_with_column_returns_lazygroupbywrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.group_by("country"), LazyGroupByWrapper)


def test_group_by_with_expr_returns_lazygroupbywrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.group_by(pl.col("country")), LazyGroupByWrapper)


def test_group_by_propagates_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    assert wrapper.group_by("country").from_pre_agg is True


# ---------------------------------------------------------------------------
# group_by_dynamic
# ---------------------------------------------------------------------------


def test_group_by_dynamic_none_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.group_by_dynamic(None, every="1d"), LazyFrameWrapper)


def test_group_by_dynamic_with_column_returns_lazygroupbywrapper():
    dates = [datetime.date(2024, 1, i + 1) for i in range(4)]
    lf = pl.LazyFrame({"date": dates, "revenue": [100, 200, 150, 300]})
    wrapper = LazyFrameWrapper(lf)
    assert isinstance(wrapper.group_by_dynamic("date", every="2d"), LazyGroupByWrapper)


# ---------------------------------------------------------------------------
# rolling
# ---------------------------------------------------------------------------


def test_rolling_none_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.rolling(None, period="1d"), LazyFrameWrapper)


def test_rolling_none_preserves_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    assert wrapper.rolling(None, period="1d").from_pre_agg is True


def test_rolling_with_column_returns_lazygroupbywrapper():
    dates = [datetime.date(2024, 1, i + 1) for i in range(4)]
    lf = pl.LazyFrame({"date": dates, "revenue": [100, 200, 150, 300]})
    wrapper = LazyFrameWrapper(lf)
    assert isinstance(wrapper.rolling("date", period="2d"), LazyGroupByWrapper)


# ---------------------------------------------------------------------------
# sort
# ---------------------------------------------------------------------------


def test_sort_none_by_returns_self():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert wrapper.sort(None) is wrapper


def test_sort_ascending():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.sort("revenue")
    revenues = [r["revenue"] for r in result.collect().to_dicts()]
    assert revenues == sorted(revenues)


def test_sort_descending():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.sort("revenue", descending=True)
    revenues = [r["revenue"] for r in result.collect().to_dicts()]
    assert revenues == sorted(revenues, reverse=True)


def test_sort_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.sort("country"), LazyFrameWrapper)


def test_sort_preserves_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    assert wrapper.sort("revenue").from_pre_agg is True


# ---------------------------------------------------------------------------
# agg (fallback — no group_by)
# ---------------------------------------------------------------------------


def test_agg_without_pre_agg_uses_select():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.agg(pl.col("revenue"))
    revenues = [r["revenue"] for r in result.collect().to_dicts()]
    assert revenues == [500, 1500, 200, 800]


def test_agg_named_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    result = wrapper.agg(total=pl.col("revenue").sum())
    assert result.collect().to_dicts() == [{"total": 3000}]


def test_agg_with_pre_agg_rewrites_sum():
    wrapper = LazyFrameWrapper(PRE_AGG_LF, from_pre_agg=True)
    result = wrapper.agg(pl.col("revenue").sum())
    # pl.col("revenue").sum() is rewritten to pl.col("revenue-sum").sum()
    assert result.collect().to_dicts() == [{"revenue-sum": 2800}]


def test_agg_with_pre_agg_named_preserves_alias():
    wrapper = LazyFrameWrapper(PRE_AGG_LF, from_pre_agg=True)
    result = wrapper.agg(total=pl.col("revenue").sum())
    assert result.collect().to_dicts() == [{"total": 2800}]


def test_agg_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.agg(pl.col("revenue")), LazyFrameWrapper)


def test_agg_preserves_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    assert wrapper.agg(pl.col("revenue")).from_pre_agg is True


# ---------------------------------------------------------------------------
# all
# ---------------------------------------------------------------------------


def test_all_reduces_boolean_lists():
    lf = pl.LazyFrame({"flags": [[True, True], [True, False]]})
    wrapper = LazyFrameWrapper(lf)
    rows = wrapper.all().collect().to_dicts()
    assert rows == [{"flags": True}, {"flags": False}]


def test_all_returns_lazyframewrapper():
    lf = pl.LazyFrame({"flags": [[True, True]]})
    assert isinstance(LazyFrameWrapper(lf).all(), LazyFrameWrapper)


# ---------------------------------------------------------------------------
# having
# ---------------------------------------------------------------------------


def test_having_is_noop_returns_self():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert wrapper.having(pl.col("revenue") > 500) is wrapper


# ---------------------------------------------------------------------------
# len
# ---------------------------------------------------------------------------


def test_len_default_name():
    lf = pl.LazyFrame({"revenue": [100, 200, 300]})
    wrapper = LazyFrameWrapper(lf)
    assert wrapper.len().collect().to_dicts() == [{"len": 3}]


def test_len_custom_name():
    lf = pl.LazyFrame({"revenue": [100, 200, 300]})
    wrapper = LazyFrameWrapper(lf)
    assert wrapper.len(name="count").collect().to_dicts() == [{"count": 3}]


def test_len_returns_lazyframewrapper():
    lf = pl.LazyFrame({"revenue": [1, 2]})
    assert isinstance(LazyFrameWrapper(lf).len(), LazyFrameWrapper)


# ---------------------------------------------------------------------------
# map_groups
# ---------------------------------------------------------------------------


def test_map_groups_applies_function():
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    wrapper = LazyFrameWrapper(lf)
    result = wrapper.map_groups(
        lambda df: df.with_columns(pl.col("a") * 2), schema=None
    )
    assert [r["a"] for r in result.collect().to_dicts()] == [2, 4, 6]


def test_map_groups_returns_lazyframewrapper():
    wrapper = LazyFrameWrapper(SAMPLE_LF)
    assert isinstance(wrapper.map_groups(lambda df: df, schema=None), LazyFrameWrapper)


def test_map_groups_preserves_from_pre_agg():
    wrapper = LazyFrameWrapper(SAMPLE_LF, from_pre_agg=True)
    assert wrapper.map_groups(lambda df: df, schema=None).from_pre_agg is True


# ---------------------------------------------------------------------------
# n_unique
# ---------------------------------------------------------------------------


def test_n_unique_correct_count():
    lf = pl.LazyFrame({"country": ["US", "CA", "US"]})
    wrapper = LazyFrameWrapper(lf)
    assert wrapper.n_unique().collect().to_dicts() == [{"country": 2}]


def test_n_unique_returns_lazyframewrapper():
    lf = pl.LazyFrame({"a": [1, 2, 3]})
    assert isinstance(LazyFrameWrapper(lf).n_unique(), LazyFrameWrapper)
