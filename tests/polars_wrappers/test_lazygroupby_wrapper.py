import polars as pl

from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
from datasubway.polars_wrappers.lazygroupby_wrapper import LazyGroupByWrapper

SAMPLE_LF = pl.LazyFrame(
    {
        "country": ["US", "CA", "UK", "CA"],
        "revenue": [500, 1500, 200, 800],
    }
)

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


def test_init_stores_lgb():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    assert wrapper.lgb is lgb


def test_init_from_pre_agg_defaults_false():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    assert wrapper.from_pre_agg is False


def test_init_from_pre_agg_can_be_true():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb, from_pre_agg=True)
    assert wrapper.from_pre_agg is True


# ---------------------------------------------------------------------------
# __getattr__ delegation
# ---------------------------------------------------------------------------


def test_getattr_delegates_head_returns_lazyframewrapper():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.head(1)
    assert isinstance(result, LazyFrameWrapper)


def test_getattr_head_correct_row_count():
    # 3 unique countries → head(1) yields 1 row per group = 3 rows
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.head(1)
    assert len(result.collect()) == 3


def test_getattr_tail_returns_lazyframewrapper():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.tail(1)
    assert isinstance(result, LazyFrameWrapper)


# ---------------------------------------------------------------------------
# agg without from_pre_agg
# ---------------------------------------------------------------------------


def test_agg_returns_lazyframewrapper():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.agg(pl.col("revenue").sum())
    assert isinstance(result, LazyFrameWrapper)


def test_agg_basic_sum():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.agg(pl.col("revenue").sum())
    rows_by_country = {r["country"]: r["revenue"] for r in result.collect().to_dicts()}
    assert rows_by_country == {"US": 500, "CA": 2300, "UK": 200}


def test_agg_named_agg():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.agg(total=pl.col("revenue").sum())
    rows_by_country = {r["country"]: r["total"] for r in result.collect().to_dicts()}
    assert rows_by_country == {"US": 500, "CA": 2300, "UK": 200}


def test_agg_multiple_positional_exprs():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.agg(
        pl.col("revenue").sum().alias("total"),
        pl.col("revenue").count().alias("cnt"),
    )
    rows_by_country = {r["country"]: r for r in result.collect().to_dicts()}
    assert rows_by_country["US"]["total"] == 500
    assert rows_by_country["CA"]["total"] == 2300
    assert rows_by_country["US"]["cnt"] == 1
    assert rows_by_country["CA"]["cnt"] == 2


def test_agg_preserves_from_pre_agg_false():
    lgb = SAMPLE_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb)
    result = wrapper.agg(pl.col("revenue").sum())
    assert result.from_pre_agg is False


# ---------------------------------------------------------------------------
# agg with from_pre_agg=True
# ---------------------------------------------------------------------------


def test_agg_from_pre_agg_rewrites_sum():
    # pl.col("revenue").sum() is rewritten to pl.col("revenue-sum").sum()
    lgb = PRE_AGG_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb, from_pre_agg=True)
    result = wrapper.agg(pl.col("revenue").sum())
    rows_by_country = {r["country"]: r["revenue-sum"] for r in result.collect().to_dicts()}
    assert rows_by_country == {"US": 500, "CA": 2300}


def test_agg_from_pre_agg_named_rewrites_expr():
    lgb = PRE_AGG_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb, from_pre_agg=True)
    result = wrapper.agg(total=pl.col("revenue").sum())
    rows_by_country = {r["country"]: r["total"] for r in result.collect().to_dicts()}
    assert rows_by_country == {"US": 500, "CA": 2300}


def test_agg_from_pre_agg_preserves_flag():
    lgb = PRE_AGG_LF.group_by("country")
    wrapper = LazyGroupByWrapper(lgb, from_pre_agg=True)
    result = wrapper.agg(pl.col("revenue").sum())
    assert result.from_pre_agg is True
