from datetime import datetime

import polars as pl
import pytest

from datasubway.pre_agg_meta import (
    AGG_EXPANSION,
    AGG_NEEDED_COMPONENTS,
    PreAggregation,
    load_metadata,
    parse_pre_aggregations,
    save_metadata,
)

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SIMPLE_CONFIG = {
    "group_by": ["orders.date", "orders.region"],
    "aggregations": {"orders.revenue": "mean"},
}

MULTI_AGG_CONFIG = {
    "group_by": ["orders.date"],
    "aggregations": {"orders.revenue": ["mean", "sum"], "orders.qty": "max"},
}


@pytest.fixture
def simple_pre_agg():
    return PreAggregation(
        name="daily_region",
        group_by=["orders.date", "orders.region"],
        raw_aggregations={"orders.revenue": "mean"},
    )


# ---------------------------------------------------------------------------
# 1. Constants
# ---------------------------------------------------------------------------


def test_agg_needed_components_all_values():
    expected = {
        "Sum": {"sum"},
        "Min": {"min"},
        "Max": {"max"},
        "Count": {"count"},
        "Len": {"len"},
        "First": {"first"},
        "Last": {"last"},
        "Product": {"product"},
        "NullCount": {"null_count"},
        "All": {"all"},
        "Any": {"any"},
        "NUnique": {"unique_set"},
        "Median": {"values_list"},
        "Mean": {"sum", "count"},
        "Std": {"sum", "sumsq", "count"},
        "Var": {"sum", "sumsq", "count"},
    }
    assert AGG_NEEDED_COMPONENTS == expected


def test_agg_expansion_all_values():
    expected = {
        "sum": {"sum"},
        "min": {"min"},
        "max": {"max"},
        "count": {"count"},
        "len": {"len"},
        "first": {"first"},
        "last": {"last"},
        "product": {"product"},
        "null_count": {"null_count"},
        "all": {"all"},
        "any": {"any"},
        "n_unique": {"unique_set"},
        "median": {"values_list"},
        "mean": {"sum", "count"},
        "std": {"sum", "sumsq", "count"},
        "var": {"sum", "sumsq", "count"},
    }
    assert AGG_EXPANSION == expected


# ---------------------------------------------------------------------------
# 2. PreAggregation.__post_init__ — aggregation expansion
# ---------------------------------------------------------------------------


def test_pre_agg_single_string_agg_expanded():
    pa = PreAggregation(name="t", group_by=["g"], raw_aggregations={"col": "mean"})
    assert pa.aggregations["col"] == ["count", "sum"]


def test_pre_agg_list_aggs_merged_and_deduplicated():
    pa = PreAggregation(
        name="t", group_by=["g"], raw_aggregations={"col": ["mean", "sum"]}
    )
    # mean → {count, sum}; sum → {sum}; merged = {count, sum}
    assert pa.aggregations["col"] == ["count", "sum"]


def test_pre_agg_compound_std_expands_to_three_components():
    pa = PreAggregation(name="t", group_by=["g"], raw_aggregations={"col": "std"})
    assert pa.aggregations["col"] == ["count", "sum", "sumsq"]


def test_pre_agg_unknown_agg_raises():
    with pytest.raises(ValueError, match="Unknown aggregation"):
        PreAggregation(name="t", group_by=["g"], raw_aggregations={"col": "custom_agg"})


def test_pre_agg_multiple_columns_expand_independently():
    pa = PreAggregation(
        name="t",
        group_by=["g"],
        raw_aggregations={"revenue": "mean", "qty": "max"},
    )
    assert pa.aggregations["revenue"] == ["count", "sum"]
    assert pa.aggregations["qty"] == ["max"]


def test_pre_agg_empty_group_by_raises():
    with pytest.raises(ValueError, match="group_by"):
        PreAggregation(name="t", group_by=[], raw_aggregations={"col": "sum"})


def test_pre_agg_empty_aggregations_raises():
    with pytest.raises(ValueError, match="raw_aggregations"):
        PreAggregation(name="t", group_by=["g"], raw_aggregations={})


# ---------------------------------------------------------------------------
# 3. PreAggregation.covers()
# ---------------------------------------------------------------------------


def test_covers_exact_match_returns_true(simple_pre_agg):
    assert simple_pre_agg.covers(
        requested_group_by=["orders.date", "orders.region"],
        requested_aggs={"orders.revenue": {"Mean"}},
    )


def test_covers_superset_group_by_returns_true(simple_pre_agg):
    # Pre-agg has more group-by columns than requested — still covers
    assert simple_pre_agg.covers(
        requested_group_by=["orders.date"],
        requested_aggs={"orders.revenue": {"Mean"}},
    )


def test_covers_missing_group_by_returns_false(simple_pre_agg):
    assert not simple_pre_agg.covers(
        requested_group_by=["orders.date", "orders.region", "orders.country"],
        requested_aggs={"orders.revenue": {"Mean"}},
    )


def test_covers_column_absent_returns_false(simple_pre_agg):
    assert not simple_pre_agg.covers(
        requested_group_by=["orders.date"],
        requested_aggs={"orders.cost": {"Sum"}},
    )


def test_covers_missing_component_returns_false():
    # Pre-agg only stores 'sum' for col, but Mean needs both sum + count
    pa = PreAggregation(
        name="t",
        group_by=["g"],
        raw_aggregations={"col": "sum"},
    )
    assert not pa.covers(
        requested_group_by=["g"],
        requested_aggs={"col": {"Mean"}},
    )


def test_covers_compound_agg_all_components_present_returns_true():
    pa = PreAggregation(
        name="t",
        group_by=["g"],
        raw_aggregations={"col": "mean"},
    )
    assert pa.covers(
        requested_group_by=["g"],
        requested_aggs={"col": {"Mean"}},
    )


def test_covers_empty_group_by_always_satisfied(simple_pre_agg):
    assert simple_pre_agg.covers(
        requested_group_by=[],
        requested_aggs={"orders.revenue": {"Mean"}},
    )


# ---------------------------------------------------------------------------
# 4. PreAggregation.load() — tmp_path
# ---------------------------------------------------------------------------


def test_pre_agg_load_returns_lazyframe(tmp_path):
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    parquet_path = tmp_path / "test.parquet"
    df.write_parquet(parquet_path)

    pa = PreAggregation(
        name="t",
        group_by=["g"],
        raw_aggregations={"a": "sum"},
        file_path=parquet_path,
    )
    lf = pa.load()
    assert isinstance(lf, pl.LazyFrame)
    assert lf.collect().equals(df)


# ---------------------------------------------------------------------------
# 5. load_metadata / save_metadata — tmp_path
# ---------------------------------------------------------------------------


def test_load_metadata_returns_empty_dict_when_no_file(tmp_path):
    assert load_metadata(tmp_path) == {}


def test_save_metadata_creates_directory_and_writes(tmp_path):
    nested = tmp_path / "sub" / "dir"
    metadata = {"pre_agg_1": {"row_count": 42}}
    save_metadata(nested, metadata)
    assert (nested / "_metadata.json").exists()


def test_save_load_metadata_round_trip(tmp_path):
    metadata = {"pre_agg_1": {"row_count": 100, "written_at": "2024-01-15T12:00:00"}}
    save_metadata(tmp_path, metadata)
    loaded = load_metadata(tmp_path)
    assert loaded == metadata


def test_save_metadata_datetime_survives_as_string(tmp_path):
    dt = datetime(2024, 6, 1, 9, 30, 0)
    metadata = {"pre_agg_1": {"written_at": dt}}
    save_metadata(tmp_path, metadata)
    loaded = load_metadata(tmp_path)
    # default=str serializes datetime as ISO-format string
    assert loaded["pre_agg_1"]["written_at"] == str(dt)


# ---------------------------------------------------------------------------
# 6. parse_pre_aggregations — tmp_path
# ---------------------------------------------------------------------------

RAW_CONFIG = {
    "daily_region": {
        "group_by": ["orders.date", "orders.region"],
        "aggregations": {"orders.revenue": "mean"},
    }
}

MULTI_CONFIG = {
    "daily_region": {
        "group_by": ["orders.date", "orders.region"],
        "aggregations": {"orders.revenue": "mean"},
    },
    "monthly": {
        "group_by": ["orders.month"],
        "aggregations": {"orders.qty": "sum"},
    },
}


def test_parse_pre_aggregations_returns_pre_agg_objects(tmp_path):
    result = parse_pre_aggregations(RAW_CONFIG, tmp_path)
    assert len(result) == 1
    pa = result[0]
    assert isinstance(pa, PreAggregation)
    assert pa.name == "daily_region"
    assert pa.group_by == ["orders.date", "orders.region"]
    assert pa.aggregations == {"orders.revenue": ["count", "sum"]}


def test_parse_pre_aggregations_file_path_correct(tmp_path):
    result = parse_pre_aggregations(RAW_CONFIG, tmp_path)
    assert result[0].file_path == tmp_path / "daily_region.parquet"


def test_parse_pre_aggregations_no_metadata_defaults(tmp_path):
    result = parse_pre_aggregations(RAW_CONFIG, tmp_path)
    pa = result[0]
    assert pa.row_count == 0
    assert pa.written_at is None


def test_parse_pre_aggregations_with_metadata_populated(tmp_path):
    written_at = "2024-03-10T08:00:00"
    metadata = {"daily_region": {"row_count": 500, "written_at": written_at}}
    save_metadata(tmp_path, metadata)

    result = parse_pre_aggregations(RAW_CONFIG, tmp_path)
    pa = result[0]
    assert pa.row_count == 500
    assert pa.written_at == datetime.fromisoformat(written_at)


def test_parse_pre_aggregations_written_at_is_datetime(tmp_path):
    metadata = {"daily_region": {"row_count": 10, "written_at": "2024-01-01T00:00:00"}}
    save_metadata(tmp_path, metadata)

    result = parse_pre_aggregations(RAW_CONFIG, tmp_path)
    assert isinstance(result[0].written_at, datetime)


def test_parse_pre_aggregations_multiple_pre_aggs_returned(tmp_path):
    result = parse_pre_aggregations(MULTI_CONFIG, tmp_path)
    assert len(result) == 2
    names = {pa.name for pa in result}
    assert names == {"daily_region", "monthly"}
