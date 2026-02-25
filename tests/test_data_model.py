import asyncio
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from datasubway.column_context import allow, exclude
from datasubway.data_model import DataModel
from datasubway.measure_decorator import measure
from datasubway.polars_wrappers.proxy import LazyFrameProxy
from datasubway.pre_agg_meta import PreAggregation
from datasubway.query_context import QueryContext

# ---------------------------------------------------------------------------
# Module-level test data
# ---------------------------------------------------------------------------

ORDERS_LF = pl.LazyFrame(
    {
        "order_id": [1, 2, 3, 4, 5],
        "customer_id": [1, 1, 2, 2, 3],
        "amount": [100.0, 200.0, 150.0, 300.0, 50.0],
        "region": ["US", "US", "CA", "CA", "US"],
    }
)

CUSTOMERS_LF = pl.LazyFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def simple_dm():
    return DataModel(tables={"orders": ORDERS_LF})


def _make_qc(**kwargs):
    """Build QueryContext object for validate_query_context tests."""
    return QueryContext(
        {
            "measures": kwargs.get("measures", []),
            "filters": kwargs.get("filters", {}),
            "groups": kwargs.get("groups", []),
            "havings": kwargs.get("havings", {}),
            "sorts": kwargs.get("sorts", []),
            "limit": kwargs.get("limit", 10000),
            "offset": kwargs.get("offset", 0),
        }
    )


def _dm_with_measures():
    """Return a DataModel with 'revenue', 'order_count', and 'unique_customers' measures registered."""
    dm = DataModel(tables={"orders": ORDERS_LF})

    @measure(dm)
    def revenue(qc: QueryContext):
        return (
            dm.table("orders")
            .filter(allow(pattern="*", context=qc.filters))
            .group_by(allow(pattern="*", context=qc.groups))
            .agg(pl.col("orders.amount").sum().alias("revenue"))
        )

    @measure(dm)
    def order_count(qc: QueryContext):
        return (
            dm.table("orders")
            .filter(allow(pattern="*", context=qc.filters))
            .group_by(allow(pattern="*", context=qc.groups))
            .agg(pl.col("orders.order_id").count().alias("order_count"))
        )

    @measure(dm)
    def unique_customers_by_region(qc: QueryContext):
        return (
            dm.table("orders")
            .filter(allow(pattern="*", context=qc.filters))
            .group_by(exclude(pattern="*", include="orders.region", context=qc.groups))
            .agg(
                pl.col("orders.customer_id")
                .n_unique()
                .alias("unique_customers_by_region")
            )
        )

    return dm


# ---------------------------------------------------------------------------
# TestDataModelInit
# ---------------------------------------------------------------------------


class TestDataModelInit:
    def test_basic_init(self):
        dm = DataModel(tables={"orders": ORDERS_LF, "customers": CUSTOMERS_LF})
        assert "orders" in dm.tables
        assert "customers" in dm.tables
        assert dm.joins == []
        assert dm.pre_agg_objects == []
        assert dm.measures == {}

    def test_table_schemas_populated(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        assert set(dm.table_schemas["orders"]) == {
            "order_id",
            "customer_id",
            "amount",
            "region",
        }

    def test_default_pre_agg_directory(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        assert dm.pre_agg_directory == Path("_pre_aggregations/")

    def test_custom_pre_agg_directory(self):
        custom = Path("/tmp/my_pre_aggs")
        dm = DataModel(tables={"orders": ORDERS_LF}, pre_agg_directory=custom)
        assert dm.pre_agg_directory == custom

    def test_logging_directory(self):
        log_dir = Path("/tmp/logs")
        dm = DataModel(tables={"orders": ORDERS_LF}, logging_directory=log_dir)
        assert dm.logging_directory == log_dir

    def test_init_with_joins(self):
        joins = [
            {
                "left": "orders",
                "right": "customers",
                "left_on": ["customer_id"],
                "right_on": ["customer_id"],
                "how": "inner",
                "direction": "right2left",
            }
        ]
        dm = DataModel(
            tables={"orders": ORDERS_LF, "customers": CUSTOMERS_LF},
            joins=joins,
        )
        assert dm.joins_lookup != {}
        assert "orders" in dm.joins_lookup
        assert "customers" in dm.joins_lookup["orders"]

    def test_init_with_pre_aggregations(self):
        pre_aggs = {
            "region_summary": {
                "group_by": ["orders.region"],
                "aggregations": {"orders.amount": "sum"},
            }
        }
        dm = DataModel(tables={"orders": ORDERS_LF}, pre_aggregations=pre_aggs)
        assert len(dm.pre_agg_objects) == 1
        assert isinstance(dm.pre_agg_objects[0], PreAggregation)
        assert dm.pre_agg_objects[0].name == "region_summary"


# ---------------------------------------------------------------------------
# TestTableMethod
# ---------------------------------------------------------------------------


class TestTableMethod:
    def test_table_returns_lazy_frame_proxy(self, simple_dm):
        result = simple_dm.table("orders")
        assert isinstance(result, LazyFrameProxy)

    def test_table_unknown_name_raises_key_error(self, simple_dm):
        with pytest.raises(KeyError):
            simple_dm.table("nonexistent")

    def test_table_error_lists_available_tables(self):
        dm = DataModel(tables={"orders": ORDERS_LF, "customers": CUSTOMERS_LF})
        with pytest.raises(KeyError, match="orders"):
            dm.table("not_a_table")


# ---------------------------------------------------------------------------
# TestFindBestPreAgg
# ---------------------------------------------------------------------------


def _make_pre_agg(name, group_by, aggregations, row_count, file_path=None):
    return PreAggregation(
        name=name,
        group_by=group_by,
        raw_aggregations=aggregations,
        file_path=file_path or Path(f"_pre_aggregations/{name}.parquet"),
        row_count=row_count,
    )


class TestFindBestPreAgg:
    def test_returns_none_when_no_pre_aggs(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        result = dm.find_best_pre_agg("orders", ["orders.region"], {})
        assert result is None

    def test_returns_none_when_no_match(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        dm.pre_agg_objects = [
            _make_pre_agg(
                "p1",
                group_by=["orders.region"],
                aggregations={"orders.amount": "sum"},
                row_count=2,
            )
        ]
        # Request a group_by that the pre-agg doesn't cover
        result = dm.find_best_pre_agg(
            "orders",
            ["orders.customer_id"],
            {"orders.amount": {"Sum"}},
        )
        assert result is None

    def test_returns_matching_pre_agg(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        pre_agg = _make_pre_agg(
            "p1",
            group_by=["orders.region"],
            aggregations={"orders.amount": "sum"},
            row_count=2,
        )
        dm.pre_agg_objects = [pre_agg]
        result = dm.find_best_pre_agg(
            "orders",
            ["orders.region"],
            {"orders.amount": {"Sum"}},
        )
        assert result is pre_agg

    def test_returns_lowest_row_count_when_multiple_match(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        large = _make_pre_agg(
            "large",
            group_by=["orders.region"],
            aggregations={"orders.amount": "sum"},
            row_count=100,
        )
        small = _make_pre_agg(
            "small",
            group_by=["orders.region"],
            aggregations={"orders.amount": "sum"},
            row_count=2,
        )
        dm.pre_agg_objects = [large, small]
        result = dm.find_best_pre_agg(
            "orders",
            ["orders.region"],
            {"orders.amount": {"Sum"}},
        )
        assert result is small

    def test_superset_group_by_covers_subset_request(self):
        dm = DataModel(tables={"orders": ORDERS_LF})
        pre_agg = _make_pre_agg(
            "p1",
            group_by=["orders.region", "orders.customer_id"],  # superset
            aggregations={"orders.amount": "sum"},
            row_count=5,
        )
        dm.pre_agg_objects = [pre_agg]
        # Request only a subset of the group_by dims
        result = dm.find_best_pre_agg(
            "orders",
            ["orders.region"],
            {"orders.amount": {"Sum"}},
        )
        assert result is pre_agg


# ---------------------------------------------------------------------------
# TestWritePreAgg
# ---------------------------------------------------------------------------


def _dm_with_pre_agg(tmp_path):
    pre_aggs = {
        "region_summary": {
            "group_by": ["orders.region"],
            "aggregations": {"orders.amount": "sum"},
        }
    }
    return DataModel(
        tables={"orders": ORDERS_LF},
        pre_aggregations=pre_aggs,
        pre_agg_directory=tmp_path,
    )


def _sample_pre_agg_lf():
    return pl.LazyFrame({"region": ["US", "CA"], "amount": [350.0, 450.0]})


class TestWritePreAgg:
    def test_write_creates_parquet_file(self, tmp_path):
        dm = _dm_with_pre_agg(tmp_path)
        dm.write_pre_agg("region_summary", _sample_pre_agg_lf())
        assert (tmp_path / "region_summary.parquet").exists()

    def test_write_returns_pre_agg_object(self, tmp_path):
        dm = _dm_with_pre_agg(tmp_path)
        result = dm.write_pre_agg("region_summary", _sample_pre_agg_lf())
        assert isinstance(result, PreAggregation)
        assert result.name == "region_summary"

    def test_write_updates_in_memory_row_count(self, tmp_path):
        dm = _dm_with_pre_agg(tmp_path)
        dm.write_pre_agg("region_summary", _sample_pre_agg_lf())
        assert dm.pre_agg_objects[0].row_count == 2

    def test_write_updates_written_at(self, tmp_path):
        dm = _dm_with_pre_agg(tmp_path)
        dm.write_pre_agg("region_summary", _sample_pre_agg_lf())
        assert isinstance(dm.pre_agg_objects[0].written_at, datetime)

    def test_write_creates_metadata_json(self, tmp_path):
        dm = _dm_with_pre_agg(tmp_path)
        dm.write_pre_agg("region_summary", _sample_pre_agg_lf())
        assert (tmp_path / "_metadata.json").exists()

    def test_write_metadata_has_row_count_and_written_at(self, tmp_path):
        dm = _dm_with_pre_agg(tmp_path)
        dm.write_pre_agg("region_summary", _sample_pre_agg_lf())
        metadata = json.loads((tmp_path / "_metadata.json").read_text())
        assert "region_summary" in metadata
        assert "row_count" in metadata["region_summary"]
        assert "written_at" in metadata["region_summary"]
        assert metadata["region_summary"]["row_count"] == 2

    def test_write_unknown_name_raises_key_error(self, tmp_path):
        dm = DataModel(tables={"orders": ORDERS_LF}, pre_agg_directory=tmp_path)
        lf = pl.LazyFrame({"region": ["US"], "amount": [100.0]})
        with pytest.raises(KeyError, match="unknown"):
            dm.write_pre_agg("unknown", lf)


# ---------------------------------------------------------------------------
# TestValidateQueryContext
# ---------------------------------------------------------------------------


class TestValidateQueryContext:
    def test_valid_context_returns_true(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["revenue"])
        assert dm.validate_query_context(qc) is True

    def test_invalid_measure_raises_key_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["bad_measure"])
        with pytest.raises(KeyError, match="bad_measure"):
            dm.validate_query_context(qc)

    def test_filter_invalid_table_raises_key_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(
            measures=["revenue"],
            filters={"AND": [("bad_table.region", "=", "US")]},
        )
        with pytest.raises(KeyError, match="bad_table"):
            dm.validate_query_context(qc)

    def test_filter_invalid_column_raises_value_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(
            measures=["revenue"],
            filters={"AND": [("orders.bad_col", "=", "US")]},
        )
        with pytest.raises(ValueError, match="bad_col"):
            dm.validate_query_context(qc)

    def test_group_invalid_table_raises_key_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["revenue"], groups=["bad_table.region"])
        with pytest.raises(KeyError, match="bad_table"):
            dm.validate_query_context(qc)

    def test_group_invalid_column_raises_value_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["revenue"], groups=["orders.bad_col"])
        with pytest.raises(ValueError, match="bad_col"):
            dm.validate_query_context(qc)

    def test_having_invalid_column_raises_value_error_unknown(self):
        dm = _dm_with_measures()
        qc = _make_qc(
            measures=["revenue"],
            havings={"AND": [("not_a_measure_or_group", ">", 0)]},
        )
        with pytest.raises(ValueError, match="not_a_measure_or_group"):
            dm.validate_query_context(qc)

    def test_having_invalid_column_raises_value_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(
            measures=["revenue"],
            havings={"AND": [("orders.bad_col", ">", 0)]},
        )
        with pytest.raises(ValueError, match="bad_col"):
            dm.validate_query_context(qc)

    def test_sort_invalid_table_raises_key_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["revenue"], sorts=[("bad_table.region", "asc")])
        with pytest.raises(KeyError, match="bad_table"):
            dm.validate_query_context(qc)

    def test_sort_invalid_column_raises_value_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["revenue"], sorts=[("orders.bad_col", "asc")])
        with pytest.raises(ValueError, match="bad_col"):
            dm.validate_query_context(qc)

    def test_sort_invalid_direction_raises_value_error(self):
        dm = _dm_with_measures()
        qc = _make_qc(measures=["revenue"], sorts=[("orders.region", "sideways")])
        with pytest.raises(ValueError, match="sideways"):
            dm.validate_query_context(qc)

    def test_invalid_limit_raises_value_error(self):
        dm = _dm_with_measures()
        for bad_limit in [0, -1, "5"]:
            with pytest.raises(ValueError):
                qc = _make_qc(measures=["revenue"], limit=bad_limit)
                dm.validate_query_context(qc)

    def test_invalid_offset_raises_value_error(self):
        dm = _dm_with_measures()
        for bad_offset in [-1, "0"]:
            with pytest.raises(ValueError):
                qc = _make_qc(measures=["revenue"], offset=bad_offset)
                dm.validate_query_context(qc)


# ---------------------------------------------------------------------------
# TestQuery
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_single_measure_no_groups(self):
        dm = _dm_with_measures()
        result = asyncio.run(dm.query({"measures": ["revenue"]}))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_query_multiple_measures_no_groups_cross_join(self):
        dm = _dm_with_measures()
        result = asyncio.run(dm.query({"measures": ["revenue", "order_count"]}))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1  # scalar × scalar cross join = 1 row
        assert "revenue" in result.columns
        assert "order_count" in result.columns

    def test_query_multiple_measures_with_groups_full_join(self):
        dm = _dm_with_measures()
        result = asyncio.run(
            dm.query(
                {
                    "measures": ["revenue", "unique_customers_by_region"],
                    "groups": ["orders.region"],
                }
            )
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 2  # US and CA

    def test_query_with_limit(self):
        dm = _dm_with_measures()
        result = asyncio.run(dm.query({"measures": ["revenue"], "limit": 1}))
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_query_with_offset(self):
        dm = _dm_with_measures()
        result_all = asyncio.run(dm.query({"measures": ["revenue"]}))
        result_offset = asyncio.run(dm.query({"measures": ["revenue"], "offset": 1}))
        assert isinstance(result_offset, pl.DataFrame)
        assert len(result_offset) == len(result_all) - 1

    def test_query_with_havings_filter(self):
        dm = _dm_with_measures()
        # US total = 350, CA total = 450; filter keeps only CA
        result = asyncio.run(
            dm.query(
                {
                    "measures": ["revenue"],
                    "havings": {"AND": [("revenue", ">", 400)]},
                }
            )
        )
        assert isinstance(result, pl.DataFrame)
        assert len(result) == 1

    def test_query_invalid_measure_raises(self):
        dm = _dm_with_measures()
        with pytest.raises(KeyError):
            asyncio.run(dm.query({"measures": ["nonexistent"]}))
