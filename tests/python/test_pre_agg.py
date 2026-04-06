"""Tests for pre-aggregation writing and usage with native DataFusion expression API."""

import asyncio
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from datafusion import col
from datafusion import functions as F
from datasubway.column_context import allow
from datasubway.data_model import DataModel
from datasubway.measure import measure

ORDERS_BATCH = pa.RecordBatch.from_pydict(
    {
        "region": ["US", "EU", "US", "EU", "US", "APAC"],
        "amount": [100, 200, 150, 250, 300, 50],
        "quantity": [10, 20, 15, 25, 30, 5],
        "date": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-02", "2024-01"],
    }
)


def run(coro):
    return asyncio.run(coro)


class TestWritePreAggs:
    def setup_method(self, tmp_path_factory=None):
        import tempfile

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.dm = DataModel(
            tables={"orders": ORDERS_BATCH},
            pre_aggregations={
                "daily_revenue": {
                    "group_by": ["orders.date", "orders.region"],
                    "aggregations": {
                        "orders.amount": ["sum", "mean"],
                        "orders.quantity": ["sum"],
                    },
                },
                "regional_revenue": {
                    "group_by": ["orders.region"],
                    "aggregations": {
                        "orders.amount": ["sum", "count"],
                    },
                },
            },
            pre_agg_directory=self.tmp_dir,
        )

    def test_write_creates_parquet(self):
        results = self.dm.write_pre_aggs(["daily_revenue"])
        assert len(results) == 1
        pa_obj = results[0]
        assert Path(pa_obj.file_path).exists()

    def test_write_correct_row_count(self):
        results = self.dm.write_pre_aggs(["daily_revenue"])
        pa_obj = results[0]
        # 5 unique (date, region) combos
        assert pa_obj.row_count == 5

    def test_write_updates_metadata(self):
        self.dm.write_pre_aggs(["daily_revenue"])
        meta_path = self.tmp_dir / "_metadata.json"
        assert meta_path.exists()
        metadata = json.loads(meta_path.read_text())
        assert "daily_revenue" in metadata
        assert metadata["daily_revenue"]["row_count"] == 5
        assert "written_at" in metadata["daily_revenue"]

    def test_write_parquet_has_correct_columns(self):
        self.dm.write_pre_aggs(["daily_revenue"])
        pa_obj = self.dm.pre_agg_objects[0]
        table = pq.read_table(pa_obj.file_path)
        cols = set(table.column_names)
        # Should have group-by cols + component cols
        assert "date" in cols or "orders.date" in cols  # SQL may drop table prefix
        assert "region" in cols or "orders.region" in cols

    def test_write_multiple(self):
        results = self.dm.write_pre_aggs(["daily_revenue", "regional_revenue"])
        assert len(results) == 2
        assert results[0].row_count == 5
        assert results[1].row_count == 3  # 3 unique regions

    def test_write_unknown_name(self):
        with pytest.raises(KeyError, match="Unknown pre-aggregation"):
            self.dm.write_pre_aggs(["nonexistent"])

    def test_written_at_set(self):
        results = self.dm.write_pre_aggs(["daily_revenue"])
        assert results[0].written_at is not None


class TestFindBestPreAgg:
    def setup_method(self):
        import tempfile

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.dm = DataModel(
            tables={"orders": ORDERS_BATCH},
            pre_aggregations={
                "daily_revenue": {
                    "group_by": ["orders.date", "orders.region"],
                    "aggregations": {
                        "orders.amount": ["sum", "mean"],
                    },
                },
                "regional_revenue": {
                    "group_by": ["orders.region"],
                    "aggregations": {
                        "orders.amount": ["sum", "count"],
                    },
                },
            },
            pre_agg_directory=self.tmp_dir,
        )
        # Write pre-aggs to set row counts
        self.dm.write_pre_aggs(["daily_revenue", "regional_revenue"])

    def test_find_covering_pre_agg(self):
        best = self.dm.find_best_pre_agg(
            group_by=["orders.region"],
            agg_components={"orders.amount": {"sum"}},
        )
        assert best is not None
        # regional_revenue has fewer rows and covers the request
        assert best.name == "regional_revenue"

    def test_find_with_filter_columns(self):
        # Filter on region — both cover it since it's in their group_by
        best = self.dm.find_best_pre_agg(
            group_by=["orders.date"],
            agg_components={"orders.amount": {"sum"}},
            filter_columns=["orders.region"],
        )
        assert best is not None
        assert best.name == "daily_revenue"

    def test_no_covering_pre_agg(self):
        best = self.dm.find_best_pre_agg(
            group_by=["orders.store"],  # nonexistent column
            agg_components={"orders.amount": {"sum"}},
        )
        assert best is None

    def test_missing_component(self):
        best = self.dm.find_best_pre_agg(
            group_by=["orders.region"],
            agg_components={"orders.amount": {"min"}},  # not stored
        )
        assert best is None

    def test_filter_column_not_in_group_by_rejected(self):
        # regional_revenue only has region in group_by, not date
        # So filtering by date should not match regional_revenue
        best = self.dm.find_best_pre_agg(
            group_by=["orders.region"],
            agg_components={"orders.amount": {"sum"}},
            filter_columns=["orders.date"],  # not in regional_revenue's group_by
        )
        # Should pick daily_revenue since it has both date and region
        assert best is not None
        assert best.name == "daily_revenue"


class TestPreAggQuery:
    """Test that queries use expression-based measures with pre-aggs available."""

    def setup_method(self):
        import tempfile

        self.tmp_dir = Path(tempfile.mkdtemp())
        self.dm = DataModel(
            tables={"orders": ORDERS_BATCH},
            pre_aggregations={
                "regional_revenue": {
                    "group_by": ["orders.region"],
                    "aggregations": {
                        "orders.amount": ["sum"],
                    },
                },
            },
            pre_agg_directory=self.tmp_dir,
        )
        self.dm.write_pre_aggs(["regional_revenue"])

        @measure(self.dm)
        def revenue(qc):
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")]
            )

    def test_query_still_works_with_pre_aggs_defined(self):
        """Pre-agg optimization is available but queries still work."""
        result = run(
            self.dm.query({"measures": ["revenue"], "groups": ["orders.region"]})
        )
        assert result.num_rows == 3  # US, EU, APAC
        rows = result.to_pydict()
        data = sorted(zip(rows["region"], rows["revenue"]))
        assert data == [("APAC", 50), ("EU", 450), ("US", 550)]

    def test_query_no_groups(self):
        result = run(self.dm.query({"measures": ["revenue"]}))
        assert result.column("revenue").to_pylist() == [1050]
