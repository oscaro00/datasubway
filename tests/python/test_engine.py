"""Tests for the Rust Engine, JoinGraph, PreAggregation, and Substrait bridge via PyO3."""

import datafusion
import pyarrow as pa
import pytest
from datafusion import col
from datafusion import functions as F
from datafusion.substrait import Producer
from datasubway._engine import Engine, JoinGraph, PreAggregation

# ── Engine tests ──


class TestEngine:
    def setup_method(self):
        self.engine = Engine()
        batch = pa.RecordBatch.from_pydict(
            {
                "region": ["US", "EU", "US", "EU"],
                "amount": [100, 200, 150, 250],
                "date": ["2024-01", "2024-01", "2024-02", "2024-02"],
            }
        )
        self.engine.register_record_batch("orders", batch)

    def test_table_names(self):
        assert "orders" in self.engine.table_names()

    def test_sql_query(self):
        result = self.engine.sql(
            "SELECT region, SUM(amount) as total FROM orders GROUP BY region ORDER BY region"
        )
        assert len(result) == 1  # one batch
        batch = result[0]
        assert batch.num_rows == 2
        assert batch.column("region").to_pylist() == ["EU", "US"]
        assert batch.column("total").to_pylist() == [450, 250]

    def test_register_multiple_tables(self):
        batch2 = pa.RecordBatch.from_pydict({"id": [1, 2], "name": ["Alice", "Bob"]})
        self.engine.register_record_batch("customers", batch2)
        names = self.engine.table_names()
        assert "orders" in names
        assert "customers" in names


# ── Substrait bridge tests ──


class TestOptimizeAndCollectSubstrait:
    def setup_method(self):
        self.engine = Engine()
        self.py_ctx = datafusion.SessionContext()
        batch = pa.RecordBatch.from_pydict(
            {
                "region": ["US", "EU", "US", "EU"],
                "amount": [100, 200, 150, 250],
            }
        )
        self.engine.register_record_batch("orders", batch)
        self.py_ctx.register_record_batches("orders", [[batch]])

    def test_substrait_round_trip(self):
        df = self.py_ctx.table("orders")
        agg = df.aggregate([col("region")], [F.sum(col("amount")).alias("total")])
        substrait_plan = Producer.to_substrait_plan(agg.logical_plan(), self.py_ctx)
        plan_bytes = substrait_plan.encode()

        result = self.engine.optimize_and_collect_substrait(plan_bytes)
        t = pa.Table.from_batches(result)
        data = sorted(
            zip(t.column("region").to_pylist(), t.column("total").to_pylist())
        )
        assert data == [("EU", 450), ("US", 250)]

    def test_substrait_no_groups(self):
        df = self.py_ctx.table("orders")
        agg = df.aggregate([], [F.sum(col("amount")).alias("total")])
        substrait_plan = Producer.to_substrait_plan(agg.logical_plan(), self.py_ctx)
        plan_bytes = substrait_plan.encode()

        result = self.engine.optimize_and_collect_substrait(plan_bytes)
        t = pa.Table.from_batches(result)
        assert t.column("total").to_pylist() == [700]

    def test_substrait_with_filter(self):
        df = self.py_ctx.table("orders")
        filtered = df.filter(col("region") == col("region").literal("US"))
        # Use lit for the filter
        filtered = self.py_ctx.table("orders").filter(
            col("region") == datafusion.lit("US")
        )
        agg = filtered.aggregate([], [F.sum(col("amount")).alias("total")])
        substrait_plan = Producer.to_substrait_plan(agg.logical_plan(), self.py_ctx)
        plan_bytes = substrait_plan.encode()

        result = self.engine.optimize_and_collect_substrait(plan_bytes)
        t = pa.Table.from_batches(result)
        assert t.column("total").to_pylist() == [250]


# ── JoinGraph tests ──


class TestJoinGraph:
    def test_basic_join(self):
        jg = JoinGraph(
            [
                {
                    "left": "orders",
                    "right": "customers",
                    "left_on": "customer_id",
                    "right_on": "id",
                    "how": "left",
                    "direction": "right2left",
                }
            ]
        )
        assert set(jg.tables()) == {"orders", "customers"}
        path = jg.find_path("orders", "customers")
        assert path is not None
        assert len(path) == 1
        assert path[0]["left"] == "orders"
        assert path[0]["right"] == "customers"

    def test_no_reverse_unidirectional(self):
        jg = JoinGraph(
            [
                {
                    "left": "a",
                    "right": "b",
                    "left_on": "id",
                    "right_on": "a_id",
                    "how": "left",
                    "direction": "right2left",
                }
            ]
        )
        assert jg.find_path("b", "a") is None

    def test_bidirectional(self):
        jg = JoinGraph(
            [
                {
                    "left": "a",
                    "right": "b",
                    "left_on": "id",
                    "right_on": "a_id",
                    "how": "inner",
                    "direction": "both",
                }
            ]
        )
        assert jg.find_path("a", "b") is not None
        assert jg.find_path("b", "a") is not None

    def test_cycle_rejection(self):
        with pytest.raises(ValueError, match="Cycle"):
            JoinGraph(
                [
                    {
                        "left": "a",
                        "right": "b",
                        "left_on": "id",
                        "right_on": "id",
                        "how": "inner",
                        "direction": "both",
                    },
                    {
                        "left": "b",
                        "right": "c",
                        "left_on": "id",
                        "right_on": "id",
                        "how": "inner",
                        "direction": "both",
                    },
                    {
                        "left": "c",
                        "right": "a",
                        "left_on": "id",
                        "right_on": "id",
                        "how": "inner",
                        "direction": "both",
                    },
                ]
            )

    def test_multi_hop_path(self):
        jg = JoinGraph(
            [
                {
                    "left": "a",
                    "right": "b",
                    "left_on": "id",
                    "right_on": "a_id",
                    "how": "left",
                    "direction": "right2left",
                },
                {
                    "left": "b",
                    "right": "c",
                    "left_on": "id",
                    "right_on": "b_id",
                    "how": "left",
                    "direction": "right2left",
                },
            ]
        )
        path = jg.find_path("a", "c")
        assert path is not None
        assert len(path) == 2


# ── PreAggregation tests ──


class TestPreAggregation:
    def test_creation(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date", "orders.region"],
            raw_aggregations={"orders.amount": ["sum", "mean"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert pa_obj.name == "daily"
        assert pa_obj.group_by == ["orders.date", "orders.region"]
        # "mean" expands to sum+count
        assert "sum" in pa_obj.aggregations["orders.amount"]
        assert "count" in pa_obj.aggregations["orders.amount"]

    def test_covers_exact(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date", "orders.region"],
            raw_aggregations={"orders.amount": ["sum"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert pa_obj.covers(
            requested_group_by=["orders.date", "orders.region"],
            requested_agg_components={"orders.amount": {"sum"}},
            filter_columns=[],
        )

    def test_covers_subset_group_by(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date", "orders.region"],
            raw_aggregations={"orders.amount": ["sum"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert pa_obj.covers(
            requested_group_by=["orders.region"],
            requested_agg_components={"orders.amount": {"sum"}},
            filter_columns=[],
        )

    def test_covers_missing_group_col(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date"],
            raw_aggregations={"orders.amount": ["sum"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert not pa_obj.covers(
            requested_group_by=["orders.region"],
            requested_agg_components={"orders.amount": {"sum"}},
            filter_columns=[],
        )

    def test_covers_filter_in_group_by(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date", "orders.region"],
            raw_aggregations={"orders.amount": ["sum"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert pa_obj.covers(
            requested_group_by=["orders.date"],
            requested_agg_components={"orders.amount": {"sum"}},
            filter_columns=["orders.region"],
        )

    def test_covers_filter_not_in_group_by(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date"],
            raw_aggregations={"orders.amount": ["sum"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert not pa_obj.covers(
            requested_group_by=["orders.date"],
            requested_agg_components={"orders.amount": {"sum"}},
            filter_columns=["orders.region"],
        )

    def test_row_count_setter(self):
        pa_obj = PreAggregation(
            name="daily",
            group_by=["orders.date"],
            raw_aggregations={"orders.amount": ["sum"]},
            file_path="_pre_aggs/daily.parquet",
        )
        assert pa_obj.row_count == 0
        pa_obj.row_count = 42
        assert pa_obj.row_count == 42

    def test_empty_group_by_rejected(self):
        with pytest.raises(ValueError):
            PreAggregation(
                name="bad",
                group_by=[],
                raw_aggregations={"orders.amount": ["sum"]},
                file_path="_pre_aggs/bad.parquet",
            )

    def test_unknown_agg_rejected(self):
        with pytest.raises(ValueError):
            PreAggregation(
                name="bad",
                group_by=["orders.date"],
                raw_aggregations={"orders.amount": ["median"]},
                file_path="_pre_aggs/bad.parquet",
            )
