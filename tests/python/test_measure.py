"""Tests for the @measure decorator with DataFusion expression-based API validation."""

import pyarrow as pa
import pytest
from datafusion import col
from datafusion import functions as F
from datasubway.column_context import allow
from datasubway.data_model import DataModel
from datasubway.dataframe import MeasureDataFrame
from datasubway.measure import measure

ORDERS_BATCH = pa.RecordBatch.from_pydict(
    {
        "region": ["US", "EU", "US", "EU"],
        "amount": [100, 200, 150, 250],
    }
)


class TestMeasureDecorator:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

    def test_basic_registration(self):
        @measure(self.dm)
        def revenue(qc):
            return self.dm.table("orders").aggregate(
                [], [F.sum(col("amount")).alias("revenue")]
            )

        assert "revenue" in self.dm.measures
        assert self.dm.measures["revenue"] is revenue

    def test_output_cols_extracted(self):
        @measure(self.dm)
        def revenue(qc):
            return self.dm.table("orders").aggregate(
                [], [F.sum(col("amount")).alias("revenue")]
            )

        assert "revenue" in self.dm.measure_output_cols["revenue"]

    def test_multiple_output_cols(self):
        @measure(self.dm)
        def stats(qc):
            return self.dm.table("orders").aggregate(
                [],
                [
                    F.sum(col("amount")).alias("total"),
                    F.count(col("amount")).alias("cnt"),
                ],
            )

        assert self.dm.measure_output_cols["stats"] == ["total", "cnt"]

    def test_duplicate_name_rejected(self):
        @measure(self.dm)
        def revenue(qc):
            return self.dm.table("orders").aggregate(
                [], [F.sum(col("amount")).alias("revenue")]
            )

        with pytest.raises(ValueError, match="already registered"):

            @measure(self.dm)
            def revenue(qc):
                return self.dm.table("orders").aggregate(
                    [], [F.sum(col("amount")).alias("revenue")]
                )

    def test_docstring_stored(self):
        @measure(self.dm)
        def revenue(qc):
            """Total revenue."""
            return self.dm.table("orders").aggregate(
                [], [F.sum(col("amount")).alias("revenue")]
            )

        assert self.dm.measure_docstrings["revenue"] == "Total revenue."

    def test_measure_with_groups(self):
        @measure(self.dm)
        def revenue_by_region(qc):
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups) if qc.groups else [],
                [F.sum(col("amount")).alias("revenue")],
            )

        assert "revenue_by_region" in self.dm.measures

    def test_measure_function_callable(self):
        @measure(self.dm)
        def revenue(qc):
            return self.dm.table("orders").aggregate(
                [], [F.sum(col("amount")).alias("revenue")]
            )

        from datasubway.query_context import QueryContext

        qc = QueryContext({"measures": ["revenue"]})
        result = revenue(qc)
        assert isinstance(result, MeasureDataFrame)
        batches = result.collect()
        t = pa.Table.from_batches(batches)
        assert t.column("revenue").to_pylist() == [700]

    def test_probe_failure_still_registers(self):
        """If probe with empty QC fails, measure is still registered."""

        @measure(self.dm)
        def conditional_measure(qc):
            if not qc.groups:
                raise ValueError("Need groups")
            return self.dm.table("orders").aggregate(
                list(qc.groups),
                [F.sum(col("amount")).alias("total")],
            )

        assert "conditional_measure" in self.dm.measures
        # output_cols will be empty since probe failed
        assert self.dm.measure_output_cols["conditional_measure"] == []


class TestMeasureValidation:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

    def test_must_end_with_aggregate(self):
        with pytest.raises(ValueError, match="must end with .aggregate"):

            @measure(self.dm)
            def bad_measure(qc):
                return self.dm.table("orders")

    def test_must_return_measure_dataframe(self):
        with pytest.raises(TypeError, match="must return a MeasureDataFrame"):

            @measure(self.dm)
            def bad_type(qc):
                return "not a dataframe"

    def test_select_without_aggregate_rejected(self):
        with pytest.raises(ValueError, match="must end with .aggregate"):

            @measure(self.dm)
            def bad_select(qc):
                return self.dm.table("orders").select(col("region"))

    def test_filter_without_aggregate_rejected(self):
        with pytest.raises(ValueError, match="must end with .aggregate"):

            @measure(self.dm)
            def bad_filter(qc):
                return self.dm.table("orders").filter_dict(
                    {"AND": [["region", "=", "US"]]}
                )


