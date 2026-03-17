"""Tests for the DataModel class."""

import pyarrow as pa
import pytest
from datasubway.data_model import DataModel
from datasubway.dataframe import MeasureDataFrame
from datasubway.query_context import QueryContext

ORDERS_BATCH = pa.RecordBatch.from_pydict(
    {
        "region": ["US", "EU", "US", "EU", "US"],
        "amount": [100, 200, 150, 250, 300],
        "date": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
        "customer_id": [1, 2, 1, 2, 3],
    }
)

CUSTOMERS_BATCH = pa.RecordBatch.from_pydict(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)


class TestDataModelInit:
    def test_record_batch_source(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH})
        assert "orders" in dm.engine.table_names()

    def test_arrow_table_source(self):
        table = pa.Table.from_batches([ORDERS_BATCH])
        dm = DataModel(tables={"orders": table})
        assert "orders" in dm.engine.table_names()

    def test_schema_stored(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH})
        assert "orders.region" in dm.table_schemas["orders"]
        assert "orders.amount" in dm.table_schemas["orders"]

    def test_multiple_tables(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH, "customers": CUSTOMERS_BATCH})
        assert "orders" in dm.engine.table_names()
        assert "customers" in dm.engine.table_names()

    def test_with_joins(self):
        dm = DataModel(
            tables={"orders": ORDERS_BATCH, "customers": CUSTOMERS_BATCH},
            joins=[
                {
                    "left": "orders",
                    "right": "customers",
                    "left_on": "customer_id",
                    "right_on": "id",
                    "how": "left",
                    "direction": "right2left",
                }
            ],
        )
        assert dm.join_graph is not None
        path = dm.join_graph.find_path("orders", "customers")
        assert path is not None

    def test_unsupported_source_type(self):
        with pytest.raises(TypeError):
            DataModel(tables={"bad": 42})


class TestDataModelTable:
    def test_table_returns_measure_dataframe(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH})
        mdf = dm.table("orders")
        assert isinstance(mdf, MeasureDataFrame)
        assert mdf._table_name == "orders"
        assert mdf._last_op == "table"

    def test_table_columns(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH})
        mdf = dm.table("orders")
        cols = mdf.columns()
        assert "region" in cols
        assert "amount" in cols


class TestDataModelPyCtx:
    def test_py_ctx_has_tables(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH, "customers": CUSTOMERS_BATCH})
        # py_ctx should also have the tables registered
        py_tables = dm.py_ctx.tables()
        assert "orders" in py_tables
        assert "customers" in py_tables


class TestDataModelAllColumns:
    def test_all_columns(self):
        dm = DataModel(tables={"orders": ORDERS_BATCH, "customers": CUSTOMERS_BATCH})
        cols = dm.all_columns()
        assert "orders.region" in cols
        assert "orders.amount" in cols
        assert "customers.id" in cols
        assert "customers.name" in cols


class TestValidateQueryContext:
    def setup_method(self):
        self.dm = DataModel(
            tables={"orders": ORDERS_BATCH, "customers": CUSTOMERS_BATCH}
        )
        # Register a dummy measure
        self.dm.measures["revenue"] = lambda qc: None
        self.dm.measure_output_cols["revenue"] = ["revenue"]

    def test_valid(self):
        qc = QueryContext(
            {
                "measures": ["revenue"],
                "groups": ["orders.region"],
                "filters": {"AND": [("orders.amount", ">", 100)]},
            }
        )
        assert self.dm.validate_query_context(qc) is True

    def test_unknown_measure(self):
        qc = QueryContext({"measures": ["nonexistent"]})
        with pytest.raises(ValueError, match="Unknown measure"):
            self.dm.validate_query_context(qc)

    def test_unknown_group_column(self):
        qc = QueryContext({"measures": ["revenue"], "groups": ["orders.nonexistent"]})
        with pytest.raises(ValueError, match="Unknown group column"):
            self.dm.validate_query_context(qc)

    def test_unknown_filter_column(self):
        qc = QueryContext(
            {
                "measures": ["revenue"],
                "filters": {"AND": [("orders.nonexistent", "=", "x")]},
            }
        )
        with pytest.raises(ValueError, match="Unknown filter column"):
            self.dm.validate_query_context(qc)

    def test_invalid_sort_direction(self):
        qc = QueryContext(
            {
                "measures": ["revenue"],
                "groups": ["orders.region"],
                "sorts": [("orders.region", "sideways")],
            }
        )
        with pytest.raises(ValueError, match="Invalid sort direction"):
            self.dm.validate_query_context(qc)

    def test_valid_having(self):
        qc = QueryContext(
            {
                "measures": ["revenue"],
                "groups": ["orders.region"],
                "havings": {"AND": [("revenue", ">", 100)]},
            }
        )
        assert self.dm.validate_query_context(qc) is True

    def test_invalid_having_column(self):
        qc = QueryContext(
            {
                "measures": ["revenue"],
                "groups": ["orders.region"],
                "havings": {"AND": [("nonexistent", ">", 100)]},
            }
        )
        with pytest.raises(ValueError, match="Invalid having column"):
            self.dm.validate_query_context(qc)
