"""Integration tests: end-to-end query execution with DataModel + native DataFusion expression measures."""

import asyncio

import pyarrow as pa
from datafusion import col
from datafusion import functions as F
from datasubway.column_context import allow
from datasubway.data_model import DataModel
from datasubway.measure import measure

ORDERS_BATCH = pa.RecordBatch.from_pydict(
    {
        "region": ["US", "EU", "US", "EU", "US"],
        "amount": [100, 200, 150, 250, 300],
        "quantity": [10, 20, 15, 25, 30],
        "date": ["2024-01", "2024-01", "2024-02", "2024-02", "2024-02"],
    }
)


def run(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


class TestSingleMeasureNoGroups:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            return self.dm.table("orders").aggregate(
                [], [F.sum(col("amount")).alias("revenue")]
            )

    def test_basic_query(self):
        result = run(self.dm.query({"measures": ["revenue"]}))
        assert isinstance(result, pa.Table)
        assert result.num_rows == 1
        assert result.column("revenue").to_pylist() == [1000]


class TestSingleMeasureWithGroups:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            groups = allow("*", qc.groups) if qc.groups else []
            return self.dm.table("orders").aggregate(
                groups, [F.sum(col("amount")).alias("revenue")]
            )

    def test_group_by_region(self):
        result = run(
            self.dm.query({"measures": ["revenue"], "groups": ["orders.region"]})
        )
        assert result.num_rows == 2
        rows = result.to_pydict()
        data = sorted(zip(rows["region"], rows["revenue"]))
        assert data == [("EU", 450), ("US", 550)]

    def test_group_by_date(self):
        result = run(
            self.dm.query({"measures": ["revenue"], "groups": ["orders.date"]})
        )
        assert result.num_rows == 2

    def test_no_groups(self):
        result = run(self.dm.query({"measures": ["revenue"]}))
        assert result.num_rows == 1
        assert result.column("revenue").to_pylist() == [1000]


class TestSingleMeasureWithFilter:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            df = self.dm.table("orders")
            if qc.filters:
                df = df.filter_dict(qc.filters)
            groups = allow("*", qc.groups) if qc.groups else []
            return df.aggregate(groups, [F.sum(col("amount")).alias("revenue")])

    def test_filter_region(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "filters": {"AND": [["orders.region", "=", "US"]]},
                }
            )
        )
        assert result.column("revenue").to_pylist() == [550]

    def test_filter_with_groups(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "filters": {"AND": [["orders.region", "=", "US"]]},
                    "groups": ["orders.date"],
                }
            )
        )
        assert result.num_rows == 2
        rows = result.to_pydict()
        data = sorted(zip(rows["date"], rows["revenue"]))
        assert data == [("2024-01", 100), ("2024-02", 450)]


class TestMultipleMeasures:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            groups = allow("*", qc.groups) if qc.groups else []
            return self.dm.table("orders").aggregate(
                groups, [F.sum(col("amount")).alias("revenue")]
            )

        @measure(self.dm)
        def total_quantity(qc):
            groups = allow("*", qc.groups) if qc.groups else []
            return self.dm.table("orders").aggregate(
                groups, [F.sum(col("quantity")).alias("total_quantity")]
            )

    def test_multi_measure_no_groups(self):
        result = run(self.dm.query({"measures": ["revenue", "total_quantity"]}))
        assert result.num_rows == 1
        assert result.column("revenue").to_pylist() == [1000]
        assert result.column("total_quantity").to_pylist() == [100]

    def test_multi_measure_with_groups(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue", "total_quantity"],
                    "groups": ["orders.region"],
                }
            )
        )
        assert result.num_rows == 2
        rows = result.to_pydict()
        data = sorted(zip(rows["region"], rows["revenue"], rows["total_quantity"]))
        assert data == [("EU", 450, 45), ("US", 550, 55)]


class TestHavings:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            groups = allow("*", qc.groups) if qc.groups else []
            return self.dm.table("orders").aggregate(
                groups, [F.sum(col("amount")).alias("revenue")]
            )

    def test_having_filter(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "groups": ["orders.region"],
                    "havings": {"AND": [("revenue", ">", 500)]},
                }
            )
        )
        assert result.num_rows == 1
        assert result.column("region").to_pylist() == ["US"]
        assert result.column("revenue").to_pylist() == [550]


class TestSorts:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            groups = allow("*", qc.groups) if qc.groups else []
            return self.dm.table("orders").aggregate(
                groups, [F.sum(col("amount")).alias("revenue")]
            )

    def test_sort_asc(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "groups": ["orders.region"],
                    "sorts": [("revenue", "asc")],
                }
            )
        )
        assert result.column("revenue").to_pylist() == [450, 550]

    def test_sort_desc(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "groups": ["orders.region"],
                    "sorts": [("revenue", "desc")],
                }
            )
        )
        assert result.column("revenue").to_pylist() == [550, 450]


class TestLimitOffset:
    def setup_method(self):
        self.dm = DataModel(tables={"orders": ORDERS_BATCH})

        @measure(self.dm)
        def revenue(qc):
            groups = allow("*", qc.groups) if qc.groups else []
            return self.dm.table("orders").aggregate(
                groups, [F.sum(col("amount")).alias("revenue")]
            )

    def test_limit(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "groups": ["orders.region"],
                    "sorts": [("revenue", "desc")],
                    "limit": 1,
                }
            )
        )
        assert result.num_rows == 1
        assert result.column("revenue").to_pylist() == [550]

    def test_offset(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "groups": ["orders.region"],
                    "sorts": [("revenue", "desc")],
                    "offset": 1,
                }
            )
        )
        assert result.num_rows == 1
        assert result.column("revenue").to_pylist() == [450]

    def test_limit_and_offset(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["revenue"],
                    "groups": ["orders.date"],
                    "sorts": [("revenue", "asc")],
                    "limit": 1,
                    "offset": 1,
                }
            )
        )
        assert result.num_rows == 1
