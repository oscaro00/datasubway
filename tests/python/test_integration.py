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

PLAYERS_BATCH = pa.RecordBatch.from_pydict(
    {
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
        "team_id": [10, 10, 20],
    }
)

PLAYER_STATS_BATCH = pa.RecordBatch.from_pydict(
    {
        "player_id": [1, 2, 3, 1, 2],
        "avg_speed": [10.0, 12.0, 8.0, 11.0, 13.0],
    }
)

TEAMS_BATCH = pa.RecordBatch.from_pydict(
    {
        "id": [10, 20],
        "team_name": ["Eagles", "Hawks"],
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
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")]
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
            return (
                self.dm.table("orders")
                .filter(allow("*", qc.filters))
                .aggregate(allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")])
            )

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
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")]
            )

        @measure(self.dm)
        def total_quantity(qc):
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("quantity")).alias("total_quantity")]
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
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")]
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
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")]
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
            return self.dm.table("orders").aggregate(
                allow("*", qc.groups), [F.sum(col("amount")).alias("revenue")]
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


# ── Auto-join tests ──


def _make_auto_join_dm():
    """Create a DataModel with players, player_stats, teams and joins."""
    return DataModel(
        tables={
            "player_stats": PLAYER_STATS_BATCH,
            "players": PLAYERS_BATCH,
            "teams": TEAMS_BATCH,
        },
        joins=[
            {
                "left": "player_stats",
                "right": "players",
                "left_on": "player_id",
                "right_on": "id",
                "how": "inner",
                "direction": "right2left",
            },
            {
                "left": "players",
                "right": "teams",
                "left_on": "team_id",
                "right_on": "id",
                "how": "inner",
                "direction": "right2left",
            },
        ],
    )


class TestAutoJoinGroupBy:
    def setup_method(self):
        self.dm = _make_auto_join_dm()

        @measure(self.dm)
        def avg_speed(qc):
            return self.dm.table("player_stats").aggregate(
                allow("*", qc.groups),
                [F.avg(col("avg_speed")).alias("avg_speed")],
            )

    def test_group_by_foreign_table(self):
        result = run(
            self.dm.query(
                {"measures": ["avg_speed"], "groups": ["players.name"]}
            )
        )
        assert result.num_rows == 3
        rows = result.to_pydict()
        data = sorted(zip(rows["name"], rows["avg_speed"]))
        assert data[0][0] == "Alice"
        assert data[1][0] == "Bob"
        assert data[2][0] == "Charlie"

    def test_no_groups_still_works(self):
        result = run(self.dm.query({"measures": ["avg_speed"]}))
        assert result.num_rows == 1


class TestAutoJoinFilter:
    def setup_method(self):
        self.dm = _make_auto_join_dm()

        @measure(self.dm)
        def avg_speed(qc):
            return (
                self.dm.table("player_stats")
                .filter(allow("*", qc.filters))
                .aggregate(
                    allow("*", qc.groups),
                    [F.avg(col("avg_speed")).alias("avg_speed")],
                )
            )

    def test_filter_on_foreign_table(self):
        result = run(
            self.dm.query(
                {
                    "measures": ["avg_speed"],
                    "filters": {"AND": [["players.name", "=", "Alice"]]},
                }
            )
        )
        assert result.num_rows == 1
        assert result.column("avg_speed").to_pylist()[0] == 10.5  # (10+11)/2


class TestAutoJoinMultiHop:
    def setup_method(self):
        self.dm = _make_auto_join_dm()

        @measure(self.dm)
        def avg_speed(qc):
            return self.dm.table("player_stats").aggregate(
                allow("*", qc.groups),
                [F.avg(col("avg_speed")).alias("avg_speed")],
            )

    def test_two_hop_join(self):
        """player_stats -> players -> teams"""
        result = run(
            self.dm.query(
                {"measures": ["avg_speed"], "groups": ["teams.team_name"]}
            )
        )
        assert result.num_rows == 2
        rows = result.to_pydict()
        data = sorted(zip(rows["team_name"], rows["avg_speed"]))
        # Eagles: Alice(10,11) + Bob(12,13) = avg 11.5
        # Hawks: Charlie(8) = avg 8.0
        assert data[0] == ("Eagles", 11.5)
        assert data[1] == ("Hawks", 8.0)
