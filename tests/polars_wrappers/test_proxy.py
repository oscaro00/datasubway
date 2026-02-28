from typing import cast

import polars as pl
import pytest

from datasubway.joins_meta import Join
from datasubway.pre_agg_meta import PreAggregation
from datasubway.polars_wrappers.proxy import (
    LazyFrameProxy,
    LazyGroupByProxy,
    RecordedOp,
    extract_col_names,
    qualify_col,
    strip_table_prefix,
)

# ---------------------------------------------------------------------------
# Helper objects
# ---------------------------------------------------------------------------


class _MockDM:
    """Minimal _DataModelLike implementation for tests.

    Mirrors DataModel.__init__ by renaming all columns to {table}.{col}
    and deriving table_schemas from the renamed tables.
    """

    def __init__(
        self,
        tables: dict[str, pl.LazyFrame],
        schemas: dict[str, list[str]] | None = None,
        joins_lookup: dict[str, dict[str, list[Join]]] | None = None,
        pre_agg: PreAggregation | None = None,
    ) -> None:
        # Rename columns to {table}.{col} like DataModel does
        self.tables: dict[str, pl.LazyFrame] = {
            name: lf.rename({col: f"{name}.{col}" for col in lf.collect_schema().names()})
            for name, lf in tables.items()
        }
        self.table_schemas: dict[str, list[str]] = schemas or {
            t: list(lf.collect_schema().names()) for t, lf in self.tables.items()
        }
        self.joins_lookup: dict[str, dict[str, list[Join]]] = joins_lookup or {}
        self._pre_agg: PreAggregation | None = pre_agg

    def find_best_pre_agg(
        self,
        table_name: str,
        group_by: list[str],
        agg_reqs: dict[str, set[str]],
    ) -> PreAggregation | None:
        return self._pre_agg


class _MockPreAgg:
    """PreAggregation-like object backed by an in-memory LazyFrame."""

    def __init__(self, lf: pl.LazyFrame) -> None:
        self._lf = lf

    def load(self) -> pl.LazyFrame:
        return self._lf


# ---------------------------------------------------------------------------
# strip_table_prefix
# ---------------------------------------------------------------------------


def test_strip_table_prefix_with_dot():
    assert strip_table_prefix("orders.id") == "id"


def test_strip_table_prefix_no_dot():
    assert strip_table_prefix("id") == "id"


def test_strip_table_prefix_list():
    assert strip_table_prefix(["orders.id", "customers.name"]) == ["id", "name"]


def test_strip_table_prefix_int_passthrough():
    assert strip_table_prefix(42) == 42


def test_strip_table_prefix_none_passthrough():
    assert strip_table_prefix(None) is None


# ---------------------------------------------------------------------------
# qualify_col
# ---------------------------------------------------------------------------


def test_qualify_col_in_schema_no_dot():
    schemas = {"orders": ["id", "amount"]}
    assert qualify_col("amount", "orders", schemas) == "orders.amount"


def test_qualify_col_already_has_dot():
    schemas = {"orders": ["orders.amount"]}
    assert qualify_col("orders.amount", "orders", schemas) == "orders.amount"


def test_qualify_col_not_in_schema():
    schemas = {"orders": ["id", "amount"]}
    assert qualify_col("unknown_col", "orders", schemas) == "unknown_col"


# ---------------------------------------------------------------------------
# extract_col_names
# ---------------------------------------------------------------------------


def test_extract_col_names_flat_strings():
    assert extract_col_names(("a", "b", "c")) == ["a", "b", "c"]


def test_extract_col_names_nested_list():
    assert extract_col_names((["a", "b"], "c")) == ["a", "b", "c"]


def test_extract_col_names_mixed_types_returns_only_strings():
    # Expr and int are ignored; only strings extracted
    result = extract_col_names(("a", pl.col("b"), 42))
    assert result == ["a"]


# ---------------------------------------------------------------------------
# RecordedOp
# ---------------------------------------------------------------------------


def test_recorded_op_stores_method_args_kwargs():
    expr = pl.col("x") > 0
    op = RecordedOp(method="filter", args=(expr,), kwargs={"k": "v"})
    assert op.method == "filter"
    assert op.args == (expr,)
    assert op.kwargs == {"k": "v"}


# ---------------------------------------------------------------------------
# LazyGroupByProxy
# ---------------------------------------------------------------------------


def test_lazy_groupby_proxy_agg_extends_agg_exprs_and_returns_parent():
    lf = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)
    gbp = proxy.group_by("tbl.a")
    expr = pl.col("tbl.b").sum()

    result = gbp.agg(expr)

    assert result is proxy
    assert expr in proxy.agg_exprs
    assert any(op.method == "agg" for op in proxy.ops)


def test_lazy_groupby_proxy_having_appends_op_and_returns_self():
    lf = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)
    gbp = proxy.group_by("tbl.a")

    result = gbp.having(pl.col("tbl.b") > 1)

    assert result is gbp
    assert any(op.method == "having" for op in proxy.ops)


def test_lazy_groupby_proxy_map_groups_appends_op_and_returns_parent():
    lf = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)
    gbp = proxy.group_by("tbl.a")

    def fn(df):
        return df

    result = gbp.map_groups(fn)

    assert result is proxy
    assert any(op.method == "map_groups" for op in proxy.ops)


# ---------------------------------------------------------------------------
# LazyFrameProxy recording
# ---------------------------------------------------------------------------


def test_lazy_frame_proxy_filter_records_op_and_returns_self():
    lf = pl.LazyFrame({"a": [1, 2]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)

    result = proxy.filter(pl.col("tbl.a") > 1)

    assert result is proxy
    assert proxy.ops[-1].method == "filter"


def test_lazy_frame_proxy_sort_records_op_and_returns_self():
    lf = pl.LazyFrame({"a": [1, 2]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)

    result = proxy.sort("tbl.a")

    assert result is proxy
    assert proxy.ops[-1].method == "sort"


def test_lazy_frame_proxy_group_by_records_op_sets_cols_returns_groupby_proxy():
    lf = pl.LazyFrame({"a": [1, 2], "b": [3, 4]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)

    gbp = proxy.group_by("tbl.a")

    assert isinstance(gbp, LazyGroupByProxy)
    assert proxy.group_by_cols == ["tbl.a"]
    assert proxy.ops[-1].method == "group_by"


def test_lazy_frame_proxy_group_by_dynamic_records_op_and_sets_cols_from_index_and_group_by():
    lf = pl.LazyFrame({"date": ["2024-01-01", "2024-01-02"], "val": [1, 2]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)

    gbp = proxy.group_by_dynamic("date", every="1d", group_by="region")

    assert isinstance(gbp, LazyGroupByProxy)
    assert "date" in proxy.group_by_cols
    assert "region" in proxy.group_by_cols
    assert proxy.ops[-1].method == "group_by_dynamic"


def test_lazy_frame_proxy_rolling_records_op_and_sets_cols_from_index_and_group_by():
    lf = pl.LazyFrame({"date": ["2024-01-01", "2024-01-02"], "val": [1, 2]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)

    gbp = proxy.rolling("date", period="1d", group_by="region")

    assert isinstance(gbp, LazyGroupByProxy)
    assert "date" in proxy.group_by_cols
    assert "region" in proxy.group_by_cols
    assert proxy.ops[-1].method == "rolling"


def test_lazy_frame_proxy_join_records_op_and_returns_self():
    lf = pl.LazyFrame({"a": [1, 2]})
    other_lf = pl.LazyFrame({"a": [1, 2], "c": [5, 6]})
    dm = _MockDM({"tbl": lf, "other": other_lf})
    proxy = LazyFrameProxy("tbl", dm)
    other_proxy = LazyFrameProxy("other", dm)

    result = proxy.join(other_proxy, left_on=["tbl.a"], right_on=["other.a"], how="inner")

    assert result is proxy
    assert proxy.ops[-1].method == "join"


def test_lazy_frame_proxy_getattr_unknown_method_records_op_and_returns_self():
    lf = pl.LazyFrame({"a": [1, 2]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)

    result = proxy.some_unknown_method("arg1", kwarg1="val1")

    assert result is proxy
    assert proxy.ops[-1].method == "some_unknown_method"
    assert proxy.ops[-1].args == ("arg1",)
    assert proxy.ops[-1].kwargs == {"kwarg1": "val1"}


def test_lazy_frame_proxy_getattr_guard_raises_attribute_error_for_internal_attr():
    bare = object.__new__(LazyFrameProxy)
    with pytest.raises(AttributeError):
        _ = bare.ops


# ---------------------------------------------------------------------------
# LazyFrameProxy._collect_foreign_tables
# ---------------------------------------------------------------------------


def test_collect_foreign_tables_no_foreign_returns_empty_set():
    lf = pl.LazyFrame({"id": [1, 2]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.group_by_cols = ["orders.id"]

    assert proxy._collect_foreign_tables() == set()


def test_collect_foreign_tables_detects_foreign_col_in_group_by():
    lf = pl.LazyFrame({"id": [1, 2]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.group_by_cols = ["customers.id"]

    assert proxy._collect_foreign_tables() == {"customers"}


def test_collect_foreign_tables_detects_foreign_col_in_agg_expr():
    lf = pl.LazyFrame({"id": [1, 2]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.agg_exprs = [pl.col("customers.name").first()]

    assert proxy._collect_foreign_tables() == {"customers"}


def test_collect_foreign_tables_ignores_own_table_prefix():
    lf = pl.LazyFrame({"id": [1, 2]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.group_by_cols = ["orders.id"]

    assert proxy._collect_foreign_tables() == set()


# ---------------------------------------------------------------------------
# LazyFrameProxy._build_joined_source
# ---------------------------------------------------------------------------


def test_build_joined_source_no_foreign_returns_raw_base_without_joining():
    lf = pl.LazyFrame({"id": [1, 2]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)

    source, unjoined = proxy._build_joined_source()

    assert source.from_pre_agg is False
    assert unjoined == set()


def test_build_joined_source_known_foreign_applies_join():
    orders = pl.LazyFrame({"order_id": [1, 2], "amount": [100, 200]})
    customers = pl.LazyFrame({"cust_id": [1, 2], "name": ["Alice", "Bob"]})
    join = Join(
        left="orders",
        right="customers",
        left_on=["order_id"],
        right_on=["cust_id"],
        how="inner",
        direction="right2left",
    )
    dm = _MockDM(
        tables={"orders": orders, "customers": customers},
        joins_lookup={"orders": {"customers": [join]}},
    )
    proxy = LazyFrameProxy("orders", dm)
    proxy.group_by_cols = ["customers.name"]

    source, unjoined = proxy._build_joined_source()

    assert unjoined == set()
    assert "customers.name" in source.lf.collect_schema().names()


def test_build_joined_source_unknown_foreign_added_to_unjoined_set():
    lf = pl.LazyFrame({"id": [1, 2]})
    dm = _MockDM({"orders": lf}, joins_lookup={})
    proxy = LazyFrameProxy("orders", dm)
    proxy.group_by_cols = ["ghost.col"]

    source, unjoined = proxy._build_joined_source()

    assert "ghost" in unjoined


def test_build_joined_source_deduplicates_shared_hop():
    """Two join paths sharing a hop: the shared hop is applied exactly once."""
    orders = pl.LazyFrame({"order_id": [1, 2]})
    customers = pl.LazyFrame(
        {"cust_id": [1, 2], "name": ["A", "B"], "region_id": [10, 20]}
    )
    regions = pl.LazyFrame({"rid": [10, 20], "region_name": ["East", "West"]})

    hop1 = Join(
        left="orders",
        right="customers",
        left_on=["order_id"],
        right_on=["cust_id"],
        how="left",
        direction="right2left",
    )
    hop2 = Join(
        left="customers",
        right="regions",
        left_on=["region_id"],
        right_on=["rid"],
        how="left",
        direction="right2left",
    )
    dm = _MockDM(
        tables={"orders": orders, "customers": customers, "regions": regions},
        joins_lookup={
            "orders": {
                "customers": [hop1],
                "regions": [hop1, hop2],  # hop1 shared with customers path
            }
        },
    )
    proxy = LazyFrameProxy("orders", dm)
    proxy.group_by_cols = ["customers.name", "regions.region_name"]

    source, unjoined = proxy._build_joined_source()

    assert unjoined == set()
    cols = source.lf.collect_schema().names()
    assert "customers.name" in cols
    assert "regions.region_name" in cols


# ---------------------------------------------------------------------------
# LazyFrameProxy.resolve — no pre-agg path
# ---------------------------------------------------------------------------


def test_resolve_simple_filter():
    lf = pl.LazyFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)
    proxy.filter(pl.col("tbl.val") > 15)

    rows = proxy.resolve().lf.collect().to_dicts()

    assert rows == [{"tbl.id": 2, "tbl.val": 20}, {"tbl.id": 3, "tbl.val": 30}]


def test_resolve_filter_groupby_agg_chain():
    lf = pl.LazyFrame({"cat": ["a", "a", "b"], "val": [1, 2, 3]})
    dm = _MockDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)
    proxy.filter(pl.col("tbl.val") >= 1).group_by("tbl.cat", maintain_order=True).agg(
        pl.col("tbl.val").sum()
    )

    rows = proxy.resolve().lf.collect().to_dicts()

    assert {"tbl.cat": "a", "tbl.val": 3} in rows
    assert {"tbl.cat": "b", "tbl.val": 3} in rows


def test_resolve_use_pre_agg_false_never_calls_find_best_pre_agg():
    called = []

    class _TrackingDM(_MockDM):
        def find_best_pre_agg(self, table_name, group_by, agg_reqs):
            called.append(True)
            return None

    lf = pl.LazyFrame({"id": [1], "val": [10]})
    dm = _TrackingDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm, use_pre_agg=False)
    proxy.filter(pl.col("tbl.val") > 5)
    proxy.resolve()

    assert called == []


# ---------------------------------------------------------------------------
# LazyFrameProxy.resolve — pre-agg path
# ---------------------------------------------------------------------------


def test_resolve_uses_pre_agg_source_when_provided():
    raw_lf = pl.LazyFrame({"cat": ["a", "a", "b"], "val": [1, 2, 3]})
    pre_agg_lf = pl.LazyFrame(
        {"cat": ["a", "b"], "val-sum": [3, 3], "val-count": [2, 1]}
    )
    dm = _MockDM({"tbl": raw_lf}, pre_agg=cast(PreAggregation, _MockPreAgg(pre_agg_lf)))
    proxy = LazyFrameProxy("tbl", dm)

    result = proxy.resolve()

    assert result.from_pre_agg is True


def test_resolve_pre_agg_source_has_pre_agg_columns_not_raw():
    raw_lf = pl.LazyFrame({"cat": ["a", "b"], "val": [1, 2]})
    pre_agg_lf = pl.LazyFrame({"cat": ["a", "b"], "val-sum": [10, 20]})
    dm = _MockDM({"tbl": raw_lf}, pre_agg=cast(PreAggregation, _MockPreAgg(pre_agg_lf)))
    proxy = LazyFrameProxy("tbl", dm)

    result = proxy.resolve()

    cols = result.lf.collect_schema().names()
    assert "val-sum" in cols
    assert "val" not in cols


def test_resolve_passes_group_by_and_agg_reqs_to_find_best_pre_agg():
    received: dict = {}

    class _TrackingDM(_MockDM):
        def find_best_pre_agg(self, table_name, group_by, agg_reqs):
            received["table_name"] = table_name
            received["group_by"] = group_by
            received["agg_reqs"] = agg_reqs
            return None

    lf = pl.LazyFrame({"cat": ["a", "b"], "val": [1, 2]})
    dm = _TrackingDM({"tbl": lf})
    proxy = LazyFrameProxy("tbl", dm)
    # Use qualified column names as real measure code would after DataModel init
    proxy.group_by("tbl.cat").agg(pl.col("tbl.val").sum())
    proxy.resolve()

    assert received["table_name"] == "tbl"
    assert received["group_by"] == ["tbl.cat"]
    # Columns already qualified — no qualify_col needed
    assert "tbl.val" in received["agg_reqs"]
    assert "Sum" in received["agg_reqs"]["tbl.val"]


# ---------------------------------------------------------------------------
# LazyFrameProxy.replay — qualified columns passed through unchanged
# ---------------------------------------------------------------------------


def test_replay_passes_qualified_string_args_unchanged():
    """Qualified string args are passed through as-is; columns are already qualified."""
    lf = pl.LazyFrame({"id": [3, 1, 2], "val": [30, 10, 20]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.sort("orders.val")

    rows = proxy.resolve().lf.collect().to_dicts()

    assert [r["orders.val"] for r in rows] == [10, 20, 30]


def test_replay_qualified_col_expr_filters_correctly():
    """Qualified pl.col expressions work directly against renamed column names."""
    lf = pl.LazyFrame({"val": [3, 1, 2]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.filter(pl.col("orders.val") > 1)

    rows = proxy.resolve().lf.collect().to_dicts()

    assert set(r["orders.val"] for r in rows) == {3, 2}


def test_replay_drops_expr_referencing_only_unjoined_tables():
    lf = pl.LazyFrame({"val": [1, 2, 3]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    # All col refs are from an unjoinable table → expression dropped silently
    proxy.filter(pl.col("ghost.col") > 0)

    rows = proxy.resolve().lf.collect().to_dicts()

    assert len(rows) == 3


def test_replay_skips_filter_op_when_all_args_dropped():
    lf = pl.LazyFrame({"val": [1, 2, 3]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    proxy.filter(pl.col("ghost.flag"))  # sole arg dropped → op skipped entirely

    rows = proxy.resolve().lf.collect().to_dicts()

    assert len(rows) == 3


# ---------------------------------------------------------------------------
# LazyFrameProxy.replay — nested proxy in join
# ---------------------------------------------------------------------------


def test_replay_join_resolves_nested_proxy_before_joining():
    orders = pl.LazyFrame({"order_id": [1, 2], "amount": [100, 200]})
    customers = pl.LazyFrame({"cust_id": [1, 2], "name": ["Alice", "Bob"]})
    dm = _MockDM({"orders": orders, "customers": customers})

    cust_proxy = LazyFrameProxy("customers", dm)
    orders_proxy = LazyFrameProxy("orders", dm)
    orders_proxy.join(
        cust_proxy,
        left_on=["orders.order_id"],
        right_on=["customers.cust_id"],
        how="inner",
    )

    result = orders_proxy.resolve().lf.collect()

    assert "customers.name" in result.columns
    assert len(result) == 2


# ---------------------------------------------------------------------------
# End-to-end integration
# ---------------------------------------------------------------------------


def test_e2e_filter_sort_groupby_agg():
    lf = pl.LazyFrame(
        {
            "region": ["N", "N", "S", "S"],
            "sales": [100, 200, 50, 150],
        }
    )
    dm = _MockDM({"facts": lf})
    proxy = LazyFrameProxy("facts", dm)

    rows = (
        proxy.filter(pl.col("facts.sales") >= 100)
        .group_by("facts.region", maintain_order=True)
        .agg(pl.col("facts.sales").sum())
        .resolve()
        .lf.collect()
        .to_dicts()
    )

    assert {"facts.region": "N", "facts.sales": 300} in rows
    assert {"facts.region": "S", "facts.sales": 150} in rows


def test_e2e_cross_table_filter_with_known_join():
    orders = pl.LazyFrame({"order_id": [1, 2, 3], "amount": [100, 200, 300]})
    customers = pl.LazyFrame({"cust_id": [1, 2, 3], "region": ["N", "S", "N"]})
    join = Join(
        left="orders",
        right="customers",
        left_on=["order_id"],
        right_on=["cust_id"],
        how="inner",
        direction="right2left",
    )
    dm = _MockDM(
        tables={"orders": orders, "customers": customers},
        joins_lookup={"orders": {"customers": [join]}},
    )
    proxy = LazyFrameProxy("orders", dm)
    proxy.filter(pl.col("customers.region") == "N")

    result = proxy.resolve().lf.collect()
    amounts = sorted(r["orders.amount"] for r in result.to_dicts())

    assert amounts == [100, 300]


def test_e2e_unknown_foreign_col_filter_partially_dropped():
    lf = pl.LazyFrame({"val": [1, 2, 3]})
    dm = _MockDM({"orders": lf})
    proxy = LazyFrameProxy("orders", dm)
    # AND expression: unknown.flag branch is dropped, orders.val > 1 branch survives
    proxy.filter((pl.col("unknown.flag")) & (pl.col("orders.val") > 1))

    rows = proxy.resolve().lf.collect().to_dicts()
    vals = sorted(r["orders.val"] for r in rows)

    assert vals == [2, 3]
