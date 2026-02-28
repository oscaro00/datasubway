from __future__ import annotations

import json
import math

import polars as pl
import pytest

from datasubway.polars_wrappers.pre_agg_expr import (
    _all_unjoinable_in_node,
    _collect_col_names_from_tree,
    all_pre_agg_expr,
    any_pre_agg_expr,
    count_pre_agg_expr,
    drop_unjoined_table_refs,
    extract_agg_requirements,
    first_pre_agg_expr,
    get_col_name,
    get_function_agg_type,
    get_pre_agg_transform,
    last_pre_agg_expr,
    len_pre_agg_expr,
    match_agg_node,
    max_pre_agg_expr,
    mean_pre_agg_expr,
    median_pre_agg_expr,
    min_pre_agg_expr,
    n_unique_pre_agg_expr,
    null_count_pre_agg_expr,
    pre_agg_transformations,
    product_pre_agg_expr,
    rewrite_agg_expr,
    serialize_expr,
    std_pre_agg_expr,
    sum_pre_agg_expr,
    var_pre_agg_expr,
)

# ─────────────────────────────────────────────────────────────────────────────
# Shared test data
# ─────────────────────────────────────────────────────────────────────────────

PRE_AGG_LF = pl.LazyFrame(
    {
        "country": ["US", "CA"],
        "revenue-sum": [500.0, 2300.0],
        "revenue-count": [1, 2],
        "revenue-min": [500.0, 1500.0],
        "revenue-max": [500.0, 1500.0],
        "revenue-sumsq": [250000.0, 5290000.0],
        "revenue-len": [1, 2],
        "revenue-first": [500.0, 1500.0],
        "revenue-last": [500.0, 800.0],
        "revenue-null_count": [0, 0],
        "revenue-product": [500.0, 1200000.0],
        "revenue-unique_set": [[500.0], [1500.0, 800.0]],
        "revenue-values_list": [[500.0], [1500.0, 800.0]],
        "flag-all": [True, True],
        "flag-any": [False, True],
    }
)


def _deserialize(d: dict) -> pl.Expr:
    return pl.Expr.deserialize(json.dumps(d).encode(), format="json")


# ─────────────────────────────────────────────────────────────────────────────
# serialize_expr
# ─────────────────────────────────────────────────────────────────────────────


def test_serialize_expr_returns_dict():
    result = serialize_expr(pl.col("x").sum())
    assert isinstance(result, dict)


def test_serialize_expr_round_trip():
    expr = pl.col("x").sum()
    serialized = serialize_expr(expr)
    recovered = _deserialize(serialized)
    assert recovered.meta.root_names() == ["x"]


# ─────────────────────────────────────────────────────────────────────────────
# pre_agg_transform decorator / get_pre_agg_transform
# ─────────────────────────────────────────────────────────────────────────────


def test_all_agg_types_registered():
    expected = {
        "Sum",
        "Min",
        "Max",
        "Count",
        "Len",
        "First",
        "Last",
        "Product",
        "NullCount",
        "All",
        "Any",
        "NUnique",
        "Median",
        "Mean",
        "Std",
        "Var",
    }
    for agg_type in expected:
        assert agg_type in pre_agg_transformations, f"{agg_type} not registered"


def test_get_pre_agg_transform_returns_callable():
    transform = get_pre_agg_transform("Sum")
    assert callable(transform)


def test_get_pre_agg_transform_unknown_raises():
    with pytest.raises(Exception):
        get_pre_agg_transform("Unknown")


# ─────────────────────────────────────────────────────────────────────────────
# Simple pre-agg expression functions
# ─────────────────────────────────────────────────────────────────────────────


def test_sum_pre_agg_expr_result():
    expr = _deserialize(sum_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 2800.0


def test_min_pre_agg_expr_result():
    expr = _deserialize(min_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 500.0


def test_max_pre_agg_expr_result():
    expr = _deserialize(max_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 1500.0


def test_count_pre_agg_expr_result():
    expr = _deserialize(count_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 3


def test_len_pre_agg_expr_result():
    expr = _deserialize(len_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 3


def test_first_pre_agg_expr_result():
    expr = _deserialize(first_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 500.0


def test_last_pre_agg_expr_result():
    expr = _deserialize(last_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 800.0


def test_null_count_pre_agg_expr_result():
    expr = _deserialize(null_count_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 0


def test_product_pre_agg_expr_result():
    expr = _deserialize(product_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == 500.0 * 1200000.0


# ─────────────────────────────────────────────────────────────────────────────
# all / any pre-agg expression functions
# ─────────────────────────────────────────────────────────────────────────────


def test_all_pre_agg_expr_result():
    expr = _deserialize(all_pre_agg_expr("flag"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result is True


def test_any_pre_agg_expr_result():
    expr = _deserialize(any_pre_agg_expr("flag"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result is True


def test_all_pre_agg_expr_ignore_nulls_default():
    d_default = all_pre_agg_expr("flag")
    d_explicit = all_pre_agg_expr("flag", ignore_nulls=True)
    assert d_default == d_explicit


def test_any_pre_agg_expr_ignore_nulls_default():
    d_default = any_pre_agg_expr("flag")
    d_explicit = any_pre_agg_expr("flag", ignore_nulls=True)
    assert d_default == d_explicit


# ─────────────────────────────────────────────────────────────────────────────
# List-based aggregation functions
# ─────────────────────────────────────────────────────────────────────────────


def test_n_unique_pre_agg_expr_result():
    expr = _deserialize(n_unique_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    # unique_set [[500.0], [1500.0, 800.0]] flattened → [500, 1500, 800] → 3 unique
    assert result == 3


def test_median_pre_agg_expr_result():
    expr = _deserialize(median_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    # values_list [[500.0], [1500.0, 800.0]] flattened → [500, 1500, 800] → median = 800.0
    assert result == 800.0


# ─────────────────────────────────────────────────────────────────────────────
# Decomposed aggregation functions
# ─────────────────────────────────────────────────────────────────────────────


def test_mean_pre_agg_expr_result():
    expr = _deserialize(mean_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    assert result == pytest.approx(2800.0 / 3, rel=1e-6)


def test_std_pre_agg_expr_ddof1():
    expr = _deserialize(std_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    sumsq = 250000.0 + 5290000.0
    s = 500.0 + 2300.0
    n = 3
    expected = math.sqrt((sumsq - s**2 / n) / (n - 1))
    assert result == pytest.approx(expected, rel=1e-6)


def test_std_pre_agg_expr_ddof0_differs_from_ddof1():
    expr_ddof0 = _deserialize(std_pre_agg_expr("revenue", ddof=0))
    expr_ddof1 = _deserialize(std_pre_agg_expr("revenue"))
    result_ddof0 = PRE_AGG_LF.select(expr_ddof0).collect().item()
    result_ddof1 = PRE_AGG_LF.select(expr_ddof1).collect().item()
    sumsq = 250000.0 + 5290000.0
    s = 500.0 + 2300.0
    n = 3
    expected_ddof0 = math.sqrt((sumsq - s**2 / n) / n)
    assert result_ddof0 == pytest.approx(expected_ddof0, rel=1e-6)
    assert result_ddof0 != pytest.approx(result_ddof1, rel=1e-3)


def test_var_pre_agg_expr_ddof1():
    expr = _deserialize(var_pre_agg_expr("revenue"))
    result = PRE_AGG_LF.select(expr).collect().item()
    sumsq = 250000.0 + 5290000.0
    s = 500.0 + 2300.0
    n = 3
    expected = (sumsq - s**2 / n) / (n - 1)
    assert result == pytest.approx(expected, rel=1e-6)


def test_var_pre_agg_expr_ddof0():
    expr = _deserialize(var_pre_agg_expr("revenue", ddof=0))
    result = PRE_AGG_LF.select(expr).collect().item()
    sumsq = 250000.0 + 5290000.0
    s = 500.0 + 2300.0
    n = 3
    expected = (sumsq - s**2 / n) / n
    assert result == pytest.approx(expected, rel=1e-6)


def test_var_pre_agg_expr_equals_std_squared():
    var_expr = _deserialize(var_pre_agg_expr("revenue")).alias("var")
    std_expr = _deserialize(std_pre_agg_expr("revenue")).alias("std")
    df = PRE_AGG_LF.select([var_expr, std_expr]).collect()
    assert df["var"].item() == pytest.approx(df["std"].item() ** 2, rel=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# get_col_name
# ─────────────────────────────────────────────────────────────────────────────


def test_get_col_name_direct():
    assert get_col_name({"Column": "foo"}) == "foo"


def test_get_col_name_nested():
    node = {"Agg": {"Sum": {"Column": "foo"}}}
    assert get_col_name(node) == "foo"


def test_get_col_name_no_column_key():
    assert get_col_name({"key": "value"}) is None


def test_get_col_name_traverses_list():
    node = [{"Column": "first"}, {"Column": "second"}]
    assert get_col_name(node) == "first"


# ─────────────────────────────────────────────────────────────────────────────
# get_function_agg_type
# ─────────────────────────────────────────────────────────────────────────────


def test_get_function_agg_type_plain_string():
    node = {"Function": {"function": "Sum", "input": []}}
    assert get_function_agg_type(node) == "Sum"


def test_get_function_agg_type_boolean():
    node = {"Function": {"function": {"Boolean": {"All": {}}}, "input": []}}
    assert get_function_agg_type(node) == "All"


def test_get_function_agg_type_unrecognized():
    node = {"Function": {"function": {"Unknown": "value"}, "input": []}}
    assert get_function_agg_type(node) is None


# ─────────────────────────────────────────────────────────────────────────────
# match_agg_node
# ─────────────────────────────────────────────────────────────────────────────


def test_match_agg_node_sum():
    node = serialize_expr(pl.col("revenue").sum())
    assert match_agg_node(node) == ("revenue", "Sum")


def test_match_agg_node_min():
    node = serialize_expr(pl.col("revenue").min())
    assert match_agg_node(node) == ("revenue", "Min")


def test_match_agg_node_max():
    node = serialize_expr(pl.col("revenue").max())
    assert match_agg_node(node) == ("revenue", "Max")


def test_match_agg_node_count():
    node = serialize_expr(pl.col("revenue").count())
    assert match_agg_node(node) == ("revenue", "Count")


def test_match_agg_node_len():
    node = serialize_expr(pl.col("revenue").len())
    assert match_agg_node(node) == ("revenue", "Len")


def test_match_agg_node_mean():
    node = serialize_expr(pl.col("revenue").mean())
    assert match_agg_node(node) == ("revenue", "Mean")


def test_match_agg_node_plain_col_returns_none():
    node = serialize_expr(pl.col("revenue"))
    assert match_agg_node(node) is None


def test_match_agg_node_unrecognized_agg_returns_none():
    node = {"Agg": {"UnknownAgg": {"Column": "revenue"}}}
    assert match_agg_node(node) is None


# ─────────────────────────────────────────────────────────────────────────────
# rewrite_agg_expr
# ─────────────────────────────────────────────────────────────────────────────


def test_rewrite_agg_expr_sum_references_pre_agg_col():
    rewritten = rewrite_agg_expr(pl.col("revenue").sum())
    assert "revenue-sum" in rewritten.meta.root_names()


def test_rewrite_agg_expr_mean_references_sum_and_count():
    rewritten = rewrite_agg_expr(pl.col("revenue").mean())
    root_names = rewritten.meta.root_names()
    assert "revenue-sum" in root_names
    assert "revenue-count" in root_names


def test_rewrite_agg_expr_std_references_sumsq_sum_count():
    rewritten = rewrite_agg_expr(pl.col("revenue").std())
    root_names = rewritten.meta.root_names()
    assert "revenue-sumsq" in root_names
    assert "revenue-sum" in root_names
    assert "revenue-count" in root_names


def test_rewrite_agg_expr_noop_returns_same_object():
    expr = pl.col("revenue")
    result = rewrite_agg_expr(expr)
    assert result is expr


def test_rewrite_agg_expr_end_to_end():
    rewritten = rewrite_agg_expr(pl.col("revenue").sum())
    result = PRE_AGG_LF.select(rewritten).collect().item()
    assert result == 2800.0


# ─────────────────────────────────────────────────────────────────────────────
# _collect_col_names_from_tree
# ─────────────────────────────────────────────────────────────────────────────


def test_collect_col_names_from_tree_single():
    assert _collect_col_names_from_tree({"Column": "revenue"}) == ["revenue"]


def test_collect_col_names_from_tree_nested():
    node = {"Agg": {"Sum": {"Column": "revenue"}}}
    assert _collect_col_names_from_tree(node) == ["revenue"]


def test_collect_col_names_from_tree_multiple():
    node = {
        "BinaryExpr": {
            "left": {"Column": "revenue"},
            "op": "Plus",
            "right": {"Column": "amount"},
        }
    }
    result = _collect_col_names_from_tree(node)
    assert "revenue" in result
    assert "amount" in result


# ─────────────────────────────────────────────────────────────────────────────
# _all_unjoinable_in_node
# ─────────────────────────────────────────────────────────────────────────────


def test_all_unjoinable_in_node_true():
    node = {"Column": "orders.revenue"}
    assert _all_unjoinable_in_node(node, {"orders"}) is True


def test_all_unjoinable_in_node_no_qualified_refs():
    node = {"Column": "revenue"}
    assert _all_unjoinable_in_node(node, {"orders"}) is False


def test_all_unjoinable_in_node_mixed_tables():
    node = {
        "left": {"Column": "orders.revenue"},
        "right": {"Column": "sales.amount"},
    }
    assert _all_unjoinable_in_node(node, {"orders"}) is False


# ─────────────────────────────────────────────────────────────────────────────
# drop_unjoined_table_refs
# ─────────────────────────────────────────────────────────────────────────────


def test_drop_unjoined_table_refs_empty_set_unchanged():
    expr = pl.col("orders.revenue") > pl.lit(100)
    result = drop_unjoined_table_refs(expr, set())
    assert result is expr


def test_drop_unjoined_table_refs_all_unjoinable_returns_none():
    expr = pl.col("orders.revenue") > pl.lit(100)
    result = drop_unjoined_table_refs(expr, {"orders"})
    assert result is None


def test_drop_unjoined_table_refs_no_unjoinable_unchanged():
    expr = pl.col("orders.revenue") > pl.lit(100)
    result = drop_unjoined_table_refs(expr, {"geo"})
    assert result is expr


def test_drop_unjoined_table_refs_and_drops_unjoinable_branch():
    expr = (pl.col("orders.revenue") > pl.lit(100)) & (
        pl.col("geo.country") == pl.lit("US")
    )
    result = drop_unjoined_table_refs(expr, {"orders"})
    assert result is not None
    root_names = result.meta.root_names()
    assert "geo.country" in root_names
    assert "orders.revenue" not in root_names


def test_drop_unjoined_table_refs_or_drops_unjoinable_branch():
    expr = (pl.col("orders.revenue") > pl.lit(100)) | (
        pl.col("geo.country") == pl.lit("US")
    )
    result = drop_unjoined_table_refs(expr, {"orders"})
    assert result is not None
    root_names = result.meta.root_names()
    assert "geo.country" in root_names
    assert "orders.revenue" not in root_names


def test_drop_unjoined_table_refs_entire_and_tree_unjoinable():
    expr = (pl.col("orders.revenue") > pl.lit(100)) & (
        pl.col("orders.amount") < pl.lit(500)
    )
    result = drop_unjoined_table_refs(expr, {"orders"})
    assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# extract_agg_requirements
# ─────────────────────────────────────────────────────────────────────────────


def test_extract_agg_requirements_single_sum():
    result = extract_agg_requirements(pl.col("revenue").sum())
    assert result == {"revenue": {"Sum"}}


def test_extract_agg_requirements_mean():
    result = extract_agg_requirements(pl.col("revenue").mean())
    assert result == {"revenue": {"Mean"}}


def test_extract_agg_requirements_multiple_aggs_same_col():
    expr = pl.col("revenue").sum() + pl.col("revenue").mean()
    result = extract_agg_requirements(expr)
    assert result == {"revenue": {"Sum", "Mean"}}


def test_extract_agg_requirements_multiple_cols():
    expr = pl.col("revenue").sum() + pl.col("amount").mean()
    result = extract_agg_requirements(expr)
    assert result == {"revenue": {"Sum"}, "amount": {"Mean"}}


def test_extract_agg_requirements_no_aggs():
    result = extract_agg_requirements(pl.col("revenue"))
    assert result == {}
