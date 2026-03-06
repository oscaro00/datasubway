"""Regression tests for pl.Expr.meta.serialize(format="json") output format.

These tests document and verify the exact JSON structures that pre_agg_expr.py
relies on when traversing serialized expression trees. A failure here means
polars has changed its internal serialization format and pre_agg_expr.py may
silently break.

Run: uv run pytest tests/test_polars_serialization.py -v
"""

from __future__ import annotations

import json

import polars as pl

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(expr: pl.Expr) -> dict:
    return json.loads(expr.meta.serialize(format="json"))


def _deserialize(tree: dict) -> pl.Expr:
    return pl.Expr.deserialize(json.dumps(tree).encode(), format="json")


# ---------------------------------------------------------------------------
# 1. Leaf nodes
# ---------------------------------------------------------------------------


def test_column_ref_top_level_key_is_column():
    tree = _serialize(pl.col("col_name"))
    assert "Column" in tree
    assert tree["Column"] == "col_name"


def test_literal_int_top_level_key_is_literal():
    tree = _serialize(pl.lit(42))
    assert "Literal" in tree


def test_literal_float_top_level_key_is_literal():
    tree = _serialize(pl.lit(3.14))
    assert "Literal" in tree


def test_literal_string_top_level_key_is_literal():
    tree = _serialize(pl.lit("hello"))
    assert "Literal" in tree


def test_literal_null_top_level_key_is_literal():
    tree = _serialize(pl.lit(None))
    assert "Literal" in tree


# ---------------------------------------------------------------------------
# 2. Simple Agg nodes
# ---------------------------------------------------------------------------


def test_sum_agg_top_level_key_is_agg():
    tree = _serialize(pl.col("x").sum())
    assert "Agg" in tree


def test_sum_agg_type_key_is_sum():
    tree = _serialize(pl.col("x").sum())
    assert "Sum" in tree["Agg"]


def test_mean_agg_top_level_key_is_agg():
    tree = _serialize(pl.col("x").mean())
    assert "Agg" in tree


def test_mean_agg_type_key_is_mean():
    tree = _serialize(pl.col("x").mean())
    assert "Mean" in tree["Agg"]


def test_min_agg_top_level_key_is_agg():
    tree = _serialize(pl.col("x").min())
    assert "Agg" in tree


def test_min_agg_type_key_is_min():
    tree = _serialize(pl.col("x").min())
    assert "Min" in tree["Agg"]


def test_max_agg_top_level_key_is_agg():
    tree = _serialize(pl.col("x").max())
    assert "Agg" in tree


def test_max_agg_type_key_is_max():
    tree = _serialize(pl.col("x").max())
    assert "Max" in tree["Agg"]


def test_first_agg_top_level_key_is_agg():
    tree = _serialize(pl.col("x").first())
    assert "Agg" in tree


def test_first_agg_type_key_is_first():
    tree = _serialize(pl.col("x").first())
    assert "First" in tree["Agg"]


def test_last_agg_top_level_key_is_agg():
    tree = _serialize(pl.col("x").last())
    assert "Agg" in tree


def test_last_agg_type_key_is_last():
    tree = _serialize(pl.col("x").last())
    assert "Last" in tree["Agg"]


def test_product_top_level_key_is_function():
    # product() serializes as Function, not Agg
    tree = _serialize(pl.col("x").product())
    assert "Function" in tree


def test_product_function_field_is_product():
    tree = _serialize(pl.col("x").product())
    assert tree["Function"]["function"] == "Product"


def test_null_count_top_level_key_is_function():
    # null_count() serializes as Function, not Agg
    tree = _serialize(pl.col("x").null_count())
    assert "Function" in tree


def test_null_count_function_field_is_null_count():
    tree = _serialize(pl.col("x").null_count())
    assert tree["Function"]["function"] == "NullCount"


def test_count_agg_type_key_is_count():
    tree = _serialize(pl.col("x").count())
    assert "Agg" in tree
    assert "Count" in tree["Agg"]


def test_count_agg_include_nulls_is_false():
    tree = _serialize(pl.col("x").count())
    agg_value = tree["Agg"]["Count"]
    include_nulls = agg_value.get("include_nulls", False)
    assert include_nulls is False


def test_len_agg_type_key_is_count():
    # len() serializes as Count (same top-level key as count())
    tree = _serialize(pl.col("x").len())
    assert "Agg" in tree
    assert "Count" in tree["Agg"]


def test_len_agg_include_nulls_is_true():
    # len() is distinguished from count() by include_nulls=True
    tree = _serialize(pl.col("x").len())
    agg_value = tree["Agg"]["Count"]
    include_nulls = agg_value.get("include_nulls", False)
    assert include_nulls is True


# ---------------------------------------------------------------------------
# 3. Function-pattern agg nodes
# ---------------------------------------------------------------------------


def test_n_unique_top_level_key_is_agg():
    # n_unique() serializes as Agg (not Function) in current polars
    tree = _serialize(pl.col("x").n_unique())
    assert "Agg" in tree


def test_n_unique_agg_type_key_is_n_unique():
    tree = _serialize(pl.col("x").n_unique())
    assert "NUnique" in tree["Agg"]


def test_n_unique_agg_inner_contains_column():
    tree = _serialize(pl.col("x").n_unique())
    assert tree["Agg"]["NUnique"]["Column"] == "x"


def test_median_top_level_key_is_agg():
    # median() serializes as Agg (not Function) in current polars
    tree = _serialize(pl.col("x").median())
    assert "Agg" in tree


def test_median_agg_type_key_is_median():
    tree = _serialize(pl.col("x").median())
    assert "Median" in tree["Agg"]


def test_median_agg_inner_contains_column():
    tree = _serialize(pl.col("x").median())
    assert tree["Agg"]["Median"]["Column"] == "x"


def test_all_top_level_key_is_function():
    tree = _serialize(pl.col("x").all())
    assert "Function" in tree


def test_all_function_field_has_boolean_key():
    tree = _serialize(pl.col("x").all())
    func = tree["Function"]["function"]
    assert isinstance(func, dict)
    assert "Boolean" in func


def test_all_boolean_subkey_is_all():
    tree = _serialize(pl.col("x").all())
    func = tree["Function"]["function"]
    assert "All" in func["Boolean"]


def test_any_top_level_key_is_function():
    tree = _serialize(pl.col("x").any())
    assert "Function" in tree


def test_any_function_field_has_boolean_key():
    tree = _serialize(pl.col("x").any())
    func = tree["Function"]["function"]
    assert isinstance(func, dict)
    assert "Boolean" in func


def test_any_boolean_subkey_is_any():
    tree = _serialize(pl.col("x").any())
    func = tree["Function"]["function"]
    assert "Any" in func["Boolean"]


# ---------------------------------------------------------------------------
# 4. BinaryExpr nodes
# ---------------------------------------------------------------------------


def test_binary_expr_top_level_key_is_binary_expr():
    tree = _serialize(pl.col("a") & pl.col("b"))
    assert "BinaryExpr" in tree


def test_binary_expr_has_left_op_right_keys():
    tree = _serialize(pl.col("a") & pl.col("b"))
    binary = tree["BinaryExpr"]
    assert "left" in binary
    assert "op" in binary
    assert "right" in binary


def test_binary_expr_and_op_is_and():
    tree = _serialize(pl.col("a") & pl.col("b"))
    assert tree["BinaryExpr"]["op"] == "And"


def test_binary_expr_or_op_is_or():
    tree = _serialize(pl.col("a") | pl.col("b"))
    assert tree["BinaryExpr"]["op"] == "Or"


# ---------------------------------------------------------------------------
# 5. Alias nodes
# ---------------------------------------------------------------------------


def test_alias_top_level_key_is_alias():
    tree = _serialize(pl.col("revenue").alias("rev"))
    assert "Alias" in tree


def test_alias_value_is_list():
    tree = _serialize(pl.col("revenue").alias("rev"))
    assert isinstance(tree["Alias"], list)
    assert len(tree["Alias"]) == 2


def test_alias_second_element_is_name():
    tree = _serialize(pl.col("revenue").alias("rev"))
    assert tree["Alias"][1] == "rev"


# ---------------------------------------------------------------------------
# 6. Column name reachability (contract tests for get_col_name traversal)
# ---------------------------------------------------------------------------


def test_agg_sum_inner_value_contains_column():
    tree = _serialize(pl.col("revenue").sum())
    assert tree["Agg"]["Sum"]["Column"] == "revenue"


def test_agg_min_inner_value_contains_column():
    # min() inner value has shape {"input": {"Column": "..."}, "propagate_nans": ...}
    tree = _serialize(pl.col("revenue").min())
    assert tree["Agg"]["Min"]["input"]["Column"] == "revenue"


def test_agg_max_inner_value_contains_column():
    # max() inner value has shape {"input": {"Column": "..."}, "propagate_nans": ...}
    tree = _serialize(pl.col("revenue").max())
    assert tree["Agg"]["Max"]["input"]["Column"] == "revenue"


def test_agg_n_unique_inner_value_contains_column():
    tree = _serialize(pl.col("revenue").n_unique())
    assert tree["Agg"]["NUnique"]["Column"] == "revenue"


def test_agg_median_inner_value_contains_column():
    tree = _serialize(pl.col("revenue").median())
    assert tree["Agg"]["Median"]["Column"] == "revenue"


# ---------------------------------------------------------------------------
# 7. Round-trip tests
# ---------------------------------------------------------------------------


def test_roundtrip_sum_produces_correct_result():
    tree = _serialize(pl.col("x").sum())
    expr = _deserialize(tree)
    df = pl.DataFrame({"x": [1, 2, 3]})
    result = df.select(expr)
    assert result["x"][0] == 6


def test_roundtrip_alias_preserves_name_and_value():
    tree = _serialize(pl.col("x").sum().alias("total"))
    expr = _deserialize(tree)
    df = pl.DataFrame({"x": [10, 20, 30]})
    result = df.select(expr)
    assert "total" in result.columns
    assert result["total"][0] == 60


def test_roundtrip_column_name_modification_works():
    # Mirrors the round-trip demo in demo_serialize.py:
    # serialize -> modify Column key -> deserialize -> execute
    tree = _serialize(pl.col("revenue").sum().round(2).alias("total"))
    tree["Alias"][0]["Function"]["input"][0]["Agg"]["Sum"]["Column"] = "revenue-sum"
    expr = _deserialize(tree)
    df = pl.DataFrame({"revenue-sum": [10, 20, 30]})
    result = df.select(expr)
    assert "total" in result.columns
    assert result["total"][0] == 60.0
