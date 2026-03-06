from __future__ import annotations

import json
from typing import Any, Callable

import polars as pl


def serialize_expr(expr: pl.Expr) -> dict:
    return json.loads(expr.meta.serialize(format="json"))


pre_agg_transformations = {}


def pre_agg_transform(agg_type: str) -> Callable:
    """Create the decorator @pre_agg_transform(agg_type) to populate get_pre_agg_transform() automatically"""

    def decorator(func: Callable) -> Callable:
        pre_agg_transformations[agg_type] = func
        return func

    return decorator


# ── Simple aggregations (direct mapping to pre-agg column) ───────────────────


@pre_agg_transform("Sum")
def sum_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-sum").sum())


@pre_agg_transform("Min")
def min_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-min").min())


@pre_agg_transform("Max")
def max_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-max").max())


@pre_agg_transform("Count")
def count_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-count").sum())


@pre_agg_transform("Len")
def len_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-len").sum())


@pre_agg_transform("First")
def first_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-first").first())


@pre_agg_transform("Last")
def last_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-last").last())


@pre_agg_transform("Product")
def product_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-product").product())


@pre_agg_transform("NullCount")
def null_count_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-null_count").sum())


@pre_agg_transform("All")
def all_pre_agg_expr(col: str, *, ignore_nulls: bool = True) -> dict:
    return serialize_expr(pl.col(f"{col}-all").all(ignore_nulls=ignore_nulls))


@pre_agg_transform("Any")
def any_pre_agg_expr(col: str, *, ignore_nulls: bool = True) -> dict:
    return serialize_expr(pl.col(f"{col}-any").any(ignore_nulls=ignore_nulls))


# ── List-based aggregations (store lists in pre-agg, flatten to re-aggregate) ─


# NOTE: storing this type of pre aggregation is expensive!
@pre_agg_transform("NUnique")
def n_unique_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-unique_set").flatten().n_unique())


# NOTE: storing this type of pre aggregation is expensive!
@pre_agg_transform("Median")
def median_pre_agg_expr(col: str) -> dict:
    return serialize_expr(pl.col(f"{col}-values_list").flatten().median())


# ── Decomposed aggregations (require multiple pre-agg columns) ───────────────


@pre_agg_transform("Mean")
def mean_pre_agg_expr(col: str) -> dict:
    expr = pl.col(f"{col}-sum").sum() / pl.col(f"{col}-count").sum()
    return serialize_expr(expr)


@pre_agg_transform("Std")
def std_pre_agg_expr(col: str, *, ddof: int = 1) -> dict:
    sumsq = pl.col(f"{col}-sumsq").sum()
    s = pl.col(f"{col}-sum").sum()
    n = pl.col(f"{col}-count").sum()
    expr = ((sumsq - s.pow(2) / n) / (n - ddof)).sqrt()
    return serialize_expr(expr)


@pre_agg_transform("Var")
def var_pre_agg_expr(col: str, *, ddof: int = 1) -> dict:
    sumsq = pl.col(f"{col}-sumsq").sum()
    s = pl.col(f"{col}-sum").sum()
    n = pl.col(f"{col}-count").sum()
    expr = (sumsq - s.pow(2) / n) / (n - ddof)
    return serialize_expr(expr)


# ── Helpers ──────────────────────────────────────────────────────────────────


def get_col_name(node: Any) -> str | None:
    """Extract the column name string from any serialized expression node."""
    if isinstance(node, dict):
        if "Column" in node:
            return node["Column"]
        for v in node.values():
            result = get_col_name(v)
            if result is not None:
                return result
    if isinstance(node, list):
        for item in node:
            result = get_col_name(item)
            if result is not None:
                return result
    return None


def get_function_agg_type(node: dict) -> str | None:
    """Extract agg type from a Function-pattern node, or None if not a recognized agg."""
    func_field = node["Function"]["function"]
    if isinstance(func_field, str):
        return func_field
    if isinstance(func_field, dict) and "Boolean" in func_field:
        return next(iter(func_field["Boolean"]))
    return None


def get_pre_agg_transform(agg_type: str) -> Callable:
    if agg_type not in pre_agg_transformations:
        raise Exception(
            f"{agg_type} not in pre agg transformations in get_pre_agg_transform()"
        )
    return pre_agg_transformations[agg_type]


def match_agg_node(node: dict) -> tuple[str, str] | None:
    """If node is a recognized Agg or Function pattern, return (col_name, agg_type), else None."""
    if "Agg" in node and len(node) == 1:
        for agg_type, agg_value in node["Agg"].items():
            col = get_col_name(agg_value)
            # Count vs Len: both serialize as Count, differentiated by include_nulls
            if agg_type == "Count":
                include_nulls = agg_value.get("include_nulls", False)
                agg_type = "Len" if include_nulls else "Count"
            if col and agg_type in pre_agg_transformations:
                return col, agg_type

    if "Function" in node and len(node) == 1:
        agg_type = get_function_agg_type(node)
        if agg_type and agg_type in pre_agg_transformations:
            col = get_col_name(node["Function"]["input"])
            if col:
                return col, agg_type

    return None


def walk_agg_expr(node: Any) -> Any:
    """Recursively walk the serialized expression tree and rewrite Agg/Function nodes."""
    if isinstance(node, dict):
        match = match_agg_node(node)
        if match:
            col, agg_type = match
            return get_pre_agg_transform(agg_type)(col)
        return {k: walk_agg_expr(v) for k, v in node.items()}
    if isinstance(node, list):
        return [walk_agg_expr(item) for item in node]
    return node


def rewrite_agg_expr(expr: pl.Expr) -> pl.Expr:
    """Rewrite a Polars expression to use pre-aggregated columns where available"""
    tree = json.loads(expr.meta.serialize(format="json"))
    rewritten = walk_agg_expr(tree)
    if (
        # TODO: check if this is inefficient (i.e. comparing json objects)
        rewritten == tree
    ):
        return expr
    return pl.Expr.deserialize(json.dumps(rewritten).encode(), format="json")


def _collect_col_names_from_tree(node: Any) -> list[str]:
    """Collect all column names from a serialized expression tree."""
    if isinstance(node, dict):
        if "Column" in node:
            return [node["Column"]]
        result = []
        for v in node.values():
            result.extend(_collect_col_names_from_tree(v))
        return result
    if isinstance(node, list):
        result = []
        for item in node:
            result.extend(_collect_col_names_from_tree(item))
        return result
    return []


def _all_unjoinable_in_node(node: Any, unjoined_tables: set[str]) -> bool:
    """True if all qualified column refs in a JSON node are from unjoinable tables."""
    names = _collect_col_names_from_tree(node)
    qualified = [n for n in names if "." in n]
    if not qualified:
        return False
    return all(n.split(".", 1)[0] in unjoined_tables for n in qualified)


def _drop_unjoined_in_tree(node: Any, unjoined_tables: set[str]) -> Any | None:
    """Recursively remove sub-trees that reference only unjoinable tables.

    Returns None if the entire node should be dropped.
    """
    if not isinstance(node, dict) or "BinaryExpr" not in node:
        if _all_unjoinable_in_node(node, unjoined_tables):
            return None
        return node

    binary = node["BinaryExpr"]
    op = binary.get("op")

    if op in ["And", "Or"]:
        left = _drop_unjoined_in_tree(binary["left"], unjoined_tables)
        right = _drop_unjoined_in_tree(binary["right"], unjoined_tables)
        if left is None and right is None:
            return None
        if left is None:
            return right
        if right is None:
            return left
        return {"BinaryExpr": {"left": left, "op": op, "right": right}}

    # Other binary expression: drop only if all refs are unjoinable
    if _all_unjoinable_in_node(node, unjoined_tables):
        return None
    return node


def drop_unjoined_table_refs(
    expr: pl.Expr,
    unjoined_tables: set[str],
) -> pl.Expr | None:
    """Remove sub-expressions that exclusively reference unjoinable tables.

    Returns None if the whole expression should be dropped.

    Rules:
    - If no column reference is from an unjoinable table → return expr unchanged.
    - If all column references are from unjoinable tables → return None.
    - AND compounds: drop branches from unjoinable tables; reconstruct from survivors.
    - OR compounds: drop branches from unjoinable tables; reconstruct from survivors.
    - Other compounds: drop if all refs unjoinable, else return unchanged.
    """
    if not unjoined_tables:
        return expr

    root_names = expr.meta.root_names()

    has_unjoinable = any(
        "." in n and n.split(".", 1)[0] in unjoined_tables for n in root_names
    )
    if not has_unjoinable:
        return expr

    all_unjoinable = len(root_names) > 0 and all(
        "." in n and n.split(".", 1)[0] in unjoined_tables for n in root_names
    )
    if all_unjoinable:
        return None

    # Mixed: use JSON tree walk to separate AND branches
    tree = json.loads(expr.meta.serialize(format="json"))
    cleaned = _drop_unjoined_in_tree(tree, unjoined_tables)
    if cleaned is None:
        return None
    if cleaned == tree:
        return expr
    return pl.Expr.deserialize(json.dumps(cleaned).encode(), format="json")


def extract_agg_requirements(expr: pl.Expr) -> dict[str, set[str]]:
    """Walk a Polars expression and return {col_name: {PolarsAggType}} pairs.

    Used by LazyFrameProxy.resolve() to determine what pre-aggregation components
    are needed to satisfy the aggregations in a measure's .agg() call.
    """
    tree = json.loads(expr.meta.serialize(format="json"))
    requirements: dict[str, set[str]] = {}
    walk_for_requirements(tree, requirements)
    return requirements


def walk_for_requirements(node: Any, requirements: dict[str, set[str]]) -> None:
    if isinstance(node, dict):
        match = match_agg_node(node)
        if match:
            col, agg_type = match
            requirements.setdefault(col, set()).add(agg_type)
            return
        for v in node.values():
            walk_for_requirements(v, requirements)
    elif isinstance(node, list):
        for item in node:
            walk_for_requirements(item, requirements)
