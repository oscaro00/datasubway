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


@pre_agg_transform("All")
def all_pre_agg_expr(col: str, *, ignore_nulls: bool = True) -> dict:
    expr = pl.col(f"{col}-all").all(ignore_nulls=ignore_nulls)
    return serialize_expr(expr)


@pre_agg_transform("Any")
def any_pre_agg_expr(col: str, *, ignore_nulls: bool = True) -> dict:
    expr = pl.col(f"{col}-any").any(ignore_nulls=ignore_nulls)
    return serialize_expr(expr)


@pre_agg_transform("Sum")
def sum_pre_agg_expr(col: str) -> dict:
    expr = pl.col(f"{col}-sum").sum()
    return serialize_expr(expr)


@pre_agg_transform("Mean")
def mean_pre_agg_expr(col: str) -> dict:
    expr = pl.col(f"{col}-sum").sum() / pl.col(f"{col}-count").sum()
    return serialize_expr(expr)


def get_col_name(node: Any) -> Any:
    if isinstance(node, dict):
        if "Column" in node.keys():
            return node["Column"]

        return {k: get_col_name(v) for k, v in node.items()}

    if isinstance(node, list):
        return [get_col_name(item) for item in node]


def get_pre_agg_transform(agg_type: str) -> Callable:
    if agg_type not in pre_agg_transformations:
        raise Exception(
            f"{agg_type} not in pre agg transformations in get_pre_agg_transform()"
        )
    return pre_agg_transformations[agg_type]


def walk_agg_expr(node: Any, schema: pl.Schema) -> Any:
    """Recursively walk the serialized expression tree and rewrite Agg nodes."""
    if isinstance(node, dict):
        # If a potential Agg node that needs to be replaced
        if "Agg" in node and len(node) == 1:
            agg_dict = node["Agg"]
            for agg_type, agg_value in agg_dict.items():
                col_name = get_col_name(agg_value)

                # column not in schema means it needs to be transformed for a pre aggregation table
                if col_name not in schema:
                    pre_agg_transform = get_pre_agg_transform(agg_type)
                    return pre_agg_transform(col_name)

        # Recurse into inner dict
        return {k: walk_agg_expr(v, schema) for k, v in node.items()}

    # Recurse into list
    if isinstance(node, list):
        return [walk_agg_expr(item, schema) for item in node]

    return node


def rewrite_agg_expr(expr: pl.Expr, schema: pl.Schema) -> pl.Expr:
    """Rewrite a Polars expression to use pre-aggregated columns where available"""
    tree = json.loads(expr.meta.serialize(format="json"))
    rewritten = walk_agg_expr(tree, schema)
    if (
        # TODO: check if this is inefficient (i.e. comparing json objects)
        rewritten == tree
    ):
        return expr
    return pl.Expr.deserialize(json.dumps(rewritten).encode(), format="json")
