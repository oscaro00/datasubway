from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Self

import numpy as np
import polars as pl
from polars._typing import EngineType, ExplainFormat, IntoExpr, IntoExprColumn
from polars.lazyframe.group_by import LazyGroupBy
from polars.lazyframe.in_process import InProcessQuery
from polars.lazyframe.opt_flags import QueryOptFlags


def serialize_expr(expr: pl.Expr) -> dict:
    return json.loads(expr.meta.serialize(format="json"))


def sum_pre_agg_expr(col: str) -> dict:
    expr = pl.col(f"{col}-sum").sum()
    return serialize_expr(expr)


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
    match agg_type:
        case "Sum":
            return sum_pre_agg_expr

        case "Mean":
            return mean_pre_agg_expr

        case _:
            raise Exception(
                f"{agg_type} not in pre agg transformations in get_pre_agg_transform()"
            )


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


class LazyFrameWrapper:
    def __init__(self, lf: pl.LazyFrame) -> None:
        self.lf = lf
        self.schema = lf.collect_schema()
        self.columns = lf.collect_schema().names()
        self.types = lf.collect_schema().dtypes()

    def filter(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> Self:

        if len(predicates) == 0 and len(constraints) == 0:
            return self
        else:
            return self.__class__(self.lf.filter(*predicates, **constraints))

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> Self | LazyGroupByWrapper:

        if len(by) == 0 and len(named_by) == 0:
            return self.__class__(self.lf)
        else:
            return LazyGroupByWrapper(
                self.lf.group_by(*by, maintain_order=maintain_order, **named_by),
                self.lf,  # need this for access to the schema
            )

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameWrapper:
        rewritten = [
            rewrite_agg_expr(a, self.schema) if isinstance(a, pl.Expr) else a
            for a in aggs
        ]
        named_rewritten = {
            k: rewrite_agg_expr(v, self.schema) if isinstance(v, pl.Expr) else v
            for k, v in named_aggs.items()
        }
        return LazyFrameWrapper(self.lf.select(*rewritten, **named_rewritten))

    def collect(
        self,
        *,
        engine: EngineType = "auto",
        background: bool = False,
        optimizations: QueryOptFlags = QueryOptFlags(),
        **_kwargs: Any,
    ) -> pl.DataFrame | InProcessQuery:
        return self.lf.collect(
            engine=engine, background=background, optimizations=optimizations, **_kwargs
        )

    def explain(
        self,
        *,
        format: ExplainFormat = "plain",
        optimized: bool = True,
        engine: EngineType = "auto",
        tree_format: bool | None = None,
        optimizations: QueryOptFlags = QueryOptFlags(),
    ) -> str:
        return self.lf.explain(
            format=format,
            optimized=optimized,
            engine=engine,
            tree_format=tree_format,
            optimizations=optimizations,
        )


class LazyGroupByWrapper:
    def __init__(self, lgb: LazyGroupBy, original_lazyframe: pl.LazyFrame) -> None:
        self.lgb = lgb
        self.schema = original_lazyframe.collect_schema()
        self.columns = original_lazyframe.collect_schema().names()
        self.types = original_lazyframe.collect_schema().dtypes()

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameWrapper:
        rewritten = [
            rewrite_agg_expr(a, self.schema) if isinstance(a, pl.Expr) else a
            for a in aggs
        ]
        named_rewritten = {
            k: rewrite_agg_expr(v, self.schema) if isinstance(v, pl.Expr) else v
            for k, v in named_aggs.items()
        }
        return LazyFrameWrapper(self.lgb.agg(*rewritten, **named_rewritten))


if __name__ == "__main__":
    lf = pl.LazyFrame(
        {
            "store_id": [1, 2, 3, 4, 5],
            "product_id": [9, 8, 7, 6, 5],
            "revenue": [23, 67, 34, 78, 34],
        }
    )

    lfw = LazyFrameWrapper(lf)

    print(lfw.group_by())
    print(lfw.filter().collect())

    print(
        lfw.filter()
        .group_by()
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )

    print(
        lfw.filter(pl.col("store_id") <= 3)
        .group_by(pl.col("product_id"))
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )

    lf_agg = pl.LazyFrame(
        {
            "store_id": [1, 2, 3, 4, 5],
            "revenue-sum": [100, 200, 300, 400, 500],
            "revenue-count": [7, 6, 5, 4, 3],
        }
    )

    lfw_agg = LazyFrameWrapper(lf_agg)

    print(
        lfw_agg.filter(pl.col("store_id") <= 3)
        .group_by()
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("revenue").mean().round(2).alias("average_revenue"),
        )
        .collect()
    )

    print(
        lfw_agg.filter()
        .group_by("store_id")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("revenue").mean().round(2).alias("average_revenue"),
        )
        .collect()
    )
