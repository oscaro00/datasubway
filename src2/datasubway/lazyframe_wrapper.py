from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

if TYPE_CHECKING:
    from datasubway.lazygroupby_wrapper import LazyGroupByWrapper

from datetime import timedelta

import numpy as np
import polars as pl
from polars._typing import (
    ClosedInterval,
    IntoExpr,
    IntoExprColumn,
    Label,
    SchemaDict,
    StartBy,
)

from datasubway.pre_agg_expr import rewrite_agg_expr


class LazyFrameWrapper:
    def __init__(self, lf: pl.LazyFrame, from_pre_agg: bool = False) -> None:
        self.lf = lf
        self.from_pre_agg = from_pre_agg

    # This allows polars LazyFrame methods that don't need custom functionality to work as expect
    # without having to explicitly write them as methods for LazyFrameWrapper
    def __getattr__(self, name: str) -> Any:
        attr = getattr(self.lf, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = attr(*args, **kwargs)
            if isinstance(result, pl.LazyFrame):
                return self.__class__(result)
            return result

        return wrapper

    def filter(
        self,
        *predicates: IntoExprColumn
        | Iterable[IntoExprColumn]
        | bool
        | list[bool]
        | np.ndarray[Any, Any],
        **constraints: Any,
    ) -> LazyFrameWrapper:

        if len(predicates) == 0 and len(constraints) == 0:
            return self
        else:
            return self.__class__(
                self.lf.filter(*predicates, **constraints), self.from_pre_agg
            )

    def group_by(
        self,
        *by: IntoExpr | Iterable[IntoExpr],
        maintain_order: bool = False,
        **named_by: IntoExpr,
    ) -> LazyFrameWrapper | LazyGroupByWrapper:

        if len(by) == 0 and len(named_by) == 0:
            return self.__class__(self.lf, self.from_pre_agg)
        else:
            from datasubway.lazygroupby_wrapper import LazyGroupByWrapper

            return LazyGroupByWrapper(
                self.lf.group_by(*by, maintain_order=maintain_order, **named_by),
                self.from_pre_agg,
            )

    def group_by_dynamic(
        self,
        index_column: IntoExpr,
        *,
        every: str | timedelta,
        period: str | timedelta | None = None,
        offset: str | timedelta | None = None,
        include_boundaries: bool = False,
        closed: ClosedInterval = "left",
        label: Label = "left",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
        start_by: StartBy = "window",
    ) -> LazyFrameWrapper | LazyGroupByWrapper:

        if index_column is None:
            return self.__class__(self.lf)
        else:
            from datasubway.lazygroupby_wrapper import LazyGroupByWrapper

            return LazyGroupByWrapper(
                self.lf.group_by_dynamic(
                    index_column,
                    every=every,
                    period=period,
                    offset=offset,
                    include_boundaries=include_boundaries,
                    closed=closed,
                    label=label,
                    group_by=group_by,
                    start_by=start_by,
                ),
                self.from_pre_agg,
            )

    def rolling(
        self,
        index_column: IntoExpr,
        *,
        period: str | timedelta,
        offset: str | timedelta | None = None,
        closed: ClosedInterval = "right",
        group_by: IntoExpr | Iterable[IntoExpr] | None = None,
    ) -> LazyFrameWrapper | LazyGroupByWrapper:

        if index_column is None:
            return self.__class__(self.lf, self.from_pre_agg)
        else:
            from datasubway.lazygroupby_wrapper import LazyGroupByWrapper

            return LazyGroupByWrapper(
                self.lf.rolling(
                    index_column,
                    period=period,
                    offset=offset,
                    closed=closed,
                    group_by=group_by,
                ),
                self.from_pre_agg,
            )

    def sort(
        self,
        by: IntoExpr | Iterable[IntoExpr],
        *more_by: IntoExpr,
        descending: bool | Sequence[bool] = False,
        nulls_last: bool | Sequence[bool] = False,
        maintain_order: bool = False,
        multithreaded: bool = True,
    ) -> LazyFrameWrapper:

        if by is None:
            return self
        else:
            return self.__class__(
                self.lf.sort(
                    by,
                    *more_by,
                    descending=descending,
                    nulls_last=nulls_last,
                    maintain_order=maintain_order,
                    multithreaded=multithreaded,
                ),
                self.from_pre_agg,
            )

    """
    The code below creates methods that normally operate on LazyGroupBy objects.
    However, if a group_by() is empty for LazyFrameWrapper, then there has to be fallback
    methods in this class to avoid errors.

    Note: Some of the LazyGroupBy methods already have LazyFrame equivalents, so
    __getattr__() will catch them
    """

    def agg(
        self,
        *aggs: IntoExpr | Iterable[IntoExpr],
        **named_aggs: IntoExpr,
    ) -> LazyFrameWrapper:
        # TODO: the table() method of DataModel could emit a LazyFrameWrapper object with a parameter
        # is_pre_agg. This would avoid a potenial expensive collect_schema() call

        if not self.from_pre_agg:
            return LazyFrameWrapper(
                self.lf.select(*aggs, **named_aggs), self.from_pre_agg
            )

        # if self.from_pre_agg == true, then all aggregations need to be rewritten to match
        # either new column names and potentially corrected calculations
        else:
            rewritten = [
                rewrite_agg_expr(a) if isinstance(a, pl.Expr) else a for a in aggs
            ]
            named_rewritten = {
                k: rewrite_agg_expr(v) if isinstance(v, pl.Expr) else v
                for k, v in named_aggs.items()
            }
            return LazyFrameWrapper(
                self.lf.select(*rewritten, **named_rewritten), self.from_pre_agg
            )

    def all(self) -> LazyFrameWrapper:
        return LazyFrameWrapper(self.lf.select(pl.all().list.all()), self.from_pre_agg)

    def having(self, *predicates: IntoExpr | Iterable[IntoExpr]) -> LazyFrameWrapper:
        """This method only makes sense on a LazyGroupBy or LazyGroupByWrapper object,
        so just do nothing if group_by() is empty and reverts back to LazyFrameWrapper"""
        return self

    def len(self, name: str | None = None) -> LazyFrameWrapper:
        if name is None:
            name = "len"
        return LazyFrameWrapper(
            self.lf.select(pl.all().len().alias(name)), self.from_pre_agg
        )

    def map_groups(
        self,
        function: Callable[[pl.DataFrame], pl.DataFrame],
        schema: SchemaDict | None,
    ) -> LazyFrameWrapper:
        return LazyFrameWrapper(
            self.lf.map_batches(function, schema=schema), self.from_pre_agg
        )

    def n_unique(self) -> LazyFrameWrapper:
        return LazyFrameWrapper(self.lf.select(pl.all().n_unique()), self.from_pre_agg)
