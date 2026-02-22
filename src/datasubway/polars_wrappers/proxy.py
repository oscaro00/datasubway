from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import polars as pl

if TYPE_CHECKING:
    from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
    from datasubway.pre_agg_meta import PreAggregation


class _DataModelLike(Protocol):
    """Structural interface describing the parts of DataModel that LazyFrameProxy needs.

    Using a Protocol instead of importing DataModel directly avoids the circular
    import between proxy.py and data_model.py.
    """

    table_schemas: dict[str, list[str]]
    tables: dict[str, pl.LazyFrame]

    def find_best_pre_agg(
        self,
        table_name: str,
        group_by: list[str],
        agg_reqs: dict[str, set[str]],
    ) -> PreAggregation | None: ...


@dataclass
class RecordedOp:
    method: str
    args: tuple
    kwargs: dict


def strip_table_prefix(col: Any) -> Any:
    """Strip 'table.' prefix from a column name string or list of strings."""
    if isinstance(col, str) and "." in col:
        return col.split(".", 1)[1]
    if isinstance(col, list):
        return [strip_table_prefix(c) for c in col]
    return col


def qualify_col(col: str, table_name: str, schemas: dict[str, list[str]]) -> str:
    """If col has no '.', prepend table_name if col is in that table's schema."""
    if "." not in col:
        if col in schemas.get(table_name, []):
            return f"{table_name}.{col}"
    return col


def extract_col_names(args: tuple) -> list[str]:
    """Flatten args recursively to extract string column names."""
    cols = []
    for arg in args:
        if isinstance(arg, str):
            cols.append(arg)
        elif isinstance(arg, (list, tuple)):
            cols.extend(extract_col_names(tuple(arg)))
    return cols


class LazyGroupByProxy:
    """Returned by LazyFrameProxy.group_by/group_by_dynamic/rolling.

    Records group-by state and captures agg expressions for pre-agg analysis.
    """

    def __init__(self, parent: LazyFrameProxy) -> None:
        self.parent = parent

    def agg(self, *aggs: pl.Expr | Any, **named_aggs: pl.Expr | Any) -> LazyFrameProxy:
        # Store Polars expressions for agg requirement analysis
        self.parent.agg_exprs.extend(a for a in aggs if isinstance(a, pl.Expr))
        self.parent.ops.append(RecordedOp("agg", aggs, named_aggs))
        return self.parent

    def having(self, *predicates: pl.Expr | Any) -> LazyGroupByProxy:
        self.parent.ops.append(RecordedOp("having", predicates, {}))
        return self

    def map_groups(
        self,
        fn: Any,
        schema: Any = None,
    ) -> LazyFrameProxy:
        self.parent.ops.append(RecordedOp("map_groups", (fn,), {"schema": schema}))
        return self.parent


class LazyFrameProxy:
    """Returned by DataModel.table(). Records the entire method chain, then resolves
    to a LazyFrameWrapper with the optimal pre-agg source on .resolve().
    """

    def __init__(self, table_name: str, dm: _DataModelLike) -> None:
        self.table_name = table_name
        self.dm = dm
        self.ops: list[RecordedOp] = []
        self.group_by_cols: list[str] = []  # captured from group_by() call
        self.agg_exprs: list[pl.Expr] = []  # captured from agg() call
        self.has_join: bool = False

    def filter(self, *predicates: Any, **constraints: Any) -> LazyFrameProxy:
        self.ops.append(RecordedOp("filter", predicates, constraints))
        return self

    def sort(self, by: Any, *more_by: Any, **kwargs: Any) -> LazyFrameProxy:
        self.ops.append(RecordedOp("sort", (by, *more_by), kwargs))
        return self

    def group_by(
        self,
        *by: Any,
        maintain_order: bool = False,
        **named_by: Any,
    ) -> LazyGroupByProxy:
        self.group_by_cols = extract_col_names(by)
        self.ops.append(
            RecordedOp("group_by", by, {"maintain_order": maintain_order, **named_by})
        )
        return LazyGroupByProxy(self)

    def group_by_dynamic(
        self,
        index_column: Any,
        *,
        every: Any,
        period: Any = None,
        offset: Any = None,
        include_boundaries: bool = False,
        closed: str = "left",
        label: str = "left",
        group_by: Any = None,
        start_by: str = "window",
    ) -> LazyGroupByProxy:
        cols = extract_col_names((index_column,))
        if group_by is not None:
            if isinstance(group_by, (list, tuple)):
                cols.extend(extract_col_names(tuple(group_by)))
            elif isinstance(group_by, str):
                cols.append(group_by)
        self.group_by_cols = cols
        self.ops.append(
            RecordedOp(
                "group_by_dynamic",
                (index_column,),
                {
                    "every": every,
                    "period": period,
                    "offset": offset,
                    "include_boundaries": include_boundaries,
                    "closed": closed,
                    "label": label,
                    "group_by": group_by,
                    "start_by": start_by,
                },
            )
        )
        return LazyGroupByProxy(self)

    def rolling(
        self,
        index_column: Any,
        *,
        period: Any,
        offset: Any = None,
        closed: str = "right",
        group_by: Any = None,
    ) -> LazyGroupByProxy:
        cols = extract_col_names((index_column,))
        if group_by is not None:
            if isinstance(group_by, (list, tuple)):
                cols.extend(extract_col_names(tuple(group_by)))
            elif isinstance(group_by, str):
                cols.append(group_by)
        self.group_by_cols = cols
        self.ops.append(
            RecordedOp(
                "rolling",
                (index_column,),
                {
                    "period": period,
                    "offset": offset,
                    "closed": closed,
                    "group_by": group_by,
                },
            )
        )
        return LazyGroupByProxy(self)

    def join(self, other: Any, **kwargs: Any) -> LazyFrameProxy:
        self.has_join = True
        self.ops.append(RecordedOp("join", (other,), kwargs))
        return self

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion for instance attributes not yet set
        if name in (
            "ops",
            "group_by_cols",
            "agg_exprs",
            "has_join",
            "table_name",
            "dm",
        ):
            raise AttributeError(name)

        def record_and_return(*args: Any, **kwargs: Any) -> LazyFrameProxy:
            self.ops.append(RecordedOp(name, args, kwargs))
            return self

        return record_and_return

    def resolve(self) -> "LazyFrameWrapper":
        """Two-phase resolution: analyze the recorded chain, select the best source,
        then replay the chain with a real LazyFrameWrapper.
        """
        from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
        from datasubway.polars_wrappers.pre_agg_expr import extract_agg_requirements

        # 1. Group-by cols (already fully-qualified via allow() with include_tables=True)
        my_cols = self.group_by_cols

        # 2. Extract agg requirements from recorded expressions
        agg_reqs: dict[str, set[str]] = {}
        for expr in self.agg_exprs:
            for col, types in extract_agg_requirements(expr).items():
                qualified = qualify_col(col, self.table_name, self.dm.table_schemas)
                agg_reqs.setdefault(qualified, set()).update(types)

        # 3. Select source
        if not self.has_join:
            pre_agg = self.dm.find_best_pre_agg(self.table_name, my_cols, agg_reqs)
            if pre_agg:
                source = LazyFrameWrapper(pre_agg.load(), from_pre_agg=True)
            else:
                source = LazyFrameWrapper(
                    self.dm.tables[self.table_name], from_pre_agg=False
                )
        else:
            # TODO: extend to join-aware pre-agg selection
            source = LazyFrameWrapper(
                self.dm.tables[self.table_name], from_pre_agg=False
            )

        return self.replay(source)

    def replay(self, source: "LazyFrameWrapper") -> "LazyFrameWrapper":
        """Replay all recorded ops against a real LazyFrameWrapper source."""
        from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
        from datasubway.polars_wrappers.proxy import LazyFrameProxy as _Proxy

        result = source
        for op in self.ops:
            # Resolve any nested LazyFrameProxy args (e.g. from join())
            resolved_args = tuple(
                arg.resolve() if isinstance(arg, _Proxy) else arg for arg in op.args
            )
            resolved_kwargs = dict(op.kwargs)

            # Polars join() expects a raw LazyFrame, not a LazyFrameWrapper
            if op.method == "join":
                resolved_args = tuple(
                    arg.lf if isinstance(arg, LazyFrameWrapper) else arg
                    for arg in resolved_args
                )

            # Strip table prefixes from all string args and kwarg values — Polars only
            # knows unqualified column names, and the analysis phase is already done.
            resolved_args = tuple(strip_table_prefix(a) for a in resolved_args)
            resolved_kwargs = {
                k: strip_table_prefix(v) for k, v in resolved_kwargs.items()
            }

            result = getattr(result, op.method)(*resolved_args, **resolved_kwargs)

        return result
