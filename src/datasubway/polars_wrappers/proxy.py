from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

import polars as pl

if TYPE_CHECKING:
    from polars._typing import JoinStrategy

    from datasubway.joins_meta import Join
    from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
    from datasubway.pre_agg_meta import PreAggregation


class _DataModelLike(Protocol):
    """Structural interface describing the parts of DataModel that LazyFrameProxy needs.

    Using a Protocol instead of importing DataModel directly avoids the circular
    import between proxy.py and data_model.py.
    """

    table_schemas: dict[str, list[str]]
    tables: dict[str, pl.LazyFrame]
    joins_lookup: dict[str, dict[str, list["Join"]]]

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

    def __init__(
        self, table_name: str, dm: _DataModelLike, *, use_pre_agg: bool = True
    ) -> None:
        self.table_name = table_name
        self.dm = dm
        self.ops: list[RecordedOp] = []
        self.group_by_cols: list[str] = []  # captured from group_by() call
        self.agg_exprs: list[pl.Expr] = []  # captured from agg() call
        self._unjoined_tables: set[str] = set()
        self.use_pre_agg = use_pre_agg

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
        self.ops.append(RecordedOp("join", (other,), kwargs))
        return self

    def __getattr__(self, name: str) -> Any:
        # Guard against infinite recursion for instance attributes not yet set
        if name in (
            "ops",
            "group_by_cols",
            "agg_exprs",
            "has_join",
            "_unjoined_tables",
            "table_name",
            "dm",
            "use_pre_agg",
        ):
            raise AttributeError(name)

        def record_and_return(*args: Any, **kwargs: Any) -> LazyFrameProxy:
            self.ops.append(RecordedOp(name, args, kwargs))
            return self

        return record_and_return

    def _collect_foreign_tables(self) -> set[str]:
        """Scan group_by_cols and recorded op expressions for foreign-table prefixes."""
        foreign: set[str] = set()

        for col in self.group_by_cols:
            if "." in col:
                tbl = col.split(".", 1)[0]
                if tbl != self.table_name:
                    foreign.add(tbl)

        all_exprs: list[pl.Expr] = list(self.agg_exprs)
        for op in self.ops:
            for arg in op.args:
                if isinstance(arg, pl.Expr):
                    all_exprs.append(arg)
                elif isinstance(arg, (list, tuple)):
                    for item in arg:
                        if isinstance(item, pl.Expr):
                            all_exprs.append(item)
            for v in op.kwargs.values():
                if isinstance(v, pl.Expr):
                    all_exprs.append(v)

        for expr in all_exprs:
            for name in expr.meta.root_names():
                if "." in name:
                    tbl = name.split(".", 1)[0]
                    if tbl != self.table_name:
                        foreign.add(tbl)

        return foreign

    def _build_joined_source(self) -> tuple["LazyFrameWrapper", set[str]]:
        """Join reachable foreign tables into the raw base source.

        Returns the built source and the set of foreign tables that could NOT be
        joined (no path in joins_lookup). Callers use this to drop unreachable
        expressions in replay.
        """
        from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper

        base = LazyFrameWrapper(self.dm.tables[self.table_name], from_pre_agg=False)
        foreign_tables = self._collect_foreign_tables()
        unjoined: set[str] = set()

        # Collect all join steps across all paths
        all_join_steps: list[Join] = []
        for foreign_table in foreign_tables:
            join_path = self.dm.joins_lookup.get(self.table_name, {}).get(foreign_table)
            if join_path is None:
                unjoined.add(foreign_table)
                continue
            all_join_steps.extend(join_path)

        # Deduplicate: remove any join step whose (left, right, left_on, right_on, how)
        # tuple has already been seen, preserving order of first occurrence.
        seen: set[tuple] = set()
        deduped: list[Join] = []
        for join in all_join_steps:
            key = (
                join.left,
                join.right,
                tuple(join.left_on),
                tuple(join.right_on),
                join.how,
            )
            if key not in seen:
                seen.add(key)
                deduped.append(join)

        def _qualify_keys(keys: Any, table: str) -> list[str]:
            if isinstance(keys, list):
                return [f"{table}.{c}" for c in keys]
            return [f"{table}.{keys}"]

        # Apply the deduplicated join sequence
        for join in deduped:
            right_lf = self.dm.tables[join.right]
            base = LazyFrameWrapper(
                base.lf.join(
                    right_lf,
                    left_on=_qualify_keys(join.left_on, join.left),
                    right_on=_qualify_keys(join.right_on, join.right),
                    how=cast("JoinStrategy", join.how),
                    coalesce=False,
                ),
                from_pre_agg=False,
            )

        return base, unjoined

    def resolve(self) -> "LazyFrameWrapper":
        """Two-phase resolution: analyze the recorded chain, select the best source,
        then replay the chain with a real LazyFrameWrapper.
        """
        from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
        from datasubway.polars_wrappers.pre_agg_expr import extract_agg_requirements

        # 1. Group-by cols (already fully-qualified via allow() with include_tables=True)
        my_cols = self.group_by_cols

        # 2. Extract agg requirements from recorded expressions
        # Columns are already qualified since tables are renamed at DataModel init
        agg_reqs: dict[str, set[str]] = {}
        for expr in self.agg_exprs:
            for col, types in extract_agg_requirements(expr).items():
                agg_reqs.setdefault(col, set()).update(types)

        # 3. Select source — try pre-agg first unless use_pre_agg is False
        pre_agg = (
            self.dm.find_best_pre_agg(self.table_name, my_cols, agg_reqs)
            if self.use_pre_agg
            else None
        )
        if pre_agg:
            source = LazyFrameWrapper(pre_agg.load(), from_pre_agg=True)
            # Tables covered by the pre-agg (group-by dims + aggregated fact tables)
            pre_agg_tables = {
                col.split(".", 1)[0]
                for col in pre_agg.group_by + list(pre_agg.aggregations.keys())
                if "." in col
            }
            self._unjoined_tables = self._collect_foreign_tables() - pre_agg_tables
        else:
            source, self._unjoined_tables = self._build_joined_source()

        return self.replay(source)

    def replay(self, source: "LazyFrameWrapper") -> "LazyFrameWrapper":
        """Replay all recorded ops against a real LazyFrameWrapper source."""
        from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper
        from datasubway.polars_wrappers.pre_agg_expr import drop_unjoined_table_refs
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

            # For Polars expressions: drop refs to unjoinable tables; pass through unchanged.
            # Columns are already fully qualified since tables are renamed at DataModel init.
            cleaned_args = []
            for a in resolved_args:
                if isinstance(a, pl.Expr):
                    cleaned = drop_unjoined_table_refs(a, self._unjoined_tables)
                    if cleaned is not None:
                        cleaned_args.append(cleaned)
                    # else: expression references only unjoinable tables — drop silently
                elif isinstance(a, list):
                    cleaned_list = []
                    for item in a:
                        if isinstance(item, pl.Expr):
                            cleaned = drop_unjoined_table_refs(item, self._unjoined_tables)
                            if cleaned is not None:
                                cleaned_list.append(cleaned)
                        else:
                            cleaned_list.append(item)
                    cleaned_args.append(cleaned_list)
                else:
                    cleaned_args.append(a)
            resolved_args = tuple(cleaned_args)

            cleaned_kwargs: dict[str, Any] = {}
            for k, v in resolved_kwargs.items():
                if isinstance(v, pl.Expr):
                    cleaned = drop_unjoined_table_refs(v, self._unjoined_tables)
                    if cleaned is not None:
                        cleaned_kwargs[k] = cleaned
                    # else: drop silently
                else:
                    cleaned_kwargs[k] = v
            resolved_kwargs = cleaned_kwargs

            # If a filter op had all its args dropped, skip the op entirely
            if (
                op.method == "filter"
                and len(resolved_args) == 0
                and len(resolved_kwargs) == 0
            ):
                continue

            result = getattr(result, op.method)(*resolved_args, **resolved_kwargs)

        return result
