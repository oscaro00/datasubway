"""MeasureDataFrame: wrapper around DataFusion Python DataFrame that tracks operations for validation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import datafusion
from datafusion import col, lit
from datafusion import functions as F

from datasubway.column_context import _AllowExcludeResult

if TYPE_CHECKING:
    from datasubway.data_model import DataModel


class MeasureDataFrame:
    """Wraps a DataFusion Python DataFrame and tracks the last operation for measure validation.

    The @measure decorator probes each measure function at registration time
    to validate that it ends with .aggregate() and to extract grouping context.
    """

    def __init__(
        self,
        inner: datafusion.DataFrame,
        table_name: str,
        data_model: DataModel | None = None,
    ) -> None:
        self._inner = inner
        self._table_name = table_name
        self._data_model = data_model
        self._joined_tables: set[str] = {table_name}
        self._last_op = "table"
        self._grouping_context: dict[str, Any] | None = None

    def _ensure_joined(self, needed_tables: set[str]) -> None:
        """Auto-join any referenced but not-yet-joined tables using the JoinGraph."""
        if self._data_model is None or self._data_model.join_graph is None:
            return
        missing = needed_tables - self._joined_tables
        if not missing:
            return
        for target in missing:
            if target in self._joined_tables:
                continue
            path = None
            for source in list(self._joined_tables):
                path = self._data_model.join_graph.find_path(source, target)
                if path is not None:
                    break
            if path is None:
                raise ValueError(
                    f"No join path to '{target}' from {self._joined_tables}"
                )
            for step in path:
                step_target = step["right"]
                if step_target in self._joined_tables:
                    continue
                right_df = self._data_model.py_ctx.table(step_target)
                self._inner = self._inner.join(
                    right_df,
                    left_on=step["left_on"].split(","),
                    right_on=step["right_on"].split(","),
                    how=step["how"],
                )
                self._joined_tables.add(step_target)

    def filter(self, *args: Any) -> MeasureDataFrame:
        if len(args) == 1:
            arg = args[0]
            # No-op for empty input (empty list, empty dict, None)
            if arg is None or (isinstance(arg, (list, dict)) and not arg):
                return self
            # Handle dict-based filter trees directly
            if isinstance(arg, dict):
                self._ensure_joined(_extract_tables_from_filter_dict(arg))
                expr = _build_filter_expr(arg)
                self._inner = self._inner.filter(expr)
                self._last_op = "filter"
                return self
        # DataFusion Expr passthrough
        needed = set()
        for arg in args:
            needed |= _extract_tables_from_expr(arg)
        self._ensure_joined(needed)
        self._inner = self._inner.filter(*args)
        self._last_op = "filter"
        return self

    def filter_dict(self, filter_tree: dict) -> MeasureDataFrame:
        return self.filter(filter_tree)

    def select(self, *args: Any) -> MeasureDataFrame:
        needed = set()
        for arg in args:
            if isinstance(arg, str) and "." in arg:
                needed.add(arg.split(".")[0])
            else:
                needed |= _extract_tables_from_expr(arg)
        self._ensure_joined(needed)
        self._inner = self._inner.select(*args)
        self._last_op = "select"
        return self

    def aggregate(
        self,
        group_by: list | _AllowExcludeResult,
        aggs: list,
    ) -> MeasureDataFrame:
        if isinstance(group_by, _AllowExcludeResult):
            self._grouping_context = {
                "type": group_by.ae_type,
                "pattern": group_by.pattern,
                "include": group_by.include,
            }
        # Extract needed tables from group_by (strings or Exprs) and agg exprs
        needed = _extract_tables_from_strings(group_by)
        for g in group_by:
            if not isinstance(g, str):
                needed |= _extract_tables_from_expr(g)
        for agg_expr in aggs:
            needed |= _extract_tables_from_expr(agg_expr)
        self._ensure_joined(needed)
        # Convert string group-by items to col() expressions
        group_exprs = [col(g) if isinstance(g, str) else g for g in group_by]
        self._inner = self._inner.aggregate(group_exprs, aggs)
        self._last_op = "aggregate"
        return self

    def join(
        self,
        right: MeasureDataFrame,
        on: str | Sequence[str] | None = None,
        how: Literal["inner", "left", "right", "full", "semi", "anti"] = "inner",
        *,
        left_on: str | Sequence[str] | None = None,
        right_on: str | Sequence[str] | None = None,
        coalesce_duplicate_keys: bool = True,
    ) -> MeasureDataFrame:
        right_df = right._inner if isinstance(right, MeasureDataFrame) else right
        if isinstance(right, MeasureDataFrame):
            self._joined_tables |= right._joined_tables
        kwargs: dict[str, Any] = {
            "how": how,
            "coalesce_duplicate_keys": coalesce_duplicate_keys,
        }
        if on is not None:
            kwargs["on"] = on
        if left_on is not None:
            kwargs["left_on"] = left_on
        if right_on is not None:
            kwargs["right_on"] = right_on
        self._inner = self._inner.join(right_df, **kwargs)
        self._last_op = "join"
        return self

    def sort(self, *args: Any) -> MeasureDataFrame:
        self._inner = self._inner.sort(*args)
        self._last_op = "sort"
        return self

    def limit(self, count: int, offset: int = 0) -> MeasureDataFrame:
        self._inner = self._inner.limit(count, offset)
        self._last_op = "limit"
        return self

    def with_column(self, name: str, expr: Any) -> MeasureDataFrame:
        needed = _extract_tables_from_expr(expr)
        self._ensure_joined(needed)
        self._inner = self._inner.with_column(name, expr)
        self._last_op = "with_column"
        return self

    def collect(self) -> Any:
        return self._inner.collect()

    def to_arrow_table(self) -> Any:
        return self._inner.to_arrow_table()

    def columns(self) -> list[str]:
        return [f.name for f in self._inner.schema()]

    def schema(self) -> Any:
        return self._inner.schema()

    def logical_plan(self) -> Any:
        return self._inner.logical_plan()


# ── Table extraction helpers ──


def _extract_tables_from_strings(items: list) -> set[str]:
    """Extract table names from 'table.column' strings."""
    tables: set[str] = set()
    for item in items:
        if isinstance(item, str) and "." in item:
            tables.add(item.split(".")[0])
    return tables


def _extract_tables_from_filter_dict(filter_tree: dict) -> set[str]:
    """Extract table names from a filter dict tree."""
    tables: set[str] = set()
    for key, conditions in filter_tree.items():
        if key in ("AND", "OR") and isinstance(conditions, list):
            for cond in conditions:
                if isinstance(cond, (list, tuple)) and len(cond) >= 1:
                    col_name = cond[0]
                    if isinstance(col_name, str) and "." in col_name:
                        tables.add(col_name.split(".")[0])
                elif isinstance(cond, dict):
                    tables |= _extract_tables_from_filter_dict(cond)
    return tables


def _extract_tables_from_expr(expr: Any) -> set[str]:
    """Extract table names from a DataFusion Expr by walking its tree."""
    tables: set[str] = set()
    try:
        _walk_expr(expr, tables)
    except Exception:
        pass
    return tables


def _walk_expr(expr: Any, tables: set[str]) -> None:
    """Recursively walk a DataFusion Expr to find Column references.

    Column nodes' rex_call_operands() returns the column itself (infinite recursion),
    so we must NOT recurse into Column nodes.
    """
    try:
        variant_name = expr.variant_name()
    except (AttributeError, TypeError):
        return

    if variant_name == "Column":
        variant = expr.to_variant()
        relation = variant.relation()
        if relation:
            tables.add(relation)
    else:
        try:
            for operand in expr.rex_call_operands():
                _walk_expr(operand, tables)
        except Exception:
            pass


def _build_filter_expr(filter_tree: dict) -> Any:
    """Convert {"AND": [["col", "op", val], ...]} to DataFusion Expr."""
    for key, conditions in filter_tree.items():
        exprs = []
        for cond in conditions:
            if isinstance(cond, dict):
                exprs.append(_build_filter_expr(cond))
            else:
                col_name, op, val = cond[0], cond[1], cond[2]
                c = col(col_name)
                if op == "in":
                    exprs.append(F.in_list(c, [lit(v) for v in val], negated=False))
                elif op == "not in":
                    exprs.append(F.in_list(c, [lit(v) for v in val], negated=True))
                else:
                    v = lit(val)
                    if op == "=":
                        exprs.append(c == v)
                    elif op == "!=":
                        exprs.append(c != v)
                    elif op == ">":
                        exprs.append(c > v)
                    elif op == ">=":
                        exprs.append(c >= v)
                    elif op == "<":
                        exprs.append(c < v)
                    elif op == "<=":
                        exprs.append(c <= v)
                    else:
                        raise ValueError(f"Unknown filter operator: '{op}'")

        combined = exprs[0]
        for e in exprs[1:]:
            combined = combined.__and__(e) if key == "AND" else combined.__or__(e)
        return combined
