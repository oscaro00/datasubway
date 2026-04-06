"""MeasureDataFrame: thin wrapper around DataFusion Python DataFrame.

Auto-join logic has been moved to Rust (AutoJoinRule optimizer).
Tables are eagerly pre-joined in DataModel.table(), so all columns
are available for filter/aggregate/select operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Sequence

import datafusion
from datafusion import col, lit
from datafusion import functions as F

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
        self._last_op = "table"

    def filter(self, *args: Any) -> MeasureDataFrame:
        if len(args) == 1:
            arg = args[0]
            # No-op for empty input (empty list, empty dict, None)
            if arg is None or (isinstance(arg, (list, dict)) and not arg):
                return self
            # Handle dict-based filter trees directly
            if isinstance(arg, dict):
                expr = _build_filter_expr(arg)
                self._inner = self._inner.filter(expr)
                self._last_op = "filter"
                return self
        # DataFusion Expr passthrough
        self._inner = self._inner.filter(*args)
        self._last_op = "filter"
        return self

    def filter_dict(self, filter_tree: dict) -> MeasureDataFrame:
        return self.filter(filter_tree)

    def select(self, *args: Any) -> MeasureDataFrame:
        self._inner = self._inner.select(*args)
        self._last_op = "select"
        return self

    def aggregate(
        self,
        group_by: list,
        aggs: list,
    ) -> MeasureDataFrame:
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


# ── Filter expression builder ──


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
