from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper

import polars as pl

from datasubway.column_context import parse_table_column, parse_table_columns
from datasubway.joins_meta import Join, parse_joins
from datasubway.libcst.measure_output_context import GroupingContext
from datasubway.polars_wrappers.filter_expr import (
    build_filter_expr,
    extract_table_columns_from_filter_dict,
)
from datasubway.polars_wrappers.proxy import LazyFrameProxy
from datasubway.pre_agg_meta import (
    PreAggregation,
    parse_pre_aggregations,
)
from datasubway.query_context import QueryContext


class DataModel:
    def __init__(
        self,
        tables: dict[str, pl.LazyFrame],
        *,
        joins: list[dict[str, Any]] | None = None,
        pre_aggregations: dict[str, Any] | None = None,
        pre_agg_directory: Path | None = None,
        logging_directory: Path | None = None,
    ) -> None:
        """
        Expected join format:
        [
            {
                'left':'table1', 'right':'table2',
                'left_on':['col1', 'col3'], 'right_on':['col1', 'col2'],
                'how':'inner', 'direction':'right2left' # direction can also be 'both'
            },
        ]

        Expected pre_aggregations format:
        {
            'pre_agg_name': {
                'group_by': ['tbl1.col1', 'tbl2.col2'],
                'aggregations': {
                    'tbl1.col3': 'sum',
                    'tbl1.col4': ['mean', 'max'],
                    'tbl2.col5': 'std',
                },
            }
        }

        Aggregation values are standard aggregation names: 'sum', 'min', 'max', 'count',
        'len', 'mean', 'std', 'var', 'first', 'last', 'product', 'null_count', 'all',
        'any', 'n_unique', 'median'. Compound aggregations like 'mean' and 'std' are
        automatically expanded to the required stored components (e.g. 'mean' → sum + count).

        row_count and written_at are recorded automatically by write_pre_agg() and
        should not be specified in the config.

        Pre-aggregation names must be unique — they determine the parquet file name
        inside pre_agg_directory.
        """
        self.joins = joins if joins is not None else []
        self.pre_agg_directory = pre_agg_directory or Path("_pre_aggregations/")
        self.logging_directory = logging_directory

        # Rename all columns to {table}.{col} for unambiguous downstream references
        self.tables: dict[str, pl.LazyFrame] = {
            name: lf.rename({col: f"{name}.{col}" for col in lf.collect_schema().names()})
            for name, lf in tables.items()
        }
        # Schemas are derived from the renamed tables; validation checks f"{table}.{column}"
        self.table_schemas: dict[str, list[str]] = {
            name: lf.collect_schema().names() for name, lf in self.tables.items()
        }

        self.joins_lookup: dict[str, dict[str, list[Join]]] = parse_joins(self.joins)

        self.measures: dict[str, Any] = {}
        self.measure_grouping_contexts: dict[str, GroupingContext] = {}
        self.measure_output_cols: dict[str, list[str]] = {}

        self.pre_agg_objects: list[PreAggregation] = parse_pre_aggregations(
            pre_aggregations if pre_aggregations is not None else {},
            self.pre_agg_directory,
        )

    def table(self, table_name: str) -> LazyFrameProxy:
        """Return a LazyFrameProxy for the named table.

        The proxy records the method chain and resolves to the optimal source
        (pre-agg or raw table) when .resolve() is called.
        """
        if table_name not in self.tables:
            raise KeyError(
                f"Table '{table_name}' not found. Available: {list(self.tables.keys())}"
            )
        return LazyFrameProxy(table_name, self)

    def find_best_pre_agg(
        self,
        table_name: str,
        group_by: list[str],
        agg_reqs: dict[str, set[str]],
    ) -> PreAggregation | None:
        """Return the most-aggregated pre-agg that covers the requested group-by and aggs.

        Returns None if no pre-agg qualifies.
        """
        candidates = [p for p in self.pre_agg_objects if p.covers(group_by, agg_reqs)]
        if not candidates:
            return None
        return min(candidates, key=lambda p: p.row_count)

    def write_pre_aggs(self, names: list[str]) -> list[PreAggregation]:
        """Compute and write the named pre-aggregations to parquet.

        names must be a list of pre-aggregation names defined in this DataModel.
        Computing pre-aggs can be expensive, so there is no default to write all.
        """
        results = []
        for name in names:
            pre_agg = next((p for p in self.pre_agg_objects if p.name == name), None)
            if pre_agg is None:
                raise KeyError(
                    f"Pre-aggregation '{name}' not defined. "
                    f"Available: {[p.name for p in self.pre_agg_objects]}"
                )
            results.append(pre_agg.write(self.tables, self.joins_lookup))
        return results

    def validate_query_context(self, qc: QueryContext) -> bool:
        for measure in qc.measures:
            if measure not in self.measures.keys():
                raise KeyError(f"measure '{measure}' not an available measure")

        filter_table_columns = (
            list(set(extract_table_columns_from_filter_dict(qc.filters)))
            if qc.filters
            else []
        )
        parsed_filters = parse_table_columns(filter_table_columns)
        for table, column in parsed_filters:
            if table not in self.table_schemas.keys():
                raise KeyError(f"filter '{table}.{column}' does not have a valid table")
            if f"{table}.{column}" not in self.table_schemas[table]:
                raise ValueError(
                    f"filter '{table}.{column}' does not have a valid column"
                )

        parsed_groups = parse_table_columns(qc.groups)
        for table, column in parsed_groups:
            if table not in self.table_schemas.keys():
                raise KeyError(
                    f"grouping '{table}.{column}' does not have a valid table"
                )
            if f"{table}.{column}" not in self.table_schemas[table]:
                raise ValueError(
                    f"grouping '{table}.{column}' does not have a valid column"
                )

        # Valid post-aggregation columns: group-by cols (column name only) + measure output cols
        valid_having_cols = set(qc.groups)
        for measure in qc.measures:
            valid_having_cols.update(self.measure_output_cols[measure])
        having_col_refs = (
            list(set(extract_table_columns_from_filter_dict(qc.havings)))
            if qc.havings
            else []
        )
        for col in having_col_refs:
            if col not in valid_having_cols:
                raise ValueError(
                    f"having column '{col}' is not a valid group-by or measure output column"
                )

        valid_sort_cols = set(qc.groups)
        for measure in qc.measures:
            valid_sort_cols.update(self.measure_output_cols[measure])

        for table_column, direction in qc.sorts:
            if table_column not in valid_sort_cols:
                table, column = parse_table_column(table_column)
                if table not in self.table_schemas.keys():
                    raise KeyError(
                        f"sorting '{table}.{column}' does not have a valid table"
                    )
                if f"{table}.{column}" not in self.table_schemas[table]:
                    raise ValueError(
                        f"sorting '{table}.{column}' does not have a valid column"
                    )

            if direction not in ["asc", "desc"]:
                raise ValueError(f"sorting direction '{direction}' is not allowed")

        if not isinstance(qc.limit, int) or qc.limit < 1:
            raise ValueError(f"limit '{qc.limit}' must be a positive integer")

        if not isinstance(qc.offset, int) or qc.offset < 0:
            raise ValueError(f"offset '{qc.offset}' must be a non-negative integer")

        return True

    async def query(self, query_context_dict: dict) -> pl.DataFrame:
        query_context = QueryContext(query_context_dict)

        self.validate_query_context(query_context)

        def _resolve_measure(name: str) -> "LazyFrameWrapper":
            proxy: LazyFrameProxy = self.measures[name](query_context)
            proxy.use_pre_agg = query_context.use_pre_agg
            return proxy.resolve()

        lazy_result = _resolve_measure(query_context.measures[0])

        for measure in query_context.measures[1:]:
            curr_resolved = _resolve_measure(measure)
            # If the query context has no groupings, assume all measures return one row,
            # so cross join the results
            if len(query_context.groups) == 0:
                lazy_result = lazy_result.join(curr_resolved.lf, how="cross")
            # Else the query context has groupings, so measures may return multiple rows,
            # so full join the results
            else:
                lazy_result = lazy_result.join(
                    curr_resolved.lf, on=query_context.groups, how="full"
                )

        if query_context.havings != {}:
            havings_filter_expr = build_filter_expr(query_context.havings, strip_prefixes=False)
            lazy_result = lazy_result.filter(havings_filter_expr)

        if len(query_context.sorts) > 0:
            sort_cols = [col for col, _ in query_context.sorts]
            descending = [dir.lower() == "desc" for _, dir in query_context.sorts]
            lazy_result = lazy_result.sort(sort_cols, descending=descending)

        lazy_result = lazy_result.slice(query_context.offset, query_context.limit)

        return await lazy_result.collect_async()
