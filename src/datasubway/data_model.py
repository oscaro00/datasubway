from __future__ import annotations

from datetime import datetime
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
    load_metadata,
    parse_pre_aggregations,
    save_metadata,
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
        self.tables = tables
        self.joins = joins if joins is not None else []
        self.pre_agg_directory = pre_agg_directory or Path("_pre_aggregations/")
        self.logging_directory = logging_directory

        self.table_schemas: dict[str, list[str]] = {
            tbl_name: lf.collect_schema().names()
            for tbl_name, lf in self.tables.items()
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

    def write_pre_agg(self, name: str, lf: pl.LazyFrame) -> PreAggregation:
        """Write a pre-aggregated LazyFrame to parquet and record metadata.

        Collects the frame, writes it to pre_agg_directory/{name}.parquet, then
        updates the metadata file with the row count and write timestamp. The
        in-memory PreAggregation object is updated to match.
        """
        pre_agg = next((p for p in self.pre_agg_objects if p.name == name), None)
        if pre_agg is None:
            raise KeyError(
                f"Pre-aggregation '{name}' not defined. "
                f"Available: {[p.name for p in self.pre_agg_objects]}"
            )

        df = lf.collect()
        row_count = len(df)
        written_at = datetime.now()

        self.pre_agg_directory.mkdir(parents=True, exist_ok=True)
        df.write_parquet(pre_agg.file_path)

        metadata = load_metadata(self.pre_agg_directory)
        metadata[name] = {
            "row_count": row_count,
            "written_at": written_at.isoformat(),
        }
        save_metadata(self.pre_agg_directory, metadata)

        pre_agg.row_count = row_count
        pre_agg.written_at = written_at

        return pre_agg

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
            if column not in self.table_schemas[table]:
                raise ValueError(
                    f"filter '{table}.{column}' does not have a valid column"
                )

        parsed_groups = parse_table_columns(qc.groups)
        for table, column in parsed_groups:
            if table not in self.table_schemas.keys():
                raise KeyError(
                    f"grouping '{table}.{column}' does not have a valid table"
                )
            if column not in self.table_schemas[table]:
                raise ValueError(
                    f"grouping '{table}.{column}' does not have a valid column"
                )

        candidate_havings_cols = list(qc.groups)
        for measure in qc.measures:
            candidate_havings_cols.extend(self.measure_output_cols[measure])
        having_table_columns = (
            list(set(extract_table_columns_from_filter_dict(qc.havings)))
            if qc.havings
            else []
        )
        parsed_havings = parse_table_columns(having_table_columns)
        for table, column in parsed_havings:
            if table not in self.table_schemas.keys():
                raise KeyError(f"having '{table}.{column}' does not have a valid table")
            if column not in self.table_schemas[table]:
                raise ValueError(
                    f"having '{table}.{column}' does not have a valid column"
                )

        for table_column, direction in qc.sorts:
            table, column = parse_table_column(table_column)
            if table not in self.table_schemas.keys():
                raise KeyError(
                    f"sorting '{table}.{column}' does not have a valid table"
                )
            if column not in self.table_schemas[table]:
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
            proxy: LazyFrameProxy = self.measures[name](self)
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
                on_cols = [
                    col.split(".", 1)[1] if "." in col else col
                    for col in query_context.groups
                ]
                lazy_result = lazy_result.join(curr_resolved.lf, on=on_cols, how="full")

        if query_context.havings != {}:
            havings_filter_expr = build_filter_expr(query_context.havings)
            lazy_result = lazy_result.filter(havings_filter_expr)

        if len(query_context.sorts) > 0:
            sort_cols = parse_table_columns([col for col, _ in query_context.sorts])
            sort_directions = parse_table_columns(
                [dir for _, dir in query_context.sorts]
            )
            lazy_result = lazy_result.sort(sort_cols, sort_directions)

        lazy_result = lazy_result.slice(query_context.offset, query_context.limit)

        return await lazy_result.collect_async()
