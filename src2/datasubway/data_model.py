from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl

from datasubway.joins_meta import Join, parse_joins
from datasubway.polars_wrappers.proxy import LazyFrameProxy
from datasubway.pre_agg_meta import (
    PreAggregation,
    load_metadata,
    parse_pre_aggregations,
    save_metadata,
)


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
