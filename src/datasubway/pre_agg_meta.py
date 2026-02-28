from __future__ import annotations

import json
from dataclasses import InitVar, dataclass, field
from datetime import datetime
from pathlib import Path

import polars as pl

# What pre-agg stored component types are needed to use each Polars agg function.
# Keys match Polars JSON serialization (capitalized). Values match metadata format (lowercase).
AGG_NEEDED_COMPONENTS: dict[str, set[str]] = {
    "Sum": {"sum"},
    "Min": {"min"},
    "Max": {"max"},
    "Count": {"count"},
    "Len": {"len"},
    "First": {"first"},
    "Last": {"last"},
    "Product": {"product"},
    "NullCount": {"null_count"},
    "All": {"all"},
    "Any": {"any"},
    "NUnique": {"unique_set"},
    "Median": {"values_list"},
    "Mean": {"sum", "count"},
    "Std": {"sum", "sumsq", "count"},
    "Var": {"sum", "sumsq", "count"},
}

METADATA_FILENAME = "_metadata.json"

# Maps user-specified aggregation names to the component column suffixes that must be stored.
# Simple aggregations are their own component; compound ones expand to multiple components.
AGG_EXPANSION: dict[str, set[str]] = {
    "sum": {"sum"},
    "min": {"min"},
    "max": {"max"},
    "count": {"count"},
    "len": {"len"},
    "first": {"first"},
    "last": {"last"},
    "product": {"product"},
    "null_count": {"null_count"},
    "all": {"all"},
    "any": {"any"},
    "n_unique": {"unique_set"},
    "median": {"values_list"},
    "mean": {"sum", "count"},
    "std": {"sum", "sumsq", "count"},
    "var": {"sum", "sumsq", "count"},
}


@dataclass
class PreAggregation:
    name: str
    group_by: list[str]  # fully-qualified: ['orders.date', 'orders.region']
    raw_aggregations: InitVar[
        dict[str, str | list[str]]
    ]  # {'orders.revenue': 'mean'} or ['mean', 'max']
    file_path: Path = field(default_factory=Path)
    row_count: int = 0  # smaller = more aggregated = preferred; set on write
    written_at: datetime | None = None  # set on write
    aggregations: dict[str, list[str]] = field(init=False)

    def __post_init__(self, raw_aggregations: dict[str, str | list[str]]) -> None:
        if not self.group_by:
            raise ValueError("PreAggregation 'group_by' must contain at least one column.")
        if not raw_aggregations:
            raise ValueError("PreAggregation 'raw_aggregations' must contain at least one aggregation.")
        for col, aggs in raw_aggregations.items():
            for agg in ([aggs] if isinstance(aggs, str) else aggs):
                if agg not in AGG_EXPANSION:
                    raise ValueError(f"Unknown aggregation '{agg}' for column '{col}'.")
        # Expand user-specified aggregation names to stored component names.
        # e.g. 'mean' → ['count', 'sum'], 'std' → ['count', 'sum', 'sumsq']
        expanded: dict[str, list[str]] = {}
        for col, aggs in raw_aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            components: set[str] = set()
            for agg in aggs:
                components.update(AGG_EXPANSION.get(agg, {agg}))
            expanded[col] = sorted(components)
        self.aggregations = expanded

    def load(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.file_path)

    def compute(
        self,
        tables: dict[str, pl.LazyFrame],
        joins_lookup: dict[str, dict[str, list]],
    ) -> pl.LazyFrame:
        """Build the pre-aggregated LazyFrame from already-qualified tables."""
        from typing import cast

        from polars._typing import JoinStrategy

        # Fact tables = tables mentioned in aggregation column names
        agg_tables: set[str] = {
            col.split(".", 1)[0] for col in self.aggregations if "." in col
        }
        group_by_cols = self.group_by

        def _qualify_keys(keys: str | list[str], table: str) -> list[str]:
            if isinstance(keys, list):
                return [f"{table}.{c}" for c in keys]
            return [f"{table}.{keys}"]

        def _build_source(fact_table: str) -> pl.LazyFrame:
            base = tables[fact_table]
            dim_tables = {
                col.split(".", 1)[0]
                for col in self.group_by
                if "." in col and col.split(".", 1)[0] != fact_table
            }
            # Collect all join steps, deduplicate to avoid re-applying shared hops
            all_join_steps = []
            for dim_table in dim_tables:
                join_path = joins_lookup.get(fact_table, {}).get(dim_table)
                if join_path:
                    all_join_steps.extend(join_path)
                else:
                    # Fallback: join on common (qualified) column names
                    base_cols = set(base.collect_schema().names())
                    dim_cols = set(tables[dim_table].collect_schema().names())
                    common = list(base_cols & dim_cols)
                    if common:
                        base = base.join(
                            tables[dim_table], on=common, how="left", coalesce=False
                        )
            seen: set[tuple] = set()
            deduped = []
            for join in all_join_steps:
                left_key = tuple(join.left_on) if isinstance(join.left_on, list) else (join.left_on,)
                right_key = tuple(join.right_on) if isinstance(join.right_on, list) else (join.right_on,)
                key = (join.left, join.right, left_key, right_key, join.how)
                if key not in seen:
                    seen.add(key)
                    deduped.append(join)
            for join in deduped:
                base = base.join(
                    tables[join.right],
                    left_on=_qualify_keys(join.left_on, join.left),
                    right_on=_qualify_keys(join.right_on, join.right),
                    how=cast("JoinStrategy", join.how),
                    coalesce=False,
                )
            return base

        def _agg_exprs(fact_table: str) -> list[pl.Expr]:
            exprs = []
            for qualified_col, components in self.aggregations.items():
                if "." in qualified_col and qualified_col.split(".", 1)[0] != fact_table:
                    continue
                alias_prefix = qualified_col
                for comp in components:
                    alias = f"{alias_prefix}-{comp}"
                    if comp == "sum":
                        exprs.append(pl.col(qualified_col).sum().alias(alias))
                    elif comp == "count":
                        exprs.append(pl.col(qualified_col).count().alias(alias))
                    elif comp == "sumsq":
                        exprs.append(
                            (pl.col(qualified_col).cast(pl.Float64) ** 2).sum().alias(alias)
                        )
                    elif comp == "min":
                        exprs.append(pl.col(qualified_col).min().alias(alias))
                    elif comp == "max":
                        exprs.append(pl.col(qualified_col).max().alias(alias))
                    elif comp == "first":
                        exprs.append(pl.col(qualified_col).first().alias(alias))
                    elif comp == "last":
                        exprs.append(pl.col(qualified_col).last().alias(alias))
                    elif comp == "product":
                        exprs.append(pl.col(qualified_col).product().alias(alias))
                    elif comp == "null_count":
                        exprs.append(pl.col(qualified_col).null_count().alias(alias))
                    elif comp == "all":
                        exprs.append(pl.col(qualified_col).all().alias(alias))
                    elif comp == "any":
                        exprs.append(pl.col(qualified_col).any().alias(alias))
                    elif comp == "unique_set":
                        exprs.append(pl.col(qualified_col).unique().alias(alias))
                    elif comp == "values_list":
                        exprs.append(pl.col(qualified_col).implode().alias(alias))
                    elif comp == "len":
                        exprs.append(pl.col(qualified_col).len().alias(alias))
            return exprs

        if len(agg_tables) == 1:
            fact_table = next(iter(agg_tables))
            source = _build_source(fact_table)
            return source.group_by(group_by_cols).agg(*_agg_exprs(fact_table))
        else:
            # Cross-table pre-agg: compute each fact table separately, join on group-by
            parts = [
                _build_source(ft).group_by(group_by_cols).agg(*_agg_exprs(ft))
                for ft in sorted(agg_tables)
            ]
            result = parts[0]
            for part in parts[1:]:
                result = result.join(part, on=group_by_cols, how="full", coalesce=True)
            return result

    def write(
        self,
        tables: dict[str, pl.LazyFrame],
        joins_lookup: dict[str, dict[str, list]],
    ) -> "PreAggregation":
        """Compute and write this pre-aggregation to parquet, updating metadata."""
        pre_agg_directory = self.file_path.parent
        pre_agg_directory.mkdir(parents=True, exist_ok=True)
        df = self.compute(tables, joins_lookup).collect()
        row_count = len(df)
        written_at = datetime.now()
        df.write_parquet(self.file_path)
        metadata = load_metadata(pre_agg_directory)
        metadata[self.name] = {"row_count": row_count, "written_at": written_at.isoformat()}
        save_metadata(pre_agg_directory, metadata)
        self.row_count = row_count
        self.written_at = written_at
        return self

    def covers(
        self,
        requested_group_by: list[str],
        requested_aggs: dict[str, set[str]],
    ) -> bool:
        """True if this pre-agg can satisfy the grouping+agg request.

        A pre-agg covers a request when:
        1. requested_group_by ⊆ pre_agg.group_by  (pre-agg is at least as granular)
        2. For each (col, agg_type) pair, the pre-agg stores all needed component types.
        """
        if not set(requested_group_by) <= set(self.group_by):
            return False
        for col, agg_types in requested_aggs.items():
            if col not in self.aggregations:
                return False
            available = set(self.aggregations[col])
            for agg_type in agg_types:
                needed = AGG_NEEDED_COMPONENTS.get(agg_type, set())
                if not needed <= available:
                    return False
        return True


def load_metadata(pre_agg_directory: Path) -> dict:
    """Read the metadata JSON file from pre_agg_directory, returning {} if absent."""
    path = pre_agg_directory / METADATA_FILENAME
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_metadata(pre_agg_directory: Path, metadata: dict) -> None:
    """Write the metadata dict to the JSON file in pre_agg_directory."""
    pre_agg_directory.mkdir(parents=True, exist_ok=True)
    path = pre_agg_directory / METADATA_FILENAME
    path.write_text(json.dumps(metadata, indent=2, default=str))


def parse_pre_aggregations(
    pre_agg_dict: dict,
    pre_agg_directory: Path,
) -> list[PreAggregation]:
    """Convert the raw pre_aggregations config to PreAggregation objects.

    row_count and written_at are loaded from the metadata file written by
    DataModel.write_pre_agg() — users should not specify them in the config.
    """
    metadata = load_metadata(pre_agg_directory)
    result = []
    for name, config in pre_agg_dict.items():
        meta = metadata.get(name, {})
        written_at_raw = meta.get("written_at")
        result.append(
            PreAggregation(
                name=name,
                group_by=config.get("group_by", []),
                raw_aggregations=config.get("aggregations", {}),
                file_path=pre_agg_directory / f"{name}.parquet",
                row_count=meta.get("row_count", 0),
                written_at=datetime.fromisoformat(written_at_raw)
                if written_at_raw
                else None,
            )
        )
    return result
