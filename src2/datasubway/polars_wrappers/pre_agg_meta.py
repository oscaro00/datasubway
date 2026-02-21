from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    group_by: list[str]              # fully-qualified: ['orders.date', 'orders.region']
    aggregations: dict[str, str | list[str]]  # {'orders.revenue': 'mean'} or ['mean', 'max']
    file_path: Path
    row_count: int = 0               # smaller = more aggregated = preferred; set on write
    written_at: datetime | None = None  # set on write

    def __post_init__(self) -> None:
        # Expand user-specified aggregation names to stored component names.
        # e.g. 'mean' → ['count', 'sum'], 'std' → ['count', 'sum', 'sumsq']
        expanded: dict[str, list[str]] = {}
        for col, aggs in self.aggregations.items():
            if isinstance(aggs, str):
                aggs = [aggs]
            components: set[str] = set()
            for agg in aggs:
                components.update(AGG_EXPANSION.get(agg, {agg}))
            expanded[col] = sorted(components)
        self.aggregations = expanded

    def load(self) -> pl.LazyFrame:
        return pl.scan_parquet(self.file_path)

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
                aggregations=config.get("aggregations", {}),
                file_path=pre_agg_directory / f"{name}.parquet",
                row_count=meta.get("row_count", 0),
                written_at=datetime.fromisoformat(written_at_raw) if written_at_raw else None,
            )
        )
    return result
