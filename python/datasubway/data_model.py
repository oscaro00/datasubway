"""DataModel: core semantic layer class using DataFusion engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datafusion as df
import pyarrow as pa
import pyarrow.compute as pc

from datasubway._engine import Engine, JoinGraph, PreAggregation
from datasubway.dataframe import MeasureDataFrame
from datasubway.query_context import QueryContext


class DataModel:
    """Semantic layer model backed by DataFusion.

    Manages data sources, joins, measures, pre-aggregations, and query execution.
    DataFusion natively tracks table.column references via qualified identifiers,
    so no manual column renaming is needed.
    """

    def __init__(
        self,
        tables: dict[str, str | pa.Table | pa.RecordBatch],
        *,
        joins: list[dict[str, Any]] | None = None,
        pre_aggregations: dict[str, Any] | None = None,
        pre_agg_directory: str | Path | None = None,
    ) -> None:
        self.engine = Engine()  # Rust side — optimizer + execution
        self.py_ctx = df.SessionContext()  # Python side — DataFrame building
        self.table_schemas: dict[str, list[str]] = {}

        # Register data sources in BOTH contexts
        for name, source in tables.items():
            if isinstance(source, str):
                path = str(source)
                if path.endswith(".parquet"):
                    self.engine.register_parquet(name, path)
                    self.py_ctx.register_parquet(name, path)
                elif path.endswith(".csv"):
                    self.engine.register_csv(name, path)
                    self.py_ctx.register_csv(name, path)
                else:
                    raise ValueError(
                        f"Unsupported file type for table '{name}': {path}"
                    )
            elif isinstance(source, pa.Table):
                for batch in source.to_batches():
                    self.engine.register_record_batch(name, batch)
                    break
                self.py_ctx.register_record_batches(name, [source.to_batches()])
            elif isinstance(source, pa.RecordBatch):
                self.engine.register_record_batch(name, source)
                self.py_ctx.register_record_batches(name, [[source]])
            else:
                raise TypeError(
                    f"Unsupported source type for table '{name}': {type(source)}"
                )

            self._store_schema(name, source)

        # Parse joins
        self.joins_list = joins or []
        self.join_graph: JoinGraph | None = None
        if self.joins_list:
            self.join_graph = JoinGraph(self.joins_list)

        # Pre-aggregation setup
        self.pre_agg_directory = Path(pre_agg_directory or "_pre_aggregations/")
        self.pre_agg_objects: list[PreAggregation] = []
        if pre_aggregations:
            self._parse_pre_aggregations(pre_aggregations)

        # Measure registry (populated by @measure decorator)
        self.measures: dict[str, Any] = {}
        self.measure_output_cols: dict[str, list[str]] = {}
        self.measure_docstrings: dict[str, str] = {}
        self.measure_grouping_contexts: dict[str, dict] = {}

    def _store_schema(self, name: str, source: str | pa.Table | pa.RecordBatch) -> None:
        """Store qualified column names for a table."""
        if isinstance(source, str):
            try:
                py_df = self.py_ctx.table(name)
                self.table_schemas[name] = [f"{name}.{f.name}" for f in py_df.schema()]
            except Exception:
                self.table_schemas[name] = []
        elif isinstance(source, pa.Table):
            self.table_schemas[name] = [f"{name}.{col}" for col in source.column_names]
        elif isinstance(source, pa.RecordBatch):
            self.table_schemas[name] = [f"{name}.{col}" for col in source.schema.names]

    def _parse_pre_aggregations(self, pre_agg_dict: dict[str, Any]) -> None:
        """Parse pre-aggregation config into PreAggregation objects."""
        metadata = self._load_pre_agg_metadata()

        for name, config in pre_agg_dict.items():
            raw_aggs = {}
            for col, agg_spec in config.get("aggregations", {}).items():
                if isinstance(agg_spec, str):
                    raw_aggs[col] = [agg_spec]
                else:
                    raw_aggs[col] = list(agg_spec)

            file_path = str(self.pre_agg_directory / f"{name}.parquet")
            pa_obj = PreAggregation(
                name=name,
                group_by=config["group_by"],
                raw_aggregations=raw_aggs,
                file_path=file_path,
            )

            if name in metadata:
                meta = metadata[name]
                if "row_count" in meta:
                    pa_obj.row_count = meta["row_count"]
                if "written_at" in meta:
                    pa_obj.written_at = meta["written_at"]

            self.pre_agg_objects.append(pa_obj)

    def _load_pre_agg_metadata(self) -> dict:
        """Load pre-agg metadata from JSON file."""
        meta_path = self.pre_agg_directory / "_metadata.json"
        if meta_path.exists():
            return json.loads(meta_path.read_text())
        return {}

    def _save_pre_agg_metadata(self, metadata: dict) -> None:
        """Save pre-agg metadata to JSON file."""
        self.pre_agg_directory.mkdir(parents=True, exist_ok=True)
        meta_path = self.pre_agg_directory / "_metadata.json"
        meta_path.write_text(json.dumps(metadata, default=str))

    def table(self, name: str) -> MeasureDataFrame:
        """Get a MeasureDataFrame for a registered table."""
        return MeasureDataFrame(self.py_ctx.table(name), name, data_model=self)

    def all_columns(self) -> list[str]:
        """Return all qualified column names across all tables."""
        cols = []
        for table_cols in self.table_schemas.values():
            cols.extend(table_cols)
        return cols

    def validate_query_context(self, qc: QueryContext) -> bool:
        """Validate all aspects of a QueryContext against this DataModel."""
        all_cols = set(self.all_columns())

        for m in qc.measures:
            if m not in self.measures:
                raise ValueError(f"Unknown measure: '{m}'")

        for g in qc.groups:
            if g not in all_cols:
                raise ValueError(f"Unknown group column: '{g}'")

        filter_cols = _extract_filter_columns(qc.filters)
        for fc in filter_cols:
            if fc not in all_cols:
                raise ValueError(f"Unknown filter column: '{fc}'")

        valid_having_cols = set(qc.groups)
        for m in qc.measures:
            valid_having_cols.update(self.measure_output_cols.get(m, []))
        having_cols = _extract_filter_columns(qc.havings)
        for hc in having_cols:
            if hc not in valid_having_cols:
                raise ValueError(f"Invalid having column: '{hc}'")

        valid_sort_cols = valid_having_cols
        for col, direction in qc.sorts:
            if col not in valid_sort_cols:
                raise ValueError(f"Invalid sort column: '{col}'")
            if direction not in ("asc", "desc"):
                raise ValueError(f"Invalid sort direction: '{direction}'")

        return True

    async def query(
        self, query_context_dict: dict[str, Any], explain: bool = False
    ) -> pa.Table:
        """Execute a query and return results as a PyArrow Table.

        1. Validates the QueryContext
        2. Calls each measure function to produce results
        3. Serializes plans to Substrait, sends to Rust for optimization + execution
        4. Joins multi-measure results on group-by columns
        5. Applies havings, sorts, limit/offset
        """
        from datafusion.substrait import Producer

        qc = QueryContext(query_context_dict)
        self.validate_query_context(qc)

        # Register optimizer rule if pre-aggs are available
        if self.pre_agg_objects and qc.use_pre_agg:
            self.engine.set_pre_aggs(self.pre_agg_objects)
            self.engine.add_pre_agg_optimizer_rule()

        # Execute each measure
        measure_results: dict[str, pa.Table] = {}
        for measure_name in qc.measures:
            measure_fn = self.measures[measure_name]
            result = measure_fn(qc)
            if isinstance(result, MeasureDataFrame):
                # Serialize plan to Substrait bytes
                substrait_plan = Producer.to_substrait_plan(
                    result.logical_plan(), self.py_ctx
                )
                plan_bytes = substrait_plan.encode()

                # Send to Rust for optimization + execution
                batches = self.engine.optimize_and_collect_substrait(plan_bytes)
                if batches:
                    measure_results[measure_name] = pa.Table.from_batches(batches)
            elif isinstance(result, pa.Table):
                measure_results[measure_name] = result
            elif isinstance(result, list):
                # List of RecordBatches from engine.sql()
                if result:
                    measure_results[measure_name] = pa.Table.from_batches(result)

        if not measure_results:
            raise RuntimeError("No measure results produced")

        # Join multiple measure results
        result = self._join_measure_results(measure_results, qc.groups)

        # Apply havings
        if qc.havings:
            result = self._apply_havings(result, qc.havings)

        # Apply sorts
        if qc.sorts:
            result = self._apply_sorts(result, qc.sorts)

        # Apply offset and limit
        if qc.offset > 0:
            result = result.slice(qc.offset)
        if qc.limit < len(result):
            result = result.slice(0, qc.limit)

        return result

    def _join_measure_results(
        self, measure_results: dict[str, pa.Table], groups: list[str]
    ) -> pa.Table:
        """Join multiple measure results.

        - No groups: cross join (each measure produces a single row)
        - With groups: full outer join on group columns
        """
        tables = list(measure_results.values())
        if len(tables) == 1:
            return tables[0]

        result = tables[0]
        for other in tables[1:]:
            if not groups:
                # Cross join — both are single-row results
                result = _cross_join(result, other)
            else:
                # Find join columns: try both qualified (orders.region) and
                # unqualified (region) names since SQL may return either
                join_cols = _find_common_group_cols(
                    result.column_names, other.column_names, groups
                )
                if join_cols:
                    result = _join_tables(result, other, join_cols)
                else:
                    result = _cross_join(result, other)

        return result

    def _apply_havings(self, table: pa.Table, havings: dict) -> pa.Table:
        """Apply having filters to the result table."""
        mask = _evaluate_filter_tree(table, havings)
        return table.filter(mask)

    def find_best_pre_agg(
        self,
        group_by: list[str],
        agg_components: dict[str, set[str]],
        filter_columns: list[str] | None = None,
    ) -> PreAggregation | None:
        """Find the best pre-aggregation covering the given requirements."""
        filter_cols = filter_columns or []
        best = None
        for pa_obj in self.pre_agg_objects:
            if pa_obj.covers(
                requested_group_by=group_by,
                requested_agg_components=agg_components,
                filter_columns=filter_cols,
            ):
                if best is None or pa_obj.row_count < best.row_count:
                    best = pa_obj
        return best

    def write_pre_aggs(self, names: list[str]) -> list[PreAggregation]:
        """Compute and write named pre-aggregations to parquet.

        Uses the DataFrame API with auto-join to support multi-table pre-aggregations.
        """
        self.pre_agg_directory.mkdir(parents=True, exist_ok=True)
        metadata = self._load_pre_agg_metadata()
        results = []

        for name in names:
            pa_obj = None
            for obj in self.pre_agg_objects:
                if obj.name == name:
                    pa_obj = obj
                    break
            if pa_obj is None:
                raise KeyError(f"Unknown pre-aggregation: '{name}'")

            # Build DataFrame with auto-join support
            mdf = self._build_pre_agg_dataframe(pa_obj)
            result_table = mdf.to_arrow_table()

            if result_table.num_rows == 0:
                raise RuntimeError(f"Pre-agg '{name}' produced no results")

            # Write to parquet
            from datetime import datetime, timezone

            import pyarrow.parquet as pq

            pq.write_table(result_table, pa_obj.file_path)

            # Update metadata
            pa_obj.row_count = result_table.num_rows
            pa_obj.written_at = datetime.now(timezone.utc).isoformat()

            metadata[name] = {
                "row_count": pa_obj.row_count,
                "written_at": pa_obj.written_at,
            }
            results.append(pa_obj)

            # Register the pre-agg table in BOTH contexts
            for batch in result_table.to_batches():
                self.engine.register_record_batch(f"_preagg_{name}", batch)
                self.py_ctx.register_record_batches(f"_preagg_{name}", [[batch]])
                break

        self._save_pre_agg_metadata(metadata)
        return results

    def _build_pre_agg_dataframe(self, pa_obj: PreAggregation) -> MeasureDataFrame:
        """Build a MeasureDataFrame that computes the pre-aggregation.

        Uses auto-join to handle multi-table group-by and aggregation columns.
        """
        from datafusion import col
        from datafusion import functions as F

        # Collect all referenced tables from group_by and aggregation columns
        all_tables: list[str] = []
        for g in pa_obj.group_by:
            if "." in g:
                all_tables.append(g.split(".")[0])
        for col_name in pa_obj.aggregations:
            if "." in col_name:
                all_tables.append(col_name.split(".")[0])

        # Pick the base table that can reach the most other tables via join graph
        # Default to first aggregation table (fact table) since it typically has
        # outgoing join paths to dimension tables
        unique_tables = list(dict.fromkeys(all_tables))
        base_table = unique_tables[0] if unique_tables else pa_obj.group_by[0]
        if self.join_graph and len(unique_tables) > 1:
            # Find the table that can reach all others
            for candidate in unique_tables:
                if all(
                    candidate == t
                    or self.join_graph.find_path(candidate, t) is not None
                    for t in unique_tables
                ):
                    base_table = candidate
                    break

        group_exprs = [col(g) for g in pa_obj.group_by]

        agg_exprs = []
        for col_name, components in pa_obj.aggregations.items():
            for comp in components:
                c = col(col_name)
                alias = f"{col_name}-{comp}"
                if comp == "sum":
                    agg_exprs.append(F.sum(c).alias(alias))
                elif comp == "count":
                    agg_exprs.append(F.count(c).alias(alias))
                elif comp == "min":
                    agg_exprs.append(F.min(c).alias(alias))
                elif comp == "max":
                    agg_exprs.append(F.max(c).alias(alias))
                elif comp == "sumsq":
                    agg_exprs.append(F.sum(c * c).alias(alias))

        return self.table(base_table).aggregate(group_by=group_exprs, aggs=agg_exprs)

    def _apply_sorts(self, table: pa.Table, sorts: list[tuple[str, str]]) -> pa.Table:
        """Apply sorting to the result table."""
        sort_keys = [
            (col, "ascending" if d == "asc" else "descending") for col, d in sorts
        ]
        indices = pc.sort_indices(table, sort_keys=sort_keys)
        return table.take(indices)


def _extract_filter_columns(filter_dict: dict) -> list[str]:
    """Extract column names from a filter tree."""
    columns = []
    for key, value in filter_dict.items():
        if key in ("AND", "OR") and isinstance(value, list):
            for item in value:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    columns.append(item[0])
                elif isinstance(item, dict):
                    columns.extend(_extract_filter_columns(item))
    return columns


def _find_common_group_cols(
    left_cols: list[str], right_cols: list[str], groups: list[str]
) -> list[str]:
    """Find group columns common to both tables.

    Handles both qualified ('orders.region') and unqualified ('region') names.
    Returns the actual column names as they appear in both tables.
    """
    left_set = set(left_cols)
    right_set = set(right_cols)

    common = []
    for g in groups:
        # Try qualified name first
        if g in left_set and g in right_set:
            common.append(g)
            continue
        # Try unqualified name (strip table prefix)
        unqualified = g.split(".", 1)[-1] if "." in g else g
        if unqualified in left_set and unqualified in right_set:
            common.append(unqualified)
    return common


def _cross_join(left: pa.Table, right: pa.Table) -> pa.Table:
    """Cross join two tables (assumes at least one is single-row)."""
    # For single-row tables, repeat each row to match the other's length
    if len(left) == 0 or len(right) == 0:
        return pa.table({})
    # Build cross product
    left_repeated = pa.concat_tables([left] * len(right))
    right_indices = []
    for i in range(len(right)):
        right_indices.extend([i] * len(left))
    right_repeated = right.take(right_indices)

    # Combine columns (skip duplicates from right)
    columns = {}
    for name in left_repeated.column_names:
        columns[name] = left_repeated.column(name)
    for name in right_repeated.column_names:
        if name not in columns:
            columns[name] = right_repeated.column(name)
    return pa.table(columns)


def _join_tables(left: pa.Table, right: pa.Table, join_cols: list[str]) -> pa.Table:
    """Full outer join two tables on specified columns."""
    # Use a simple hash join approach via PyArrow
    # Build index from right table
    right_index: dict[tuple, int] = {}
    for i in range(len(right)):
        key = tuple(right.column(c)[i].as_py() for c in join_cols)
        right_index[key] = i

    # Collect all columns for the result
    result_cols: dict[str, list] = {name: [] for name in left.column_names}
    for name in right.column_names:
        if name not in result_cols:
            result_cols[name] = []

    matched_right = set()

    # Process left rows
    for i in range(len(left)):
        key = tuple(left.column(c)[i].as_py() for c in join_cols)
        for name in left.column_names:
            result_cols[name].append(left.column(name)[i].as_py())

        if key in right_index:
            j = right_index[key]
            matched_right.add(j)
            for name in right.column_names:
                if name not in left.column_names:
                    result_cols[name].append(right.column(name)[j].as_py())
        else:
            for name in right.column_names:
                if name not in left.column_names:
                    result_cols[name].append(None)

    # Add unmatched right rows
    for j in range(len(right)):
        if j not in matched_right:
            for name in left.column_names:
                if name in right.column_names:
                    result_cols[name].append(right.column(name)[j].as_py())
                else:
                    result_cols[name].append(None)
            for name in right.column_names:
                if name not in left.column_names:
                    result_cols[name].append(right.column(name)[j].as_py())

    return pa.table(result_cols)


def _evaluate_filter_tree(table: pa.Table, filter_tree: dict) -> pa.ChunkedArray:
    """Evaluate a filter tree against a PyArrow Table, returning a boolean mask."""
    for key, conditions in filter_tree.items():
        if key == "AND":
            masks = [_evaluate_condition(table, cond) for cond in conditions]
            result = masks[0]
            for m in masks[1:]:
                result = pc.and_(result, m)
            return result
        elif key == "OR":
            masks = [_evaluate_condition(table, cond) for cond in conditions]
            result = masks[0]
            for m in masks[1:]:
                result = pc.or_(result, m)
            return result
    # Empty filter = all True
    return pc.equal(pa.array([True] * len(table)), True)


def _evaluate_condition(table: pa.Table, condition: Any) -> pa.ChunkedArray:
    """Evaluate a single condition (tuple) or nested filter dict."""
    if isinstance(condition, dict):
        return _evaluate_filter_tree(table, condition)

    col_name, op, value = condition[0], condition[1], condition[2]
    col = table.column(col_name)

    if op == "=":
        return pc.equal(col, value)
    elif op == "!=":
        return pc.not_equal(col, value)
    elif op == ">":
        return pc.greater(col, value)
    elif op == ">=":
        return pc.greater_equal(col, value)
    elif op == "<":
        return pc.less(col, value)
    elif op == "<=":
        return pc.less_equal(col, value)
    elif op == "in":
        return pc.is_in(col, value_set=pa.array(value))
    elif op == "not in":
        return pc.invert(pc.is_in(col, value_set=pa.array(value)))
    else:
        raise ValueError(f"Unknown filter operator: '{op}'")
