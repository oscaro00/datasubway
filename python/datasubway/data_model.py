"""DataModel: core semantic layer class using DataFusion engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import datafusion as df
import pyarrow as pa

from datasubway._engine import Engine, JoinGraph, PreAggregation, QueryContext


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
        self.engine = Engine()  # Rust side — optimizer + execution + schema tracking
        self.py_ctx = df.SessionContext()  # Python side — DataFrame building

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

    def table(self, name: str) -> df.DataFrame:
        """Get a DataFusion DataFrame for a registered table.

        Eagerly pre-joins all reachable tables via the JoinGraph so that
        cross-table column references work without lazy auto-join logic.
        The join decision (which tables, what order) comes from Rust's JoinGraph.
        """
        inner = self.py_ctx.table(name)
        if self.join_graph is not None:
            joined_tables = {name}
            for target in self.join_graph.tables():
                if target in joined_tables:
                    continue
                path = self.join_graph.find_path(name, target)
                if path is None:
                    continue
                for step in path:
                    step_target = step["right"]
                    if step_target in joined_tables:
                        continue
                    right_df = self.py_ctx.table(step_target)
                    left_on = [
                        f"{step['left']}.{c}" for c in step["left_on"].split(",")
                    ]
                    right_on = [
                        f"{step_target}.{c}"
                        for c in step["right_on"].split(",")
                    ]
                    inner = inner.join(
                        right_df,
                        left_on=left_on,
                        right_on=right_on,
                        how=step["how"],
                    )
                    joined_tables.add(step_target)
        return inner

    def all_columns(self) -> list[str]:
        """Return all qualified column names across all tables (from Rust Engine)."""
        return self.engine.all_columns()

    async def query(
        self, query_context_dict: dict[str, Any], explain: bool = False
    ) -> pa.Table:
        """Execute a query and return results as a PyArrow Table.

        1. Validates the QueryContext (in Rust)
        2. Calls each measure function to produce results
        3. Serializes plans to Substrait, sends to Rust for optimization + execution
        4. Sends all measure results to Rust for post-processing
           (joining, havings, sorts, limit/offset)
        """
        from datafusion.substrait import Producer

        qc = QueryContext(query_context_dict)

        # Validate in Rust
        qc.validate(
            list(self.measures.keys()),
            self.measure_output_cols,
            self.all_columns(),
        )

        # Register optimizer rule if pre-aggs are available
        if self.pre_agg_objects and qc.use_pre_agg:
            self.engine.set_pre_aggs(self.pre_agg_objects)
            self.engine.add_pre_agg_optimizer_rule()

        # Execute each measure and collect batches
        measure_batches: list[tuple[str, list]] = []
        for measure_name in qc.measures:
            measure_fn = self.measures[measure_name]
            result = measure_fn(qc)
            if isinstance(result, df.DataFrame):
                print("PY COLLECT:", result.collect())

            if isinstance(result, df.DataFrame):
                # Serialize plan to Substrait bytes
                substrait_plan = Producer.to_substrait_plan(
                    result.logical_plan(), self.py_ctx
                )
                plan_bytes = substrait_plan.encode()

                # Send to Rust for optimization + execution
                batches = self.engine.optimize_and_collect_substrait(plan_bytes)
                print(
                    f"BATCHES for {measure_name}: {batches}, type={type(batches)}, len={len(batches)}"
                )

                if batches:
                    measure_batches.append((measure_name, list(batches)))
            elif isinstance(result, pa.Table):
                measure_batches.append((measure_name, result.to_batches()))
            elif isinstance(result, list):
                # List of RecordBatches from engine.sql()
                if result:
                    measure_batches.append((measure_name, result))

        if not measure_batches:
            raise RuntimeError("No measure results produced")

        # Post-process in Rust: join measures, apply havings, sorts, limit/offset
        result_batches = self.engine.post_process_measures(measure_batches, qc)
        return pa.Table.from_batches(result_batches)

    def find_best_pre_agg(
        self,
        group_by: list[str],
        agg_components: dict[str, set[str]],
        filter_columns: list[str] | None = None,
    ) -> PreAggregation | None:
        """Find the best pre-aggregation covering the given requirements (Rust)."""
        return self.engine.find_best_pre_agg(
            group_by, agg_components, filter_columns or []
        )

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
        self.engine.set_pre_aggs(self.pre_agg_objects)
        return results

    def _build_pre_agg_dataframe(self, pa_obj: PreAggregation) -> df.DataFrame:
        """Build a DataFrame that computes the pre-aggregation.

        Uses dm.table() which eagerly pre-joins all reachable tables,
        so cross-table group-by and aggregation columns work automatically.
        """
        from datafusion import col
        from datafusion import functions as F

        # Pick a base table that can reach all referenced tables via join graph
        all_tables: list[str] = []
        for g in pa_obj.group_by:
            if "." in g:
                all_tables.append(g.split(".")[0])
        for col_name in pa_obj.aggregations:
            if "." in col_name:
                all_tables.append(col_name.split(".")[0])
        unique_tables = list(dict.fromkeys(all_tables))
        base_table = unique_tables[0] if unique_tables else pa_obj.group_by[0]
        if self.join_graph and len(unique_tables) > 1:
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
