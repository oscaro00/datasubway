//! Standalone Rust DataModel — semantic layer using DataFusion DataFrame API.
//!
//! Enables pure-Rust usage without Python. Measures are closures that build
//! DataFusion DataFrames using `allow()`/`exclude()` from `column_context`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::datasource::DefaultTableSource;
use datafusion::execution::context::{SessionConfig, SessionContext};
use datafusion::prelude::*;
use datafusion_expr::{LogicalPlan, TableSource};
use futures::future::BoxFuture;

use crate::model::column_context;
use crate::model::combine_measures;
use crate::model::data_sources;
use crate::model::joins::{Join, JoinGraph, JoinHow};
use crate::model::pre_agg::PreAggregation;
use crate::model::query_context::{MeasureMetadata, QueryContext};
use crate::optimizer::eliminate_joins_rule::EliminateUnusedJoins;
use crate::optimizer::pre_agg_rule::PreAggSubstitution;

/// Records a single `allow()`/`exclude()` call made during measure probing.
#[derive(Debug, Clone)]
pub struct AllowExcludeRecord {
    pub kind: AllowExcludeKind,
    pub patterns: Vec<String>,
    pub input_type: ColumnInputType,
    pub columns: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AllowExcludeKind {
    Allow,
    Exclude,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ColumnInputType {
    Columns,
    FilterTree,
}

/// A measure is a closure that takes a QueryContext and DataModel reference,
/// and returns a DataFusion DataFrame.
///
/// Measures should use `dm.allow()`/`dm.exclude()` (not `column_context::allow/exclude` directly)
/// so that probe recording captures the call metadata.
///
/// ```rust,ignore
/// Arc::new(|qc: &QueryContext, dm: &DataModel| Box::pin(async move {
///     let filter = dm.allow(&["*".into()], ColumnInput::FilterTree(&qc.filters), None)?.into_filter_expr();
///     let groups = dm.allow(&["*".into()], ColumnInput::Columns(&qc.groups), None)?.into_exprs();
///     dm.table("orders").await?
///         .filter(filter)?
///         .aggregate(groups, vec![sum(col("amount")).alias("revenue")])
/// }))
/// ```
pub type MeasureFn = Arc<
    dyn for<'a> Fn(
            &'a QueryContext,
            &'a DataModel,
        ) -> BoxFuture<'a, Result<DataFrame, DataFusionError>>
        + Send
        + Sync,
>;

/// Extract aggregate column names from the outermost Aggregate node in the plan.
///
/// Only the final (outermost) aggregate matters — inner aggregates are not
/// traversed. Returns the output names (aliases) of `aggr_expr` entries.
fn extract_outermost_aggregate_columns(plan: &LogicalPlan) -> Vec<String> {
    match plan {
        LogicalPlan::Aggregate(agg) => agg
            .aggr_expr
            .iter()
            .filter_map(extract_expr_output_name)
            .collect(),
        _ => {
            for child in plan.inputs() {
                let result = extract_outermost_aggregate_columns(child);
                if !result.is_empty() {
                    return result;
                }
            }
            vec![]
        }
    }
}

fn extract_expr_output_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Alias(alias) => Some(alias.name.clone()),
        Expr::Column(col) => Some(col.name.clone()),
        _ => Some(format!("{}", expr)),
    }
}

/// Infer the base table from the first aggregation column name.
///
/// Uses the table prefix from the first aggregation key (e.g. "orders.amount" → "orders").
/// The aggregation table is chosen as the base because join directions may not be
/// bidirectional — the fact table that owns the measures is the natural entry point.
fn infer_source_table(
    aggregations: &HashMap<String, Vec<String>>,
) -> Result<String, DataFusionError> {
    for col_name in aggregations.keys() {
        if col_name.contains('.') {
            return Ok(col_name.split('.').next().unwrap().to_string());
        }
    }

    Err(DataFusionError::Plan(
        "Cannot infer source table: no qualified column names (expected table.column format)"
            .into(),
    ))
}

/// Standalone Rust semantic layer model backed by DataFusion.
///
/// Manages data sources, joins, measures, pre-aggregations, and query execution.
pub struct DataModel {
    ctx: SessionContext,
    /// Pre-aggregation optimizer rule, applied per-measure in `prepare_measure_dfs`.
    /// Built by `build_pre_agg_optimizer()` when pre-agg objects are present.
    pre_agg_rule: Option<PreAggSubstitution>,
    join_graph: Option<JoinGraph>,
    table_schemas: HashMap<String, Vec<String>>,
    measures: HashMap<String, MeasureFn>,
    measure_metadata: HashMap<String, MeasureMetadata>,
    pre_agg_objects: Vec<PreAggregation>,
    pre_agg_path: Option<String>,
    probe_recorder: Option<Arc<Mutex<Vec<AllowExcludeRecord>>>>,
}

impl DataModel {
    /// Create a new DataModel with a default SessionContext.
    pub fn new() -> Self {
        // Disable Utf8View for parquet so string types stay consistent across
        // data sources (MemTable uses Utf8, Parquet defaults to Utf8View).
        // This matters for pre-aggregation where the optimizer swaps table scans.
        let config = SessionConfig::new().set_bool(
            "datafusion.execution.parquet.schema_force_view_types",
            false,
        );
        DataModel {
            ctx: SessionContext::new_with_config(config),
            pre_agg_rule: None,
            join_graph: None,
            table_schemas: HashMap::new(),
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
            pre_agg_objects: Vec::new(),
            pre_agg_path: None,
            probe_recorder: None,
        }
    }

    /// Create a DataModel with a caller-provided SessionContext.
    ///
    /// Use this when you already have a runtime (e.g., axum, actix-web)
    /// and want to share the same SessionContext.
    pub fn with_context(ctx: SessionContext) -> Self {
        DataModel {
            ctx,
            pre_agg_rule: None,
            join_graph: None,
            table_schemas: HashMap::new(),
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
            pre_agg_objects: Vec::new(),
            pre_agg_path: None,
            probe_recorder: None,
        }
    }

    /// Register a RecordBatch as an in-memory table.
    pub fn register_record_batch(
        &mut self,
        name: &str,
        batch: RecordBatch,
    ) -> Result<(), DataFusionError> {
        let schema_names = data_sources::register_record_batch(&self.ctx, name, batch)?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register a Parquet file as a named table.
    pub async fn register_parquet(
        &mut self,
        name: &str,
        path: &str,
    ) -> Result<(), DataFusionError> {
        let schema_names = data_sources::register_parquet(&self.ctx, name, path).await?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register a CSV file as a named table.
    pub async fn register_csv(&mut self, name: &str, path: &str) -> Result<(), DataFusionError> {
        let schema_names = data_sources::register_csv(&self.ctx, name, path).await?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register a newline-delimited JSON file as a named table.
    pub async fn register_json(&mut self, name: &str, path: &str) -> Result<(), DataFusionError> {
        let schema_names = data_sources::register_json(&self.ctx, name, path).await?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register an Arrow IPC file as a named table.
    pub async fn register_arrow(&mut self, name: &str, path: &str) -> Result<(), DataFusionError> {
        let schema_names = data_sources::register_arrow(&self.ctx, name, path).await?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Set the join graph from a list of Join specifications.
    pub fn set_joins(&mut self, joins: &[Join]) -> Result<(), String> {
        self.join_graph = Some(JoinGraph::new(joins)?);
        let elim_rule = EliminateUnusedJoins::new(self.join_graph.as_ref().unwrap().clone());
        self.ctx.add_optimizer_rule(Arc::new(elim_rule));
        Ok(())
    }

    /// Set pre-aggregation objects.
    pub fn set_pre_aggregations(&mut self, pre_aggs: Vec<PreAggregation>) {
        self.pre_agg_objects = pre_aggs;
    }

    /// Set the base directory for pre-aggregation parquet files.
    pub fn set_pre_agg_path(&mut self, path: &str) {
        self.pre_agg_path = Some(path.to_string());
    }

    /// Register pre-aggregation definitions without writing them.
    pub fn register_pre_agg(&mut self, pre_aggs: Vec<PreAggregation>) {
        self.pre_agg_objects.extend(pre_aggs);
    }

    /// Compute, write, and activate pre-aggregations.
    ///
    /// For each provided `PreAggregation`:
    /// 1. Infers the source table from qualified column names
    /// 2. Builds and executes the aggregation query
    /// 3. Writes results to `{pre_agg_path}/{name}.parquet`
    /// 4. Registers the parquet as a named table
    /// 5. Rebuilds the pre-aggregation optimizer context
    pub async fn write_pre_agg(
        &mut self,
        pre_aggs: Vec<PreAggregation>,
    ) -> Result<(), DataFusionError> {
        let base_path = self
            .pre_agg_path
            .as_ref()
            .ok_or_else(|| {
                DataFusionError::Plan("pre_agg_path not set — call set_pre_agg_path() first".into())
            })?
            .clone();

        let base = std::path::Path::new(&base_path);
        std::fs::create_dir_all(base).map_err(|e| {
            DataFusionError::Execution(format!("Failed to create pre-agg directory: {}", e))
        })?;

        for pa in pre_aggs {
            // Infer source table from qualified column names
            let source_table = infer_source_table(&pa.aggregations)?;

            // Build the aggregation query
            let df = self.table(&source_table).await?;

            // Use bare column names in parquet — DataFusion's col() resolves
            // "orders.region" to Column { relation: Some("orders"), name: "region" },
            // and the Arrow output field name is just "region".
            // The optimizer wraps the pre-agg scan in a SubqueryAlias to restore
            // the original table qualifier.
            let group_exprs: Vec<Expr> = pa.group_by.iter().map(|c| col(c.as_str())).collect();

            let mut agg_exprs: Vec<Expr> = Vec::new();
            for (col_name, components) in &pa.aggregations {
                for comp in components {
                    let src_col = col(col_name.as_str());
                    let agg_expr = match comp.as_str() {
                        "sum" => datafusion_functions_aggregate::sum::sum(src_col),
                        "count" => datafusion_functions_aggregate::count::count(src_col),
                        "min" => datafusion_functions_aggregate::min_max::min(src_col),
                        "max" => datafusion_functions_aggregate::min_max::max(src_col),
                        "sumsq" => {
                            datafusion_functions_aggregate::sum::sum(src_col.clone() * src_col)
                        }
                        other => {
                            return Err(DataFusionError::Plan(format!(
                                "Unsupported pre-agg component: {}",
                                other
                            )));
                        }
                    };
                    let alias = PreAggregation::component_column(col_name, comp);
                    agg_exprs.push(agg_expr.alias(&alias));
                }
            }

            let result_df = df.aggregate(group_exprs, agg_exprs)?;
            let batches = result_df.collect().await?;

            // Write to parquet
            let parquet_path = base.join(format!("{}.parquet", pa.name));
            let mut total_rows: u64 = 0;
            if let Some(batch) = batches.first() {
                let schema = batch.schema();
                let file = std::fs::File::create(&parquet_path).map_err(|e| {
                    DataFusionError::Execution(format!("Failed to create parquet file: {}", e))
                })?;
                let mut writer = parquet::arrow::ArrowWriter::try_new(file, schema.clone(), None)?;
                for b in &batches {
                    writer.write(b)?;
                    total_rows += b.num_rows() as u64;
                }
                writer.close()?;
            }

            // Store PreAggregation with qualified column names
            let stored_pa = PreAggregation {
                name: pa.name.clone(),
                group_by: pa.group_by.clone(),
                aggregations: pa.aggregations.clone(),
                row_count: total_rows,
                written_at: Some(format!("{:?}", std::time::SystemTime::now())),
            };

            // Register the parquet file as a table
            self.register_parquet(&pa.name, parquet_path.to_str().unwrap())
                .await?;

            self.pre_agg_objects.push(stored_pa);
        }

        // Rebuild the pre-agg optimizer context
        self.build_pre_agg_optimizer().await?;

        Ok(())
    }

    /// Register a measure function.
    ///
    /// The measure is probed with an empty QueryContext to extract output columns,
    /// classify aggregate vs group-by columns, and record allow/exclude call metadata.
    pub async fn register_measure(
        &mut self,
        name: &str,
        measure_fn: MeasureFn,
    ) -> Result<(), DataFusionError> {
        // Probe with empty context to extract metadata
        let probe_qc = QueryContext::new(
            vec![name.to_string()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .map_err(|e| DataFusionError::Plan(e))?;

        // Enable recording
        let recorder = Arc::new(Mutex::new(Vec::new()));
        self.probe_recorder = Some(recorder.clone());

        let mut output_cols = Vec::new();
        let mut aggregate_cols = Vec::new();

        if let Ok(df) = measure_fn(&probe_qc, self).await {
            output_cols = df
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            aggregate_cols = extract_outermost_aggregate_columns(df.logical_plan());
        }

        // Collect recordings and disable
        let calls = Arc::try_unwrap(recorder)
            .map(|mutex| mutex.into_inner().unwrap())
            .unwrap_or_default();
        self.probe_recorder = None;

        let metadata = MeasureMetadata {
            name: name.to_string(),
            output_columns: output_cols,
            aggregate_columns: aggregate_cols,
            allow_exclude_calls: calls,
        };

        self.measures.insert(name.to_string(), measure_fn);
        self.measure_metadata.insert(name.to_string(), metadata);
        Ok(())
    }

    /// Get a DataFusion DataFrame for a registered table.
    ///
    /// Eagerly pre-joins all reachable tables via the JoinGraph so that
    /// cross-table column references work in the DataFrame API (which validates
    /// schemas at expression-building time, before the optimizer runs).
    pub async fn table(&self, name: &str) -> Result<DataFrame, DataFusionError> {
        let ctx = &self.ctx;
        let mut inner = ctx.table(name).await?;

        if let Some(ref join_graph) = self.join_graph {
            let mut joined_tables: HashSet<String> = HashSet::new();
            joined_tables.insert(name.to_string());

            for target in join_graph.tables() {
                if joined_tables.contains(target) {
                    continue;
                }
                let path = match join_graph.find_path(name, target) {
                    Some(p) => p,
                    None => continue,
                };
                for step in &path {
                    if joined_tables.contains(&step.right) {
                        continue;
                    }
                    let right_df = ctx.table(&step.right).await?;
                    let left_on_qualified: Vec<String> = step
                        .left_on
                        .iter()
                        .map(|s| format!("{}.{}", step.left, s))
                        .collect();
                    let right_on_qualified: Vec<String> = step
                        .right_on
                        .iter()
                        .map(|s| format!("{}.{}", step.right, s))
                        .collect();
                    let left_on: Vec<&str> = left_on_qualified.iter().map(|s| s.as_str()).collect();
                    let right_on: Vec<&str> =
                        right_on_qualified.iter().map(|s| s.as_str()).collect();
                    let join_type = match step.how {
                        JoinHow::Inner => JoinType::Inner,
                        JoinHow::Left => JoinType::Left,
                    };
                    inner = inner.join(right_df, join_type, &left_on, &right_on, None)?;
                    joined_tables.insert(step.right.clone());
                }
            }
        }
        Ok(inner)
    }

    /// Return all qualified column names across all tables.
    pub fn all_columns(&self) -> Vec<String> {
        self.table_schemas
            .values()
            .flat_map(|cols| cols.iter().cloned())
            .collect()
    }

    /// Access the SessionContext (for advanced use cases).
    pub fn session_context(&self) -> &SessionContext {
        &self.ctx
    }

    /// Apply `allow()` column context. Use this in measure closures instead of
    /// calling `column_context::allow` directly so that probe recording works.
    pub fn allow(
        &self,
        patterns: &[String],
        input: column_context::ColumnInput,
        columns: Option<&[String]>,
    ) -> Result<column_context::ColumnOutput, DataFusionError> {
        if let Some(ref recorder) = self.probe_recorder {
            recorder.lock().unwrap().push(AllowExcludeRecord {
                kind: AllowExcludeKind::Allow,
                patterns: patterns.to_vec(),
                input_type: match &input {
                    column_context::ColumnInput::Columns(_) => ColumnInputType::Columns,
                    column_context::ColumnInput::FilterTree(_) => ColumnInputType::FilterTree,
                },
                columns: columns.map(|c| c.to_vec()),
            });
        }
        column_context::allow(patterns, input, columns)
    }

    /// Apply `exclude()` column context. Use this in measure closures instead of
    /// calling `column_context::exclude` directly so that probe recording works.
    pub fn exclude(
        &self,
        patterns: &[String],
        input: column_context::ColumnInput,
        columns: Option<&[String]>,
    ) -> Result<column_context::ColumnOutput, DataFusionError> {
        if let Some(ref recorder) = self.probe_recorder {
            recorder.lock().unwrap().push(AllowExcludeRecord {
                kind: AllowExcludeKind::Exclude,
                patterns: patterns.to_vec(),
                input_type: match &input {
                    column_context::ColumnInput::Columns(_) => ColumnInputType::Columns,
                    column_context::ColumnInput::FilterTree(_) => ColumnInputType::FilterTree,
                },
                columns: columns.map(|c| c.to_vec()),
            });
        }
        column_context::exclude(patterns, input, columns)
    }

    /// Build (or rebuild) the pre-aggregation rule.
    ///
    /// Creates a `PreAggSubstitution` rule that can rewrite individual measure
    /// plans to use pre-aggregated tables. The rule is applied per-measure in
    /// `prepare_measure_dfs` rather than as a session-level optimizer, so it
    /// never sees the combined multi-measure plan.
    pub async fn build_pre_agg_optimizer(&mut self) -> Result<(), DataFusionError> {
        if !self.pre_agg_objects.is_empty() {
            let mut table_sources: HashMap<String, Arc<dyn TableSource>> = HashMap::new();
            for pa in &self.pre_agg_objects {
                let provider = self.ctx.table_provider(&pa.name).await?;
                table_sources.insert(pa.name.clone(), Arc::new(DefaultTableSource::new(provider)));
            }
            self.pre_agg_rule =
                Some(PreAggSubstitution::new(self.pre_agg_objects.clone(), table_sources));
        }

        Ok(())
    }

    /// Validate the QueryContext and build a DataFrame for each requested
    /// measure (without collecting).
    async fn prepare_measure_dfs(
        &self,
        qc: &QueryContext,
    ) -> Result<Vec<(String, DataFrame)>, DataFusionError> {
        // Validate
        let measure_metadata: Vec<MeasureMetadata> =
            self.measure_metadata.values().cloned().collect();
        let all_cols: HashSet<String> = self.all_columns().into_iter().collect();
        qc.validate(&measure_metadata, &all_cols)
            .map_err(|e| DataFusionError::Plan(e))?;

        let use_pre_agg = qc.use_pre_agg && self.pre_agg_rule.is_some();

        // Build each measure DataFrame (no collect)
        let mut measure_dfs: Vec<(String, DataFrame)> = Vec::new();
        for measure_name in &qc.measures {
            let measure_fn = self.measures.get(measure_name).ok_or_else(|| {
                DataFusionError::Plan(format!("Unknown measure: '{}'", measure_name))
            })?;
            let df = measure_fn(qc, self).await?;

            if use_pre_agg {
                // Apply the pre-agg rule to each measure's plan individually.
                // This ensures the rule only sees one measure at a time and never
                // attempts to rewrite the combined multi-measure plan.
                let rule = self.pre_agg_rule.as_ref().unwrap();
                let rewritten_plan = rule.try_rewrite(df.logical_plan().clone())?;
                let rewritten_df = DataFrame::new(self.ctx.state(), rewritten_plan);
                measure_dfs.push((measure_name.clone(), rewritten_df));
            } else {
                measure_dfs.push((measure_name.clone(), df));
            }
        }

        if measure_dfs.is_empty() {
            return Err(DataFusionError::Plan("No measure results produced".into()));
        }

        Ok(measure_dfs)
    }

    /// Collect query results as RecordBatches.
    ///
    /// 1. Validates the QueryContext
    /// 2. Calls each measure function to produce DataFrames
    /// 3. Combines measures (joins, havings, sorts, limit/offset)
    /// 4. Collects and returns results
    pub async fn collect(&self, qc: &QueryContext) -> Result<Vec<RecordBatch>, DataFusionError> {
        let measure_dfs = self.prepare_measure_dfs(qc).await?;
        let borrowed: Vec<(&str, DataFrame)> = measure_dfs
            .iter()
            .map(|(n, df)| (n.as_str(), df.clone()))
            .collect();
        combine_measures::combine_measure_results(borrowed, qc).await
    }

    /// Return an explain DataFrame for the query plan without executing it.
    ///
    /// Works like `collect()` but calls DataFusion's `explain()` instead of
    /// collecting results. Accepts the same `verbose` and `analyze` flags
    /// as `DataFrame::explain()`.
    pub async fn explain(
        &self,
        qc: &QueryContext,
        verbose: bool,
        analyze: bool,
    ) -> Result<DataFrame, DataFusionError> {
        let measure_dfs = self.prepare_measure_dfs(qc).await?;
        let borrowed: Vec<(&str, DataFrame)> = measure_dfs
            .iter()
            .map(|(n, df)| (n.as_str(), df.clone()))
            .collect();
        let combined_df = combine_measures::combine_measure_dfs(borrowed, qc)?;
        combined_df.explain(verbose, analyze)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::column_context::ColumnInput::*;
    use crate::model::joins::JoinDirection;
    use crate::model::pre_agg::PreAggregation;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::functions_aggregate::count::count;
    use datafusion::functions_aggregate::sum::sum;

    fn make_orders_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Int64, false),
            Field::new("quantity", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec![
                    "US", "EU", "US", "EU", "US",
                ])),
                Arc::new(arrow::array::Int64Array::from(vec![
                    100, 200, 150, 250, 300,
                ])),
                Arc::new(arrow::array::Int64Array::from(vec![10, 20, 15, 25, 30])),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn test_basic_query_no_groups() {
        let mut dm = DataModel::new();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|_qc, dm| {
                Box::pin(async move {
                    dm.table("orders")
                        .await?
                        .aggregate(vec![], vec![sum(col("amount")).alias("revenue")])
                })
            }),
        )
        .await
        .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 1);
        let col = result[0]
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 1000);
    }

    #[tokio::test]
    async fn test_query_with_groups() {
        let mut dm = DataModel::new();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|qc, dm| {
                Box::pin(async move {
                    let group_exprs = dm
                        .allow(
                            &["*".into()],
                            column_context::ColumnInput::Columns(&qc.groups),
                            None,
                        )?
                        .into_exprs();
                    dm.table("orders")
                        .await?
                        .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
                })
            }),
        )
        .await
        .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[tokio::test]
    async fn test_query_with_having_and_sort() {
        let mut dm = DataModel::new();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|qc, dm| {
                Box::pin(async move {
                    let group_exprs = dm
                        .allow(
                            &["*".into()],
                            column_context::ColumnInput::Columns(&qc.groups),
                            None,
                        )?
                        .into_exprs();
                    dm.table("orders")
                        .await?
                        .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
                })
            }),
        )
        .await
        .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            Some(serde_json::json!({"AND": [["revenue", ">", 500]]})),
            Some(vec![("revenue".into(), "desc".into())]),
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
        let col = result[0]
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 550);
    }

    #[tokio::test]
    async fn test_multi_measure() {
        let mut dm = DataModel::new();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|_qc, dm| {
                Box::pin(async move {
                    dm.table("orders")
                        .await?
                        .aggregate(vec![], vec![sum(col("amount")).alias("revenue")])
                })
            }),
        )
        .await
        .unwrap();

        dm.register_measure(
            "total_quantity",
            Arc::new(|_qc, dm| {
                Box::pin(async move {
                    dm.table("orders")
                        .await?
                        .aggregate(vec![], vec![sum(col("quantity")).alias("total_quantity")])
                })
            }),
        )
        .await
        .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into(), "total_quantity".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_auto_join() {
        let mut dm = DataModel::new();

        let players_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));
        let players = RecordBatch::try_new(
            players_schema,
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2])),
                Arc::new(arrow::array::StringArray::from(vec!["Alice", "Bob"])),
            ],
        )
        .unwrap();

        let stats_schema = Arc::new(Schema::new(vec![
            Field::new("player_id", DataType::Int64, false),
            Field::new("score", DataType::Int64, false),
        ]));
        let stats = RecordBatch::try_new(
            stats_schema,
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2, 1])),
                Arc::new(arrow::array::Int64Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        dm.register_record_batch("players", players).unwrap();
        dm.register_record_batch("player_stats", stats).unwrap();

        dm.set_joins(&[Join {
            left: "player_stats".into(),
            right: "players".into(),
            left_on: vec!["player_id".into()],
            right_on: vec!["id".into()],
            how: JoinHow::Inner,
            direction: JoinDirection::Right2Left,
        }])
        .unwrap();

        dm.register_measure(
            "total_score",
            Arc::new(|qc, dm| {
                Box::pin(async move {
                    let group_exprs = dm
                        .allow(
                            &["*".into()],
                            column_context::ColumnInput::Columns(&qc.groups),
                            None,
                        )?
                        .into_exprs();
                    dm.table("player_stats")
                        .await?
                        .aggregate(group_exprs, vec![sum(col("score")).alias("total_score")])
                })
            }),
        )
        .await
        .unwrap();

        let qc = QueryContext::new(
            vec!["total_score".into()],
            None,
            Some(vec!["players.name".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[tokio::test]
    async fn test_unreferenced_join_eliminated() {
        // Setup: 3 tables, but the query only references player_stats and players.
        // "teams" is joined into the graph but never referenced in the query.
        // The EliminateUnusedJoins optimizer rule should remove the teams join.
        let mut dm = DataModel::new();

        let players = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("team_id", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2])),
                Arc::new(arrow::array::StringArray::from(vec!["Alice", "Bob"])),
                Arc::new(arrow::array::Int64Array::from(vec![10, 20])),
            ],
        )
        .unwrap();

        let stats = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("player_id", DataType::Int64, false),
                Field::new("score", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![1, 2, 1])),
                Arc::new(arrow::array::Int64Array::from(vec![10, 20, 30])),
            ],
        )
        .unwrap();

        let teams = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("team_name", DataType::Utf8, false),
            ])),
            vec![
                Arc::new(arrow::array::Int64Array::from(vec![10, 20])),
                Arc::new(arrow::array::StringArray::from(vec!["Red", "Blue"])),
            ],
        )
        .unwrap();

        dm.register_record_batch("players", players).unwrap();
        dm.register_record_batch("player_stats", stats).unwrap();
        dm.register_record_batch("teams", teams).unwrap();

        dm.set_joins(&[
            Join {
                left: "player_stats".into(),
                right: "players".into(),
                left_on: vec!["player_id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Inner,
                direction: JoinDirection::Right2Left,
            },
            Join {
                left: "players".into(),
                right: "teams".into(),
                left_on: vec!["team_id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Inner,
                direction: JoinDirection::Right2Left,
            },
        ])
        .unwrap();

        // Register a measure that references player_stats.score and players.name
        // but NOT teams
        dm.register_measure(
            "total_score",
            Arc::new(|qc, dm| {
                Box::pin(async move {
                    let group_exprs = dm
                        .allow(
                            &["*".into()],
                            column_context::ColumnInput::Columns(&qc.groups),
                            None,
                        )?
                        .into_exprs();
                    dm.table("player_stats")
                        .await?
                        .aggregate(group_exprs, vec![sum(col("score")).alias("total_score")])
                })
            }),
        )
        .await
        .unwrap();

        // Query through DataModel::collect()
        let qc = QueryContext::new(
            vec!["total_score".into()],
            None,
            Some(vec!["players.name".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();

        // Verify correct results: Alice=40, Bob=20
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "Should have 2 rows (Alice and Bob)");

        // Also verify the plan does NOT contain "teams"
        let df = dm.table("player_stats").await.unwrap();
        let group_exprs = vec![col("players.name")];
        let agg_df = df
            .aggregate(group_exprs, vec![sum(col("score")).alias("total_score")])
            .unwrap();

        let plan = agg_df.into_optimized_plan().unwrap();
        let plan_str = format!("{}", plan.display_indent());
        println!("OPTIMIZED PLAN:\n{}", plan_str);

        assert!(
            !plan_str.contains("teams"),
            "Optimized plan should eliminate unreferenced 'teams' join.\nPlan:\n{}",
            plan_str
        );
    }

    // ── Pre-aggregation integration tests (parquet round-trip) ─────────

    fn write_batch_to_parquet(batch: &RecordBatch, path: &std::path::Path) {
        let file = std::fs::File::create(path).unwrap();
        let mut writer = parquet::arrow::ArrowWriter::try_new(file, batch.schema(), None).unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }

    /// Build a DataModel with orders table + a pre-agg parquet registered.
    /// Returns (DataModel, TempDir) — keep TempDir alive for the test duration.
    async fn setup_pre_agg_dm(
        preagg_batch: RecordBatch,
        preagg_name: &str,
        pre_agg: PreAggregation,
    ) -> (DataModel, tempfile::TempDir) {
        let tmp_dir = tempfile::tempdir().unwrap();
        let preagg_path = tmp_dir.path().join(format!("{}.parquet", preagg_name));
        write_batch_to_parquet(&preagg_batch, &preagg_path);

        let mut dm = DataModel::new();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();
        dm.register_parquet(preagg_name, preagg_path.to_str().unwrap())
            .await
            .unwrap();
        dm.set_pre_aggregations(vec![pre_agg]);
        dm.build_pre_agg_optimizer().await.unwrap();

        (dm, tmp_dir)
    }

    fn make_sum_preagg_batch() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("amount-sum", DataType::Int64, false),
        ]));
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["EU", "US"])),
                Arc::new(arrow::array::Int64Array::from(vec![450, 550])),
            ],
        )
        .unwrap()
    }

    fn make_sum_pre_agg_object(preagg_name: &str) -> PreAggregation {
        let mut pa = PreAggregation::new(
            preagg_name.into(),
            vec!["orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["sum".into()])]),
        )
        .unwrap();
        pa.row_count = 2;
        pa
    }

    /// Helper to register the standard revenue measure with filter and group support.
    fn revenue_measure() -> MeasureFn {
        Arc::new(|qc, dm| {
            Box::pin(async move {
                let filter_expr = dm
                    .allow(&["*".into()], FilterTree(&qc.filters), None)?
                    .into_filter_expr();
                let group_exprs = dm
                    .allow(&["*".into()], Columns(&qc.groups), None)?
                    .into_exprs();
                dm.table("orders")
                    .await?
                    .filter(filter_expr)?
                    .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
            })
        })
    }

    #[tokio::test]
    async fn test_pre_agg_sum_from_parquet() {
        let preagg_batch = make_sum_preagg_batch();
        let pa = make_sum_pre_agg_object("regional_preagg");
        let (mut dm, _tmp) = setup_pre_agg_dm(preagg_batch, "regional_preagg", pa).await;

        dm.register_measure("revenue", revenue_measure())
            .await
            .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("orders.region".into(), "asc".into())]),
            None,
            None,
            None,
        )
        .unwrap();

        // Verify explain references pre-agg table
        let explain_df = dm.explain(&qc, false, false).await.unwrap();
        let explain_batches = explain_df.collect().await.unwrap();
        let explain_str = format!("{:?}", explain_batches);
        assert!(
            explain_str.contains("regional_preagg"),
            "Explain should reference pre-agg table.\nExplain:\n{}",
            explain_str
        );

        // Need a fresh DataModel for collect (optimizer rules accumulate)
        let preagg_batch = make_sum_preagg_batch();
        let pa = make_sum_pre_agg_object("regional_preagg");
        let (mut dm2, _tmp2) = setup_pre_agg_dm(preagg_batch, "regional_preagg", pa).await;

        dm2.register_measure("revenue", revenue_measure())
            .await
            .unwrap();

        let result = dm2.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "Should have 2 rows (EU and US)");

        // Check values: EU=450, US=550 (sorted by region asc)
        let batch = &result[0];
        let revenue_col = batch
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        let region_col = batch
            .column_by_name("region")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        assert_eq!(region_col.value(0), "EU");
        assert_eq!(revenue_col.value(0), 450);
        assert_eq!(region_col.value(1), "US");
        assert_eq!(revenue_col.value(1), 550);
    }

    #[tokio::test]
    async fn test_pre_agg_with_filter() {
        let preagg_batch = make_sum_preagg_batch();
        let pa = make_sum_pre_agg_object("regional_preagg");
        let (mut dm, _tmp) = setup_pre_agg_dm(preagg_batch, "regional_preagg", pa).await;

        dm.register_measure("revenue", revenue_measure())
            .await
            .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into()],
            Some(serde_json::json!({"AND": [["orders.region", "=", "US"]]})),
            Some(vec!["orders.region".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1, "Should have 1 row (US only)");

        let batch = &result[0];
        let revenue_col = batch
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(revenue_col.value(0), 550);
    }

    #[tokio::test]
    async fn test_pre_agg_disabled_falls_back() {
        let preagg_batch = make_sum_preagg_batch();
        let pa = make_sum_pre_agg_object("regional_preagg");
        let (mut dm, _tmp) = setup_pre_agg_dm(preagg_batch, "regional_preagg", pa).await;

        dm.register_measure("revenue", revenue_measure())
            .await
            .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("orders.region".into(), "asc".into())]),
            None,
            None,
            Some(false), // disable pre-agg
        )
        .unwrap();

        // Verify explain does NOT reference pre-agg table
        let explain_df = dm.explain(&qc, false, false).await.unwrap();
        let explain_batches = explain_df.collect().await.unwrap();
        let explain_str = format!("{:?}", explain_batches);
        assert!(
            !explain_str.contains("regional_preagg"),
            "Explain should NOT reference pre-agg when disabled.\nExplain:\n{}",
            explain_str
        );
    }

    #[tokio::test]
    async fn test_pre_agg_no_coverage_falls_back() {
        // Pre-agg only covers "region", but query groups by "category" (not covered)
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("category", DataType::Utf8, false),
            Field::new("amount", DataType::Int64, false),
        ]));
        let orders_with_category = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU", "US"])),
                Arc::new(arrow::array::StringArray::from(vec!["A", "B", "A"])),
                Arc::new(arrow::array::Int64Array::from(vec![100, 200, 150])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let preagg_path = tmp_dir.path().join("regional_preagg.parquet");
        let preagg_batch = make_sum_preagg_batch();
        write_batch_to_parquet(&preagg_batch, &preagg_path);

        let mut dm = DataModel::new();
        dm.register_record_batch("orders", orders_with_category)
            .unwrap();
        dm.register_parquet("regional_preagg", preagg_path.to_str().unwrap())
            .await
            .unwrap();

        let mut pa = make_sum_pre_agg_object("regional_preagg");
        pa.row_count = 2;
        dm.set_pre_aggregations(vec![pa]);
        dm.build_pre_agg_optimizer().await.unwrap();

        dm.register_measure("revenue", revenue_measure())
            .await
            .unwrap();

        // Group by category — pre-agg doesn't cover this
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.category".into()]),
            None,
            Some(vec![("orders.category".into(), "asc".into())]),
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "Should have 2 rows (A and B)");

        let batch = &result[0];
        let revenue_col = batch
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        let category_col = batch
            .column_by_name("category")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();
        assert_eq!(category_col.value(0), "A");
        assert_eq!(revenue_col.value(0), 250); // 100+150
        assert_eq!(category_col.value(1), "B");
        assert_eq!(revenue_col.value(1), 200);
    }

    #[tokio::test]
    async fn test_pre_agg_count_rewrite() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("amount-count", DataType::Int64, false),
        ]));
        let preagg_batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["EU", "US"])),
                Arc::new(arrow::array::Int64Array::from(vec![2, 3])),
            ],
        )
        .unwrap();

        let tmp_dir = tempfile::tempdir().unwrap();
        let preagg_path = tmp_dir.path().join("count_preagg.parquet");
        write_batch_to_parquet(&preagg_batch, &preagg_path);

        let mut dm = DataModel::new();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();
        dm.register_parquet("count_preagg", preagg_path.to_str().unwrap())
            .await
            .unwrap();

        let mut pa = PreAggregation::new(
            "count_preagg".into(),
            vec!["orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["count".into()])]),
        )
        .unwrap();
        pa.row_count = 2;
        dm.set_pre_aggregations(vec![pa]);
        dm.build_pre_agg_optimizer().await.unwrap();

        dm.register_measure(
            "order_count",
            Arc::new(|qc, dm| {
                Box::pin(async move {
                    let filter_expr = dm
                        .allow(&["*".into()], FilterTree(&qc.filters), None)?
                        .into_filter_expr();
                    let group_exprs = dm
                        .allow(&["*".into()], Columns(&qc.groups), None)?
                        .into_exprs();
                    dm.table("orders")
                        .await?
                        .filter(filter_expr)?
                        .aggregate(group_exprs, vec![count(col("amount")).alias("order_count")])
                })
            }),
        )
        .await
        .unwrap();

        let qc = QueryContext::new(
            vec!["order_count".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("orders.region".into(), "asc".into())]),
            None,
            None,
            None,
        )
        .unwrap();

        let result = dm.collect(&qc).await.unwrap();
        let batch = &result[0];
        let count_col = batch
            .column_by_name("order_count")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        let region_col = batch
            .column_by_name("region")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::StringArray>()
            .unwrap();

        assert_eq!(region_col.value(0), "EU");
        assert_eq!(count_col.value(0), 2);
        assert_eq!(region_col.value(1), "US");
        assert_eq!(count_col.value(1), 3);
    }
}
