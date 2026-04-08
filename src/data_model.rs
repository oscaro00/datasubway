//! Standalone Rust DataModel — semantic layer using DataFusion DataFrame API.
//!
//! Enables pure-Rust usage without Python. Measures are closures that build
//! DataFusion DataFrames using `allow()`/`exclude()` from `column_context`.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::execution::context::SessionContext;
use datafusion::prelude::*;
use tokio::runtime::Runtime;

use crate::model::column_context;
use crate::model::combine_measures;
use crate::model::joins::{Join, JoinGraph, JoinHow};
use crate::model::pre_agg::PreAggregation;
use crate::model::query_context::{MeasureMetadata, QueryContext};
use crate::optimizer::auto_join_rule::AutoJoinRule;
use crate::optimizer::eliminate_joins_rule::EliminateUnusedJoins;
use crate::optimizer::pre_agg_rule::PreAggSubstitution;

/// A measure is a closure that takes a QueryContext and DataModel reference,
/// and returns a DataFusion DataFrame.
///
/// Measures use the DataFusion DataFrame API with `allow()`/`exclude()`:
/// ```rust,ignore
/// Arc::new(|qc: &QueryContext, dm: &DataModel| {
///     let table = dm.table("orders")?;
///     let filter = allow(&["*".into()], ColumnInput::FilterTree(&qc.filters))?.into_filter_expr();
///     let groups = allow(&["*".into()], ColumnInput::Columns(&qc.groups))?.into_exprs();
///     Ok(table
///         .filter(filter)?
///         .aggregate(groups, vec![sum(col("amount")).alias("revenue")])?)
/// })
/// ```
pub type MeasureFn =
    Arc<dyn Fn(&QueryContext, &DataModel) -> Result<DataFrame, DataFusionError> + Send + Sync>;

/// Standalone Rust semantic layer model backed by DataFusion.
///
/// Manages data sources, joins, measures, pre-aggregations, and query execution.
pub struct DataModel {
    ctx: SessionContext,
    rt: Arc<Runtime>,
    join_graph: Option<JoinGraph>,
    table_schemas: HashMap<String, Vec<String>>,
    measures: HashMap<String, MeasureFn>,
    measure_output_cols: HashMap<String, Vec<String>>,
    pre_agg_objects: Vec<PreAggregation>,
}

impl DataModel {
    /// Create a new DataModel with an empty SessionContext.
    pub fn new() -> Result<Self, DataFusionError> {
        let rt = Runtime::new().map_err(|e| DataFusionError::External(Box::new(e)))?;
        Ok(DataModel {
            ctx: SessionContext::new(),
            rt: Arc::new(rt),
            join_graph: None,
            table_schemas: HashMap::new(),
            measures: HashMap::new(),
            measure_output_cols: HashMap::new(),
            pre_agg_objects: Vec::new(),
        })
    }

    /// Register a RecordBatch as a named table.
    pub fn register_record_batch(
        &mut self,
        name: &str,
        batch: RecordBatch,
    ) -> Result<(), DataFusionError> {
        let schema_names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| format!("{}.{}", name, f.name()))
            .collect();

        let schema = batch.schema();
        let mem_table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])?;
        self.rt
            .block_on(async { self.ctx.register_table(name, Arc::new(mem_table)) })?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register a Parquet file as a named table.
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<(), DataFusionError> {
        self.rt.block_on(async {
            self.ctx
                .register_parquet(name, path, Default::default())
                .await
        })?;
        // Extract schema from the registered table
        let schema_names = self.rt.block_on(async {
            let df = self.ctx.table(name).await?;
            Ok::<Vec<String>, DataFusionError>(
                df.schema()
                    .fields()
                    .iter()
                    .map(|f| format!("{}.{}", name, f.name().clone()))
                    .collect(),
            )
        })?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register a CSV file as a named table.
    pub fn register_csv(&mut self, name: &str, path: &str) -> Result<(), DataFusionError> {
        self.rt
            .block_on(async { self.ctx.register_csv(name, path, Default::default()).await })?;
        let schema_names = self.rt.block_on(async {
            let df = self.ctx.table(name).await?;
            Ok::<Vec<String>, DataFusionError>(
                df.schema()
                    .fields()
                    .iter()
                    .map(|f| format!("{}.{}", name, f.name().clone()))
                    .collect(),
            )
        })?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Set the join graph from a list of Join specifications.
    pub fn set_joins(&mut self, joins: &[Join]) -> Result<(), String> {
        self.join_graph = Some(JoinGraph::new(joins)?);
        Ok(())
    }

    /// Set pre-aggregation objects.
    pub fn set_pre_aggregations(&mut self, pre_aggs: Vec<PreAggregation>) {
        self.pre_agg_objects = pre_aggs;
    }

    /// Register a measure function.
    ///
    /// The measure is probed with an empty QueryContext to extract output columns.
    pub fn register_measure(
        &mut self,
        name: &str,
        measure_fn: MeasureFn,
    ) -> Result<(), DataFusionError> {
        // Probe with empty context to extract output columns
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

        let mut output_cols = Vec::new();
        if let Ok(df) = measure_fn(&probe_qc, self) {
            output_cols = df
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
        }

        self.measures.insert(name.to_string(), measure_fn);
        self.measure_output_cols
            .insert(name.to_string(), output_cols);
        Ok(())
    }

    /// Get a DataFusion DataFrame for a registered table.
    ///
    /// Eagerly pre-joins all reachable tables via the JoinGraph so that
    /// cross-table column references work in the DataFrame API (which validates
    /// schemas at expression-building time, before the optimizer runs).
    pub fn table(&self, name: &str) -> Result<DataFrame, DataFusionError> {
        let mut inner = self.rt.block_on(self.ctx.table(name))?;

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
                    let right_df = self.rt.block_on(self.ctx.table(&step.right))?;
                    let left_on: Vec<&str> = step.left_on.iter().map(|s| s.as_str()).collect();
                    let right_on: Vec<&str> = step.right_on.iter().map(|s| s.as_str()).collect();
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

    /// Access column_context::allow for use in measures.
    pub fn allow(
        &self,
        patterns: &[String],
        input: column_context::ColumnInput,
    ) -> Result<column_context::ColumnOutput, DataFusionError> {
        column_context::allow(patterns, input)
    }

    /// Access column_context::exclude for use in measures.
    pub fn exclude(
        &self,
        patterns: &[String],
        input: column_context::ColumnInput,
    ) -> Result<column_context::ColumnOutput, DataFusionError> {
        column_context::exclude(patterns, input)
    }

    /// Validate the QueryContext, register optimizer rules, and build a
    /// DataFrame for each requested measure (without collecting).
    fn prepare_measure_dfs(
        &self,
        qc: &QueryContext,
    ) -> Result<Vec<(String, DataFrame)>, DataFusionError> {
        // Validate
        let measure_metadata: Vec<MeasureMetadata> = self
            .measures
            .keys()
            .map(|name| MeasureMetadata {
                name: name.clone(),
                output_columns: self
                    .measure_output_cols
                    .get(name)
                    .cloned()
                    .unwrap_or_default(),
            })
            .collect();
        let all_cols: HashSet<String> = self.all_columns().into_iter().collect();
        qc.validate(&measure_metadata, &all_cols)
            .map_err(|e| DataFusionError::Plan(e))?;

        // Register optimizer rules if applicable
        if !self.pre_agg_objects.is_empty() && qc.use_pre_agg {
            let rule = PreAggSubstitution::new(self.pre_agg_objects.clone());
            self.ctx.add_optimizer_rule(Arc::new(rule));
        }
        if let Some(ref jg) = self.join_graph {
            let rule = AutoJoinRule::new(jg.clone(), self.table_schemas.clone());
            self.ctx.add_optimizer_rule(Arc::new(rule));
            let elim_rule = EliminateUnusedJoins::new(jg.clone());
            self.ctx.add_optimizer_rule(Arc::new(elim_rule));
        }

        // Build each measure DataFrame (no collect)
        let mut measure_dfs: Vec<(String, DataFrame)> = Vec::new();
        for measure_name in &qc.measures {
            let measure_fn = self.measures.get(measure_name).ok_or_else(|| {
                DataFusionError::Plan(format!("Unknown measure: '{}'", measure_name))
            })?;
            let df = measure_fn(qc, self)?;
            measure_dfs.push((measure_name.clone(), df));
        }

        if measure_dfs.is_empty() {
            return Err(DataFusionError::Plan("No measure results produced".into()));
        }

        Ok(measure_dfs)
    }

    /// Collect query results as RecordBatches.
    ///
    /// 1. Validates the QueryContext
    /// 2. Registers optimizer rules
    /// 3. Calls each measure function to produce DataFrames
    /// 4. Combines measures (joins, havings, sorts, limit/offset)
    /// 5. Collects and returns results
    pub fn collect(&self, qc: &QueryContext) -> Result<Vec<RecordBatch>, DataFusionError> {
        let measure_dfs = self.prepare_measure_dfs(qc)?;
        let borrowed: Vec<(&str, DataFrame)> = measure_dfs
            .iter()
            .map(|(n, df)| (n.as_str(), df.clone()))
            .collect();
        combine_measures::combine_measure_results(&self.rt, borrowed, qc)
    }

    /// Return an explain DataFrame for the query plan without executing it.
    ///
    /// Works like `collect()` but calls DataFusion's `explain()` instead of
    /// collecting results. Accepts the same `verbose` and `analyze` flags
    /// as `DataFrame::explain()`.
    pub fn explain(
        &self,
        qc: &QueryContext,
        verbose: bool,
        analyze: bool,
    ) -> Result<DataFrame, DataFusionError> {
        let measure_dfs = self.prepare_measure_dfs(qc)?;
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
    use crate::model::joins::JoinDirection;
    use arrow::datatypes::{DataType, Field, Schema};
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

    #[test]
    fn test_basic_query_no_groups() {
        let mut dm = DataModel::new().unwrap();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|_qc, dm| {
                dm.table("orders")?
                    .aggregate(vec![], vec![sum(col("amount")).alias("revenue")])
            }),
        )
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

        let result = dm.collect(&qc).unwrap();
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

    #[test]
    fn test_query_with_groups() {
        let mut dm = DataModel::new().unwrap();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|qc, dm| {
                let group_exprs = column_context::allow(
                    &["*".into()],
                    column_context::ColumnInput::Columns(&qc.groups),
                )
                .unwrap()
                .into_exprs();
                dm.table("orders")?
                    .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
            }),
        )
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

        let result = dm.collect(&qc).unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[test]
    fn test_query_with_having_and_sort() {
        let mut dm = DataModel::new().unwrap();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|qc, dm| {
                let group_exprs = column_context::allow(
                    &["*".into()],
                    column_context::ColumnInput::Columns(&qc.groups),
                )
                .unwrap()
                .into_exprs();
                dm.table("orders")?
                    .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
            }),
        )
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

        let result = dm.collect(&qc).unwrap();
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

    #[test]
    fn test_multi_measure() {
        let mut dm = DataModel::new().unwrap();
        dm.register_record_batch("orders", make_orders_batch())
            .unwrap();

        dm.register_measure(
            "revenue",
            Arc::new(|_qc, dm| {
                dm.table("orders")?
                    .aggregate(vec![], vec![sum(col("amount")).alias("revenue")])
            }),
        )
        .unwrap();

        dm.register_measure(
            "total_quantity",
            Arc::new(|_qc, dm| {
                dm.table("orders")?
                    .aggregate(vec![], vec![sum(col("quantity")).alias("total_quantity")])
            }),
        )
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

        let result = dm.collect(&qc).unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[test]
    fn test_auto_join() {
        let mut dm = DataModel::new().unwrap();

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
                let group_exprs = column_context::allow(
                    &["*".into()],
                    column_context::ColumnInput::Columns(&qc.groups),
                )
                .unwrap()
                .into_exprs();
                dm.table("player_stats")?
                    .aggregate(group_exprs, vec![sum(col("score")).alias("total_score")])
            }),
        )
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

        let result = dm.collect(&qc).unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[test]
    fn test_unreferenced_join_eliminated() {
        // Setup: 3 tables, but the query only references player_stats and players.
        // "teams" is joined into the graph but never referenced in the query.
        // The EliminateUnusedJoins optimizer rule should remove the teams join.
        let mut dm = DataModel::new().unwrap();

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
                let group_exprs = column_context::allow(
                    &["*".into()],
                    column_context::ColumnInput::Columns(&qc.groups),
                )
                .unwrap()
                .into_exprs();
                dm.table("player_stats")?
                    .aggregate(group_exprs, vec![sum(col("score")).alias("total_score")])
            }),
        )
        .unwrap();

        // Query through DataModel::collect() which registers optimizer rules
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

        let result = dm.collect(&qc).unwrap();

        // Verify correct results: Alice=40, Bob=20
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2, "Should have 2 rows (Alice and Bob)");

        // Also verify the plan does NOT contain "teams" by building a DF
        // with the optimizer registered (it was registered by collect() above)
        let df = dm.table("player_stats").unwrap();
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
}
