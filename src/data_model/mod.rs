use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::datasource::provider_as_source;
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::ExplainOption;
use datafusion::logical_expr::{JoinType, LogicalPlan, LogicalPlanBuilder, SubqueryAlias};
use datafusion::prelude::{DataFrame, ParquetReadOptions, SessionContext};
use tracing::debug;

use crate::model_components::{
    agg_context::AggContext,
    column_values_context::ColumnValuesContext,
    joins::{JoinGraph, JoinHow},
    measures::{
        DfMeasure, MeasureMetadata, extract_df_measure_metadata, validate_df_measure_structure,
    },
    pre_aggregations::{PreAggregation, agg_needed_components},
    select_context::SelectContext,
};
use crate::wrappers::datafusion::{
    aggregate_with_metadata::AggregateWithMetadataPlanner, dataframe_recorder::DataFrameRecorder,
    dataframe_wrapper::DataFrameWrapper,
};

mod agg_builder;
mod column_values_builder;
mod merge_optimizer;
mod select_builder;

// Re-export flatten_df under the historic name for callers (e.g. integration tests)
// that need to bring a raw DataFusion DataFrame into the same flat-alias schema
// that DataModel.execute() produces.
pub use agg_builder::flatten_df as normalize_schema;

// ── Public types ─────────────────────────────────────────────────────────────

pub enum DataOutput {
    Data(Vec<RecordBatch>),
    Explanation(String),
}

impl std::fmt::Debug for DataOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataOutput::Data(batches) => write!(f, "DataOutput::Data({} batches)", batches.len()),
            DataOutput::Explanation(s) => write!(f, "DataOutput::Explanation({s:?})"),
        }
    }
}

pub enum DataQuery {
    Agg(AggContext),
    View(SelectContext),
    ColumnValues(ColumnValuesContext),
}

// ── Internal struct ───────────────────────────────────────────────────────────

pub(crate) struct DataModelInner {
    pub(crate) ctx: SessionContext,
    pub(crate) table_providers: HashMap<String, Arc<dyn TableProvider>>,
    pub(crate) joins: JoinGraph,
    pub(crate) measures: HashMap<String, DfMeasure>,
    pub(crate) measure_metadata: HashMap<String, MeasureMetadata>,
    pub(crate) pre_aggs: Option<Vec<PreAggregation>>,
    pub(crate) pre_agg_path: Option<String>,
}

#[derive(Clone)]
pub struct DataModel(pub(crate) Arc<DataModelInner>);

impl DataModel {
    pub fn new(
        tables: HashMap<String, Arc<dyn TableProvider>>,
        joins: JoinGraph,
        pre_aggs: Vec<PreAggregation>,
        pre_agg_path: Option<String>,
    ) -> DataModel {
        let state = SessionStateBuilder::new_with_default_features()
            .with_query_planner(Arc::new(AggregateWithMetadataPlanner))
            .build();
        let ctx = SessionContext::new_with_state(state);

        for (name, provider) in &tables {
            ctx.register_table(name, Arc::clone(provider))
                .unwrap_or_else(|e| panic!("failed to register table '{name}': {e}"));
        }

        let mut seen = HashSet::new();
        for pa in &pre_aggs {
            if !seen.insert(pa.name.as_str()) {
                panic!("duplicate pre-aggregation name '{}'", pa.name);
            }
        }

        DataModel(Arc::new(DataModelInner {
            ctx,
            table_providers: tables,
            joins,
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
            pre_aggs: if pre_aggs.is_empty() {
                None
            } else {
                Some(pre_aggs)
            },
            pre_agg_path,
        }))
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn table(&self, table_name: &str) -> DataFrameRecorder {
        assert!(
            self.0.table_providers.contains_key(table_name),
            "table '{table_name}' not found in DataModel"
        );
        DataFrameRecorder::new(table_name.to_string(), self.clone())
    }

    pub fn add_measure(&mut self, measure: DfMeasure) -> Result<(), String> {
        if self.0.measures.contains_key(&measure.name) {
            return Err(format!("measure '{}' already registered", measure.name));
        }
        let name = measure.name.clone();
        let metadata = self.validate_and_extract_measure(&measure)?;
        let inner = Arc::get_mut(&mut self.0).expect("DataModel cannot be modified while shared");
        inner.measures.insert(name.clone(), measure);
        inner.measure_metadata.insert(name, metadata);
        Ok(())
    }

    pub fn measure_metadata(&self, name: &str) -> Option<&MeasureMetadata> {
        self.0.measure_metadata.get(name)
    }

    pub fn can_join(&self, base: &str, target: &str) -> bool {
        self.0.joins.find_path(base, target).is_some()
    }

    pub async fn execute(&self, q: &DataQuery) -> Result<DataOutput, String> {
        let df = match q {
            DataQuery::Agg(ctx) => self.build_agg_frame(ctx)?,
            DataQuery::View(ctx) => self.build_select_frame(ctx)?,
            DataQuery::ColumnValues(ctx) => self.build_column_values_frame(ctx)?,
        };

        df.collect()
            .await
            .map(DataOutput::Data)
            .map_err(|e| format!("execution failed: {e}"))
    }

    pub fn explain(&self, q: &DataQuery, options: ExplainOption) -> Result<DataFrame, String> {
        let df = match q {
            DataQuery::Agg(ctx) => self.build_agg_frame(ctx)?,
            DataQuery::View(ctx) => self.build_select_frame(ctx)?,
            DataQuery::ColumnValues(ctx) => self.build_column_values_frame(ctx)?,
        };
        df.explain_with_options(options)
            .map_err(|e| format!("explain failed: {e}"))
    }

    pub fn display_graphviz(&self, q: &DataQuery) -> Result<String, String> {
        let df = match q {
            DataQuery::Agg(ctx) => self.build_agg_frame(ctx)?,
            DataQuery::View(ctx) => self.build_select_frame(ctx)?,
            DataQuery::ColumnValues(ctx) => self.build_column_values_frame(ctx)?,
        };
        Ok(df.into_unoptimized_plan().display_graphviz().to_string())
    }

    // ── Internal ──────────────────────────────────────────────────────────────

    fn validate_and_extract_measure(&self, measure: &DfMeasure) -> Result<MeasureMetadata, String> {
        let stub_qc = AggContext::stub();
        let df = measure
            .call(self, &stub_qc)
            .map_err(|e| format!("measure '{}' failed to build: {e}", measure.name))?;
        validate_df_measure_structure(&df)?;
        extract_df_measure_metadata(&df, &measure.name)
    }

    /// Build a base DataFrame for a table, applying joins for any external columns.
    /// Uses a pre-aggregation file if available and allowed.
    pub(crate) fn get_df_table(
        &self,
        table_name: &str,
        non_agg_cols: &HashSet<String>,
        agg_cols: &HashMap<String, Vec<String>>,
        pre_agg_allowed: bool,
    ) -> datafusion::common::Result<DataFrameWrapper> {
        if pre_agg_allowed && !agg_cols.is_empty() {
            if let (Some(pre_aggs), Some(path)) = (&self.0.pre_aggs, self.0.pre_agg_path.as_deref())
            {
                let non_agg_vec: Vec<String> = non_agg_cols.iter().cloned().collect();
                let agg_map: HashMap<String, HashSet<String>> = agg_cols
                    .iter()
                    .map(|(col, agg_names)| {
                        let components: HashSet<String> = agg_names
                            .iter()
                            .filter_map(|name| agg_needed_components(name))
                            .flatten()
                            .map(|s| s.to_string())
                            .collect();
                        (col.clone(), components)
                    })
                    .collect();

                let mut candidates: Vec<&PreAggregation> = pre_aggs
                    .iter()
                    .filter(|pa| pa.covers(&non_agg_vec, &agg_map))
                    .collect();
                candidates.sort_by_key(|pa| pa.row_count);

                'candidates: for candidate in &candidates {
                    let pre_agg_file = format!("{path}/{}.parquet", candidate.name);
                    if !std::path::Path::new(&pre_agg_file).exists() {
                        debug!(pre_agg = %candidate.name, "parquet not found, trying next");
                        continue 'candidates;
                    }
                    debug!(pre_agg = %candidate.name, table = %table_name, "using pre-aggregation");
                    if let Ok(raw_df) = self.read_parquet_sync(&pre_agg_file) {
                        match SubqueryAlias::try_new(
                            Arc::new(raw_df.into_unoptimized_plan()),
                            candidate.name.as_str(),
                        ) {
                            Ok(alias) => {
                                let df = DataFrame::new(
                                    self.0.ctx.state(),
                                    LogicalPlan::SubqueryAlias(alias),
                                );
                                return Ok(DataFrameWrapper {
                                    inner: df,
                                    from_pre_agg: true,
                                    pre_agg_name: Some(candidate.name.clone()),
                                });
                            }
                            Err(e) => {
                                debug!(pre_agg = %candidate.name, err = %e, "SubqueryAlias failed, trying next");
                            }
                        }
                    }
                    debug!(pre_agg = %candidate.name, "failed to read parquet, trying next");
                }
            }
        }

        debug!(table = %table_name, "scanning base table");
        let mut df = self.scan_table(table_name)?;

        let all_col_names = non_agg_cols
            .iter()
            .map(|s| s.as_str())
            .chain(agg_cols.keys().map(|s| s.as_str()));
        let mut external_tables: Vec<String> = all_col_names
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .filter(|t| t != table_name)
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        external_tables.sort();

        let mut joined: HashSet<String> = HashSet::from([table_name.to_string()]);
        for ext in &external_tables {
            if let Some(path) = self.0.joins.find_path(table_name, ext) {
                for join in path {
                    if joined.contains(&join.right) {
                        continue;
                    }
                    let right_df = self.scan_table(&join.right)?;
                    let join_type = match join.how {
                        JoinHow::Left => JoinType::Left,
                        JoinHow::Inner => JoinType::Inner,
                    };
                    let left_on: Vec<&str> = join.left_on.iter().map(|s| s.as_str()).collect();
                    let right_on: Vec<&str> = join.right_on.iter().map(|s| s.as_str()).collect();
                    df = df.join(right_df, join_type, &left_on, &right_on, None)?;
                    joined.insert(join.right.clone());
                }
            }
        }

        Ok(DataFrameWrapper {
            inner: df,
            from_pre_agg: false,
            pre_agg_name: None,
        })
    }

    pub(crate) fn scan_table(&self, table_name: &str) -> datafusion::common::Result<DataFrame> {
        let provider = self
            .0
            .table_providers
            .get(table_name)
            .unwrap_or_else(|| panic!("table '{table_name}' not found in DataModel"));

        let source = provider_as_source(Arc::clone(provider));
        let plan = LogicalPlanBuilder::scan(table_name, source, None)?.build()?;
        Ok(DataFrame::new(self.0.ctx.state(), plan))
    }

    pub(crate) fn read_parquet_sync(
        &self,
        file_path: &str,
    ) -> datafusion::common::Result<DataFrame> {
        let ctx = self.0.ctx.clone();
        let path = file_path.to_string();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(ctx.read_parquet(path, ParquetReadOptions::default()))
            }),
            Err(_) => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(ctx.read_parquet(path, ParquetReadOptions::default())),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::arrow::array::{Float64Array, StringArray};
    use datafusion::arrow::datatypes::{DataType, Field, Schema};
    use datafusion::arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use tempfile::TempDir;

    fn make_orders_dm(pre_agg_path: Option<String>, pre_aggs: Vec<PreAggregation>) -> DataModel {
        let schema = Arc::new(Schema::new(vec![
            Field::new("date", DataType::Utf8, true),
            Field::new("region", DataType::Utf8, true),
            Field::new("amount", DataType::Float64, true),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec![
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-02",
                    "2024-01-02",
                ])),
                Arc::new(StringArray::from(vec!["north", "south", "north", "south"])),
                Arc::new(Float64Array::from(vec![100.0, 200.0, 150.0, 250.0])),
            ],
        )
        .unwrap();
        let provider: Arc<dyn TableProvider> =
            Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
        DataModel::new(
            HashMap::from([("orders".to_string(), provider)]),
            JoinGraph::new(&[]).unwrap(),
            pre_aggs,
            pre_agg_path,
        )
    }

    fn daily_revenue_pre_agg() -> PreAggregation {
        PreAggregation::new(
            "daily_revenue".into(),
            vec!["orders.date".into(), "orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["sum".into(), "mean".into()])]),
        )
        .unwrap()
    }

    fn write_and_get_tmp(pa: PreAggregation) -> (DataModel, TempDir) {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        let dm = make_orders_dm(Some(path), vec![pa]);
        dm.write_pre_aggs(&["daily_revenue"]).unwrap();
        (dm, tmp)
    }

    #[test]
    fn test_write_pre_agg_creates_file() {
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        assert!(tmp.path().join("daily_revenue.parquet").exists());
    }
}
