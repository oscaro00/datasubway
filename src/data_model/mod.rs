use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
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
    pre_aggregations::{
        PreAggregation, agg_needed_components, read_parquet_row_count, resolve_fresh_pre_agg_path,
        resolve_pre_agg_path,
    },
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
    pub(crate) pre_aggs: Option<std::sync::RwLock<Vec<PreAggregation>>>,
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

        let dm = DataModel(Arc::new(DataModelInner {
            ctx,
            table_providers: tables,
            joins,
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
            pre_aggs: if pre_aggs.is_empty() {
                None
            } else {
                Some(std::sync::RwLock::new(pre_aggs))
            },
            pre_agg_path,
        }));

        // Best-effort: restore bookkeeping from any existing pointer files on disk.
        if let (Some(path), Some(lock)) = (dm.0.pre_agg_path.as_deref(), dm.0.pre_aggs.as_ref()) {
            let mut pre_aggs = lock.write().unwrap();
            for pa in pre_aggs.iter_mut() {
                if let Some(versioned) = resolve_pre_agg_path(path, &pa.name) {
                    if let Some(rc) = read_parquet_row_count(&versioned) {
                        pa.row_count = rc;
                    }
                    if let Ok(meta) = std::fs::metadata(&versioned)
                        && let Ok(modified) = meta.modified()
                    {
                        let secs = modified
                            .duration_since(std::time::UNIX_EPOCH)
                            .unwrap_or_default()
                            .as_secs();
                        pa.written_at = Some(secs.to_string());
                    }
                }
            }
        }

        dm
    }

    // ── Public API ────────────────────────────────────────────────────────────

    pub fn table(&self, table_name: &str, use_pre_agg: bool) -> DataFrameRecorder {
        assert!(
            self.0.table_providers.contains_key(table_name),
            "table '{table_name}' not found in DataModel"
        );
        DataFrameRecorder::new(table_name.to_string(), self.clone(), use_pre_agg)
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

    /// Returns each registered table's name paired with its Arrow schema
    /// (column names and types), sorted by table name.
    pub fn schemas(&self) -> Vec<(String, SchemaRef)> {
        let mut schemas: Vec<(String, SchemaRef)> = self
            .0
            .table_providers
            .iter()
            .map(|(name, provider)| (name.clone(), provider.schema()))
            .collect();
        schemas.sort_by(|a, b| a.0.cmp(&b.0));
        schemas
    }

    /// Flatten every registered table's schema into the full set of qualified
    /// `table.column` names known to this model.
    pub(crate) fn known_qualified_columns(&self) -> HashSet<String> {
        self.0
            .table_providers
            .iter()
            .flat_map(|(name, provider)| {
                let prefix = format!("{name}.");
                provider
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| {
                        if f.name().starts_with(&prefix) {
                            f.name().to_string()
                        } else {
                            format!("{prefix}{}", f.name())
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
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
        if pre_agg_allowed
            && !agg_cols.is_empty()
            && let (Some(pre_aggs_lock), Some(path)) =
                (&self.0.pre_aggs, self.0.pre_agg_path.as_deref())
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

            let pre_aggs = pre_aggs_lock.read().unwrap();
            let mut candidates: Vec<&PreAggregation> = pre_aggs
                .iter()
                .filter(|pa| pa.covers(&non_agg_vec, &agg_map))
                .collect();
            candidates.sort_by_key(|pa| pa.row_count);

            'candidates: for candidate in &candidates {
                let Some(pre_agg_file) = resolve_fresh_pre_agg_path(path, &candidate.name, None)
                else {
                    debug!(pre_agg = %candidate.name, "no current pointer or file missing, trying next");
                    continue 'candidates;
                };
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

/// Convert validated `(column, direction)` sort pairs into DataFusion sort expressions.
pub(crate) fn sort_exprs(sorts: &[(String, String)]) -> Vec<datafusion::logical_expr::SortExpr> {
    sorts
        .iter()
        .map(|(c, d)| {
            datafusion::logical_expr::Expr::Column(datafusion::common::Column::from_name(
                c.as_str(),
            ))
            .sort(d != "desc", true)
        })
        .collect()
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
    fn test_schemas_returns_sorted_table_schemas() {
        let orders_schema = Arc::new(Schema::new(vec![
            Field::new("date", DataType::Utf8, true),
            Field::new("amount", DataType::Float64, true),
        ]));
        let orders_provider: Arc<dyn TableProvider> =
            Arc::new(MemTable::try_new(orders_schema.clone(), vec![vec![]]).unwrap());

        let customers_schema =
            Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, true)]));
        let customers_provider: Arc<dyn TableProvider> =
            Arc::new(MemTable::try_new(customers_schema.clone(), vec![vec![]]).unwrap());

        let dm = DataModel::new(
            HashMap::from([
                ("orders".to_string(), orders_provider),
                ("customers".to_string(), customers_provider),
            ]),
            JoinGraph::new(&[]).unwrap(),
            vec![],
            None,
        );

        let schemas = dm.schemas();
        let names: Vec<&str> = schemas.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, vec!["customers", "orders"]);

        let (_, customers_out) = &schemas[0];
        assert_eq!(customers_out, &customers_schema);

        let (_, orders_out) = &schemas[1];
        assert_eq!(orders_out, &orders_schema);
    }

    #[test]
    fn test_write_pre_agg_creates_versioned_file() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let base = tmp.path().to_str().unwrap();

        // Pointer file must exist.
        assert!(tmp.path().join("daily_revenue.current").exists());

        // Pointer must resolve to an existing versioned parquet.
        let versioned = resolve_pre_agg_path(base, "daily_revenue")
            .expect("pointer should resolve to a versioned path");
        assert!(
            std::path::Path::new(&versioned).exists(),
            "versioned path does not exist: {versioned}"
        );

        // Bookkeeping must be populated after write.
        let pre_aggs = dm.0.pre_aggs.as_ref().unwrap().read().unwrap();
        let pa = pre_aggs.iter().find(|p| p.name == "daily_revenue").unwrap();
        assert!(pa.row_count > 0, "row_count should be > 0 after write");
        assert!(
            pa.written_at.is_some(),
            "written_at should be Some after write"
        );
    }

    #[test]
    fn test_write_twice_keeps_two_versions_then_purge() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        std::thread::sleep(std::time::Duration::from_millis(5));
        dm.write_pre_aggs(&["daily_revenue"]).unwrap();

        let base = tmp.path().to_str().unwrap();
        let versioned_entries: Vec<_> = std::fs::read_dir(base)
            .unwrap()
            .flatten()
            .filter(|e| {
                let s = e.file_name();
                let s = s.to_string_lossy();
                s.starts_with("daily_revenue.") && s.ends_with(".parquet")
            })
            .collect();
        assert!(
            versioned_entries.len() >= 2,
            "expected at least two versioned entries after two writes"
        );

        dm.purge_old_pre_agg_versions(&["daily_revenue"]).unwrap();

        let remaining: Vec<_> = std::fs::read_dir(base)
            .unwrap()
            .flatten()
            .filter(|e| {
                let s = e.file_name();
                let s = s.to_string_lossy();
                s.starts_with("daily_revenue.") && s.ends_with(".parquet")
            })
            .collect();
        assert_eq!(
            remaining.len(),
            1,
            "expected exactly one versioned entry after purge"
        );
    }

    #[test]
    fn test_new_restores_bookkeeping_from_existing_pointer() {
        // Write a pre-agg with one DataModel instance, then construct a fresh DataModel
        // pointing at the same path and verify row_count / written_at are restored from disk.
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let path = tmp.path().to_str().unwrap().to_string();

        let dm2 = make_orders_dm(Some(path), vec![daily_revenue_pre_agg()]);

        let pre_aggs = dm2.0.pre_aggs.as_ref().unwrap().read().unwrap();
        let pa = pre_aggs.iter().find(|p| p.name == "daily_revenue").unwrap();
        assert!(
            pa.row_count > 0,
            "row_count should be restored from parquet metadata on DataModel::new"
        );
        assert!(
            pa.written_at.is_some(),
            "written_at should be restored from file mtime on DataModel::new"
        );
    }

    #[test]
    fn test_corrupt_pointer_resolves_to_missing_path() {
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let base = tmp.path().to_str().unwrap();

        // Overwrite the pointer to reference a non-existent versioned file.
        std::fs::write(
            format!("{base}/daily_revenue.current"),
            "daily_revenue.0.parquet",
        )
        .unwrap();

        // resolve_pre_agg_path returns the path named in the pointer even when it doesn't exist.
        // The read path's subsequent exists() check will see it as missing and fall through.
        let resolved =
            resolve_pre_agg_path(base, "daily_revenue").expect("pointer file should still parse");
        assert!(
            !std::path::Path::new(&resolved).exists(),
            "resolved path should not exist for corrupt pointer"
        );
    }
}
