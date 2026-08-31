use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::datatypes::SchemaRef;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion::datasource::{provider_as_source, source_as_provider};
use datafusion::execution::session_state::SessionStateBuilder;
use datafusion::logical_expr::ExplainOption;
use datafusion::logical_expr::{JoinType, LogicalPlan, LogicalPlanBuilder};
use datafusion::prelude::{DataFrame, SessionConfig, SessionContext};
use tracing::debug;

use crate::model_components::{
    agg_context::AggContext,
    column_values_context::ColumnValuesContext,
    joins::{JoinGraph, JoinHow},
    measures::{
        DfMeasure, MeasureMetadata, extract_df_measure_metadata, validate_df_measure_structure,
    },
    pre_agg_store::{
        LeaseScope, PRE_AGG_SCHEMA, PreAggSchemaProvider, PreAggStore, PreAggVersion,
        active_max_age, record_lease,
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
    /// Version registry + reclamation for pre-aggregation files. `None` when the
    /// model has no pre-aggregations or no path to write them to.
    pub(crate) pre_agg_store: Option<Arc<PreAggStore>>,
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
            // So `information_schema.tables` and `SHOW TABLES` can see what is
            // registered — the point of giving pre-aggs a schema of their own is
            // being able to enumerate and query them.
            .with_config(SessionConfig::new().with_information_schema(true))
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

        // `PreAggStore::load` rehydrates the current version of each definition
        // from the `<name>.current` pointers already on disk.
        let pre_agg_store = match (pre_aggs.is_empty(), pre_agg_path) {
            (false, Some(path)) => Some(Arc::new(PreAggStore::load(path.into(), pre_aggs))),
            _ => None,
        };

        // Pre-aggs get a schema of their own rather than sitting alongside the
        // source tables, so a pre-agg and a table may share a name without their
        // `TableScan`s sharing a qualifier. It also makes them addressable:
        // `SELECT * FROM pre_agg.<name>` returns whatever version is current.
        if let Some(store) = &pre_agg_store {
            let default_catalog = ctx.state().config_options().catalog.default_catalog.clone();
            let catalog = ctx
                .catalog(&default_catalog)
                .unwrap_or_else(|| panic!("default catalog '{default_catalog}' not registered"));
            catalog
                .register_schema(
                    PRE_AGG_SCHEMA,
                    Arc::new(PreAggSchemaProvider(Arc::clone(store))),
                )
                .unwrap_or_else(|e| panic!("failed to register '{PRE_AGG_SCHEMA}' schema: {e}"));
        }

        DataModel(Arc::new(DataModelInner {
            ctx,
            table_providers: tables,
            joins,
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
            pre_agg_store,
        }))
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
        // Any pre-agg version bound while building the plan is pinned by the
        // scope, and `_leases` keeps it alive across the collect below — the file
        // is not opened until then, so reclamation must not unlink it in between.
        let scope = LeaseScope::enter(query_max_age(q));
        let df = self.build_frame(q);
        let _leases = scope.take();
        let df = df?;

        df.collect()
            .await
            .map(DataOutput::Data)
            .map_err(|e| format!("execution failed: {e}"))
    }

    /// Run SQL against this model's session.
    ///
    /// Source tables are addressable by their registered name; pre-aggregations
    /// live in the `pre_agg` schema (`SELECT * FROM pre_agg.daily_revenue`) and
    /// resolve to whichever version is current, over the physical column names.
    /// Mainly a window onto what the optimizer is substituting — the measure path
    /// does not go through here.
    pub async fn sql(&self, query: &str) -> Result<Vec<RecordBatch>, String> {
        let df = self
            .0
            .ctx
            .sql(query)
            .await
            .map_err(|e| format!("sql failed: {e}"))?;

        // Planning is async here, so the thread-local `LeaseScope` the DataFrame
        // path relies on cannot span it. Pin the versions straight out of the plan
        // that resolution just built instead, and hold them across `collect()` —
        // the physical plan carries file paths, not providers, so past this point
        // nothing else keeps the files from being reclaimed.
        let _leases = pre_agg_providers(df.logical_plan());

        df.collect()
            .await
            .map_err(|e| format!("execution failed: {e}"))
    }

    pub fn explain(&self, q: &DataQuery, options: ExplainOption) -> Result<DataFrame, String> {
        let scope = LeaseScope::enter(query_max_age(q));
        let df = self.build_frame(q);
        let _leases = scope.take();

        df?.explain_with_options(options)
            .map_err(|e| format!("explain failed: {e}"))
    }

    pub fn display_graphviz(&self, q: &DataQuery) -> Result<String, String> {
        let scope = LeaseScope::enter(query_max_age(q));
        let df = self.build_frame(q);
        drop(scope);
        Ok(df?.into_unoptimized_plan().display_graphviz().to_string())
    }

    fn build_frame(&self, q: &DataQuery) -> Result<DataFrame, String> {
        match q {
            DataQuery::Agg(ctx) => self.build_agg_frame(ctx),
            DataQuery::View(ctx) => self.build_select_frame(ctx),
            DataQuery::ColumnValues(ctx) => self.build_column_values_frame(ctx),
        }
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
        // `AggContext::pre_agg_valid_secs`, carried on the active query scope —
        // `DataFrameRecorder::build()` gets here without the context in hand.
        let max_age = active_max_age();
        if pre_agg_allowed
            && !agg_cols.is_empty()
            && let Some(store) = &self.0.pre_agg_store
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

            for version in store.covering_versions(&non_agg_vec, &agg_map, max_age) {
                debug!(pre_agg = %version.name, table = %table_name, "using pre-aggregation");
                match self.scan_pre_agg(&version) {
                    Ok(df) => {
                        return Ok(DataFrameWrapper {
                            inner: df,
                            pre_agg: Some(version),
                        });
                    }
                    Err(e) => {
                        debug!(pre_agg = %version.name, err = %e, "failed to scan pre-agg, trying next");
                    }
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
            pre_agg: None,
        })
    }

    /// Scan one pinned pre-aggregation version, pinning it for the active query.
    ///
    /// Scans the version directly rather than resolving `pre_agg.<name>` through
    /// the schema provider: `covering_versions` has already chosen between
    /// candidates on coverage and row count, and name resolution would only ever
    /// hand back the current one. The `TableScan` still carries the `pre_agg.<name>`
    /// qualifier that resolution would produce, so a column reference reads the
    /// same either way — and `rewrite_*_for_pre_agg` builds exactly that.
    pub(crate) fn scan_pre_agg(
        &self,
        version: &Arc<PreAggVersion>,
    ) -> datafusion::common::Result<DataFrame> {
        record_lease(version);
        let source = provider_as_source(Arc::clone(version) as Arc<dyn TableProvider>);
        let plan = LogicalPlanBuilder::scan(version.table_ref(), source, None)?.build()?;
        Ok(DataFrame::new(self.0.ctx.state(), plan))
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
}

/// Every pre-aggregation version a plan scans, as `Arc`s that pin the underlying
/// files: `PreAggStore::reclaim` deletes a retired version only once its strong
/// count proves nobody holds it, so holding these is holding the files.
fn pre_agg_providers(plan: &LogicalPlan) -> Vec<Arc<dyn TableProvider>> {
    let mut out = Vec::new();
    let _ = plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node
            && let Ok(provider) = source_as_provider(&scan.source)
            && provider.as_ref().downcast_ref::<PreAggVersion>().is_some()
        {
            out.push(provider);
        }
        Ok(TreeNodeRecursion::Continue)
    });
    out
}

/// The staleness bound a query places on any pre-aggregation it accepts.
fn query_max_age(q: &DataQuery) -> Option<u64> {
    match q {
        DataQuery::Agg(ctx) => ctx.pre_agg_valid_secs,
        DataQuery::ColumnValues(ctx) => ctx.pre_agg_valid_secs,
        DataQuery::View(_) => None,
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
    use std::time::Duration;
    use tempfile::TempDir;

    use crate::model_components::pre_agg_store::{resolve_pointer, version_of_filename};

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

    /// Every `<name>.<millis>.parquet` currently on disk for a pre-agg.
    fn version_files(base: &std::path::Path, name: &str) -> Vec<std::path::PathBuf> {
        let mut out: Vec<_> = std::fs::read_dir(base)
            .unwrap()
            .flatten()
            .filter(|e| {
                e.file_name()
                    .to_str()
                    .and_then(|f| version_of_filename(name, f))
                    .is_some()
            })
            .map(|e| e.path())
            .collect();
        out.sort();
        out
    }

    #[test]
    fn test_write_pre_agg_creates_versioned_file() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());

        // Pointer file must exist and resolve to an existing versioned parquet.
        assert!(tmp.path().join("daily_revenue.current").exists());
        let versioned = resolve_pointer(tmp.path(), "daily_revenue")
            .expect("pointer should resolve to a versioned path");
        assert!(versioned.exists(), "missing versioned path: {versioned:?}");

        // The store must hold that version, with its row count read from the footer.
        let store = dm.0.pre_agg_store.as_ref().unwrap();
        let v = store.acquire("daily_revenue").expect("version registered");
        assert_eq!(v.path, versioned);
        assert!(v.row_count > 0, "row_count should be > 0 after write");
        assert!(
            v.version > 0,
            "version millis should be parsed from filename"
        );
    }

    #[test]
    fn test_reclaim_deletes_superseded_version_once_unreferenced() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        dm.set_pre_agg_retired_grace(Duration::ZERO);
        let first = version_files(tmp.path(), "daily_revenue");
        assert_eq!(first.len(), 1);

        // write_pre_aggs reclaims on its own, so after a rebuild with nothing
        // holding a lease the old version is already gone.
        dm.write_pre_aggs(&["daily_revenue"]).unwrap();
        let after = version_files(tmp.path(), "daily_revenue");
        assert_eq!(
            after.len(),
            1,
            "superseded version should have been reclaimed: {after:?}"
        );
        assert_ne!(after[0], first[0], "a new version should be current");
        assert!(!first[0].exists());
    }

    #[test]
    fn test_reclaim_spares_a_version_a_query_still_holds() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        dm.set_pre_agg_retired_grace(Duration::ZERO);
        let original = version_files(tmp.path(), "daily_revenue")[0].clone();

        // Stand in for an in-flight query: a lease taken at plan-build time and
        // held while the rebuild + reclaim runs.
        let scope = LeaseScope::enter(None);
        let store = dm.0.pre_agg_store.as_ref().unwrap();
        let version = store.acquire("daily_revenue").unwrap();
        record_lease(&version);
        let leases = scope.take();
        drop(version);

        dm.write_pre_aggs(&["daily_revenue"]).unwrap();
        assert!(
            original.exists(),
            "reclaim must not unlink a version an in-flight query is pinned to"
        );

        // Query finishes; the next sweep may take it.
        drop(leases);
        let report = dm.reclaim_pre_agg_versions().unwrap();
        assert!(report.errors.is_empty(), "{:?}", report.errors);
        assert!(
            !original.exists(),
            "unreferenced version should be reclaimed"
        );
    }

    #[test]
    fn test_retired_grace_floor_delays_reclaim() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        dm.set_pre_agg_retired_grace(Duration::from_secs(3600));
        let original = version_files(tmp.path(), "daily_revenue")[0].clone();

        dm.write_pre_aggs(&["daily_revenue"]).unwrap();
        assert!(
            original.exists(),
            "grace floor should hold the old version even with no readers"
        );

        dm.set_pre_agg_retired_grace(Duration::ZERO);
        dm.reclaim_pre_agg_versions().unwrap();
        assert!(!original.exists());
    }

    #[test]
    fn test_orphan_sweep_respects_grace_and_name_anchoring() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        dm.set_pre_agg_retired_grace(Duration::ZERO);

        // An untracked versioned file, as a crashed process would leave behind.
        let orphan = tmp.path().join("daily_revenue.1.parquet");
        std::fs::write(&orphan, b"not really parquet").unwrap();
        // A file whose name only *starts* with the pre-agg name. The old prefix
        // match would have deleted this; the anchored one must not.
        let sibling = tmp.path().join("daily_revenue.other.7.parquet");
        std::fs::write(&sibling, b"someone else's").unwrap();
        let tmp_pointer = tmp.path().join("daily_revenue.current.tmp");
        std::fs::write(&tmp_pointer, b"crashed mid-swap").unwrap();

        dm.set_pre_agg_orphan_grace(Duration::from_secs(3600));
        dm.reclaim_pre_agg_versions().unwrap();
        assert!(orphan.exists(), "orphan is still inside its grace window");
        assert!(tmp_pointer.exists());

        dm.set_pre_agg_orphan_grace(Duration::ZERO);
        dm.reclaim_pre_agg_versions().unwrap();
        assert!(!orphan.exists(), "orphan past grace should be swept");
        assert!(!tmp_pointer.exists(), "stale pointer temp should be swept");
        assert!(
            sibling.exists(),
            "a file belonging to a different pre-agg name must never be swept"
        );
        assert!(
            resolve_pointer(tmp.path(), "daily_revenue")
                .unwrap()
                .exists()
        );
    }

    #[test]
    fn test_covering_versions_prefers_smallest_row_count() {
        // Two definitions covering the same request; the one with fewer stored
        // rows must be offered first.
        let by_date_region = daily_revenue_pre_agg();
        let by_region = PreAggregation::new(
            "regional_revenue".into(),
            vec!["orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["sum".into()])]),
        )
        .unwrap();

        let tmp = TempDir::new().unwrap();
        let dm = make_orders_dm(
            Some(tmp.path().to_str().unwrap().to_string()),
            vec![by_date_region, by_region],
        );
        dm.write_pre_aggs(&["daily_revenue", "regional_revenue"])
            .unwrap();

        let store = dm.0.pre_agg_store.as_ref().unwrap();
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        let versions = store.covering_versions(&["orders.region".to_string()], &agg_cols, None);
        let names: Vec<&str> = versions.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, vec!["regional_revenue", "daily_revenue"]);
        assert!(versions[0].row_count < versions[1].row_count);
    }

    #[test]
    fn test_max_age_rejects_a_stale_version() {
        let (dm, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let store = dm.0.pre_agg_store.as_ref().unwrap();
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        let cols = ["orders.date".to_string()];

        assert_eq!(store.covering_versions(&cols, &agg_cols, None).len(), 1);
        assert_eq!(
            store.covering_versions(&cols, &agg_cols, Some(3600)).len(),
            1,
            "a freshly written version is well inside an hour"
        );
        assert!(
            store
                .covering_versions(&cols, &agg_cols, Some(0))
                .is_empty(),
            "max_age of 0 must reject even a just-written version"
        );
        let _ = tmp;
    }

    #[test]
    fn test_new_restores_current_version_from_existing_pointer() {
        // Write with one DataModel, then construct a fresh one over the same path
        // and verify the current version is rehydrated from the pointer.
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let path = tmp.path().to_str().unwrap().to_string();

        let dm2 = make_orders_dm(Some(path), vec![daily_revenue_pre_agg()]);

        let v = dm2
            .0
            .pre_agg_store
            .as_ref()
            .unwrap()
            .acquire("daily_revenue")
            .expect("current version should be restored on DataModel::new");
        assert!(
            v.row_count > 0,
            "row_count should come from parquet metadata"
        );
        assert_eq!(
            v.path,
            resolve_pointer(tmp.path(), "daily_revenue").unwrap()
        );
    }

    #[test]
    fn test_corrupt_pointer_leaves_no_current_version() {
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());

        // Point at a versioned file that does not exist, then reload.
        std::fs::write(
            tmp.path().join("daily_revenue.current"),
            "daily_revenue.0.parquet",
        )
        .unwrap();

        let dm2 = make_orders_dm(
            Some(tmp.path().to_str().unwrap().to_string()),
            vec![daily_revenue_pre_agg()],
        );
        assert!(
            dm2.0
                .pre_agg_store
                .as_ref()
                .unwrap()
                .acquire("daily_revenue")
                .is_none(),
            "a pointer to a missing file must not register a version"
        );
    }
}
