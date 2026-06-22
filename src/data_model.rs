use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::catalog::TableProvider;
use datafusion::datasource::provider_as_source;
use datafusion::logical_expr::{JoinType, LogicalPlanBuilder, SortExpr};
use datafusion::prelude::{DataFrame, Expr, ParquetReadOptions, SessionContext, col};
use tracing::{debug, trace};

use crate::{
    column_expressions::filter_expr::json_to_expr,
    model_components::{
        agg_context::AggContext,
        column_values_context::ColumnValuesContext,
        joins::JoinHow,
        measures::{
            DfMeasure, MeasureMetadata, extract_df_measure_metadata, resolve_group_by_cols,
            validate_df_measure_structure,
        },
        pre_aggregations::{PreAggregation, agg_needed_components, component_col_name},
        select_context::SelectContext,
    },
    wrappers::datafusion::{
        dataframe_recorder::DataFrameRecorder, dataframe_wrapper::DataFrameWrapper,
    },
};

use super::model_components::joins::JoinGraph;

// ── Public types ─────────────────────────────────────────────────────────────

pub enum DataOutput {
    Data(Vec<RecordBatch>),
    Explanation(String),
}

impl std::fmt::Debug for DataOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataOutput::Data(batches) => {
                write!(f, "DataOutput::Data({} batches)", batches.len())
            }
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

struct DataModelInner {
    /// DataFusion session context with all base tables registered.
    ctx: SessionContext,
    /// Raw table providers, keyed by table name. Used to build scan plans.
    table_providers: HashMap<String, Arc<dyn TableProvider>>,
    joins: JoinGraph,
    pub(crate) measures: HashMap<String, DfMeasure>,
    pub measure_metadata: HashMap<String, MeasureMetadata>,
    pre_aggs: Option<Vec<PreAggregation>>,
    pre_agg_path: Option<String>,
    /// Tokio runtime for executing async DataFusion operations from sync code.
    rt: Arc<tokio::runtime::Runtime>,
}

#[derive(Clone)]
pub struct DataModel(Arc<DataModelInner>);

impl DataModel {
    /// Create a DataModel from a map of table providers.
    ///
    /// Column names are prefixed with the table name (e.g. `id` → `orders.id`)
    /// at query time via a projection, so providers may have unqualified names.
    pub fn new(
        tables: HashMap<String, Arc<dyn TableProvider>>,
        joins: JoinGraph,
        pre_aggs: Vec<PreAggregation>,
        pre_agg_path: Option<String>,
    ) -> DataModel {
        let ctx = SessionContext::new();

        let rt = Arc::new(tokio::runtime::Runtime::new().expect("tokio runtime for DataModel"));

        // Register all tables so the session context can resolve them.
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
            rt,
        }))
    }

    // ── Public API ────────────────────────────────────────────────────────────

    /// Begin a recorder chain for the named table.
    pub fn table(&self, table_name: &str) -> DataFrameRecorder {
        assert!(
            self.0.table_providers.contains_key(table_name),
            "table '{table_name}' not found in DataModel"
        );
        DataFrameRecorder::new(table_name.to_string(), self.clone())
    }

    /// Register a measure with the DataModel. Validates structure at registration time.
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

    /// Returns true if `target` is reachable from `base` via the join graph.
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

    // ── Internal ──────────────────────────────────────────────────────────────

    fn validate_and_extract_measure(&self, measure: &DfMeasure) -> Result<MeasureMetadata, String> {
        let stub_qc = AggContext::stub();
        let df = measure
            .call(self, &stub_qc)
            .map_err(|e| format!("measure '{}' failed to build: {e}", measure.name))?;
        validate_df_measure_structure(&df)?;
        extract_df_measure_metadata(&df, &measure.name)
    }

    fn build_agg_frame(&self, qc: &AggContext) -> Result<DataFrame, String> {
        let known_measures: Vec<MeasureMetadata> =
            self.0.measure_metadata.values().cloned().collect();
        let all_columns: HashSet<String> = self
            .0
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
            .collect();
        qc.validate(&known_measures, &all_columns)?;

        let mut measure_dfs: Vec<DataFrame> = Vec::new();
        let mut expected_cols: Option<Vec<String>> = None;

        for m_name in &qc.measures {
            let measure = self.0.measures.get(m_name).unwrap();
            let df = measure
                .call(self, qc)
                .map_err(|e| format!("measure '{m_name}' failed: {e}"))?;
            let cols = resolve_group_by_cols(&df)?;

            if let Some(ref prev) = expected_cols {
                if &cols != prev {
                    return Err(format!(
                        "incompatible group-by columns across measures: {prev:?} vs {cols:?}"
                    ));
                }
            } else {
                expected_cols = Some(cols);
            }
            measure_dfs.push(df);
        }

        let join_cols: Vec<String> = expected_cols.unwrap_or_default();

        let mut combined = measure_dfs.remove(0);
        for frame in measure_dfs {
            combined = if join_cols.is_empty() {
                let (state, left_plan) = combined.into_parts();
                let right_plan = frame.into_unoptimized_plan();
                let cross_plan = LogicalPlanBuilder::from(left_plan)
                    .cross_join(right_plan)
                    .map_err(|e| format!("failed to cross join measures: {e}"))?
                    .build()
                    .map_err(|e| format!("failed to build cross join: {e}"))?;
                DataFrame::new(state, cross_plan)
            } else {
                let left_on: Vec<&str> = join_cols.iter().map(|s| s.as_str()).collect();
                let right_on: Vec<&str> = join_cols.iter().map(|s| s.as_str()).collect();
                combined
                    .join(frame, JoinType::Full, &left_on, &right_on, None)
                    .map_err(|e| format!("failed to join measures: {e}"))?
            };
        }

        if let Some(having_expr) = json_to_expr(&qc.havings) {
            combined = combined
                .filter(having_expr)
                .map_err(|e| format!("having filter failed: {e}"))?;
        }

        if !qc.sorts.is_empty() {
            let sort_exprs: Vec<SortExpr> = qc
                .sorts
                .iter()
                .map(|(c, d)| col(c.as_str()).sort(d != "desc", true))
                .collect();
            combined = combined
                .sort(sort_exprs)
                .map_err(|e| format!("sort failed: {e}"))?;
        }

        combined
            .limit(qc.offset, Some(qc.limit))
            .map_err(|e| format!("limit failed: {e}"))
    }

    fn build_select_frame(&self, vc: &SelectContext) -> Result<DataFrame, String> {
        let all_columns: HashSet<String> = self
            .0
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
            .collect();
        vc.validate(&all_columns)?;

        let mut all_needed: HashSet<String> = vc.columns.iter().cloned().collect();
        for fc in vc.filter_columns() {
            all_needed.insert(fc);
        }

        let mut referenced_tables: Vec<String> = all_needed
            .iter()
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        referenced_tables.sort();

        if referenced_tables.is_empty() {
            return Err("columns must be table-qualified (e.g. table.column)".into());
        }

        let base_table = referenced_tables
            .iter()
            .find(|candidate| {
                referenced_tables
                    .iter()
                    .all(|t| t == *candidate || self.0.joins.find_path(candidate, t).is_some())
            })
            .ok_or_else(|| {
                format!(
                    "no single base table can reach all tables {referenced_tables:?} via join graph"
                )
            })?
            .clone();

        let non_agg_str: HashSet<String> = all_needed.clone();
        let mut df = self
            .get_df_table(&base_table, &non_agg_str, &HashMap::new(), false)
            .map_err(|e| e.to_string())?
            .inner;

        if let Some(filter_expr) = json_to_expr(&vc.filters) {
            df = df.filter(filter_expr).map_err(|e| e.to_string())?;
        }

        let select_exprs: Vec<Expr> = vc.columns.iter().map(|c| col(c.as_str())).collect();
        df = df.select(select_exprs).map_err(|e| e.to_string())?;

        if !vc.sorts.is_empty() {
            let sort_exprs: Vec<SortExpr> = vc
                .sorts
                .iter()
                .map(|(c, d)| col(c.as_str()).sort(d != "desc", true))
                .collect();
            df = df.sort(sort_exprs).map_err(|e| e.to_string())?;
        }

        df.limit(vc.offset, Some(vc.limit))
            .map_err(|e| e.to_string())
    }

    fn build_column_values_frame(&self, ctx: &ColumnValuesContext) -> Result<DataFrame, String> {
        let (table_name, _) = ctx.column.split_once('.').unwrap();
        if !self.0.table_providers.contains_key(table_name) {
            return Err(format!("unknown table: '{table_name}'"));
        }

        if ctx.use_pre_agg {
            if let (Some(pre_aggs), Some(path)) = (&self.0.pre_aggs, self.0.pre_agg_path.as_deref())
            {
                let mut candidates: Vec<&PreAggregation> = pre_aggs
                    .iter()
                    .filter(|pa| pa.group_by.contains(&ctx.column))
                    .collect();
                candidates.sort_by_key(|pa| pa.row_count);

                'candidates: for candidate in candidates {
                    let pre_agg_file = format!("{path}/{}.parquet", candidate.name);
                    if let Some(max_age) = ctx.pre_agg_valid_secs {
                        match std::fs::metadata(&pre_agg_file)
                            .ok()
                            .and_then(|m| m.modified().ok())
                        {
                            Some(modified) => {
                                let age = modified.elapsed().unwrap_or(std::time::Duration::MAX);
                                if age > std::time::Duration::from_secs(max_age) {
                                    continue 'candidates;
                                }
                            }
                            None => continue 'candidates,
                        }
                    }
                    if std::path::Path::new(&pre_agg_file).exists() {
                        debug!(column = %ctx.column, pre_agg = %candidate.name, "using pre-agg for column values");
                        if let Ok(df) = self.read_parquet_sync(&pre_agg_file) {
                            return df
                                .select(vec![col(ctx.column.as_str())])
                                .and_then(|d| d.distinct())
                                .map_err(|e| e.to_string());
                        }
                    }
                    debug!(pre_agg = %candidate.name, "pre-agg not found, trying next");
                }
                trace!(column = %ctx.column, "no valid pre-agg, falling back");
            }
        }

        let base = self
            .get_df_table(
                table_name,
                &HashSet::from([ctx.column.clone()]),
                &HashMap::new(),
                false,
            )
            .map_err(|e| e.to_string())?
            .inner;

        base.select(vec![col(ctx.column.as_str())])
            .and_then(|d| d.distinct())
            .map_err(|e| e.to_string())
    }

    /// Compute and write parquet files for the named pre-aggregations.
    pub fn write_pre_aggs(&self, names: &[&str]) -> Result<(), String> {
        let pre_aggs = self
            .0
            .pre_aggs
            .as_ref()
            .ok_or("no pre-aggregations registered on this DataModel")?;
        for &name in names {
            let pa = pre_aggs
                .iter()
                .find(|pa| pa.name == name)
                .ok_or_else(|| format!("pre-aggregation '{name}' not found"))?;
            self.write_pre_agg(pa)?;
        }
        Ok(())
    }

    fn write_pre_agg(&self, pa: &PreAggregation) -> Result<(), String> {
        use datafusion::dataframe::DataFrameWriteOptions;
        use datafusion::functions_aggregate::expr_fn::{count, max, min, sum};

        let path = self
            .0
            .pre_agg_path
            .as_deref()
            .ok_or("pre_agg_path not set on DataModel")?;

        let all_col_names = pa.group_by.iter().chain(pa.aggregations.keys());
        let mut referenced_tables: Vec<String> = all_col_names
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        referenced_tables.sort();

        if referenced_tables.is_empty() {
            return Err("all columns must be table-qualified (e.g. orders.amount)".into());
        }

        let base_table = referenced_tables
            .iter()
            .find(|candidate| {
                referenced_tables
                    .iter()
                    .all(|t| t == *candidate || self.0.joins.find_path(candidate, t).is_some())
            })
            .ok_or_else(|| {
                format!(
                    "no single base table can reach all tables {referenced_tables:?} via join graph"
                )
            })?
            .clone();

        let non_agg_str: HashSet<String> = pa.group_by.iter().cloned().collect();
        let agg_str: HashMap<String, Vec<String>> = pa
            .aggregations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut df = self
            .get_df_table(&base_table, &non_agg_str, &agg_str, false)
            .map_err(|e| e.to_string())?
            .inner;

        let group_by_exprs: Vec<Expr> = pa.group_by.iter().map(|c| col(c.as_str())).collect();

        let mut agg_exprs: Vec<Expr> = Vec::new();
        for (qcol, components) in &pa.aggregations {
            for component in components {
                let alias = component_col_name(qcol, component);
                let expr = match component.as_str() {
                    "sum" => sum(col(qcol.as_str())).alias(&alias),
                    "count" => count(col(qcol.as_str())).alias(&alias),
                    "min" => min(col(qcol.as_str())).alias(&alias),
                    "max" => max(col(qcol.as_str())).alias(&alias),
                    "sumsq" => sum(col(qcol.as_str()) * col(qcol.as_str())).alias(&alias),
                    other => return Err(format!("unknown pre-agg component '{other}'")),
                };
                agg_exprs.push(expr);
            }
        }

        df = df
            .aggregate(group_by_exprs, agg_exprs)
            .map_err(|e| format!("failed to aggregate for pre-agg: {e}"))?;

        let file_path = format!("{path}/{}.parquet", pa.name);
        self.0
            .rt
            .block_on(df.write_parquet(&file_path, DataFrameWriteOptions::new(), None))
            .map_err(|e| format!("failed to write parquet: {e}"))?;

        Ok(())
    }

    /// Build a base DataFrame for a table, applying joins for any external columns
    /// needed by the caller. Uses a pre-aggregation file if available and allowed.
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
                    if let Ok(df) = self.read_parquet_sync(&pre_agg_file) {
                        return Ok(DataFrameWrapper {
                            inner: df,
                            from_pre_agg: true,
                        });
                    }
                    debug!(pre_agg = %candidate.name, "failed to read parquet, trying next");
                }
                trace!(table = %table_name, "no valid pre-agg, falling back to base table");
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
        })
    }

    /// Build a scan DataFrame for a single table with column prefixing applied.
    fn scan_table(&self, table_name: &str) -> datafusion::common::Result<DataFrame> {
        let provider = self
            .0
            .table_providers
            .get(table_name)
            .unwrap_or_else(|| panic!("table '{table_name}' not found in DataModel"));

        let schema = provider.schema();
        let prefix = format!("{table_name}.");
        let source = provider_as_source(Arc::clone(provider));

        let builder = LogicalPlanBuilder::scan(table_name, source, None)?;

        let needs_prefix = schema
            .fields()
            .iter()
            .any(|f| !f.name().starts_with(&prefix));
        let plan = if needs_prefix {
            let rename_exprs: Vec<Expr> = schema
                .fields()
                .iter()
                .map(|f| {
                    let new_name = if f.name().starts_with(&prefix) {
                        f.name().to_string()
                    } else {
                        format!("{prefix}{}", f.name())
                    };
                    col(f.name()).alias(new_name)
                })
                .collect();
            builder.project(rename_exprs)?.build()?
        } else {
            builder.build()?
        };

        Ok(DataFrame::new(self.0.ctx.state(), plan))
    }

    /// Read a parquet file synchronously using the stored tokio runtime.
    /// Falls back gracefully when called from within an existing async runtime.
    fn read_parquet_sync(&self, file_path: &str) -> datafusion::common::Result<DataFrame> {
        let ctx = self.0.ctx.clone();
        let path = file_path.to_string();
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => {
                // Inside an async runtime — use block_in_place to avoid blocking the executor.
                tokio::task::block_in_place(|| {
                    handle.block_on(ctx.read_parquet(path, ParquetReadOptions::default()))
                })
            }
            Err(_) => self
                .0
                .rt
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
