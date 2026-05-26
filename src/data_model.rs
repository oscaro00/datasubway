use std::collections::{HashMap, HashSet};

use polars::prelude::{
    col, lit, DataFrame, Expr, JoinArgs, JoinType, LazyFrame, PlRefPath, PlSmallStr, Schema,
    SortMultipleOptions,
};

use crate::{
    column_expressions::{
        column_context::{allow, exclude, AllowExcludeKind, ColumnReturn},
        filter_expr::json_to_expr,
    },
    model_components::{
        joins::JoinHow,
        measures::{
            extract_measure_metadata, validate_measure_structure, Measure, MeasureMetadata,
        },
        pre_aggregations::{agg_expansion, component_col_name, find_best_pre_agg, PreAggregation},
        query_context::QueryContext,
    },
    wrappers::polars::{
        lazyframe_recorder::LazyFrameRecorder, lazyframe_wrapper::LazyFrameWrapper,
    },
};

use super::model_components::joins::JoinGraph;

pub enum QueryOutput {
    Data(DataFrame),
    Explanation(String),
}

impl std::fmt::Debug for QueryOutput {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueryOutput::Data(df) => write!(f, "QueryOutput::Data({:?})", df.head(Some(5))),
            QueryOutput::Explanation(s) => write!(f, "QueryOutput::Explanation({:?})", s),
        }
    }
}

pub struct DataModel {
    tables: HashMap<String, LazyFrame>,
    joins: JoinGraph,
    pub(crate) measures: HashMap<String, Measure>,
    pub measure_metadata: HashMap<String, MeasureMetadata>,
    pre_aggs: Option<Vec<PreAggregation>>,
    pre_agg_path: Option<String>,
    table_schemas: HashMap<String, Schema>,
}

impl DataModel {
    pub fn new(
        tables: HashMap<String, LazyFrame>,
        joins: JoinGraph,
        pre_aggs: Vec<PreAggregation>,
        pre_agg_path: Option<String>,
    ) -> DataModel {
        let mut tables: HashMap<String, LazyFrame> = tables
            .into_iter()
            .map(|(name, lf)| {
                let lf = Self::prefix_columns_if_needed(&name, lf);
                (name, lf)
            })
            .collect();

        let table_schemas = tables
            .iter_mut()
            .map(|(name, lf)| {
                let schema = lf
                    .collect_schema()
                    .unwrap_or_else(|e| panic!("failed to collect schema for table '{name}': {e}"));
                (name.clone(), (*schema).clone())
            })
            .collect();

        let mut seen = HashSet::new();
        for pa in &pre_aggs {
            if !seen.insert(pa.name.as_str()) {
                panic!("duplicate pre-aggregation name '{}'", pa.name);
            }
        }

        DataModel {
            tables,
            joins,
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
            pre_aggs: if pre_aggs.is_empty() {
                None
            } else {
                Some(pre_aggs)
            },
            pre_agg_path,
            table_schemas,
        }
    }

    fn prefix_columns_if_needed(table_name: &str, mut lf: LazyFrame) -> LazyFrame {
        let schema = lf
            .collect_schema()
            .unwrap_or_else(|e| panic!("failed to get schema for '{table_name}': {e}"));
        let prefix = format!("{}.", table_name);
        let to_rename: Vec<(String, String)> = schema
            .iter_names()
            .filter(|name| !name.starts_with(prefix.as_str()))
            .map(|name| (name.to_string(), format!("{}{}", prefix, name)))
            .collect();
        if to_rename.is_empty() {
            return lf;
        }
        let existing: Vec<String> = to_rename.iter().map(|(e, _)| e.clone()).collect();
        let new_names: Vec<String> = to_rename.iter().map(|(_, n)| n.clone()).collect();
        lf.rename(existing, new_names, false)
    }

    pub fn table(&self, table_name: &str) -> LazyFrameRecorder<'_> {
        assert!(
            self.tables.contains_key(table_name),
            "table '{table_name}' not found in DataModel"
        );
        LazyFrameRecorder {
            table_name: table_name.to_string(),
            data_model: self,
            lazy_ops: Vec::new(),
            non_agg_cols: HashSet::new(),
            agg_cols: HashMap::new(),
            non_base_tables: HashSet::new(),
            use_pre_agg: true,
            allow_exclude_records: Vec::new(),
        }
    }

    /// Returns the schema for a registered table.
    pub fn table_schema(&self, table_name: &str) -> &Schema {
        self.table_schemas
            .get(table_name)
            .unwrap_or_else(|| panic!("schema for table '{table_name}' not found"))
    }

    /// Register a measure with the DataModel. Validates structure at registration time.
    pub fn add_measure(&mut self, measure: Measure) -> Result<(), String> {
        if self.measures.contains_key(&measure.name) {
            return Err(format!("measure '{}' already registered", measure.name));
        }
        let name = measure.name.clone();
        let metadata = self.validate_and_extract_measure(&measure)?;
        self.measures.insert(name.clone(), measure);
        self.measure_metadata.insert(name, metadata);
        Ok(())
    }

    fn validate_and_extract_measure(&self, measure: &Measure) -> Result<MeasureMetadata, String> {
        let stub_qc = QueryContext::stub();
        let recorder = measure.call(self, &stub_qc);
        validate_measure_structure(&recorder)?;
        Ok(extract_measure_metadata(&recorder, &measure.name))
    }

    /// Returns the metadata for a registered measure, if it exists.
    pub fn measure_metadata(&self, name: &str) -> Option<&MeasureMetadata> {
        self.measure_metadata.get(name)
    }

    /// Returns true if `target` is reachable from `base` via the join graph.
    pub fn can_join(&self, base: &str, target: &str) -> bool {
        self.joins.find_path(base, target).is_some()
    }

    fn eval_last_allow_exclude(
        &self,
        recorder: &LazyFrameRecorder<'_>,
    ) -> Result<Vec<String>, String> {
        match recorder.allow_exclude_records.last() {
            None => {
                let mut cols: Vec<String> = recorder
                    .non_agg_cols
                    .iter()
                    .map(|s| s.to_string())
                    .collect();
                cols.sort();
                Ok(cols)
            }
            Some(record) => {
                let result = match record.kind {
                    AllowExcludeKind::Allow => allow(
                        record.pattern.clone(),
                        record.context.clone(),
                        record.include.clone(),
                    ),
                    AllowExcludeKind::Exclude => exclude(
                        record.pattern.clone(),
                        record.context.clone(),
                        record.include.clone(),
                    ),
                };
                match result.inner {
                    ColumnReturn::Strings(cols) => Ok(cols),
                    ColumnReturn::PolarsExpr(_) => Err(
                        "JSON-context allow/exclude cannot produce group-by column names".into(),
                    ),
                }
            }
        }
    }

    pub fn query(&self, qc: &QueryContext, explain: bool) -> Result<QueryOutput, String> {
        let known_measures: Vec<MeasureMetadata> =
            self.measure_metadata.values().cloned().collect();
        let all_columns: HashSet<String> = self
            .table_schemas
            .values()
            .flat_map(|s| s.iter_names().map(|n| n.to_string()))
            .collect();
        qc.validate(&known_measures, &all_columns)?;

        let mut recorders = Vec::new();
        let mut expected_cols: Option<Vec<String>> = None;

        for m_name in &qc.measures {
            let measure = self.measures.get(m_name).unwrap();
            let mut recorder = measure.call(self, qc);
            recorder.use_pre_agg = qc.use_pre_agg;
            let cols = self.eval_last_allow_exclude(&recorder)?;

            if let Some(ref prev) = expected_cols {
                if &cols != prev {
                    return Err(format!(
                        "incompatible group-by columns across measures: {:?} vs {:?}",
                        prev, cols
                    ));
                }
            } else {
                expected_cols = Some(cols);
            }
            recorders.push(recorder);
        }

        let join_cols: Vec<String> = expected_cols.unwrap_or_default();

        let mut frames: Vec<LazyFrame> =
            recorders.into_iter().map(|r| r.build().lazyframe).collect();

        let mut combined = frames.remove(0);
        for frame in frames {
            let (left_on, right_on): (Vec<Expr>, Vec<Expr>) = if join_cols.is_empty() {
                (vec![], vec![])
            } else {
                let exprs: Vec<Expr> = join_cols.iter().map(|c| col(c.as_str())).collect();
                (exprs.clone(), exprs)
            };
            let join_type = if join_cols.is_empty() {
                JoinType::Cross
            } else {
                JoinType::Full
            };
            combined = combined.join(frame, left_on, right_on, JoinArgs::new(join_type));
        }

        if let Some(having_expr) = json_to_expr(&qc.havings) {
            combined = combined.filter(having_expr);
        }

        if !qc.sorts.is_empty() {
            let (sort_cols, descs): (Vec<PlSmallStr>, Vec<bool>) = qc
                .sorts
                .iter()
                .map(|(c, d)| (PlSmallStr::from(c.as_str()), d == "desc"))
                .unzip();
            let opts = SortMultipleOptions::default().with_order_descending_multi(descs);
            combined = combined.sort(sort_cols, opts);
        }

        combined = combined.slice(qc.offset as i64, qc.limit as u32);

        if explain {
            combined
                .explain(true)
                .map(QueryOutput::Explanation)
                .map_err(|e| format!("explain failed: {e}"))
        } else {
            combined
                .collect()
                .map(QueryOutput::Data)
                .map_err(|e| format!("query execution failed: {e}"))
        }
    }

    /// Compute and write parquet files for the named pre-aggregations.
    ///
    /// Looks up each name in the pre-aggregations registered at construction,
    /// then writes `{pre_agg_path}/{name}.parquet` for each.
    pub fn write_pre_aggs(&self, names: &[&str]) -> Result<(), String> {
        let pre_aggs = self
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
        let path = self
            .pre_agg_path
            .as_deref()
            .ok_or("pre_agg_path not set on DataModel")?;

        // --- 1. Identify referenced tables ---
        let all_col_names = pa.group_by.iter().chain(pa.aggregations.keys());
        let mut referenced_tables: Vec<String> = all_col_names
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        referenced_tables.sort(); // deterministic order

        if referenced_tables.is_empty() {
            return Err("all columns must be table-qualified (e.g. orders.amount)".into());
        }

        // --- 2. Find a base table that can reach all others ---
        let base_table = referenced_tables
            .iter()
            .find(|candidate| {
                referenced_tables
                    .iter()
                    .all(|t| t == *candidate || self.joins.find_path(candidate, t).is_some())
            })
            .ok_or_else(|| {
                format!(
                    "no single base table can reach all tables {:?} via join graph",
                    referenced_tables
                )
            })?
            .clone();

        // --- 3. Build joined LazyFrame ---
        let mut lf = self
            .tables
            .get(&base_table)
            .ok_or_else(|| format!("table '{base_table}' not found in DataModel"))?
            .clone();

        for other in &referenced_tables {
            if other == &base_table {
                continue;
            }
            let join_path = self.joins.find_path(&base_table, other).unwrap();
            for join in join_path {
                let right_lf = self
                    .tables
                    .get(&join.right)
                    .ok_or_else(|| format!("table '{}' not found in DataModel", join.right))?
                    .clone();
                let join_type = match join.how {
                    JoinHow::Left => JoinType::Left,
                    JoinHow::Inner => JoinType::Inner,
                };
                let left_on: Vec<Expr> = join.left_on.iter().map(|c| col(c.as_str())).collect();
                let right_on: Vec<Expr> = join.right_on.iter().map(|c| col(c.as_str())).collect();
                lf = lf.join(right_lf, left_on, right_on, JoinArgs::new(join_type));
            }
        }

        // --- 4. Build group_by expressions ---
        let group_by_exprs: Vec<Expr> = pa.group_by.iter().map(|qcol| col(qcol.as_str())).collect();

        // --- 5. Build agg expressions, one per component ---
        let mut agg_exprs: Vec<Expr> = Vec::new();
        for (qcol, components) in &pa.aggregations {
            for component in components {
                let alias = component_col_name(qcol, component);
                let expr = match component.as_str() {
                    "sum" => col(qcol.as_str()).sum().alias(&alias),
                    "count" => col(qcol.as_str()).count().alias(&alias),
                    "min" => col(qcol.as_str()).min().alias(&alias),
                    "max" => col(qcol.as_str()).max().alias(&alias),
                    "sumsq" => col(qcol.as_str()).pow(lit(2.0f64)).sum().alias(&alias),
                    other => {
                        return Err(format!("unknown pre-agg component '{other}'"));
                    }
                };
                agg_exprs.push(expr);
            }
        }

        // --- 6. Execute group_by + agg ---
        let mut df = lf
            .group_by(group_by_exprs)
            .agg(agg_exprs)
            .collect()
            .map_err(|e| format!("failed to collect pre-agg: {e}"))?;

        // --- 8. Write parquet ---
        let file_path = format!("{path}/{}.parquet", pa.name);
        let file = std::fs::File::create(&file_path)
            .map_err(|e| format!("failed to create '{file_path}': {e}"))?;
        polars::prelude::ParquetWriter::new(file)
            .finish(&mut df)
            .map_err(|e| format!("failed to write parquet: {e}"))?;

        Ok(())
    }

    pub(crate) fn get_table(
        &self,
        table_name: &str,
        non_agg_cols: &HashSet<PlSmallStr>,
        agg_cols: &HashMap<PlSmallStr, Vec<String>>,
        use_pre_agg: bool,
    ) -> LazyFrameWrapper {
        if use_pre_agg {
            if let (Some(pre_aggs), Some(path)) = (&self.pre_aggs, self.pre_agg_path.as_deref()) {
                let non_agg_vec: Vec<String> = non_agg_cols.iter().map(|s| s.to_string()).collect();

                let agg_map: HashMap<String, HashSet<String>> = agg_cols
                    .iter()
                    .map(|(col, agg_names)| {
                        let components: HashSet<String> = agg_names
                            .iter()
                            .filter_map(|name| agg_expansion(&name.to_lowercase()).ok())
                            .flatten()
                            .map(|s| s.to_string())
                            .collect();
                        (col.to_string(), components)
                    })
                    .collect();

                if let Some(best) = find_best_pre_agg(pre_aggs, &non_agg_vec, &agg_map) {
                    let pre_agg_file = format!("{}/{}.parquet", path, best.name);
                    if let Ok(lf) = LazyFrame::scan_parquet(
                        PlRefPath::from(pre_agg_file.as_str()),
                        Default::default(),
                    ) {
                        return LazyFrameWrapper {
                            lazyframe: lf,
                            from_pre_agg: true,
                        };
                    }
                }
            }
        } // end if use_pre_agg

        let mut lf = self
            .tables
            .get(table_name)
            .unwrap_or_else(|| panic!("table '{table_name}' not found in DataModel"))
            .clone();

        // Collect external table names referenced by non-aggregated or aggregated columns
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

        // Tracks columns dropped by prior joins: dropped_name → surviving_name.
        // Polars drops the right-side key after each join, so any subsequent join
        // that references the dropped name must be redirected to the surviving key.
        let mut col_remap: HashMap<String, String> = HashMap::new();

        let mut joined: HashSet<String> = HashSet::from([table_name.to_string()]);
        for ext in &external_tables {
            if let Some(path) = self.joins.find_path(table_name, ext) {
                for join in path {
                    if joined.contains(&join.right) {
                        continue;
                    }
                    let right_lf = self
                        .tables
                        .get(&join.right)
                        .unwrap_or_else(|| panic!("table '{}' not found in DataModel", join.right))
                        .clone();
                    let join_type = match join.how {
                        JoinHow::Left => JoinType::Left,
                        JoinHow::Inner => JoinType::Inner,
                    };
                    let left_on: Vec<Expr> = join
                        .left_on
                        .iter()
                        .map(|c| {
                            let name = col_remap
                                .get(c.as_str())
                                .map(|s| s.as_str())
                                .unwrap_or(c.as_str());
                            col(name)
                        })
                        .collect();
                    let right_on: Vec<Expr> =
                        join.right_on.iter().map(|c| col(c.as_str())).collect();
                    // After the join, each right key is dropped; record the surviving left key.
                    for (left_col, right_col) in join.left_on.iter().zip(join.right_on.iter()) {
                        let surviving = col_remap
                            .get(left_col.as_str())
                            .cloned()
                            .unwrap_or_else(|| left_col.clone());
                        col_remap.insert(right_col.clone(), surviving);
                    }
                    lf = lf.join(right_lf, left_on, right_on, JoinArgs::new(join_type));
                    joined.insert(join.right.clone());
                }
            }
        }

        LazyFrameWrapper {
            lazyframe: lf,
            from_pre_agg: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;
    use tempfile::TempDir;

    fn make_orders_dm(pre_agg_path: Option<String>, pre_aggs: Vec<PreAggregation>) -> DataModel {
        let orders = df![
            "date"   => ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "region" => ["north", "south", "north", "south"],
            "amount" => [100.0f64, 200.0, 150.0, 250.0],
        ]
        .unwrap()
        .lazy();
        DataModel::new(
            HashMap::from([("orders".to_string(), orders)]),
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

    #[test]
    fn test_write_pre_agg_correct_schema() {
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let path = tmp.path().join("daily_revenue.parquet");
        let mut lf =
            LazyFrame::scan_parquet(PlRefPath::from(path.to_str().unwrap()), Default::default())
                .unwrap();
        let schema = lf.collect_schema().unwrap();
        let cols: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
        assert!(cols.contains(&"orders.date"), "missing orders.date");
        assert!(cols.contains(&"orders.region"), "missing orders.region");
        assert!(
            cols.contains(&"orders.amount-sum"),
            "missing orders.amount-sum"
        );
        assert!(
            cols.contains(&"orders.amount-count"),
            "missing orders.amount-count"
        );
    }

    #[test]
    fn test_write_pre_agg_correct_data() {
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let path = tmp.path().join("daily_revenue.parquet");
        let df =
            LazyFrame::scan_parquet(PlRefPath::from(path.to_str().unwrap()), Default::default())
                .unwrap()
                .sort(
                    ["orders.date", "orders.region"],
                    SortMultipleOptions::default(),
                )
                .collect()
                .unwrap();

        // 2 dates × 2 regions = 4 rows
        assert_eq!(df.height(), 4);

        // north/2024-01-01 has amount=100 → sum=100, count=1
        let sums = df
            .column("orders.amount-sum")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        assert!(sums.contains(&100.0_f64));
        assert!(sums.contains(&200.0_f64));
        assert!(sums.contains(&150.0_f64));
        assert!(sums.contains(&250.0_f64));
    }

    #[test]
    fn test_pre_agg_selected_when_query_covers() {
        let (dm, _tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let result = dm
            .table("orders")
            .group_by(vec![col("orders.date"), col("orders.region")])
            .agg(vec![col("orders.amount").sum()])
            .build();
        assert!(result.from_pre_agg);
    }

    #[test]
    fn test_pre_agg_not_selected_when_query_does_not_cover() {
        let (dm, _tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        // "orders.store" is not in the pre-agg group_by
        let result = dm
            .table("orders")
            .group_by(vec![col("orders.store")])
            .agg(vec![col("orders.amount").sum()])
            .build();
        assert!(!result.from_pre_agg);
    }

    #[test]
    fn test_pre_agg_end_to_end_sum_matches() {
        let (dm, _tmp) = write_and_get_tmp(daily_revenue_pre_agg());

        let mut result = dm
            .table("orders")
            .group_by(vec![col("orders.date"), col("orders.region")])
            .agg(vec![col("orders.amount").sum().alias("total")])
            .build()
            .collect()
            .unwrap();
        result = result
            .sort(
                ["orders.date", "orders.region"],
                SortMultipleOptions::default(),
            )
            .unwrap();

        let totals: Vec<f64> = result
            .column("total")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();

        // Sorted by date+region: north/01, south/01, north/02, south/02
        assert_eq!(totals, vec![100.0, 200.0, 150.0, 250.0]);
    }

    #[test]
    #[should_panic(expected = "duplicate pre-aggregation name")]
    fn test_duplicate_pre_agg_name_panics() {
        make_orders_dm(None, vec![daily_revenue_pre_agg(), daily_revenue_pre_agg()]);
    }

    #[test]
    fn test_write_unknown_name_errors() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().to_str().unwrap().to_string();
        let dm = make_orders_dm(Some(path), vec![daily_revenue_pre_agg()]);
        assert!(dm.write_pre_aggs(&["nonexistent"]).is_err());
    }

    #[test]
    fn test_write_pre_agg_no_path_errors() {
        let dm = make_orders_dm(None, vec![daily_revenue_pre_agg()]);
        assert!(dm.write_pre_aggs(&["daily_revenue"]).is_err());
    }

    // ── query() tests ─────────────────────────────────────────────────────────

    use crate::model_components::measures::Measure;
    use crate::model_components::query_context::QueryContext;
    use crate::wrappers::polars::lazyframe_recorder::LazyFrameRecorder;

    // Groups by whatever qc.groups says; aggregates orders.amount (no alias so
    // output_columns == ["orders.amount"] which validates correctly).
    fn revenue_measure<'a>(dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
        let group_cols: Vec<Expr> = qc.groups.iter().map(|c| col(c.as_str())).collect();
        dm.table("orders")
            .group_by(group_cols)
            .agg(vec![col("orders.amount").sum()])
    }

    // Same group-by logic; aggregates orders.count.
    fn order_count_measure<'a>(dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
        let group_cols: Vec<Expr> = qc.groups.iter().map(|c| col(c.as_str())).collect();
        dm.table("orders")
            .group_by(group_cols)
            .agg(vec![col("orders.count").sum()])
    }

    fn make_query_dm() -> DataModel {
        let orders = df![
            "orders.date"   => ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
            "orders.region" => ["north", "south", "north", "south"],
            "orders.amount" => [100.0f64, 200.0, 150.0, 250.0],
            "orders.count"  => [1i64, 2, 3, 4],
        ]
        .unwrap()
        .lazy();
        let mut dm = DataModel::new(
            HashMap::from([("orders".to_string(), orders)]),
            JoinGraph::new(&[]).unwrap(),
            vec![],
            None,
        );
        dm.add_measure(Measure::new("revenue", revenue_measure))
            .unwrap();
        dm.add_measure(Measure::new("order_count", order_count_measure))
            .unwrap();
        dm
    }

    #[test]
    fn test_query_single_measure() {
        let dm = make_query_dm();
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
        let df = match dm.query(&qc, false).unwrap() {
            QueryOutput::Data(df) => df,
            _ => panic!("expected Data"),
        };
        assert_eq!(df.height(), 2);
        let total: f64 = df
            .column("orders.amount")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .sum();
        assert!((total - 700.0).abs() < 1e-9);
    }

    #[test]
    fn test_query_two_compatible_measures() {
        let dm = make_query_dm();
        let qc = QueryContext::new(
            vec!["revenue".into(), "order_count".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let df = match dm.query(&qc, false).unwrap() {
            QueryOutput::Data(df) => df,
            _ => panic!("expected Data"),
        };
        assert_eq!(df.height(), 2);
        assert!(df.column("orders.amount").is_ok());
        assert!(df.column("orders.count").is_ok());
    }

    #[test]
    fn test_query_incompatible_measures_errors() {
        let orders = df![
            "orders.date"   => ["2024-01-01"],
            "orders.region" => ["north"],
            "orders.amount" => [100.0f64],
        ]
        .unwrap()
        .lazy();
        let mut dm = DataModel::new(
            HashMap::from([("orders".to_string(), orders)]),
            JoinGraph::new(&[]).unwrap(),
            vec![],
            None,
        );

        fn by_region<'a>(dm: &'a DataModel, _qc: &QueryContext) -> LazyFrameRecorder<'a> {
            dm.table("orders")
                .group_by(vec![col("orders.region")])
                .agg(vec![col("orders.amount").sum()])
        }

        fn by_date<'a>(dm: &'a DataModel, _qc: &QueryContext) -> LazyFrameRecorder<'a> {
            dm.table("orders")
                .group_by(vec![col("orders.date")])
                .agg(vec![col("orders.amount").sum()])
        }

        dm.add_measure(Measure::new("revenue", by_region)).unwrap();
        dm.add_measure(Measure::new("alt_revenue", by_date))
            .unwrap();

        let qc = QueryContext::new(
            vec!["revenue".into(), "alt_revenue".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let err = dm.query(&qc, false).unwrap_err();
        assert!(
            err.contains("incompatible"),
            "expected incompatible error, got: {err}"
        );
    }

    #[test]
    fn test_query_with_having() {
        let dm = make_query_dm();
        // north: 100+150=250, south: 200+250=450 — only south passes > 300
        let havings = serde_json::json!({
            "left": {"col": "orders.amount"},
            "op": ">",
            "right": {"lit": 300.0}
        });
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            Some(havings),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let df = match dm.query(&qc, false).unwrap() {
            QueryOutput::Data(df) => df,
            _ => panic!("expected Data"),
        };
        let amounts: Vec<f64> = df
            .column("orders.amount")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert!(amounts.iter().all(|&v| v > 300.0));
    }

    #[test]
    fn test_query_with_sort() {
        let dm = make_query_dm();
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("orders.amount".into(), "desc".into())]),
            None,
            None,
            None,
        )
        .unwrap();
        let df = match dm.query(&qc, false).unwrap() {
            QueryOutput::Data(df) => df,
            _ => panic!("expected Data"),
        };
        let amounts: Vec<f64> = df
            .column("orders.amount")
            .unwrap()
            .f64()
            .unwrap()
            .into_iter()
            .flatten()
            .collect();
        assert!(amounts.windows(2).all(|w| w[0] >= w[1]));
    }

    #[test]
    fn test_query_limit_offset() {
        let dm = make_query_dm();
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            None,
            Some(1),
            Some(1),
            None,
        )
        .unwrap();
        let df = match dm.query(&qc, false).unwrap() {
            QueryOutput::Data(df) => df,
            _ => panic!("expected Data"),
        };
        assert_eq!(df.height(), 1);
    }
}
