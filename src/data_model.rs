use std::collections::{HashMap, HashSet};

use polars::prelude::{
    col, lit, Expr, JoinArgs, JoinType, LazyFrame, PlRefPath, PlSmallStr, Schema,
};

use crate::{
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
        mut tables: HashMap<String, LazyFrame>,
        joins: JoinGraph,
        pre_aggs: Vec<PreAggregation>,
        pre_agg_path: Option<String>,
    ) -> DataModel {
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
            use_pre_agg: false,
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

        // --- 4. Build group_by expressions (unqualified column name) ---
        let group_by_exprs: Vec<Expr> = pa
            .group_by
            .iter()
            .map(|qcol| {
                let col_name = qcol
                    .split_once('.')
                    .map(|(_, c)| c)
                    .unwrap_or(qcol.as_str());
                col(col_name)
            })
            .collect();

        // --- 5. Build agg expressions, one per component ---
        let mut agg_exprs: Vec<Expr> = Vec::new();
        for (qcol, components) in &pa.aggregations {
            let col_name = qcol
                .split_once('.')
                .map(|(_, c)| c)
                .unwrap_or(qcol.as_str());
            for component in components {
                let alias = component_col_name(qcol, component);
                let expr = match component.as_str() {
                    "sum" => col(col_name).sum().alias(&alias),
                    "count" => col(col_name).count().alias(&alias),
                    "min" => col(col_name).min().alias(&alias),
                    "max" => col(col_name).max().alias(&alias),
                    "sumsq" => col(col_name).pow(lit(2.0f64)).sum().alias(&alias),
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

        // --- 7. Rename group-by columns to their qualified form ---
        for qcol in &pa.group_by {
            if let Some((_, col_name)) = qcol.split_once('.') {
                if col_name != qcol.as_str() {
                    df.rename(col_name, qcol.as_str().into())
                        .map_err(|e| format!("failed to rename '{col_name}' → '{qcol}': {e}"))?;
                }
            }
        }

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
    ) -> LazyFrameWrapper {
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

        LazyFrameWrapper {
            lazyframe: self
                .tables
                .get(table_name)
                .unwrap_or_else(|| panic!("table '{table_name}' not found in DataModel"))
                .clone(),
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
            HashMap::from([(
                "orders.amount".into(),
                vec!["sum".into(), "mean".into()],
            )]),
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
        let mut lf = LazyFrame::scan_parquet(
            PlRefPath::from(path.to_str().unwrap()),
            Default::default(),
        )
        .unwrap();
        let schema = lf.collect_schema().unwrap();
        let cols: Vec<&str> = schema.iter_names().map(|n| n.as_str()).collect();
        assert!(cols.contains(&"orders.date"), "missing orders.date");
        assert!(cols.contains(&"orders.region"), "missing orders.region");
        assert!(cols.contains(&"orders.amount-sum"), "missing orders.amount-sum");
        assert!(cols.contains(&"orders.amount-count"), "missing orders.amount-count");
    }

    #[test]
    fn test_write_pre_agg_correct_data() {
        let (_, tmp) = write_and_get_tmp(daily_revenue_pre_agg());
        let path = tmp.path().join("daily_revenue.parquet");
        let df = LazyFrame::scan_parquet(
            PlRefPath::from(path.to_str().unwrap()),
            Default::default(),
        )
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
            .sort(["orders.date", "orders.region"], SortMultipleOptions::default())
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
        make_orders_dm(
            None,
            vec![daily_revenue_pre_agg(), daily_revenue_pre_agg()],
        );
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
}
