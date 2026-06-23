use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

use datafusion::prelude::col;

use crate::data_model::DataModel;

/// Maps user-facing agg names to stored component column suffixes.
/// e.g. "mean" needs both "sum" and "count" components stored.
pub fn agg_expansion(agg_name: &str) -> Result<Vec<&'static str>, String> {
    match agg_name {
        "sum" => Ok(vec!["sum"]),
        "count" => Ok(vec!["count"]),
        "min" => Ok(vec!["min"]),
        "max" => Ok(vec!["max"]),
        "mean" => Ok(vec!["sum", "count"]),
        "std" | "var" => Ok(vec!["sum", "sumsq", "count"]),
        _ => Err(format!("Unknown aggregation type: {}", agg_name)),
    }
}

/// Returns the column name for a pre-agg component stored in parquet.
/// e.g. ("orders.amount", "sum") → "orders.amount-sum"
pub fn component_col_name(col: &str, component: &str) -> String {
    format!("{col}-{component}")
}

/// Convert a qualified column name to a parquet-safe dunder name.
/// Replaces `.` with `__` to avoid clashing with DataFusion's `table.column` separator.
/// e.g. "players.player_name" → "players__player_name"
pub fn to_pre_agg_col_name(qualified: &str) -> String {
    qualified.replace('.', "__")
}

/// Generate the parquet column name for a pre-agg component column.
/// e.g. ("player_stats.goals", "sum") → "player_stats__goals__sum"
pub fn pre_agg_component_col_name(qualified_col: &str, component: &str) -> String {
    format!("{}__{component}", to_pre_agg_col_name(qualified_col))
}

/// Maps a DataFusion aggregate function name to the pre-agg components it requires.
pub fn agg_needed_components(agg_fn_name: &str) -> Option<Vec<&'static str>> {
    match agg_fn_name {
        "sum" | "SUM" => Some(vec!["sum"]),
        "count" | "COUNT" => Some(vec!["count"]),
        "min" | "MIN" => Some(vec!["min"]),
        "max" | "MAX" => Some(vec!["max"]),
        "avg" | "AVG" | "mean" => Some(vec!["sum", "count"]),
        "stddev" | "STDDEV" | "stddev_pop" | "STDDEV_POP" => Some(vec!["sum", "sumsq", "count"]),
        "variance" | "VARIANCE" | "var_pop" | "VAR_POP" => Some(vec!["sum", "sumsq", "count"]),
        _ => None,
    }
}

/// A pre-aggregation definition with its metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreAggregation {
    pub name: String,
    pub group_by: Vec<String>,
    /// col_name -> list of component suffixes stored (e.g. "orders.amount" -> ["sum", "count"])
    pub aggregations: HashMap<String, Vec<String>>,
    pub row_count: u64,
    pub written_at: Option<String>,
}

impl PreAggregation {
    /// Create a new PreAggregation, expanding raw agg specs to component lists.
    pub fn new(
        name: String,
        group_by: Vec<String>,
        raw_aggregations: HashMap<String, Vec<String>>,
    ) -> Result<Self, String> {
        if group_by.is_empty() {
            return Err("group_by must not be empty".into());
        }
        if raw_aggregations.is_empty() {
            return Err("aggregations must not be empty".into());
        }

        let mut aggregations: HashMap<String, Vec<String>> = HashMap::new();
        for (col, agg_names) in &raw_aggregations {
            let mut components = HashSet::new();
            for agg_name in agg_names {
                for comp in agg_expansion(agg_name)? {
                    components.insert(comp.to_string());
                }
            }
            let mut sorted: Vec<String> = components.into_iter().collect();
            sorted.sort();
            aggregations.insert(col.clone(), sorted);
        }

        Ok(PreAggregation {
            name,
            group_by,
            aggregations,
            row_count: 0,
            written_at: None,
        })
    }

    /// Check if this pre-agg covers the requested column requirements.
    ///
    /// A pre-agg covers a request if:
    /// 1. All non-aggregate column references (group-by, filter, projection, join keys, etc.)
    ///    are in self.group_by
    /// 2. For each (col, needed_components) aggregate column reference, all components are stored
    pub fn covers(
        &self,
        non_agg_cols: &[String],
        agg_cols: &HashMap<String, HashSet<String>>,
    ) -> bool {
        let my_group_set: HashSet<&str> = self.group_by.iter().map(|s| s.as_str()).collect();

        // Every non-aggregate column reference must be in this pre-agg's group_by
        for col in non_agg_cols {
            if !my_group_set.contains(col.as_str()) {
                return false;
            }
        }

        // Check agg component coverage
        for (col, needed) in agg_cols {
            match self.aggregations.get(col) {
                None => {
                    return false;
                }
                Some(stored) => {
                    let stored_set: HashSet<&str> = stored.iter().map(|s| s.as_str()).collect();
                    for comp in needed {
                        if !stored_set.contains(comp.as_str()) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

// ── DataModel write methods ───────────────────────────────────────────────────

impl DataModel {
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
        use datafusion::prelude::Expr;

        let path = self
            .0
            .pre_agg_path
            .as_deref()
            .ok_or("pre_agg_path not set on DataModel")?;

        let all_col_names = pa.group_by.iter().chain(pa.aggregations.keys());
        let mut referenced_tables: Vec<String> = all_col_names
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .collect::<std::collections::HashSet<_>>()
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

        let non_agg_str: std::collections::HashSet<String> = pa.group_by.iter().cloned().collect();
        let agg_str: HashMap<String, Vec<String>> = pa
            .aggregations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut df = self
            .get_df_table(&base_table, &non_agg_str, &agg_str, false)
            .map_err(|e| e.to_string())?
            .inner;

        let group_by_exprs: Vec<Expr> = pa
            .group_by
            .iter()
            .map(|c| col(c.as_str()).alias(to_pre_agg_col_name(c).as_str()))
            .collect();

        let mut agg_exprs: Vec<Expr> = Vec::new();
        for (qcol, components) in &pa.aggregations {
            for component in components {
                let alias = pre_agg_component_col_name(qcol, component);
                let qcol_expr = || col(qcol.as_str());
                let expr = match component.as_str() {
                    "sum" => sum(qcol_expr()).alias(&alias),
                    "count" => count(qcol_expr()).alias(&alias),
                    "min" => min(qcol_expr()).alias(&alias),
                    "max" => max(qcol_expr()).alias(&alias),
                    "sumsq" => sum(qcol_expr() * qcol_expr()).alias(&alias),
                    other => return Err(format!("unknown pre-agg component '{other}'")),
                };
                agg_exprs.push(expr);
            }
        }

        df = df
            .aggregate(group_by_exprs, agg_exprs)
            .map_err(|e| format!("failed to aggregate for pre-agg: {e}"))?;

        let file_path = format!("{path}/{}.parquet", pa.name);
        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(df.write_parquet(&file_path, DataFrameWriteOptions::new(), None))
            }),
            Err(_) => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(df.write_parquet(&file_path, DataFrameWriteOptions::new(), None)),
        }
        .map_err(|e| format!("failed to write parquet: {e}"))?;

        Ok(())
    }
}

// ── Coverage helpers ──────────────────────────────────────────────────────────

/// Find the best (smallest row_count) pre-agg that covers the request.
pub fn find_best_pre_agg<'a>(
    pre_aggs: &'a [PreAggregation],
    non_agg_cols: &[String],
    agg_cols: &HashMap<String, HashSet<String>>,
) -> Option<&'a PreAggregation> {
    pre_aggs
        .iter()
        .filter(|pa| pa.covers(non_agg_cols, agg_cols))
        .min_by_key(|pa| pa.row_count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pre_agg() -> PreAggregation {
        PreAggregation::new(
            "daily_revenue".into(),
            vec!["orders.date".into(), "orders.region".into()],
            HashMap::from([
                ("orders.amount".into(), vec!["sum".into(), "mean".into()]),
                ("orders.quantity".into(), vec!["sum".into()]),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn test_agg_expansion() {
        assert_eq!(agg_expansion("sum").unwrap(), vec!["sum"]);
        assert!(agg_expansion("mean").unwrap().contains(&"sum"));
        assert!(agg_expansion("mean").unwrap().contains(&"count"));
        assert!(agg_expansion("unknown").is_err());
    }

    #[test]
    fn test_pre_agg_creation() {
        let pa = sample_pre_agg();
        // "mean" expands to sum+count, "sum" adds sum → merged = {sum, count}
        assert!(pa.aggregations["orders.amount"].contains(&"sum".to_string()));
        assert!(pa.aggregations["orders.amount"].contains(&"count".to_string()));
        assert_eq!(pa.aggregations["orders.quantity"], vec!["sum".to_string()]);
    }

    #[test]
    fn test_covers_exact() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.date".into(), "orders.region".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_subset_group_by() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.region".into()]; // subset
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_missing_group_col() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.store".into()]; // not in pre-agg
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(!pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_missing_agg_component() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.date".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sumsq".to_string()]), // not stored
        )]);
        assert!(!pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_filter_column_in_group_by() {
        let pa = sample_pre_agg();
        // Filter on region (non-agg, in group_by) → ok
        let non_agg_cols = vec!["orders.date".into(), "orders.region".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_filter_column_not_in_group_by() {
        let pa = sample_pre_agg();
        // Filter on store, which is NOT in group_by → rejected
        let non_agg_cols = vec!["orders.date".into(), "orders.store".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(!pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_find_best() {
        let mut pa1 = sample_pre_agg();
        pa1.row_count = 1000;

        let mut pa2 = PreAggregation::new(
            "monthly_revenue".into(),
            vec!["orders.date".into(), "orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["sum".into()])]),
        )
        .unwrap();
        pa2.row_count = 100; // smaller

        let pre_aggs = vec![pa1, pa2];
        let non_agg_cols = vec!["orders.date".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);

        let best = find_best_pre_agg(&pre_aggs, &non_agg_cols, &agg_cols);
        assert_eq!(best.unwrap().name, "monthly_revenue");
    }
}
