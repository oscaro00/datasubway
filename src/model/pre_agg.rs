use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("DATASUBWAY_DEBUG").is_ok() {
            eprintln!("[datasubway pre_agg] {}", format!($($arg)*));
        }
    };
}

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
    /// 2. For each (col, needed_components), all components are stored
    pub fn covers(
        &self,
        non_agg_cols: &[String],
        agg_cols: &HashMap<String, HashSet<String>>,
    ) -> bool {
        debug_log!("covers: checking '{}' (group_by={:?}) against non_agg={:?}", self.name, self.group_by, non_agg_cols);
        let my_group_set: HashSet<&str> = self.group_by.iter().map(|s| s.as_str()).collect();

        // Every non-aggregate column reference must be in this pre-agg's group_by
        for col in non_agg_cols {
            if !my_group_set.contains(col.as_str()) {
                debug_log!("covers: '{}' rejected — non_agg col '{}' not in group_by", self.name, col);
                return false;
            }
        }

        // Check agg component coverage
        for (col, needed) in agg_cols {
            match self.aggregations.get(col) {
                None => {
                    debug_log!("covers: '{}' rejected — agg col '{}' not in aggregations", self.name, col);
                    return false;
                }
                Some(stored) => {
                    let stored_set: HashSet<&str> = stored.iter().map(|s| s.as_str()).collect();
                    for comp in needed {
                        if !stored_set.contains(comp.as_str()) {
                            debug_log!("covers: '{}' rejected — component '{}' for '{}' not stored", self.name, comp, col);
                            return false;
                        }
                    }
                }
            }
        }

        debug_log!("covers: '{}' ACCEPTED", self.name);
        true
    }

    /// Parquet field name for a group-by column.
    /// Encodes the table qualifier using "__" to avoid dot ambiguity.
    /// e.g. "orders.region" → "orders__region"
    pub fn group_by_column_name(col: &str) -> String {
        col.replace('.', "__")
    }

    /// Parquet field name for a stored aggregate component.
    /// Encodes the table qualifier and column name, then appends the component.
    /// e.g. "orders.amount", "sum" → "orders__amount-sum"
    pub fn component_column(col: &str, component: &str) -> String {
        let encoded = col.replace('.', "__");
        format!("{}-{}", encoded, component)
    }

    /// Decode a parquet column name back to (table, bare_name).
    /// Splits on the first "__" separator.
    /// e.g. "orders__amount-sum" → ("orders", "amount-sum")
    /// e.g. "orders__region"     → ("orders", "region")
    pub fn decode_parquet_col_name(name: &str) -> (String, String) {
        match name.find("__") {
            Some(i) => (name[..i].to_string(), name[i + 2..].to_string()),
            None => (String::new(), name.to_string()),
        }
    }

    /// Extract the source table name from the qualified group-by columns.
    /// Returns the table qualifier of the first group-by column (e.g. "orders").
    pub fn source_table(&self) -> Option<&str> {
        self.group_by.first()?.split('.').next()
    }
}

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

    #[test]
    fn test_group_by_column_name() {
        assert_eq!(
            PreAggregation::group_by_column_name("orders.region"),
            "orders__region"
        );
    }

    #[test]
    fn test_component_column_naming() {
        assert_eq!(
            PreAggregation::component_column("orders.amount", "sum"),
            "orders__amount-sum"
        );
    }

    #[test]
    fn test_decode_parquet_col_name() {
        let (table, bare) = PreAggregation::decode_parquet_col_name("orders__amount-sum");
        assert_eq!(table, "orders");
        assert_eq!(bare, "amount-sum");

        let (table, bare) = PreAggregation::decode_parquet_col_name("orders__region");
        assert_eq!(table, "orders");
        assert_eq!(bare, "region");
    }
}
