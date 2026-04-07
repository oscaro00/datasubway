use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

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
    pub file_path: String,
    pub row_count: u64,
    pub written_at: Option<String>,
}

impl PreAggregation {
    /// Create a new PreAggregation, expanding raw agg specs to component lists.
    pub fn new(
        name: String,
        group_by: Vec<String>,
        raw_aggregations: HashMap<String, Vec<String>>,
        file_path: String,
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
            file_path,
            row_count: 0,
            written_at: None,
        })
    }

    /// Check if this pre-agg covers the requested group_by and aggregation needs.
    ///
    /// A pre-agg covers a request if:
    /// 1. requested_group_by ⊆ self.group_by
    /// 2. For each (col, needed_components), all components are stored
    /// 3. All filter_columns are in self.group_by (filters need raw values)
    pub fn covers(
        &self,
        requested_group_by: &[String],
        requested_agg_components: &HashMap<String, HashSet<String>>,
        filter_columns: &[String],
    ) -> bool {
        let my_group_set: HashSet<&str> = self.group_by.iter().map(|s| s.as_str()).collect();

        // Check group_by coverage
        for col in requested_group_by {
            if !my_group_set.contains(col.as_str()) {
                return false;
            }
        }

        // Check filter columns are in group_by (filters need raw values at the right granularity)
        for col in filter_columns {
            if !my_group_set.contains(col.as_str()) {
                return false;
            }
        }

        // Check agg component coverage
        for (col, needed) in requested_agg_components {
            match self.aggregations.get(col) {
                None => return false,
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

    /// Column name for a stored component, e.g. "orders.amount-sum"
    pub fn component_column(col: &str, component: &str) -> String {
        format!("{}-{}", col, component)
    }
}

/// Find the best (smallest row_count) pre-agg that covers the request.
pub fn find_best_pre_agg<'a>(
    pre_aggs: &'a [PreAggregation],
    requested_group_by: &[String],
    requested_agg_components: &HashMap<String, HashSet<String>>,
    filter_columns: &[String],
) -> Option<&'a PreAggregation> {
    pre_aggs
        .iter()
        .filter(|pa| pa.covers(requested_group_by, requested_agg_components, filter_columns))
        .min_by_key(|pa| pa.row_count)
}

// ── PyO3 wrapper ──

#[cfg(feature = "python")]
pub use py_wrapper::*;

#[cfg(feature = "python")]
mod py_wrapper {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    #[pyclass(name = "PreAggregation")]
    #[derive(Debug, Clone)]
    pub struct PyPreAggregation {
        pub inner: PreAggregation,
    }

    #[pymethods]
    impl PyPreAggregation {
        #[new]
        fn new(
            name: String,
            group_by: Vec<String>,
            raw_aggregations: HashMap<String, Vec<String>>,
            file_path: String,
        ) -> PyResult<Self> {
            let inner = PreAggregation::new(name, group_by, raw_aggregations, file_path)
                .map_err(|e| PyValueError::new_err(e))?;
            Ok(PyPreAggregation { inner })
        }

        #[getter]
        fn name(&self) -> &str {
            &self.inner.name
        }

        #[getter]
        fn group_by(&self) -> Vec<String> {
            self.inner.group_by.clone()
        }

        #[getter]
        fn aggregations(&self) -> HashMap<String, Vec<String>> {
            self.inner.aggregations.clone()
        }

        #[getter]
        fn file_path(&self) -> &str {
            &self.inner.file_path
        }

        #[getter]
        fn row_count(&self) -> u64 {
            self.inner.row_count
        }

        #[setter]
        fn set_row_count(&mut self, count: u64) {
            self.inner.row_count = count;
        }

        #[getter]
        fn written_at(&self) -> Option<String> {
            self.inner.written_at.clone()
        }

        #[setter]
        fn set_written_at(&mut self, val: Option<String>) {
            self.inner.written_at = val;
        }

        fn covers(
            &self,
            requested_group_by: Vec<String>,
            requested_agg_components: HashMap<String, HashSet<String>>,
            filter_columns: Vec<String>,
        ) -> bool {
            self.inner.covers(
                &requested_group_by,
                &requested_agg_components,
                &filter_columns,
            )
        }
    }
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
            "_pre_aggregations/daily_revenue.parquet".into(),
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
        let group_by = vec!["orders.date".into(), "orders.region".into()];
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&group_by, &agg_components, &[]));
    }

    #[test]
    fn test_covers_subset_group_by() {
        let pa = sample_pre_agg();
        let group_by = vec!["orders.region".into()]; // subset
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&group_by, &agg_components, &[]));
    }

    #[test]
    fn test_covers_missing_group_col() {
        let pa = sample_pre_agg();
        let group_by = vec!["orders.store".into()]; // not in pre-agg
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(!pa.covers(&group_by, &agg_components, &[]));
    }

    #[test]
    fn test_covers_missing_agg_component() {
        let pa = sample_pre_agg();
        let group_by = vec!["orders.date".into()];
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sumsq".to_string()]), // not stored
        )]);
        assert!(!pa.covers(&group_by, &agg_components, &[]));
    }

    #[test]
    fn test_covers_filter_column_in_group_by() {
        let pa = sample_pre_agg();
        let group_by = vec!["orders.date".into()];
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        // Filter on region, which IS in group_by → ok
        assert!(pa.covers(&group_by, &agg_components, &["orders.region".into()]));
    }

    #[test]
    fn test_covers_filter_column_not_in_group_by() {
        let pa = sample_pre_agg();
        let group_by = vec!["orders.date".into()];
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        // Filter on store, which is NOT in group_by → rejected
        assert!(!pa.covers(&group_by, &agg_components, &["orders.store".into()]));
    }

    #[test]
    fn test_find_best() {
        let mut pa1 = sample_pre_agg();
        pa1.row_count = 1000;

        let mut pa2 = PreAggregation::new(
            "monthly_revenue".into(),
            vec!["orders.date".into(), "orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["sum".into()])]),
            "_pre_aggregations/monthly_revenue.parquet".into(),
        )
        .unwrap();
        pa2.row_count = 100; // smaller

        let pre_aggs = vec![pa1, pa2];
        let group_by = vec!["orders.date".into()];
        let agg_components = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);

        let best = find_best_pre_agg(&pre_aggs, &group_by, &agg_components, &[]);
        assert_eq!(best.unwrap().name, "monthly_revenue");
    }

    #[test]
    fn test_component_column_naming() {
        assert_eq!(
            PreAggregation::component_column("orders.amount", "sum"),
            "orders.amount-sum"
        );
    }
}
