use serde::{Deserialize, Serialize};

/// Validated query context — mirrors the Python QueryContext but validated in Rust.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryContext {
    pub measures: Vec<String>,
    pub filters: serde_json::Value,
    pub groups: Vec<String>,
    pub havings: serde_json::Value,
    pub sorts: Vec<(String, String)>,
    pub limit: usize,
    pub offset: usize,
    pub use_pre_agg: bool,
}

impl QueryContext {
    pub fn new(
        measures: Vec<String>,
        filters: Option<serde_json::Value>,
        groups: Option<Vec<String>>,
        havings: Option<serde_json::Value>,
        sorts: Option<Vec<(String, String)>>,
        limit: Option<usize>,
        offset: Option<usize>,
        use_pre_agg: Option<bool>,
    ) -> Result<Self, String> {
        if measures.is_empty() {
            return Err("measures must not be empty".into());
        }

        let limit = limit.unwrap_or(10000);
        if limit == 0 {
            return Err("limit must be > 0".into());
        }

        Ok(QueryContext {
            measures,
            filters: filters.unwrap_or(serde_json::Value::Object(Default::default())),
            groups: groups.unwrap_or_default(),
            havings: havings.unwrap_or(serde_json::Value::Object(Default::default())),
            sorts: sorts.unwrap_or_default(),
            limit,
            offset: offset.unwrap_or(0),
            use_pre_agg: use_pre_agg.unwrap_or(true),
        })
    }

    /// Extract all column references from filters (leaf tuples in AND/OR tree).
    /// Filter format: {"AND": [("table.col", "=", value), ...]} or nested.
    pub fn filter_columns(&self) -> Vec<String> {
        extract_columns_from_filter_tree(&self.filters)
    }

    /// Extract all column references from havings filter tree.
    pub fn having_columns(&self) -> Vec<String> {
        extract_columns_from_filter_tree(&self.havings)
    }

    /// Validate the QueryContext against the data model's metadata.
    ///
    /// Checks:
    /// - All measures are known
    /// - All group columns exist in the model
    /// - All filter columns exist in the model
    /// - All having columns are valid (group cols or measure output cols)
    /// - All sort columns are valid and directions are "asc"/"desc"
    pub fn validate(
        &self,
        known_measures: &[MeasureMetadata],
        all_columns: &std::collections::HashSet<String>,
    ) -> Result<(), String> {
        let measure_map: std::collections::HashMap<&str, &MeasureMetadata> = known_measures
            .iter()
            .map(|m| (m.name.as_str(), m))
            .collect();

        // Check measures exist
        for m in &self.measures {
            if !measure_map.contains_key(m.as_str()) {
                return Err(format!("Unknown measure: '{}'", m));
            }
        }

        // Check group columns exist
        for g in &self.groups {
            if !all_columns.contains(g) {
                return Err(format!("Unknown group column: '{}'", g));
            }
        }

        // Check filter columns exist
        let filter_cols = self.filter_columns();
        for fc in &filter_cols {
            if !all_columns.contains(fc) {
                return Err(format!("Unknown filter column: '{}'", fc));
            }
        }

        // Build valid having columns: groups + measure output columns
        let mut valid_having_cols: std::collections::HashSet<String> =
            self.groups.iter().cloned().collect();
        for m in &self.measures {
            if let Some(meta) = measure_map.get(m.as_str()) {
                for col in &meta.output_columns {
                    valid_having_cols.insert(col.clone());
                }
            }
        }

        // Check having columns
        let having_cols = self.having_columns();
        for hc in &having_cols {
            if !valid_having_cols.contains(hc) {
                return Err(format!("Invalid having column: '{}'", hc));
            }
        }

        // Check sort columns and directions
        let valid_sort_cols = &valid_having_cols;
        for (col, direction) in &self.sorts {
            if !valid_sort_cols.contains(col) {
                return Err(format!("Invalid sort column: '{}'", col));
            }
            if direction != "asc" && direction != "desc" {
                return Err(format!("Invalid sort direction: '{}'", direction));
            }
        }

        Ok(())
    }
}

/// Metadata about a registered measure, used for validation.
#[derive(Debug, Clone)]
pub struct MeasureMetadata {
    pub name: String,
    pub output_columns: Vec<String>,
}

fn extract_columns_from_filter_tree(value: &serde_json::Value) -> Vec<String> {
    let mut columns = Vec::new();
    match value {
        serde_json::Value::Object(map) => {
            for (_key, val) in map {
                columns.extend(extract_columns_from_filter_tree(val));
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let serde_json::Value::Array(tuple) = item {
                    // Leaf: ["table.col", "op", value]
                    if tuple.len() >= 2 {
                        if let Some(col) = tuple[0].as_str() {
                            columns.push(col.to_string());
                        }
                    }
                } else {
                    columns.extend(extract_columns_from_filter_tree(item));
                }
            }
        }
        _ => {}
    }
    columns
}

// ── PyO3 wrapper ──

#[cfg(feature = "python")]
pub use py_wrapper::*;

#[cfg(feature = "python")]
mod py_wrapper {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use pythonize::{depythonize, pythonize};

    #[pyclass(name = "QueryContext")]
    #[derive(Debug, Clone)]
    pub struct PyQueryContext {
        pub inner: QueryContext,
    }

    #[pymethods]
    impl PyQueryContext {
        #[new]
        fn new(qc_dict: &Bound<'_, PyDict>) -> PyResult<Self> {
            // measures — required
            let measures: Vec<String> = match qc_dict.get_item("measures")? {
                Some(val) => val.extract().map_err(|_| {
                    PyValueError::new_err("measures must be a non-empty list of strings")
                })?,
                None => {
                    return Err(PyValueError::new_err(
                        "measures must be a non-empty list of strings",
                    ))
                }
            };

            // filters — optional dict, default {}
            let filters: Option<serde_json::Value> = match qc_dict.get_item("filters")? {
                Some(val) => Some(
                    depythonize(&val)
                        .map_err(|_| PyValueError::new_err("filters must be a dict"))?,
                ),
                None => None,
            };

            // groups — optional list, default []
            let groups: Option<Vec<String>> = match qc_dict.get_item("groups")? {
                Some(val) => Some(
                    val.extract()
                        .map_err(|_| PyValueError::new_err("groups must be a list of strings"))?,
                ),
                None => None,
            };

            // havings — optional dict, default {}
            let havings: Option<serde_json::Value> = match qc_dict.get_item("havings")? {
                Some(val) => Some(
                    depythonize(&val)
                        .map_err(|_| PyValueError::new_err("havings must be a dict"))?,
                ),
                None => None,
            };

            // sorts — optional list of (str, str) tuples
            let sorts: Option<Vec<(String, String)>> = match qc_dict.get_item("sorts")? {
                Some(val) => Some(val.extract().map_err(|_| {
                    PyValueError::new_err("sorts must be a list of (column, direction) pairs")
                })?),
                None => None,
            };

            // limit — optional int
            let limit: Option<usize> = match qc_dict.get_item("limit")? {
                Some(val) => Some(
                    val.extract()
                        .map_err(|_| PyValueError::new_err("limit must be a positive integer"))?,
                ),
                None => None,
            };

            // offset — optional int
            let offset: Option<usize> = match qc_dict.get_item("offset")? {
                Some(val) => {
                    let v: i64 = val.extract().map_err(|_| {
                        PyValueError::new_err("offset must be a non-negative integer")
                    })?;
                    if v < 0 {
                        return Err(PyValueError::new_err(
                            "offset must be a non-negative integer",
                        ));
                    }
                    Some(v as usize)
                }
                None => None,
            };

            // use_pre_agg — optional bool
            let use_pre_agg: Option<bool> = match qc_dict.get_item("use_pre_agg")? {
                Some(val) => Some(
                    val.extract()
                        .map_err(|_| PyValueError::new_err("use_pre_agg must be a bool"))?,
                ),
                None => None,
            };

            let inner = QueryContext::new(
                measures, filters, groups, havings, sorts, limit, offset, use_pre_agg,
            )
            .map_err(|e| PyValueError::new_err(e))?;
            Ok(PyQueryContext { inner })
        }

        #[getter]
        fn measures(&self) -> Vec<String> {
            self.inner.measures.clone()
        }

        #[getter]
        fn filters(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            pythonize(py, &self.inner.filters)
                .map(|v| v.unbind())
                .map_err(|e| PyValueError::new_err(format!("failed to convert filters: {e}")))
        }

        #[getter]
        fn groups(&self) -> Vec<String> {
            self.inner.groups.clone()
        }

        #[getter]
        fn havings(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
            pythonize(py, &self.inner.havings)
                .map(|v| v.unbind())
                .map_err(|e| PyValueError::new_err(format!("failed to convert havings: {e}")))
        }

        #[getter]
        fn sorts(&self) -> Vec<(String, String)> {
            self.inner.sorts.clone()
        }

        #[getter]
        fn limit(&self) -> usize {
            self.inner.limit
        }

        #[getter]
        fn offset(&self) -> usize {
            self.inner.offset
        }

        #[getter]
        fn use_pre_agg(&self) -> bool {
            self.inner.use_pre_agg
        }

        fn filter_columns(&self) -> Vec<String> {
            self.inner.filter_columns()
        }

        /// Validate the query context against the data model's metadata.
        fn validate(
            &self,
            measure_names: Vec<String>,
            measure_output_cols: std::collections::HashMap<String, Vec<String>>,
            all_columns: Vec<String>,
        ) -> PyResult<bool> {
            let metadata: Vec<MeasureMetadata> = measure_names
                .into_iter()
                .map(|name| {
                    let output_columns = measure_output_cols
                        .get(&name)
                        .cloned()
                        .unwrap_or_default();
                    MeasureMetadata {
                        name,
                        output_columns,
                    }
                })
                .collect();

            let all_cols_set: std::collections::HashSet<String> =
                all_columns.into_iter().collect();

            self.inner
                .validate(&metadata, &all_cols_set)
                .map(|()| true)
                .map_err(|e| PyValueError::new_err(e))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_creation() {
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
        assert_eq!(qc.measures, vec!["revenue"]);
        assert_eq!(qc.limit, 10000);
        assert_eq!(qc.offset, 0);
        assert!(qc.use_pre_agg);
    }

    #[test]
    fn test_empty_measures_rejected() {
        let result = QueryContext::new(vec![], None, None, None, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_limit_rejected() {
        let result = QueryContext::new(
            vec!["revenue".into()],
            None,
            None,
            None,
            None,
            Some(0),
            None,
            None,
        );
        assert!(result.is_err());
    }

    fn make_measure_meta(name: &str, output_cols: &[&str]) -> MeasureMetadata {
        MeasureMetadata {
            name: name.into(),
            output_columns: output_cols.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn make_all_columns() -> std::collections::HashSet<String> {
        ["orders.region", "orders.amount", "orders.date"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_validate_ok() {
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
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        assert!(qc.validate(&measures, &make_all_columns()).is_ok());
    }

    #[test]
    fn test_validate_unknown_measure() {
        let qc = QueryContext::new(
            vec!["unknown".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown measure"));
    }

    #[test]
    fn test_validate_unknown_group_column() {
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.nonexistent".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown group column"));
    }

    #[test]
    fn test_validate_unknown_filter_column() {
        let filters = json!({"AND": [["orders.nonexistent", "=", "US"]]});
        let qc = QueryContext::new(
            vec!["revenue".into()],
            Some(filters),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown filter column"));
    }

    #[test]
    fn test_validate_invalid_having_column() {
        let havings = json!({"AND": [["nonexistent", ">", 500]]});
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
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid having column"));
    }

    #[test]
    fn test_validate_valid_having_on_measure_output() {
        let havings = json!({"AND": [["revenue", ">", 500]]});
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
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        assert!(qc.validate(&measures, &make_all_columns()).is_ok());
    }

    #[test]
    fn test_validate_invalid_sort_direction() {
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("revenue".into(), "up".into())]),
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid sort direction"));
    }

    #[test]
    fn test_validate_invalid_sort_column() {
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("nonexistent".into(), "asc".into())]),
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid sort column"));
    }

    #[test]
    fn test_filter_column_extraction() {
        let filters = json!({
            "AND": [
                ["orders.region", "=", "US"],
                ["orders.date", ">", "2024-01-01"]
            ]
        });
        let qc = QueryContext::new(
            vec!["revenue".into()],
            Some(filters),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let cols = qc.filter_columns();
        assert!(cols.contains(&"orders.region".to_string()));
        assert!(cols.contains(&"orders.date".to_string()));
    }
}
