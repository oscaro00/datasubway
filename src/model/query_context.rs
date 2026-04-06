use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use pythonize::{depythonize, pythonize};
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
                depythonize(&val).map_err(|_| PyValueError::new_err("filters must be a dict"))?,
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
                depythonize(&val).map_err(|_| PyValueError::new_err("havings must be a dict"))?,
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
                let v: i64 = val
                    .extract()
                    .map_err(|_| PyValueError::new_err("offset must be a non-negative integer"))?;
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
            measures,
            filters,
            groups,
            havings,
            sorts,
            limit,
            offset,
            use_pre_agg,
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
