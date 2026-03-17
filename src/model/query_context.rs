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
