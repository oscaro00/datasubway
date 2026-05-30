use serde::{Deserialize, Serialize};

use crate::column_expressions::filter_expr::extract_filter_cols;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectContext {
    pub columns: Vec<String>,
    pub filters: serde_json::Value,
    pub sorts: Vec<(String, String)>,
    pub limit: usize,
    pub offset: usize,
}

impl SelectContext {
    pub fn new(
        columns: Vec<String>,
        filters: Option<serde_json::Value>,
        sorts: Option<Vec<(String, String)>>,
        limit: Option<usize>,
        offset: Option<usize>,
    ) -> Result<Self, String> {
        if columns.is_empty() {
            return Err("columns must not be empty".into());
        }

        let limit = limit.unwrap_or(10000);
        if limit == 0 {
            return Err("limit must be > 0".into());
        }

        Ok(SelectContext {
            columns,
            filters: filters.unwrap_or(serde_json::Value::Object(Default::default())),
            sorts: sorts.unwrap_or_default(),
            limit,
            offset: offset.unwrap_or(0),
        })
    }

    pub fn filter_columns(&self) -> Vec<String> {
        extract_filter_cols(&self.filters)
    }

    pub fn validate(&self, all_columns: &std::collections::HashSet<String>) -> Result<(), String> {
        for c in &self.columns {
            if !all_columns.contains(c) {
                return Err(format!("Unknown column: '{}'", c));
            }
        }

        for fc in &self.filter_columns() {
            if !all_columns.contains(fc) {
                return Err(format!("Unknown filter column: '{}'", fc));
            }
        }

        let col_set: std::collections::HashSet<&String> = self.columns.iter().collect();
        for (col, direction) in &self.sorts {
            if !col_set.contains(col) {
                return Err(format!("Sort column not in selected columns: '{}'", col));
            }
            if direction != "asc" && direction != "desc" {
                return Err(format!("Invalid sort direction: '{}'", direction));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_all_columns() -> std::collections::HashSet<String> {
        ["orders.region", "orders.amount", "orders.date"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_basic_creation() {
        let sc = SelectContext::new(
            vec!["orders.region".into(), "orders.amount".into()],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(sc.columns, vec!["orders.region", "orders.amount"]);
        assert_eq!(sc.limit, 10000);
        assert_eq!(sc.offset, 0);
    }

    #[test]
    fn test_empty_columns_rejected() {
        let result = SelectContext::new(vec![], None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_limit_rejected() {
        let result = SelectContext::new(vec!["orders.region".into()], None, None, Some(0), None);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_ok() {
        let sc = SelectContext::new(
            vec!["orders.region".into(), "orders.amount".into()],
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert!(sc.validate(&make_all_columns()).is_ok());
    }

    #[test]
    fn test_validate_unknown_column() {
        let sc =
            SelectContext::new(vec!["orders.nonexistent".into()], None, None, None, None).unwrap();
        let err = sc.validate(&make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown column"));
    }

    #[test]
    fn test_validate_unknown_filter_column() {
        let filters = json!({"and": [{"left": {"col": "orders.nonexistent"}, "op": "=", "right": {"lit": "US"}}]});
        let sc = SelectContext::new(
            vec!["orders.region".into()],
            Some(filters),
            None,
            None,
            None,
        )
        .unwrap();
        let err = sc.validate(&make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown filter column"));
    }

    #[test]
    fn test_validate_sort_not_in_columns() {
        let sc = SelectContext::new(
            vec!["orders.region".into()],
            None,
            Some(vec![("orders.amount".into(), "asc".into())]),
            None,
            None,
        )
        .unwrap();
        let err = sc.validate(&make_all_columns()).unwrap_err();
        assert!(err.contains("Sort column not in selected columns"));
    }

    #[test]
    fn test_validate_invalid_sort_direction() {
        let sc = SelectContext::new(
            vec!["orders.region".into()],
            None,
            Some(vec![("orders.region".into(), "up".into())]),
            None,
            None,
        )
        .unwrap();
        let err = sc.validate(&make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid sort direction"));
    }

    #[test]
    fn test_filter_column_extraction() {
        let filters = json!({
            "and": [
                {"left": {"col": "orders.region"}, "op": "=", "right": {"lit": "US"}},
                {"left": {"col": "orders.date"}, "op": ">", "right": {"lit": "2024-01-01"}}
            ]
        });
        let sc = SelectContext::new(
            vec!["orders.region".into()],
            Some(filters),
            None,
            None,
            None,
        )
        .unwrap();
        let cols = sc.filter_columns();
        assert!(cols.contains(&"orders.region".to_string()));
        assert!(cols.contains(&"orders.date".to_string()));
    }
}
