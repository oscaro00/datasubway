//! Convert JSON filter trees to DataFusion expressions.
//!
//! Filter tree format:
//! ```json
//! {"AND": [["col", "op", value], {"OR": [...]}]}
//! ```
//!
//! Values can be literals (strings, numbers, booleans, null) or column references:
//! ```json
//! ["col1", "<=", {"column": "col2"}]
//! ```
//!
//! Supported operators: =, !=, >, >=, <, <=, in, not in

use datafusion::common::DataFusionError;
use datafusion::prelude::*;
use datafusion_expr::Expr;

/// Parse a JSON filter tree into a DataFusion `Expr`.
///
/// Expects a JSON object with a single key ("AND" or "OR") whose value is an array
/// of conditions (leaf tuples or nested filter trees). Returns `lit(true)` for empty objects.
pub fn filter_tree_to_expr(filter_tree: &serde_json::Value) -> Result<Expr, DataFusionError> {
    match filter_tree {
        serde_json::Value::Object(map) => {
            for (key, conditions) in map {
                let conditions = conditions.as_array().ok_or_else(|| {
                    DataFusionError::Plan(format!("Filter tree '{}' value must be an array", key))
                })?;

                let exprs: Vec<Expr> = conditions
                    .iter()
                    .map(|cond| condition_to_expr(cond))
                    .collect::<Result<Vec<_>, _>>()?;

                if exprs.is_empty() {
                    return Ok(lit(true));
                }

                let combined = match key.to_uppercase().as_str() {
                    "AND" => exprs.into_iter().reduce(|a, b| a.and(b)).unwrap(),
                    "OR" => exprs.into_iter().reduce(|a, b| a.or(b)).unwrap(),
                    _ => {
                        return Err(DataFusionError::Plan(format!(
                            "Unknown filter operator: '{}'. Expected 'AND' or 'OR'",
                            key
                        )))
                    }
                };
                return Ok(combined);
            }
            // Empty object = no filter
            Ok(lit(true))
        }
        _ => Err(DataFusionError::Plan(
            "Filter tree must be a JSON object".into(),
        )),
    }
}

/// Convert a single condition (leaf tuple or nested object) to a DataFusion Expr.
fn condition_to_expr(condition: &serde_json::Value) -> Result<Expr, DataFusionError> {
    match condition {
        // Nested filter tree: {"AND": [...]} or {"OR": [...]}
        serde_json::Value::Object(_) => filter_tree_to_expr(condition),

        // Leaf condition: ["col", "op", value]
        serde_json::Value::Array(arr) => {
            if arr.len() < 3 {
                return Err(DataFusionError::Plan(
                    "Filter condition must have at least 3 elements: [col, op, value]".into(),
                ));
            }

            let col_name = arr[0].as_str().ok_or_else(|| {
                DataFusionError::Plan("Filter column name must be a string".into())
            })?;
            let op = arr[1]
                .as_str()
                .ok_or_else(|| DataFusionError::Plan("Filter operator must be a string".into()))?;
            let value = &arr[2];

            let column = col(col_name);

            match op {
                "=" => Ok(column.eq(json_to_expr(value)?)),
                "!=" => Ok(column.not_eq(json_to_expr(value)?)),
                ">" => Ok(column.gt(json_to_expr(value)?)),
                ">=" => Ok(column.gt_eq(json_to_expr(value)?)),
                "<" => Ok(column.lt(json_to_expr(value)?)),
                "<=" => Ok(column.lt_eq(json_to_expr(value)?)),
                "in" => {
                    let list = json_to_expr_list(value)?;
                    Ok(column.in_list(list, false))
                }
                "not in" => {
                    let list = json_to_expr_list(value)?;
                    Ok(column.in_list(list, true))
                }
                _ => Err(DataFusionError::Plan(format!(
                    "Unknown filter operator: '{}'",
                    op
                ))),
            }
        }
        _ => Err(DataFusionError::Plan(
            "Filter condition must be an array or object".into(),
        )),
    }
}

/// Convert a JSON value to a DataFusion expression (literal or column reference).
///
/// Column references use the tagged object format: `{"column": "col_name"}`.
/// All other values are treated as literals.
fn json_to_expr(value: &serde_json::Value) -> Result<Expr, DataFusionError> {
    match value {
        serde_json::Value::Object(map) => {
            if let Some(col_name) = map.get("column") {
                let col_name = col_name.as_str().ok_or_else(|| {
                    DataFusionError::Plan("Column reference name must be a string".into())
                })?;
                Ok(col(col_name))
            } else {
                Err(DataFusionError::Plan(format!(
                    "Unsupported object in filter value: {:?}. Use {{\"column\": \"name\"}} for column references",
                    map
                )))
            }
        }
        serde_json::Value::String(s) => Ok(lit(s.as_str())),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Ok(lit(i))
            } else if let Some(f) = n.as_f64() {
                Ok(lit(f))
            } else {
                Err(DataFusionError::Plan(format!(
                    "Unsupported numeric value: {}",
                    n
                )))
            }
        }
        serde_json::Value::Bool(b) => Ok(lit(*b)),
        serde_json::Value::Null => Ok(lit(datafusion::scalar::ScalarValue::Null)),
        _ => Err(DataFusionError::Plan(format!(
            "Unsupported literal value type: {:?}",
            value
        ))),
    }
}

/// Convert a JSON array to a list of DataFusion expressions (for in/not in).
fn json_to_expr_list(value: &serde_json::Value) -> Result<Vec<Expr>, DataFusionError> {
    let arr = value
        .as_array()
        .ok_or_else(|| DataFusionError::Plan("'in'/'not in' value must be an array".into()))?;
    arr.iter().map(json_to_expr).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_filter_tree_to_expr_simple_and() {
        let tree = json!({
            "AND": [
                ["revenue", ">", 500],
                ["region", "=", "US"]
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("revenue"));
        assert!(display.contains("region"));
    }

    #[test]
    fn test_filter_tree_to_expr_or() {
        let tree = json!({
            "OR": [
                ["region", "=", "US"],
                ["region", "=", "EU"]
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("OR"));
    }

    #[test]
    fn test_filter_tree_to_expr_nested() {
        let tree = json!({
            "AND": [
                ["revenue", ">", 100],
                {"OR": [
                    ["region", "=", "US"],
                    ["region", "=", "EU"]
                ]}
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("revenue"));
        assert!(display.contains("OR"));
    }

    #[test]
    fn test_filter_tree_to_expr_in_operator() {
        let tree = json!({
            "AND": [
                ["region", "in", ["US", "EU"]]
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("region"));
        assert!(display.contains("IN"));
    }

    #[test]
    fn test_filter_tree_to_expr_not_in_operator() {
        let tree = json!({
            "AND": [
                ["region", "not in", ["APAC"]]
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("NOT"));
    }

    #[test]
    fn test_filter_tree_empty_object() {
        let tree = json!({});
        let expr = filter_tree_to_expr(&tree).unwrap();
        assert_eq!(format!("{}", expr), "Boolean(true)");
    }

    #[test]
    fn test_filter_tree_column_to_column() {
        let tree = json!({
            "AND": [
                ["col1", "<=", {"column": "col2"}]
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("col1"), "Expected col1 in: {}", display);
        assert!(display.contains("col2"), "Expected col2 in: {}", display);
    }

    #[test]
    fn test_filter_tree_mixed_literal_and_column_ref() {
        let tree = json!({
            "AND": [
                ["revenue", ">", 500],
                ["col1", "=", {"column": "col2"}]
            ]
        });
        let expr = filter_tree_to_expr(&tree).unwrap();
        let display = format!("{}", expr);
        assert!(display.contains("revenue"));
        assert!(display.contains("col1"));
        assert!(display.contains("col2"));
    }

    #[test]
    fn test_filter_tree_all_operators() {
        for (op, _) in &[
            ("=", "eq"),
            ("!=", "noteq"),
            (">", "gt"),
            (">=", "gteq"),
            ("<", "lt"),
            ("<=", "lteq"),
        ] {
            let tree = json!({"AND": [["col", op, 42]]});
            assert!(filter_tree_to_expr(&tree).is_ok(), "Failed for op: {}", op);
        }
    }
}
