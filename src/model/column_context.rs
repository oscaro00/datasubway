use crate::post_process::filter_tree_to_expr;
use datafusion::common::DataFusionError;
use datafusion_expr::Expr;

/// Input to allow/exclude: either a column list or a filter tree.
pub enum ColumnInput<'a> {
    Columns(&'a [String]),
    FilterTree(&'a serde_json::Value),
}

/// Output from allow/exclude: either column expressions or a filter expression.
pub enum ColumnOutput {
    Exprs(Vec<Expr>),
    FilterExpr(Expr),
}

impl ColumnOutput {
    /// Unwrap as `Vec<Expr>`, panics if this is a `FilterExpr`.
    pub fn into_exprs(self) -> Vec<Expr> {
        match self {
            ColumnOutput::Exprs(exprs) => exprs,
            ColumnOutput::FilterExpr(_) => panic!("Expected ColumnOutput::Exprs, got FilterExpr"),
        }
    }

    /// Unwrap as a single filter `Expr`, panics if this is `Exprs`.
    pub fn into_filter_expr(self) -> Expr {
        match self {
            ColumnOutput::FilterExpr(expr) => expr,
            ColumnOutput::Exprs(_) => panic!("Expected ColumnOutput::FilterExpr, got Exprs"),
        }
    }
}

/// Parsed pattern: (table_part, column_part) where "*" means match any.
#[derive(Debug, Clone)]
struct Pattern {
    table: String,
    column: String,
}

/// Parse a pattern string like "*", "orders.*", "*.amount", or "orders.amount".
fn parse_pattern(pattern: &str) -> Result<Pattern, DataFusionError> {
    if pattern == "*" {
        return Ok(Pattern {
            table: "*".into(),
            column: "*".into(),
        });
    }
    let parts: Vec<&str> = pattern.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err(DataFusionError::Plan(format!(
            "Invalid pattern: '{}'. Expected '*', 'table.*', or 'table.col'.",
            pattern
        )));
    }
    Ok(Pattern {
        table: parts[0].into(),
        column: parts[1].into(),
    })
}

/// Parse a "table.column" string into (table, column).
fn parse_table_column(tc: &str) -> Result<(&str, &str), String> {
    let parts: Vec<&str> = tc.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid table.column format: '{}'. Expected 'table_name.column_name'.",
            tc
        ));
    }
    Ok((parts[0], parts[1]))
}

/// Check if a "table.column" string matches a parsed pattern.
fn matches_pattern(col_str: &str, pattern: &Pattern) -> bool {
    if let Ok((table, column)) = parse_table_column(col_str) {
        let t_match = pattern.table == "*" || pattern.table == table;
        let c_match = pattern.column == "*" || pattern.column == column;
        t_match && c_match
    } else {
        false
    }
}

/// Check if a "table.column" string matches any of the patterns.
fn matches_any(col_str: &str, patterns: &[Pattern]) -> bool {
    patterns.iter().any(|p| matches_pattern(col_str, p))
}

/// Parse one or more pattern strings.
fn parse_patterns(patterns: &[String]) -> Result<Vec<Pattern>, DataFusionError> {
    patterns.iter().map(|p| parse_pattern(p)).collect()
}

/// Prune a JSON filter tree to only keep conditions whose column matches (or doesn't match) the patterns.
///
/// - `invert = false` (allow): keep conditions where the column matches a pattern
/// - `invert = true` (exclude): keep conditions where the column does NOT match any pattern
fn prune_filter_tree(
    tree: &serde_json::Value,
    patterns: &[Pattern],
    invert: bool,
) -> serde_json::Value {
    match tree {
        serde_json::Value::Object(map) => {
            let mut result = serde_json::Map::new();
            for (key, conditions) in map {
                if let Some(arr) = conditions.as_array() {
                    let pruned: Vec<serde_json::Value> = arr
                        .iter()
                        .filter_map(|cond| {
                            let pruned_cond = prune_condition(cond, patterns, invert);
                            // Drop null (pruned leaves) and empty objects (pruned branches)
                            match &pruned_cond {
                                serde_json::Value::Null => None,
                                serde_json::Value::Object(m) if m.is_empty() => None,
                                _ => Some(pruned_cond),
                            }
                        })
                        .collect();
                    if !pruned.is_empty() {
                        result.insert(key.clone(), serde_json::Value::Array(pruned));
                    }
                }
            }
            serde_json::Value::Object(result)
        }
        _ => serde_json::Value::Object(serde_json::Map::new()),
    }
}

/// Prune a single condition (leaf array or nested object).
fn prune_condition(
    condition: &serde_json::Value,
    patterns: &[Pattern],
    invert: bool,
) -> serde_json::Value {
    match condition {
        // Nested filter tree: {"AND": [...]} or {"OR": [...]}
        serde_json::Value::Object(_) => prune_filter_tree(condition, patterns, invert),
        // Leaf condition: ["col", "op", value]
        serde_json::Value::Array(arr) => {
            if arr.len() >= 3 {
                if let Some(col_name) = arr[0].as_str() {
                    let matched = matches_any(col_name, patterns);
                    let keep = if invert { !matched } else { matched };
                    if keep {
                        return condition.clone();
                    }
                }
            }
            serde_json::Value::Null
        }
        _ => serde_json::Value::Null,
    }
}

/// Return matching columns as expressions, or prune a filter tree to matching conditions.
///
/// - `ColumnInput::Columns`: returns `ColumnOutput::Exprs` with col() expressions for columns matching the patterns
/// - `ColumnInput::FilterTree`: returns `ColumnOutput::FilterExpr` with an Expr built from the pruned filter tree
pub fn allow(patterns: &[String], input: ColumnInput) -> Result<ColumnOutput, DataFusionError> {
    let parsed = parse_patterns(patterns)?;
    match input {
        ColumnInput::Columns(context) => {
            let cols: Vec<String> = context
                .iter()
                .filter(|c| matches_any(c, &parsed))
                .cloned()
                .collect();
            Ok(ColumnOutput::Exprs(
                cols.into_iter().map(datafusion_expr::col).collect(),
            ))
        }
        ColumnInput::FilterTree(tree) => {
            let pruned = prune_filter_tree(tree, &parsed, false);
            let expr = filter_tree_to_expr(&pruned)?;
            Ok(ColumnOutput::FilterExpr(expr))
        }
    }
}

/// Return non-matching columns as expressions, or prune a filter tree to non-matching conditions.
///
/// - `ColumnInput::Columns`: returns `ColumnOutput::Exprs` with col() expressions for columns NOT matching the patterns
/// - `ColumnInput::FilterTree`: returns `ColumnOutput::FilterExpr` with an Expr built from conditions whose columns do NOT match
pub fn exclude(patterns: &[String], input: ColumnInput) -> Result<ColumnOutput, DataFusionError> {
    let parsed = parse_patterns(patterns)?;
    match input {
        ColumnInput::Columns(context) => {
            let cols: Vec<String> = context
                .iter()
                .filter(|c| !matches_any(c, &parsed))
                .cloned()
                .collect();
            Ok(ColumnOutput::Exprs(
                cols.into_iter().map(datafusion_expr::col).collect(),
            ))
        }
        ColumnInput::FilterTree(tree) => {
            let pruned = prune_filter_tree(tree, &parsed, true);
            let expr = filter_tree_to_expr(&pruned)?;
            Ok(ColumnOutput::FilterExpr(expr))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ColumnInput::*;
    use serde_json::json;

    fn sample_context() -> Vec<String> {
        vec![
            "orders.region".into(),
            "orders.amount".into(),
            "orders.customer_id".into(),
            "customers.name".into(),
            "customers.country".into(),
            "products.category".into(),
        ]
    }

    // ── Column-based tests ──

    #[test]
    fn test_allow_wildcard() {
        let result = allow(&["*".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_allow_table_wildcard() {
        let result = allow(&["orders.*".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_allow_column_wildcard() {
        let result = allow(&["*.country".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_allow_exact() {
        let result = allow(&["orders.amount".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_exclude_table_wildcard() {
        let result = exclude(&["orders.*".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_exclude_wildcard() {
        let result = exclude(&["*".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_allow_no_match() {
        let result = allow(&["nonexistent.*".into()], Columns(&sample_context())).unwrap().into_exprs();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_allow_multiple_patterns() {
        let result = allow(
            &["orders.region".into(), "customers.name".into()],
            Columns(&sample_context()),
        )
        .unwrap()
        .into_exprs();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_invalid_pattern() {
        let result = parse_pattern("no_dot_no_star");
        assert!(result.is_err());
    }

    // ── Filter tree tests ──

    #[test]
    fn test_allow_filter_tree_prunes_non_matching() {
        let tree = json!({"AND": [
            ["orders.region", "=", "US"],
            ["customers.name", "=", "Bob"]
        ]});
        let expr = allow(&["orders.*".into()], FilterTree(&tree)).unwrap().into_filter_expr();
        let s = format!("{}", expr);
        assert!(s.contains("orders.region"), "Expected orders.region in: {}", s);
        assert!(!s.contains("customers.name"), "Should not contain customers.name in: {}", s);
    }

    #[test]
    fn test_exclude_filter_tree_keeps_non_matching() {
        let tree = json!({"AND": [
            ["orders.region", "=", "US"],
            ["customers.name", "=", "Bob"]
        ]});
        let expr = exclude(&["orders.*".into()], FilterTree(&tree)).unwrap().into_filter_expr();
        let s = format!("{}", expr);
        assert!(!s.contains("orders.region"), "Should not contain orders.region in: {}", s);
        assert!(s.contains("customers.name"), "Expected customers.name in: {}", s);
    }

    #[test]
    fn test_allow_filter_tree_all_pruned_returns_true() {
        let tree = json!({"AND": [
            ["customers.name", "=", "Bob"]
        ]});
        let expr = allow(&["orders.*".into()], FilterTree(&tree)).unwrap().into_filter_expr();
        let s = format!("{}", expr);
        assert!(s.contains("true"), "All pruned should return lit(true), got: {}", s);
    }

    #[test]
    fn test_allow_filter_tree_nested() {
        let tree = json!({"AND": [
            ["orders.region", "=", "US"],
            {"OR": [
                ["orders.amount", ">", 100],
                ["customers.country", "=", "CA"]
            ]}
        ]});
        let expr = allow(&["orders.*".into()], FilterTree(&tree)).unwrap().into_filter_expr();
        let s = format!("{}", expr);
        assert!(s.contains("orders.region"), "Expected orders.region in: {}", s);
        assert!(s.contains("orders.amount"), "Expected orders.amount in: {}", s);
        assert!(!s.contains("customers.country"), "Should not contain customers.country in: {}", s);
    }

    #[test]
    fn test_allow_filter_tree_wildcard_keeps_all() {
        let tree = json!({"AND": [
            ["orders.region", "=", "US"],
            ["customers.name", "=", "Bob"]
        ]});
        let expr = allow(&["*".into()], FilterTree(&tree)).unwrap().into_filter_expr();
        let s = format!("{}", expr);
        assert!(s.contains("orders.region"), "Expected orders.region in: {}", s);
        assert!(s.contains("customers.name"), "Expected customers.name in: {}", s);
    }
}
