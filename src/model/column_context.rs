use datafusion_expr::Expr;

/// Parsed pattern: (table_part, column_part) where "*" means match any.
#[derive(Debug, Clone)]
struct Pattern {
    table: String,
    column: String,
}

/// Parse a pattern string like "*", "orders.*", "*.amount", or "orders.amount".
fn parse_pattern(pattern: &str) -> Result<Pattern, String> {
    if pattern == "*" {
        return Ok(Pattern {
            table: "*".into(),
            column: "*".into(),
        });
    }
    let parts: Vec<&str> = pattern.splitn(2, '.').collect();
    if parts.len() != 2 {
        return Err(format!(
            "Invalid pattern: '{}'. Expected '*', 'table.*', '*.col', or 'table.col'.",
            pattern
        ));
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
fn parse_patterns(patterns: &[String]) -> Result<Vec<Pattern>, String> {
    patterns.iter().map(|p| parse_pattern(p)).collect()
}

/// Return columns from `context` that match the pattern(s).
/// Returns qualified "table.column" strings.
pub fn allow_columns(patterns: &[String], context: &[String]) -> Result<Vec<String>, String> {
    let parsed = parse_patterns(patterns)?;
    Ok(context
        .iter()
        .filter(|c| matches_any(c, &parsed))
        .cloned()
        .collect())
}

/// Return columns from `context` that do NOT match the pattern(s).
/// Returns qualified "table.column" strings.
pub fn exclude_columns(patterns: &[String], context: &[String]) -> Result<Vec<String>, String> {
    let parsed = parse_patterns(patterns)?;
    Ok(context
        .iter()
        .filter(|c| !matches_any(c, &parsed))
        .cloned()
        .collect())
}

/// Return DataFusion col() expressions for columns matching the pattern(s).
pub fn allow_exprs(patterns: &[String], context: &[String]) -> Result<Vec<Expr>, String> {
    let cols = allow_columns(patterns, context)?;
    Ok(cols.into_iter().map(|c| datafusion_expr::col(c)).collect())
}

/// Return DataFusion col() expressions for columns NOT matching the pattern(s).
pub fn exclude_exprs(patterns: &[String], context: &[String]) -> Result<Vec<Expr>, String> {
    let cols = exclude_columns(patterns, context)?;
    Ok(cols.into_iter().map(|c| datafusion_expr::col(c)).collect())
}

// ── Convenience wrappers for single-pattern usage ──

/// Single-pattern allow: return col expressions matching the pattern.
pub fn allow(pattern: &str, context: &[String]) -> Result<Vec<Expr>, String> {
    allow_exprs(&[pattern.to_string()], context)
}

/// Single-pattern exclude: return col expressions NOT matching the pattern.
pub fn exclude(pattern: &str, context: &[String]) -> Result<Vec<Expr>, String> {
    exclude_exprs(&[pattern.to_string()], context)
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn test_allow_wildcard() {
        let result = allow_columns(&["*".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 6);
    }

    #[test]
    fn test_allow_table_wildcard() {
        let result = allow_columns(&["orders.*".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains(&"orders.region".to_string()));
        assert!(result.contains(&"orders.amount".to_string()));
        assert!(result.contains(&"orders.customer_id".to_string()));
    }

    #[test]
    fn test_allow_column_wildcard() {
        let result = allow_columns(&["*.country".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result.contains(&"customers.country".to_string()));
    }

    #[test]
    fn test_allow_exact() {
        let result = allow_columns(&["orders.amount".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0], "orders.amount");
    }

    #[test]
    fn test_exclude_table_wildcard() {
        let result = exclude_columns(&["orders.*".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains(&"customers.name".to_string()));
        assert!(result.contains(&"customers.country".to_string()));
        assert!(result.contains(&"products.category".to_string()));
    }

    #[test]
    fn test_exclude_wildcard() {
        let result = exclude_columns(&["*".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_allow_no_match() {
        let result = allow_columns(&["nonexistent.*".into()], &sample_context()).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_allow_returns_exprs() {
        let exprs = allow("orders.*", &sample_context()).unwrap();
        assert_eq!(exprs.len(), 3);
    }

    #[test]
    fn test_exclude_returns_exprs() {
        let exprs = exclude("orders.*", &sample_context()).unwrap();
        assert_eq!(exprs.len(), 3);
    }

    #[test]
    fn test_invalid_pattern() {
        let result = parse_pattern("no_dot_no_star");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_patterns() {
        let result = allow_columns(
            &["orders.region".into(), "customers.name".into()],
            &sample_context(),
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert!(result.contains(&"orders.region".to_string()));
        assert!(result.contains(&"customers.name".to_string()));
    }
}
