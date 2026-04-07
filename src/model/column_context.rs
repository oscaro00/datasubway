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
    Ok(cols
        .into_iter()
        .map(|c| datafusion_expr::col(c))
        .collect())
}

/// Return DataFusion col() expressions for columns NOT matching the pattern(s).
pub fn exclude_exprs(patterns: &[String], context: &[String]) -> Result<Vec<Expr>, String> {
    let cols = exclude_columns(patterns, context)?;
    Ok(cols
        .into_iter()
        .map(|c| datafusion_expr::col(c))
        .collect())
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

// ── Filter expression building (used by Python PyO3 functions) ──

/// Extract column names from a filter tree (dict).
fn extract_filter_columns(value: &serde_json::Value) -> Vec<String> {
    let mut columns = Vec::new();
    match value {
        serde_json::Value::Object(map) => {
            for (_key, val) in map {
                columns.extend(extract_filter_columns(val));
            }
        }
        serde_json::Value::Array(arr) => {
            for item in arr {
                if let serde_json::Value::Array(tuple) = item {
                    if tuple.len() >= 2 {
                        if let Some(col) = tuple[0].as_str() {
                            columns.push(col.to_string());
                        }
                    }
                } else {
                    columns.extend(extract_filter_columns(item));
                }
            }
        }
        _ => {}
    }
    columns
}

// ── PyO3 functions ──

#[cfg(feature = "python")]
pub use py_wrapper::*;

#[cfg(feature = "python")]
mod py_wrapper {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;
    use pythonize::depythonize;

    /// Build a Python DataFusion Expr from a filter dict tree, scoped to allowed columns.
    fn build_filter_py_expr<'py>(
        py: Python<'py>,
        filter_json: &serde_json::Value,
        allowed_columns: &[String],
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let df_mod = py.import("datafusion")?;
        let df_col = df_mod.getattr("col")?;
        let df_lit = df_mod.getattr("lit")?;
        let df_functions = py.import("datafusion.functions")?;
        let df_in_list = df_functions.getattr("in_list")?;

        build_filter_node(
            py,
            filter_json,
            allowed_columns,
            &df_col,
            &df_lit,
            &df_in_list,
        )
    }

    fn build_filter_node<'py>(
        py: Python<'py>,
        value: &serde_json::Value,
        allowed_columns: &[String],
        df_col: &Bound<'py, PyAny>,
        df_lit: &Bound<'py, PyAny>,
        df_in_list: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        match value {
            serde_json::Value::Object(map) => {
                for (key, conditions) in map {
                    let is_and = key == "AND";
                    if let serde_json::Value::Array(arr) = conditions {
                        let mut exprs: Vec<Bound<'py, PyAny>> = Vec::new();
                        for item in arr {
                            match item {
                                serde_json::Value::Array(tuple) if tuple.len() >= 3 => {
                                    let col_name = tuple[0].as_str().ok_or_else(|| {
                                        PyValueError::new_err("filter column must be a string")
                                    })?;
                                    if !allowed_columns.contains(&col_name.to_string()) {
                                        continue;
                                    }
                                    let expr = build_leaf_expr(
                                        py, col_name, &tuple[1], &tuple[2], df_col, df_lit,
                                        df_in_list,
                                    )?;
                                    exprs.push(expr);
                                }
                                serde_json::Value::Object(_) => {
                                    if let Some(nested) = build_filter_node(
                                        py,
                                        item,
                                        allowed_columns,
                                        df_col,
                                        df_lit,
                                        df_in_list,
                                    )? {
                                        exprs.push(nested);
                                    }
                                }
                                _ => {}
                            }
                        }
                        if exprs.is_empty() {
                            return Ok(None);
                        }
                        let mut combined = exprs.remove(0);
                        for e in exprs {
                            combined = if is_and {
                                combined.call_method1("__and__", (e,))?
                            } else {
                                combined.call_method1("__or__", (e,))?
                            };
                        }
                        return Ok(Some(combined));
                    }
                }
                Ok(None)
            }
            _ => Ok(None),
        }
    }

    fn build_leaf_expr<'py>(
        py: Python<'py>,
        col_name: &str,
        op_val: &serde_json::Value,
        value: &serde_json::Value,
        df_col: &Bound<'py, PyAny>,
        df_lit: &Bound<'py, PyAny>,
        df_in_list: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let op = op_val
            .as_str()
            .ok_or_else(|| PyValueError::new_err("filter operator must be a string"))?;
        let c = df_col.call1((col_name,))?;

        match op {
            "in" | "not in" => {
                let arr = value.as_array().ok_or_else(|| {
                    PyValueError::new_err(format!("'{op}' operator requires a list value"))
                })?;
                let py_list: Vec<Bound<'py, PyAny>> = arr
                    .iter()
                    .map(|v| json_value_to_py(py, v, df_lit))
                    .collect::<PyResult<Vec<_>>>()?;
                let negated = op == "not in";
                df_in_list.call1((c, py_list, negated))
            }
            _ => {
                let v = json_value_to_py(py, value, df_lit)?;
                match op {
                    "=" => c.call_method1("__eq__", (v,)),
                    "!=" => c.call_method1("__ne__", (v,)),
                    ">" => c.call_method1("__gt__", (v,)),
                    ">=" => c.call_method1("__ge__", (v,)),
                    "<" => c.call_method1("__lt__", (v,)),
                    "<=" => c.call_method1("__le__", (v,)),
                    _ => Err(PyValueError::new_err(format!(
                        "Unknown filter operator: '{op}'"
                    ))),
                }
            }
        }
    }

    /// Convert a serde_json::Value to a Python DataFusion lit() expression.
    fn json_value_to_py<'py>(
        py: Python<'py>,
        value: &serde_json::Value,
        df_lit: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        match value {
            serde_json::Value::String(s) => df_lit.call1((s.as_str(),)),
            serde_json::Value::Number(n) => {
                if let Some(i) = n.as_i64() {
                    df_lit.call1((i,))
                } else if let Some(f) = n.as_f64() {
                    df_lit.call1((f,))
                } else {
                    Err(PyValueError::new_err(format!(
                        "Unsupported number in filter: {n}"
                    )))
                }
            }
            serde_json::Value::Bool(b) => df_lit.call1((*b,)),
            serde_json::Value::Null => df_lit.call1((py.None(),)),
            _ => Err(PyValueError::new_err(format!(
                "Unsupported filter value type: {value}"
            ))),
        }
    }

    /// Accepts either a single string or a list of strings from Python.
    pub enum PatternArg {
        Single(String),
        Multiple(Vec<String>),
    }

    impl PatternArg {
        fn into_vec(self) -> Vec<String> {
            match self {
                PatternArg::Single(s) => vec![s],
                PatternArg::Multiple(v) => v,
            }
        }
    }

    impl<'py> pyo3::FromPyObject<'py> for PatternArg {
        fn extract_bound(ob: &pyo3::Bound<'py, pyo3::PyAny>) -> PyResult<Self> {
            if let Ok(s) = ob.extract::<String>() {
                Ok(PatternArg::Single(s))
            } else {
                Ok(PatternArg::Multiple(ob.extract::<Vec<String>>()?))
            }
        }
    }

    /// Python-exposed allow function. Polymorphic based on context type:
    /// - list[str] context → returns list of matching column strings
    /// - dict context (filter tree) → returns a DataFusion Expr
    #[pyfunction]
    #[pyo3(name = "allow", signature = (pattern, context, include=None))]
    pub fn py_allow<'py>(
        py: Python<'py>,
        pattern: PatternArg,
        context: &Bound<'py, PyAny>,
        include: Option<PatternArg>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(dict) = context.downcast::<PyDict>() {
            // Dict context → build filter Expr
            let filter_json: serde_json::Value = depythonize(dict)
                .map_err(|e| PyValueError::new_err(format!("Invalid filter dict: {e}")))?;
            let all_columns = extract_filter_columns(&filter_json);
            let patterns = pattern.into_vec();
            let matched =
                allow_columns(&patterns, &all_columns).map_err(|e| PyValueError::new_err(e))?;
            match build_filter_py_expr(py, &filter_json, &matched)? {
                Some(expr) => Ok(expr),
                None => {
                    let df_mod = py.import("datafusion")?;
                    let df_lit = df_mod.getattr("lit")?;
                    df_lit.call1((true,))
                }
            }
        } else {
            // List context → return column strings
            let context_cols: Vec<String> = context.extract()?;
            let patterns = pattern.into_vec();
            let mut result =
                allow_columns(&patterns, &context_cols).map_err(|e| PyValueError::new_err(e))?;
            if let Some(extras) = include {
                for extra in extras.into_vec() {
                    if !result.contains(&extra) {
                        result.push(extra);
                    }
                }
            }
            Ok(result.into_pyobject(py)?.into_any())
        }
    }

    /// Python-exposed exclude function. Polymorphic based on context type:
    /// - list[str] context → returns list of non-matching column strings
    /// - dict context (filter tree) → returns a DataFusion Expr (excluding matched columns)
    #[pyfunction]
    #[pyo3(name = "exclude", signature = (pattern, context, include=None))]
    pub fn py_exclude<'py>(
        py: Python<'py>,
        pattern: PatternArg,
        context: &Bound<'py, PyAny>,
        include: Option<PatternArg>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if let Ok(dict) = context.downcast::<PyDict>() {
            let filter_json: serde_json::Value = depythonize(dict)
                .map_err(|e| PyValueError::new_err(format!("Invalid filter dict: {e}")))?;
            let all_columns = extract_filter_columns(&filter_json);
            let patterns = pattern.into_vec();
            let excluded =
                exclude_columns(&patterns, &all_columns).map_err(|e| PyValueError::new_err(e))?;
            match build_filter_py_expr(py, &filter_json, &excluded)? {
                Some(expr) => Ok(expr),
                None => {
                    let df_mod = py.import("datafusion")?;
                    let df_lit = df_mod.getattr("lit")?;
                    df_lit.call1((true,))
                }
            }
        } else {
            let context_cols: Vec<String> = context.extract()?;
            let patterns = pattern.into_vec();
            let mut result =
                exclude_columns(&patterns, &context_cols).map_err(|e| PyValueError::new_err(e))?;
            if let Some(extras) = include {
                for extra in extras.into_vec() {
                    if !result.contains(&extra) {
                        result.push(extra);
                    }
                }
            }
            Ok(result.into_pyobject(py)?.into_any())
        }
    }
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
