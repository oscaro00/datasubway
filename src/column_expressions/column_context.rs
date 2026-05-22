use polars::lazy::dsl::{col, lit, Expr};

use crate::column_expressions::filter_expr::{filter_expr_to_polars, FilterExpr};

use super::column::{validate_table_columns, ReturnKind, TableColumn, TableColumnsReturn};

#[derive(Clone, Debug)]
pub enum ColumnPattern {
    OnePattern(String),
    MultiplePatterns(Vec<String>),
}

#[derive(Clone, Debug)]
pub enum ColumnContext {
    OneString(String),
    MultipleStrings(Vec<String>),
    Json(FilterExpr),
}

#[derive(Clone, Debug)]
pub enum ColumnInclude {
    None,
    OneString(String),
    MultipleStrings(Vec<String>),
}

pub enum ColumnReturn {
    Strings(Vec<String>),
    PolarsExpr(Expr),
}

// ── AllowExclude metadata ─────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub enum AllowExcludeKind {
    Allow,
    Exclude,
}

/// Stores the arguments passed to `allow()` or `exclude()` (schema excluded).
#[derive(Clone, Debug)]
pub struct AllowExcludeRecord {
    pub kind: AllowExcludeKind,
    pub pattern: ColumnPattern,
    pub context: ColumnContext,
    pub include: ColumnInclude,
}

/// Returned by `allow()` and `exclude()`: carries both the computed result and
/// the call metadata for recording purposes.
pub struct AllowExcludeResult {
    pub inner: ColumnReturn,
    pub record: AllowExcludeRecord,
}

// ── Traits for generic recorder methods ──────────────────────────────────────

/// Implemented by types that can supply multiple Polars column expressions,
/// optionally pushing an `AllowExcludeRecord` onto the recorder's log.
/// Used by `LazyFrameRecorder::group_by()` and similar multi-column methods.
pub trait IntoPolarsColsExpr {
    fn into_exprs_with_record(self, records: &mut Vec<AllowExcludeRecord>) -> Vec<Expr>;
}

impl IntoPolarsColsExpr for Vec<Expr> {
    fn into_exprs_with_record(self, _records: &mut Vec<AllowExcludeRecord>) -> Vec<Expr> {
        self
    }
}

impl IntoPolarsColsExpr for AllowExcludeResult {
    fn into_exprs_with_record(self, records: &mut Vec<AllowExcludeRecord>) -> Vec<Expr> {
        records.push(self.record);
        match self.inner {
            ColumnReturn::Strings(cols) => cols.iter().map(|c| col(c.as_str())).collect(),
            ColumnReturn::PolarsExpr(_) => {
                panic!("IntoPolarsColsExpr requires a string-context AllowExcludeResult (got PolarsExpr)")
            }
        }
    }
}

/// Implemented by types that can supply a single optional Polars filter
/// expression, optionally pushing an `AllowExcludeRecord` onto the recorder's log.
/// Used by `LazyFrameRecorder::filter()`.
pub trait IntoFilterExpr {
    fn into_filter(self, records: &mut Vec<AllowExcludeRecord>) -> Option<Expr>;
}

impl IntoFilterExpr for Option<Expr> {
    fn into_filter(self, _records: &mut Vec<AllowExcludeRecord>) -> Option<Expr> {
        self
    }
}

impl IntoFilterExpr for Expr {
    fn into_filter(self, _records: &mut Vec<AllowExcludeRecord>) -> Option<Expr> {
        Some(self)
    }
}

impl IntoFilterExpr for AllowExcludeResult {
    fn into_filter(self, records: &mut Vec<AllowExcludeRecord>) -> Option<Expr> {
        records.push(self.record);
        match self.inner {
            ColumnReturn::PolarsExpr(e) => Some(e),
            ColumnReturn::Strings(_) => {
                panic!("IntoFilterExpr requires a JSON-context AllowExcludeResult (got Strings)")
            }
        }
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

fn normalize_column_pattern(pattern: ColumnPattern) -> Vec<TableColumn> {
    let pattern_vec = match pattern {
        ColumnPattern::OnePattern(s) => vec![s],
        ColumnPattern::MultiplePatterns(v) => v,
    };
    pattern_vec
        .into_iter()
        .map(|p| parse_column_pattern(&p).unwrap())
        .collect()
}

pub fn parse_column_pattern(pattern: &str) -> Option<TableColumn> {
    if pattern == "*" {
        return TableColumn::new("*", "*").ok();
    }
    let (table, col) = pattern.split_once('.')?;
    TableColumn::new(table, col).ok()
}

pub fn match_context_pattern(context_column: &TableColumn, patterns: &[TableColumn]) -> bool {
    for pat in patterns.iter() {
        if pat.table() == "*" {
            return true;
        } else if pat.table() == context_column.table()
            && (pat.column() == "*" || pat.column() == context_column.column())
        {
            return true;
        }
    }
    false
}

fn include_to_strings(include: ColumnInclude) -> Vec<String> {
    let include_return = match include {
        ColumnInclude::None => return vec![],
        ColumnInclude::OneString(s) => {
            validate_table_columns(vec![s.as_str()], ReturnKind::Strings)
        }
        ColumnInclude::MultipleStrings(v) => {
            validate_table_columns(v.iter().map(|s| s.as_str()).collect(), ReturnKind::Strings)
        }
    };
    match include_return {
        TableColumnsReturn::Strings(v) => v,
        _ => unreachable!(),
    }
}

fn context_to_table_columns(context: ColumnContext) -> Vec<TableColumn> {
    let context_return = match context {
        ColumnContext::OneString(s) => {
            validate_table_columns(vec![s.as_str()], ReturnKind::TableColumns)
        }
        ColumnContext::MultipleStrings(v) => validate_table_columns(
            v.iter().map(|s| s.as_str()).collect(),
            ReturnKind::TableColumns,
        ),
        ColumnContext::Json(_) => unreachable!(),
    };
    match context_return {
        TableColumnsReturn::TableColumns(v) => v,
        _ => unreachable!(),
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

pub fn allow(
    pattern: ColumnPattern,
    context: ColumnContext,
    include: ColumnInclude,
) -> AllowExcludeResult {
    let record = AllowExcludeRecord {
        kind: AllowExcludeKind::Allow,
        pattern: pattern.clone(),
        context: context.clone(),
        include: include.clone(),
    };

    let normalized_patterns = normalize_column_pattern(pattern);

    if let ColumnContext::Json(filter_expr) = context {
        let pattern_owned: Vec<String> = normalized_patterns
            .iter()
            .map(|tc| tc.table_column())
            .collect();
        let all_patterns: Vec<&str> = pattern_owned.iter().map(String::as_str).collect();
        let expr =
            filter_expr_to_polars(filter_expr, &all_patterns, true).unwrap_or_else(|| lit(true));
        return AllowExcludeResult {
            inner: ColumnReturn::PolarsExpr(expr),
            record,
        };
    }

    let mut allowed_columns = include_to_strings(include);

    let context_vec = context_to_table_columns(context);
    let mut allowed_context: Vec<String> = context_vec
        .into_iter()
        .filter(|c| match_context_pattern(c, &normalized_patterns))
        .map(|c| c.table_column())
        .collect();

    allowed_columns.append(&mut allowed_context);
    allowed_columns.sort();
    allowed_columns.dedup();

    AllowExcludeResult {
        inner: ColumnReturn::Strings(allowed_columns),
        record,
    }
}

pub fn exclude(
    pattern: ColumnPattern,
    context: ColumnContext,
    include: ColumnInclude,
) -> AllowExcludeResult {
    let record = AllowExcludeRecord {
        kind: AllowExcludeKind::Exclude,
        pattern: pattern.clone(),
        context: context.clone(),
        include: include.clone(),
    };

    let normalized_patterns = normalize_column_pattern(pattern);

    if let ColumnContext::Json(filter_expr) = context {
        let pattern_owned: Vec<String> = normalized_patterns
            .iter()
            .map(|tc| tc.table_column())
            .collect();
        let all_patterns: Vec<&str> = pattern_owned.iter().map(String::as_str).collect();
        let expr =
            filter_expr_to_polars(filter_expr, &all_patterns, false).unwrap_or_else(|| lit(true));
        return AllowExcludeResult {
            inner: ColumnReturn::PolarsExpr(expr),
            record,
        };
    }

    let mut allowed_columns = include_to_strings(include);

    let context_vec = context_to_table_columns(context);
    let mut excluded_context: Vec<String> = context_vec
        .into_iter()
        .filter(|c| !match_context_pattern(c, &normalized_patterns))
        .map(|c| c.table_column())
        .collect();

    allowed_columns.append(&mut excluded_context);
    allowed_columns.sort();
    allowed_columns.dedup();

    AllowExcludeResult {
        inner: ColumnReturn::Strings(allowed_columns),
        record,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use polars::lazy::dsl::{col, lit};
    #[test]
    fn test_json_context_prunes_non_matching_columns() {
        let filter_expr = FilterExpr::And {
            and: vec![
                FilterExpr::Comparison {
                    left: crate::column_expressions::filter_expr::Operand::Col {
                        col: "sales.amount".to_string(),
                    },
                    op: crate::column_expressions::filter_expr::CompareOp::Gt,
                    right: crate::column_expressions::filter_expr::Operand::Lit {
                        lit: serde_json::Value::Number(serde_json::Number::from(0)),
                    },
                },
                FilterExpr::Comparison {
                    left: crate::column_expressions::filter_expr::Operand::Col {
                        col: "date.year".to_string(),
                    },
                    op: crate::column_expressions::filter_expr::CompareOp::Gt,
                    right: crate::column_expressions::filter_expr::Operand::Lit {
                        lit: serde_json::Value::Number(serde_json::Number::from(2020)),
                    },
                },
            ],
        };

        let result = allow(
            ColumnPattern::OnePattern("sales.*".into()),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount".into()),
        );

        let expected = col("sales.amount").gt(lit(0i64));
        assert!(matches!(result.inner, ColumnReturn::PolarsExpr(ref e) if e == &expected));
    }

    #[test]
    fn test_json_context_wildcard_pattern_keeps_all() {
        let filter_expr = FilterExpr::And {
            and: vec![
                FilterExpr::Comparison {
                    left: crate::column_expressions::filter_expr::Operand::Col {
                        col: "sales.amount".to_string(),
                    },
                    op: crate::column_expressions::filter_expr::CompareOp::Gt,
                    right: crate::column_expressions::filter_expr::Operand::Lit {
                        lit: serde_json::Value::Number(serde_json::Number::from(0)),
                    },
                },
                FilterExpr::Comparison {
                    left: crate::column_expressions::filter_expr::Operand::Col {
                        col: "date.year".to_string(),
                    },
                    op: crate::column_expressions::filter_expr::CompareOp::Gt,
                    right: crate::column_expressions::filter_expr::Operand::Lit {
                        lit: serde_json::Value::Number(serde_json::Number::from(2020)),
                    },
                },
            ],
        };

        let result = allow(
            ColumnPattern::OnePattern("*".into()),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount".into()),
        );

        let expected = col("sales.amount")
            .gt(lit(0i64))
            .and(col("date.year").gt(lit(2020i64)));
        assert!(matches!(result.inner, ColumnReturn::PolarsExpr(ref e) if e == &expected));
    }

    #[test]
    fn test_json_context_all_pruned_returns_lit_true() {
        let filter_expr = FilterExpr::Comparison {
            left: crate::column_expressions::filter_expr::Operand::Col {
                col: "unknown.col".to_string(),
            },
            op: crate::column_expressions::filter_expr::CompareOp::Eq,
            right: crate::column_expressions::filter_expr::Operand::Lit {
                lit: serde_json::Value::Number(serde_json::Number::from(1)),
            },
        };

        let result = allow(
            ColumnPattern::OnePattern("sales.*".into()),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount".into()),
        );

        assert!(matches!(result.inner, ColumnReturn::PolarsExpr(ref e) if e == &lit(true)));
    }

    #[test]
    fn test_string_context_regression() {
        let result = allow(
            ColumnPattern::OnePattern("sales.*".into()),
            ColumnContext::OneString("sales.amount".into()),
            ColumnInclude::OneString("sales.amount".into()),
        );

        assert!(
            matches!(result.inner, ColumnReturn::Strings(ref v) if v == &vec!["sales.amount".to_string()])
        );
    }

    #[test]
    fn test_exclude_string_context_drops_matching_columns() {
        let result = exclude(
            ColumnPattern::OnePattern("date.*".into()),
            ColumnContext::MultipleStrings(vec!["sales.amount".into(), "date.year".into()]),
            ColumnInclude::OneString("sales.amount".into()),
        );

        assert!(
            matches!(result.inner, ColumnReturn::Strings(ref v) if v == &vec!["sales.amount".to_string()])
        );
    }

    #[test]
    fn test_exclude_string_context_include_always_kept() {
        let result = exclude(
            ColumnPattern::OnePattern("date.*".into()),
            ColumnContext::MultipleStrings(vec!["sales.amount".into(), "date.year".into()]),
            ColumnInclude::OneString("date.year".into()),
        );

        assert!(
            matches!(result.inner, ColumnReturn::Strings(ref v) if v == &vec!["date.year".to_string(), "sales.amount".to_string()])
        );
    }

    #[test]
    fn test_exclude_json_context_keeps_non_matching_columns() {
        let filter_expr = FilterExpr::And {
            and: vec![
                FilterExpr::Comparison {
                    left: crate::column_expressions::filter_expr::Operand::Col {
                        col: "sales.amount".to_string(),
                    },
                    op: crate::column_expressions::filter_expr::CompareOp::Gt,
                    right: crate::column_expressions::filter_expr::Operand::Lit {
                        lit: serde_json::Value::Number(serde_json::Number::from(0)),
                    },
                },
                FilterExpr::Comparison {
                    left: crate::column_expressions::filter_expr::Operand::Col {
                        col: "date.year".to_string(),
                    },
                    op: crate::column_expressions::filter_expr::CompareOp::Gt,
                    right: crate::column_expressions::filter_expr::Operand::Lit {
                        lit: serde_json::Value::Number(serde_json::Number::from(2020)),
                    },
                },
            ],
        };

        let result = exclude(
            ColumnPattern::OnePattern("date.*".into()),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount".into()),
        );

        let expected = col("sales.amount").gt(lit(0i64));
        assert!(matches!(result.inner, ColumnReturn::PolarsExpr(ref e) if e == &expected));
    }

    #[test]
    fn test_exclude_json_context_all_match_returns_lit_true() {
        let filter_expr = FilterExpr::Comparison {
            left: crate::column_expressions::filter_expr::Operand::Col {
                col: "sales.amount".to_string(),
            },
            op: crate::column_expressions::filter_expr::CompareOp::Gt,
            right: crate::column_expressions::filter_expr::Operand::Lit {
                lit: serde_json::Value::Number(serde_json::Number::from(0)),
            },
        };

        let result = exclude(
            ColumnPattern::OnePattern("sales.*".into()),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount".into()),
        );

        assert!(matches!(result.inner, ColumnReturn::PolarsExpr(ref e) if e == &lit(true)));
    }
}
