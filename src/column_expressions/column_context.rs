use polars::lazy::dsl::{lit, Expr};
use polars::prelude::Schema;

use crate::column_expressions::filter_expr::{filter_expr_to_polars, FilterExpr};

use super::column::{validate_table_columns, ReturnKind, TableColumn, TableColumnsReturn};

pub enum ColumnPattern {
    OnePattern(&'static str),
    MultiplePatterns(Vec<&'static str>),
}

pub enum ColumnContext {
    OneString(&'static str),
    MultipleStrings(Vec<&'static str>),
    Json(FilterExpr),
}

pub enum ColumnInclude {
    OneString(&'static str),
    MultipleStrings(Vec<&'static str>),
}

pub enum ColumnReturn {
    Strings(Vec<String>),
    PolarsExpr(Expr),
}

fn normalize_column_pattern(pattern: ColumnPattern) -> Vec<TableColumn> {
    let pattern_vec = match pattern {
        ColumnPattern::OnePattern(s) => vec![s],
        ColumnPattern::MultiplePatterns(v) => v,
    };

    pattern_vec
        .into_iter()
        .map(|p| parse_column_pattern(p).unwrap())
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

pub fn allow(
    pattern: ColumnPattern,
    context: ColumnContext,
    include: ColumnInclude,
    schema: &Schema,
) -> ColumnReturn {
    let normalized_patterns = normalize_column_pattern(pattern);

    if let ColumnContext::Json(filter_expr) = context {
        // include is ignored for Json context — there is no meaningful way to
        // inject arbitrary column names into an already-structured filter expression
        let pattern_owned: Vec<String> = normalized_patterns
            .iter()
            .map(|tc| tc.table_column())
            .collect();
        let all_patterns: Vec<&str> = pattern_owned.iter().map(String::as_str).collect();

        // lit(true) when all conditions pruned = safe pass-through (no filtering)
        let expr = filter_expr_to_polars(filter_expr, &all_patterns, schema, true)
            .unwrap_or_else(|| lit(true));
        return ColumnReturn::PolarsExpr(expr);
    }

    let include_return = match include {
        ColumnInclude::OneString(s) => validate_table_columns(vec![s], ReturnKind::Strings),
        ColumnInclude::MultipleStrings(v) => validate_table_columns(v, ReturnKind::Strings),
    };
    let mut allowed_columns = match include_return {
        TableColumnsReturn::Strings(v) => v,
        _ => unreachable!(),
    };

    let context_return = match context {
        ColumnContext::OneString(s) => validate_table_columns(vec![s], ReturnKind::TableColumns),
        ColumnContext::MultipleStrings(v) => validate_table_columns(v, ReturnKind::TableColumns),
        ColumnContext::Json(_) => unreachable!(),
    };
    let context_vec = match context_return {
        TableColumnsReturn::TableColumns(v) => v,
        _ => unreachable!(),
    };

    let mut allowed_context: Vec<String> = context_vec
        .into_iter()
        .filter(|c| match_context_pattern(c, &normalized_patterns))
        .map(|c| c.table_column())
        .collect();

    allowed_columns.append(&mut allowed_context);
    allowed_columns.sort();
    allowed_columns.dedup();

    ColumnReturn::Strings(allowed_columns)
}

// Exclude should work very similarly to allow(). Instead of the context matching the patterns, they should not match.
// The include columns should still be included automatically
pub fn exclude(
    pattern: ColumnPattern,
    context: ColumnContext,
    include: ColumnInclude,
    schema: &Schema,
) -> ColumnReturn {
    let normalized_patterns = normalize_column_pattern(pattern);

    if let ColumnContext::Json(filter_expr) = context {
        let pattern_owned: Vec<String> =
            normalized_patterns.iter().map(|tc| tc.table_column()).collect();
        let all_patterns: Vec<&str> = pattern_owned.iter().map(String::as_str).collect();
        let expr = filter_expr_to_polars(filter_expr, &all_patterns, schema, false)
            .unwrap_or_else(|| lit(true));
        return ColumnReturn::PolarsExpr(expr);
    }

    let include_return = match include {
        ColumnInclude::OneString(s) => validate_table_columns(vec![s], ReturnKind::Strings),
        ColumnInclude::MultipleStrings(v) => validate_table_columns(v, ReturnKind::Strings),
    };
    let mut allowed_columns = match include_return {
        TableColumnsReturn::Strings(v) => v,
        _ => unreachable!(),
    };

    let context_return = match context {
        ColumnContext::OneString(s) => validate_table_columns(vec![s], ReturnKind::TableColumns),
        ColumnContext::MultipleStrings(v) => validate_table_columns(v, ReturnKind::TableColumns),
        ColumnContext::Json(_) => unreachable!(),
    };
    let context_vec = match context_return {
        TableColumnsReturn::TableColumns(v) => v,
        _ => unreachable!(),
    };

    let mut excluded_context: Vec<String> = context_vec
        .into_iter()
        .filter(|c| !match_context_pattern(c, &normalized_patterns))
        .map(|c| c.table_column())
        .collect();

    allowed_columns.append(&mut excluded_context);
    allowed_columns.sort();
    allowed_columns.dedup();

    ColumnReturn::Strings(allowed_columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::lazy::dsl::{col, lit};
    use polars::prelude::{DataType, Field, Schema};

    fn test_schema() -> Schema {
        Schema::from_iter([
            Field::new("sales.amount".into(), DataType::Float64),
            Field::new("date.year".into(), DataType::Int64),
        ])
    }

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

        let schema = test_schema();
        let result = allow(
            ColumnPattern::OnePattern("sales.*"),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        // date.year is pruned; only sales.amount > 0 survives
        let expected = col("sales.amount").gt(lit(0i64));
        assert!(matches!(result, ColumnReturn::PolarsExpr(ref e) if e == &expected));
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

        let schema = test_schema();
        let result = allow(
            ColumnPattern::OnePattern("*"),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        let expected = col("sales.amount")
            .gt(lit(0i64))
            .and(col("date.year").gt(lit(2020i64)));
        assert!(matches!(result, ColumnReturn::PolarsExpr(ref e) if e == &expected));
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

        let schema = test_schema();
        let result = allow(
            ColumnPattern::OnePattern("sales.*"),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        assert!(matches!(result, ColumnReturn::PolarsExpr(ref e) if e == &lit(true)));
    }

    #[test]
    fn test_string_context_regression() {
        let schema = test_schema();
        let result = allow(
            ColumnPattern::OnePattern("sales.*"),
            ColumnContext::OneString("sales.amount"),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        assert!(
            matches!(result, ColumnReturn::Strings(ref v) if v == &vec!["sales.amount".to_string()])
        );
    }

    #[test]
    fn test_exclude_string_context_drops_matching_columns() {
        let schema = test_schema();
        let result = exclude(
            ColumnPattern::OnePattern("date.*"),
            ColumnContext::MultipleStrings(vec!["sales.amount", "date.year"]),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        // date.year matches pattern → excluded; sales.amount survives
        assert!(
            matches!(result, ColumnReturn::Strings(ref v) if v == &vec!["sales.amount".to_string()])
        );
    }

    #[test]
    fn test_exclude_string_context_include_always_kept() {
        let schema = test_schema();
        let result = exclude(
            ColumnPattern::OnePattern("date.*"),
            ColumnContext::MultipleStrings(vec!["sales.amount", "date.year"]),
            ColumnInclude::OneString("date.year"),
            &schema,
        );

        // date.year is in include so it appears even though the pattern would exclude it from context
        assert!(
            matches!(result, ColumnReturn::Strings(ref v) if v == &vec!["date.year".to_string(), "sales.amount".to_string()])
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

        let schema = test_schema();
        let result = exclude(
            ColumnPattern::OnePattern("date.*"),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        // date.year matches exclude pattern → pruned; only sales.amount > 0 survives
        let expected = col("sales.amount").gt(lit(0i64));
        assert!(matches!(result, ColumnReturn::PolarsExpr(ref e) if e == &expected));
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

        let schema = test_schema();
        let result = exclude(
            ColumnPattern::OnePattern("sales.*"),
            ColumnContext::Json(filter_expr),
            ColumnInclude::OneString("sales.amount"),
            &schema,
        );

        // sales.amount matches exclude pattern → all pruned → lit(true)
        assert!(matches!(result, ColumnReturn::PolarsExpr(ref e) if e == &lit(true)));
    }
}
