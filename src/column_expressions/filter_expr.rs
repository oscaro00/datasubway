use std::str::FromStr;

use serde::{Deserialize, Deserializer};
use serde_json::Value;

use polars::lazy::dsl::{col, lit, Expr};
use polars::prelude::{NamedFrom, Schema, Series};

use super::column::TableColumn;
use super::column_context::{match_context_pattern, parse_column_pattern};

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum FilterExpr {
    And {
        and: Vec<FilterExpr>,
    },
    Or {
        or: Vec<FilterExpr>,
    },
    Comparison {
        left: Operand,
        op: CompareOp,
        right: Operand,
    },
}

#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum Operand {
    Col { col: String },
    Lit { lit: Value },
}

#[derive(Debug, Clone)]
pub enum CompareOp {
    Eq,
    Ne,
    Gt,
    Gte,
    Lt,
    Lte,
    In,
    NotIn,
}

impl<'de> Deserialize<'de> for CompareOp {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl FromStr for CompareOp {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "=" | "eq" => Ok(Self::Eq),
            "!=" | "ne" => Ok(Self::Ne),
            ">" | "gt" => Ok(Self::Gt),
            ">=" | "gte" => Ok(Self::Gte),
            "<" | "lt" => Ok(Self::Lt),
            "<=" | "lte" => Ok(Self::Lte),
            "in" => Ok(Self::In),
            "not_in" => Ok(Self::NotIn),
            other => Err(format!("unknown operator: {other}")),
        }
    }
}

fn col_is_valid(col_name: &str, patterns: &[&str], schema: &Schema, keep_matching: bool) -> bool {
    if schema.get(col_name).is_none() {
        return false;
    }
    let (table, column) = match col_name.split_once('.') {
        Some(parts) => parts,
        None => return false,
    };
    let col_tc = match TableColumn::new(table, column) {
        Ok(tc) => tc,
        Err(_) => return false,
    };
    let pattern_tcs: Vec<TableColumn> = patterns
        .iter()
        .filter_map(|p| parse_column_pattern(p))
        .collect();
    let matches = match_context_pattern(&col_tc, &pattern_tcs);
    if keep_matching {
        matches
    } else {
        !matches
    }
}

fn prune(
    expr: FilterExpr,
    patterns: &[&str],
    schema: &Schema,
    keep_matching: bool,
) -> Option<FilterExpr> {
    match expr {
        FilterExpr::Comparison {
            ref left,
            ref right,
            ..
        } => {
            let cols_valid = [left, right].iter().all(|op| match op {
                Operand::Col { col: name } => col_is_valid(name, patterns, schema, keep_matching),
                Operand::Lit { .. } => true,
            });
            if cols_valid {
                Some(expr)
            } else {
                None
            }
        }
        FilterExpr::And { and } => {
            let children: Vec<_> = and
                .into_iter()
                .filter_map(|e| prune(e, patterns, schema, keep_matching))
                .collect();
            if children.is_empty() {
                None
            } else {
                Some(FilterExpr::And { and: children })
            }
        }
        FilterExpr::Or { or } => {
            let children: Vec<_> = or
                .into_iter()
                .filter_map(|e| prune(e, patterns, schema, keep_matching))
                .collect();
            if children.is_empty() {
                None
            } else {
                Some(FilterExpr::Or { or: children })
            }
        }
    }
}

fn value_to_lit(value: Value) -> Expr {
    match value {
        Value::Bool(b) => lit(b),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                lit(i)
            } else {
                lit(n.as_f64().unwrap_or(f64::NAN))
            }
        }
        Value::String(s) => lit(s),
        Value::Array(arr) if !arr.is_empty() && arr[0].is_string() => {
            let v: Vec<String> = arr
                .into_iter()
                .filter_map(|x| x.as_str().map(str::to_owned))
                .collect();
            lit(Series::new("".into(), v))
        }
        Value::Array(arr) if !arr.is_empty() && arr[0].as_i64().is_some() => {
            let v: Vec<i64> = arr.into_iter().filter_map(|x| x.as_i64()).collect();
            lit(Series::new("".into(), v))
        }
        Value::Array(arr) if !arr.is_empty() && arr[0].is_number() => {
            let v: Vec<f64> = arr.into_iter().filter_map(|x| x.as_f64()).collect();
            lit(Series::new("".into(), v))
        }
        _ => lit(false),
    }
}

fn operand_to_expr(op: Operand) -> Expr {
    match op {
        Operand::Col { col: name } => col(&name),
        Operand::Lit { lit: value } => value_to_lit(value),
    }
}

fn to_expr(expr: FilterExpr) -> Expr {
    match expr {
        FilterExpr::Comparison { left, op, right } => {
            let l = operand_to_expr(left);
            let r = operand_to_expr(right);
            match op {
                CompareOp::Eq => l.eq(r),
                CompareOp::Ne => l.neq(r),
                CompareOp::Gt => l.gt(r),
                CompareOp::Gte => l.gt_eq(r),
                CompareOp::Lt => l.lt(r),
                CompareOp::Lte => l.lt_eq(r),
                CompareOp::In => l.is_in(r, false),
                CompareOp::NotIn => l.is_in(r, false).eq(lit(false)),
            }
        }
        FilterExpr::And { and } => and
            .into_iter()
            .map(to_expr)
            .reduce(|a, b| a.and(b))
            .unwrap(),
        FilterExpr::Or { or } => or.into_iter().map(to_expr).reduce(|a, b| a.or(b)).unwrap(),
    }
}

pub(crate) fn filter_expr_to_polars(
    expr: FilterExpr,
    patterns: &[&str],
    schema: &Schema,
    keep_matching: bool,
) -> Option<Expr> {
    let pruned = prune(expr, patterns, schema, keep_matching)?;
    Some(to_expr(pruned))
}

pub fn filter_to_expr(filter: &Value, patterns: &[&str], schema: &Schema) -> Option<Expr> {
    let parsed: FilterExpr = serde_json::from_value(filter.clone()).ok()?;
    let pruned = prune(parsed, patterns, schema, true)?;
    Some(to_expr(pruned))
}

/// Convert a JSON filter value directly to a polars Expr without schema-based pruning.
/// Use this for post-aggregation filters (havings) where columns may not exist in any
/// table schema (e.g. measure output column aliases).
pub fn json_to_expr(filter: &Value) -> Option<Expr> {
    if filter.is_null() || filter.as_object().map_or(false, |m| m.is_empty()) {
        return None;
    }
    let parsed: FilterExpr = serde_json::from_value(filter.clone()).ok()?;
    Some(to_expr(parsed))
}

/// Recursively collect all column names referenced in a FilterExpr.
pub fn collect_col_names(expr: &FilterExpr) -> Vec<String> {
    match expr {
        FilterExpr::And { and } => and.iter().flat_map(collect_col_names).collect(),
        FilterExpr::Or { or } => or.iter().flat_map(collect_col_names).collect(),
        FilterExpr::Comparison { left, right, .. } => [left, right]
            .iter()
            .filter_map(|op| match op {
                Operand::Col { col } => Some(col.clone()),
                Operand::Lit { .. } => None,
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::lazy::dsl::{col, lit};
    use polars::prelude::{DataType, Field, NamedFrom, Schema, Series};
    use serde_json::json;

    fn test_schema() -> Schema {
        Schema::from_iter([
            Field::new("geography.state".into(), DataType::String),
            Field::new("date.year".into(), DataType::Int64),
            Field::new("sales.ty_sales".into(), DataType::Float64),
            Field::new("sales.ly_sales".into(), DataType::Float64),
        ])
    }

    #[test]
    fn test_parse_and_convert() {
        let filter = json!({
            "and": [
                {
                    "and": [
                        {"left": {"col": "geography.state"}, "op": "in",  "right": {"lit": ["MN", "WI"]}},
                        {"left": {"col": "date.year"},       "op": ">",   "right": {"lit": 2024}}
                    ]
                },
                {
                    "or": [
                        {"left": {"col": "sales.ty_sales"}, "op": "<=", "right": {"col": "sales.ly_sales"}}
                    ]
                }
            ]
        });

        let schema = test_schema();
        let patterns = &["geography.*", "date.year", "sales.*"];
        let result = filter_to_expr(&filter, patterns, &schema);

        let expected = col("geography.state")
            .is_in(
                lit(Series::new(
                    "".into(),
                    vec!["MN".to_string(), "WI".to_string()],
                )),
                false,
            )
            .and(col("date.year").gt(lit(2024i64)))
            .and(col("sales.ty_sales").lt_eq(col("sales.ly_sales")));

        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_prunes_invalid_column() {
        let filter = json!({
            "and": [
                {"left": {"col": "sales.ty_sales"}, "op": ">",  "right": {"lit": 0}},
                {"left": {"col": "unknown.col"},    "op": "=",  "right": {"lit": 1}}
            ]
        });

        let schema = test_schema();
        let patterns = &["sales.*"];
        let result = filter_to_expr(&filter, patterns, &schema);

        // "unknown.col" pruned; only "sales.ty_sales > 0" survives
        let expected = col("sales.ty_sales").gt(lit(0i64));
        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_returns_none_when_all_pruned() {
        let filter = json!({
            "and": [
                {"left": {"col": "unknown.col"}, "op": "=", "right": {"lit": 1}}
            ]
        });

        let schema = test_schema();
        let patterns = &["sales.*"];
        assert_eq!(filter_to_expr(&filter, patterns, &schema), None);
    }
}
