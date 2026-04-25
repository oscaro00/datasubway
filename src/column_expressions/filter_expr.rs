use std::str::FromStr;

use serde::{Deserialize, Deserializer};
use serde_json::Value;

use polars::lazy::dsl::{col, lit, Expr};
use polars::prelude::{NamedFrom, Schema, Series};

#[derive(Deserialize, Debug)]
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

#[derive(Deserialize, Debug)]
#[serde(untagged)]
pub enum Operand {
    Col { col: String },
    Lit { lit: Value },
}

#[derive(Debug)]
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

fn col_matches_pattern(col_name: &str, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    let col_parts: Vec<&str> = col_name.splitn(2, '.').collect();
    let pat_parts: Vec<&str> = pattern.splitn(2, '.').collect();
    if col_parts.len() != 2 || pat_parts.len() != 2 {
        return false;
    }
    (pat_parts[0] == "*" || pat_parts[0] == col_parts[0])
        && (pat_parts[1] == "*" || pat_parts[1] == col_parts[1])
}

fn col_is_valid(col_name: &str, patterns: &[&str], schema: &Schema) -> bool {
    schema.get(col_name).is_some() && patterns.iter().any(|p| col_matches_pattern(col_name, p))
}

fn prune(expr: FilterExpr, patterns: &[&str], schema: &Schema) -> Option<FilterExpr> {
    match expr {
        FilterExpr::Comparison {
            ref left,
            ref right,
            ..
        } => {
            let cols_valid = [left, right].iter().all(|op| match op {
                Operand::Col { col: name } => col_is_valid(name, patterns, schema),
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
                .filter_map(|e| prune(e, patterns, schema))
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
                .filter_map(|e| prune(e, patterns, schema))
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

pub fn filter_to_expr(filter: &Value, patterns: &[&str], schema: &Schema) -> Option<Expr> {
    let parsed: FilterExpr = serde_json::from_value(filter.clone()).ok()?;
    let pruned = prune(parsed, patterns, schema)?;
    Some(to_expr(pruned))
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::{DataType, Field, Schema};
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
        assert!(filter_to_expr(&filter, patterns, &schema).is_some());
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
        // "unknown.col" pruned; "sales.ty_sales" survives → Some
        assert!(filter_to_expr(&filter, patterns, &schema).is_some());
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
        assert!(filter_to_expr(&filter, patterns, &schema).is_none());
    }
}
