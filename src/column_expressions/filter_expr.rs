use std::str::FromStr;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

use datafusion::common::Column;
use datafusion::prelude::{Expr, col, in_list, lit};

use super::column::TableColumn;
use super::column_context::{match_context_pattern, parse_column_pattern};

#[derive(Serialize, Deserialize, Debug, Clone)]
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

#[derive(Serialize, Deserialize, Debug, Clone)]
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

impl Serialize for CompareOp {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let str_val = match self {
            CompareOp::Eq => "eq",
            CompareOp::Ne => "ne",
            CompareOp::Gt => "gt",
            CompareOp::Gte => "gte",
            CompareOp::Lt => "lt",
            CompareOp::Lte => "lte",
            CompareOp::In => "in",
            CompareOp::NotIn => "not_in",
        };
        s.serialize_str(str_val)
    }
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

fn col_is_valid(col_name: &str, patterns: &[&str], keep_matching: bool) -> bool {
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
    if keep_matching { matches } else { !matches }
}

fn prune(expr: FilterExpr, patterns: &[&str], keep_matching: bool) -> Option<FilterExpr> {
    match expr {
        FilterExpr::Comparison {
            ref left,
            ref right,
            ..
        } => {
            let cols_valid = [left, right].iter().all(|op| match op {
                Operand::Col { col: name } => col_is_valid(name, patterns, keep_matching),
                Operand::Lit { .. } => true,
            });
            if cols_valid { Some(expr) } else { None }
        }
        FilterExpr::And { and } => {
            let children: Vec<_> = and
                .into_iter()
                .filter_map(|e| prune(e, patterns, keep_matching))
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
                .filter_map(|e| prune(e, patterns, keep_matching))
                .collect();
            if children.is_empty() {
                None
            } else {
                Some(FilterExpr::Or { or: children })
            }
        }
    }
}

fn value_to_lit_scalar(value: Value) -> Expr {
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
        _ => lit(false),
    }
}

fn value_to_lit_list(arr: Vec<Value>) -> Vec<Expr> {
    arr.into_iter().map(value_to_lit_scalar).collect()
}

/// How a `{"col": "..."}` operand becomes a column reference.
///
/// The two callers of [`to_expr`] target differently shaped schemas, and a dotted
/// name means opposite things in each — so the constructor is a parameter rather
/// than a default, and every call site has to say which schema it is building
/// against.
type ColumnRef = fn(&str) -> Expr;

/// A table-qualified reference, for filters applied *before* aggregation.
///
/// `col` splits on the dot: `"orders.amount"` becomes
/// `Column { relation: Some("orders"), name: "amount" }`, which is exactly right
/// against a joined scan, where `orders` is a real relation.
fn qualified_column(name: &str) -> Expr {
    col(name)
}

/// One unqualified column whose *name* contains the dot, for filters applied
/// *after* aggregation.
///
/// `flatten_df` collapses the aggregate's schema to unqualified fields literally
/// named `"orders.amount"`, so the qualified form above resolves against nothing
/// there. `sort_exprs` builds post-aggregation references the same way.
fn flat_column(name: &str) -> Expr {
    Expr::Column(Column::from_name(name))
}

fn operand_to_expr(op: Operand, column: ColumnRef) -> Expr {
    match op {
        Operand::Col { col: name } => column(&name),
        Operand::Lit { lit: value } => value_to_lit_scalar(value),
    }
}

fn to_expr(expr: FilterExpr, column: ColumnRef) -> Expr {
    match expr {
        FilterExpr::Comparison { left, op, right } => {
            let l = operand_to_expr(left, column);
            match op {
                CompareOp::Eq => l.eq(operand_to_expr(right, column)),
                CompareOp::Ne => l.not_eq(operand_to_expr(right, column)),
                CompareOp::Gt => l.gt(operand_to_expr(right, column)),
                CompareOp::Gte => l.gt_eq(operand_to_expr(right, column)),
                CompareOp::Lt => l.lt(operand_to_expr(right, column)),
                CompareOp::Lte => l.lt_eq(operand_to_expr(right, column)),
                CompareOp::In => {
                    let list = match right {
                        Operand::Lit {
                            lit: Value::Array(arr),
                        } => value_to_lit_list(arr),
                        other => vec![operand_to_expr(other, column)],
                    };
                    in_list(l, list, false)
                }
                CompareOp::NotIn => {
                    let list = match right {
                        Operand::Lit {
                            lit: Value::Array(arr),
                        } => value_to_lit_list(arr),
                        other => vec![operand_to_expr(other, column)],
                    };
                    in_list(l, list, true)
                }
            }
        }
        FilterExpr::And { and } => and
            .into_iter()
            .map(|e| to_expr(e, column))
            .reduce(|a, b| a.and(b))
            .unwrap(),
        FilterExpr::Or { or } => or
            .into_iter()
            .map(|e| to_expr(e, column))
            .reduce(|a, b| a.or(b))
            .unwrap(),
    }
}

pub(crate) fn filter_expr_to_df(
    expr: FilterExpr,
    patterns: &[&str],
    keep_matching: bool,
) -> Option<Expr> {
    let pruned = prune(expr, patterns, keep_matching)?;
    Some(to_expr(pruned, qualified_column))
}

pub fn filter_to_expr(filter: &Value, patterns: &[&str]) -> Option<Expr> {
    let parsed: FilterExpr = serde_json::from_value(filter.clone()).ok()?;
    let pruned = prune(parsed, patterns, true)?;
    Some(to_expr(pruned, qualified_column))
}

/// A filter value that carries no filter: `null`, or `{}`.
fn is_empty_filter(filter: &Value) -> bool {
    filter.is_null() || filter.as_object().is_some_and(|m| m.is_empty())
}

/// Convert a JSON filter value directly to a DataFusion Expr without schema-based pruning.
/// Use this for post-aggregation filters (havings) where columns may not exist in any
/// table schema (e.g. measure output column aliases).
///
/// Column references are built flat, because by the time a having is applied
/// `flatten_df` has already turned the aggregate's schema into unqualified
/// fields whose names carry the dot. Building them qualified instead — which is
/// what this did before — left every having on a dotted measure alias or group
/// column resolving against a relation the flattened schema no longer has, and
/// the query failed in type coercion with "No field named orders.amount ...
/// Valid fields are "orders.amount"".
pub fn json_to_expr(filter: &Value) -> Option<Expr> {
    if is_empty_filter(filter) {
        return None;
    }
    let parsed: FilterExpr = serde_json::from_value(filter.clone()).ok()?;
    Some(to_expr(parsed, flat_column))
}

/// The same conversion, against a schema that has *not* been flattened.
///
/// `build_select_frame` filters the joined scan before it projects, so `games`
/// there is a real relation and `"games.date"` has to split on the dot — the
/// opposite of what a having needs. The two cases share everything but that one
/// decision, so they are two entry points over one builder rather than a flag
/// callers could forget to set.
pub fn json_to_expr_qualified(filter: &Value) -> Option<Expr> {
    if is_empty_filter(filter) {
        return None;
    }
    let parsed: FilterExpr = serde_json::from_value(filter.clone()).ok()?;
    Some(to_expr(parsed, qualified_column))
}

/// Extract column names from a raw JSON filter value.
pub fn extract_filter_cols(value: &serde_json::Value) -> Vec<String> {
    if value.is_null() || value.as_object().is_some_and(|m| m.is_empty()) {
        return Vec::new();
    }
    match serde_json::from_value::<FilterExpr>(value.clone()) {
        Ok(expr) => collect_col_names(&expr),
        Err(_) => Vec::new(),
    }
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
    use datafusion::common::tree_node::TreeNode;
    use datafusion::prelude::{col, in_list, lit};
    use serde_json::json;

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

        let patterns = &["geography.*", "date.year", "sales.*"];
        let result = filter_to_expr(&filter, patterns);

        let expected = in_list(col("geography.state"), vec![lit("MN"), lit("WI")], false)
            .and(col("date.year").gt(lit(2024i64)))
            .and(col("sales.ty_sales").lt_eq(col("sales.ly_sales")));

        assert_eq!(result, Some(expected));
    }

    #[test]
    fn test_prunes_non_matching_column() {
        let filter = json!({
            "and": [
                {"left": {"col": "sales.ty_sales"}, "op": ">",  "right": {"lit": 0}},
                {"left": {"col": "unknown.col"},    "op": "=",  "right": {"lit": 1}}
            ]
        });

        let patterns = &["sales.*"];
        let result = filter_to_expr(&filter, patterns);

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

        let patterns = &["sales.*"];
        assert_eq!(filter_to_expr(&filter, patterns), None);
    }

    /// A having is applied after `flatten_df`, where `"orders.amount"` is one
    /// unqualified field whose name contains a dot — not table `orders`, field
    /// `amount`. Building it qualified resolves against nothing and the query
    /// dies in type coercion, which is what every having on a dotted measure
    /// alias or group column used to do.
    #[test]
    fn test_json_to_expr_builds_flat_columns_for_havings() {
        let having = json!({"left": {"col": "orders.amount"}, "op": ">", "right": {"lit": 500}});
        let expr = json_to_expr(&having).unwrap();

        let expected = Expr::Column(Column::from_name("orders.amount")).gt(lit(500i64));
        assert_eq!(expr, expected);

        // Spelled out, because this is the whole bug: no relation, and the dot
        // stays inside the name.
        let Expr::BinaryExpr(binary) = &expr else {
            panic!("expected a comparison, got {expr:?}")
        };
        let Expr::Column(column) = binary.left.as_ref() else {
            panic!("expected a column on the left, got {:?}", binary.left)
        };
        assert_eq!(column.relation, None);
        assert_eq!(column.name, "orders.amount");
    }

    /// The other half of the same decision: a pre-aggregation filter runs
    /// against a joined scan, where `orders` really is a relation. Splitting the
    /// dot is correct there, and must stay that way.
    #[test]
    fn test_filter_to_expr_still_builds_qualified_columns() {
        let filter = json!({"left": {"col": "orders.amount"}, "op": ">", "right": {"lit": 500}});
        let expr = filter_to_expr(&filter, &["orders.*"]).unwrap();
        assert_eq!(expr, col("orders.amount").gt(lit(500i64)));

        let Expr::BinaryExpr(binary) = &expr else {
            panic!("expected a comparison, got {expr:?}")
        };
        let Expr::Column(column) = binary.left.as_ref() else {
            panic!("expected a column on the left, got {:?}", binary.left)
        };
        assert_eq!(
            column.relation.as_ref().map(ToString::to_string),
            Some("orders".to_string())
        );
        assert_eq!(column.name, "amount");
    }

    /// A select filter runs before the projection, against the joined scan —
    /// the same shape a pre-aggregation filter sees, and the opposite of a
    /// having. It must keep splitting the dot.
    #[test]
    fn test_json_to_expr_qualified_builds_qualified_columns_for_select_filters() {
        let filter = json!({"left": {"col": "orders.amount"}, "op": ">", "right": {"lit": 500}});
        let expr = json_to_expr_qualified(&filter).unwrap();
        assert_eq!(expr, col("orders.amount").gt(lit(500i64)));

        let Expr::BinaryExpr(binary) = &expr else {
            panic!("expected a comparison, got {expr:?}")
        };
        let Expr::Column(column) = binary.left.as_ref() else {
            panic!("expected a column on the left, got {:?}", binary.left)
        };
        assert_eq!(
            column.relation.as_ref().map(ToString::to_string),
            Some("orders".to_string())
        );
        assert_eq!(column.name, "amount");
    }

    /// Both entry points share the empty guard, so an absent filter stays absent
    /// rather than becoming a filter that matches nothing.
    #[test]
    fn test_an_empty_filter_yields_no_expression_either_way() {
        for filter in [json!(null), json!({})] {
            assert_eq!(json_to_expr(&filter), None);
            assert_eq!(json_to_expr_qualified(&filter), None);
        }
    }

    /// A dot-free measure alias resolves either way, which is why the existing
    /// coverage never caught this.
    #[test]
    fn test_a_dotless_having_column_is_unaffected() {
        let having = json!({"left": {"col": "revenue"}, "op": ">", "right": {"lit": 500}});
        assert_eq!(
            json_to_expr(&having).unwrap(),
            col("revenue").gt(lit(500i64))
        );
    }

    /// The constructor has to reach every operand, not just the first: nested
    /// boolean nodes, both sides of a column-to-column comparison, and the
    /// `in`/`not_in` lists all go through separate arms.
    #[test]
    fn test_flat_columns_reach_every_operand_of_a_having() {
        let having = json!({"and": [
            {"left": {"col": "orders.amount"}, "op": ">", "right": {"col": "orders.cost"}},
            {"or": [
                {"left": {"col": "orders.region"}, "op": "in", "right": {"lit": ["north"]}},
                {"left": {"col": "orders.region"}, "op": "not_in", "right": {"lit": ["south"]}}
            ]}
        ]});

        let mut columns = Vec::new();
        json_to_expr(&having)
            .unwrap()
            .apply(|e| {
                if let Expr::Column(c) = e {
                    columns.push((c.relation.as_ref().map(ToString::to_string), c.name.clone()));
                }
                Ok(datafusion::common::tree_node::TreeNodeRecursion::Continue)
            })
            .unwrap();

        assert_eq!(columns.len(), 4, "{columns:?}");
        assert!(
            columns.iter().all(|(relation, _)| relation.is_none()),
            "every having column must be unqualified, got {columns:?}"
        );
        assert!(
            columns.iter().all(|(_, name)| name.starts_with("orders.")),
            "the dot belongs inside the name, got {columns:?}"
        );
    }
}
