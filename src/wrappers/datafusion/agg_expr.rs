use std::collections::HashMap;

use datafusion::common::Column;
use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::prelude::{Expr, col};

use crate::model_components::pre_aggregations::{pre_agg_component_col_name, to_pre_agg_col_name};

// ── Parsing ───────────────────────────────────────────────────────────────────

/// Flatten a (possibly qualified) `Column` into a `table.col` / `col` string.
pub(crate) fn qualified_name(c: &Column) -> String {
    match &c.relation {
        Some(rel) => format!("{rel}.{}", c.name),
        None => c.name.clone(),
    }
}

/// Walk a DataFusion `Expr` and return every `(source_column, agg_function_name)`
/// pair found in it. Pattern-matches the Expr tree directly — no JSON round-trip.
pub fn extract_agg_exprs(expr: &Expr) -> Vec<(String, String)> {
    match expr {
        Expr::AggregateFunction(AggregateFunction { func, params, .. }) => {
            let agg_name = func.name().to_lowercase();
            let col_name = params.args.first().and_then(extract_col_name);
            match col_name {
                Some(c) => vec![(c, agg_name)],
                None => vec![],
            }
        }
        Expr::Alias(a) => extract_agg_exprs(&a.expr),
        Expr::BinaryExpr(b) => {
            let mut v = extract_agg_exprs(&b.left);
            v.extend(extract_agg_exprs(&b.right));
            v
        }
        _ => vec![],
    }
}

/// Recursively find the first plain column name in an expression,
/// looking through casts and aliases.
pub(crate) fn extract_col_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(c) => Some(qualified_name(c)),
        Expr::Alias(a) => extract_col_name(&a.expr),
        Expr::Cast(c) => extract_col_name(&c.expr),
        Expr::TryCast(c) => extract_col_name(&c.expr),
        _ => None,
    }
}

/// Resolve an alias chain: follow the alias_map until we reach an entry
/// that has no further mapping (the ultimate source column name).
pub fn resolve_source_col(name: &str, alias_map: &HashMap<String, String>) -> String {
    let mut current = name;
    let mut seen = std::collections::HashSet::new();
    loop {
        if !seen.insert(current) {
            break; // cycle guard
        }
        match alias_map.get(current) {
            Some(parent) => current = parent.as_str(),
            None => break,
        }
    }
    current.to_string()
}

// ── Rewriting ─────────────────────────────────────────────────────────────────

/// Rewrite a group-by column expression to reference the pre-aggregation table.
///
/// References the dunder column in the aliased pre-agg table and aliases the result
/// back to the original qualified name — using a *qualified* alias so the output
/// field is `(Some("players"), "player_name")`, structurally identical to what a
/// raw-table-sourced group-by column produces (a bare `col("players.player_name")`
/// keeps its table's real qualifier). This keeps the aggregate output schema
/// consistent with the non-pre-agg path regardless of source, so downstream code
/// (`flatten_df`, the FULL JOIN merge path in `agg_builder.rs`) doesn't need to
/// special-case "did this measure come from a pre-agg or a raw table".
///
/// e.g. `col("players.player_name")` →
///      `col("player_goals_by_player.players__player_name").alias_qualified(Some("players"), "player_name")`
pub fn rewrite_group_for_pre_agg(
    expr: Expr,
    alias_map: &HashMap<String, String>,
    pre_agg_name: &str,
) -> Expr {
    match expr {
        Expr::Column(c) => {
            let flat = qualified_name(&c);
            let resolved = resolve_source_col(&flat, alias_map);
            let source = col(format!("{pre_agg_name}.{}", to_pre_agg_col_name(&resolved)).as_str());
            match flat.split_once('.') {
                Some((table, col_name)) => {
                    source.alias_qualified(Some(table.to_string()), col_name.to_string())
                }
                None => source.alias(flat.as_str()),
            }
        }
        other => other,
    }
}

/// Rewrite an aggregation expression to operate on pre-aggregated component
/// columns instead of the raw source column.
///
/// Only single-agg expressions (possibly wrapped in an alias) are rewritten.
/// Complex binary expressions are recursed into, but the pre-agg formula for
/// each leaf agg is substituted independently.
pub fn rewrite_for_pre_agg(
    expr: Expr,
    alias_map: &HashMap<String, String>,
    pre_agg_name: &str,
) -> Expr {
    match expr {
        Expr::Alias(a) => {
            let rewritten = rewrite_for_pre_agg(*a.expr, alias_map, pre_agg_name);
            rewritten.alias(a.name.as_str())
        }
        Expr::AggregateFunction(ref agg) => {
            let agg_name = agg.func.name().to_lowercase();
            let col_name = agg
                .params
                .args
                .first()
                .and_then(|a| extract_col_name(a).map(|n| resolve_source_col(&n, alias_map)));
            match col_name {
                Some(c) => build_pre_agg_expr(&c, &agg_name, pre_agg_name).unwrap_or(expr),
                None => expr,
            }
        }
        Expr::BinaryExpr(b) => {
            let left = rewrite_for_pre_agg(*b.left, alias_map, pre_agg_name);
            let right = rewrite_for_pre_agg(*b.right, alias_map, pre_agg_name);
            Expr::BinaryExpr(datafusion::logical_expr::BinaryExpr {
                left: Box::new(left),
                op: b.op,
                right: Box::new(right),
            })
        }
        other => other,
    }
}

fn build_pre_agg_expr(col_name: &str, agg_name: &str, pre_agg_name: &str) -> Option<Expr> {
    use datafusion::functions_aggregate::expr_fn::{max, min, sum};

    // col("pre_agg_name.table__col__component") — DataFusion splits at the first dot,
    // yielding Column{relation: Some(pre_agg_name), name: "table__col__component"}.
    let c = |component: &str| {
        col(format!(
            "{pre_agg_name}.{}",
            pre_agg_component_col_name(col_name, component)
        )
        .as_str())
    };

    Some(match agg_name {
        "sum" => sum(c("sum")),
        "count" => sum(c("count")),
        "min" => min(c("min")),
        "max" => max(c("max")),
        "avg" | "mean" => sum(c("sum")) / sum(c("count")),
        "stddev" | "stddev_pop" | "std" => {
            use datafusion::functions::expr_fn::sqrt;
            let n = sum(c("count"));
            let mean = sum(c("sum")) / n.clone();
            let variance = sum(c("sumsq")) / n.clone() - mean.clone() * mean;
            sqrt(variance)
        }
        "variance" | "var_pop" | "var" => {
            let n = sum(c("count"));
            let mean = sum(c("sum")) / n.clone();
            sum(c("sumsq")) / n - mean.clone() * mean
        }
        _ => return None,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::functions_aggregate::expr_fn::sum;
    use datafusion::prelude::col;

    fn single(expr: &Expr) -> (String, String) {
        let mut v = extract_agg_exprs(expr);
        assert_eq!(v.len(), 1, "expected exactly one agg pair, got {v:?}");
        v.remove(0)
    }

    #[test]
    fn test_sum() {
        assert_eq!(single(&sum(col("x"))), ("x".into(), "sum".into()));
    }

    #[test]
    fn test_alias_wrapper() {
        let expr = sum(col("x")).alias("total");
        assert_eq!(extract_agg_exprs(&expr), vec![("x".into(), "sum".into())]);
    }

    #[test]
    fn test_resolve_source_col_chain() {
        let mut map = HashMap::new();
        map.insert("b".to_string(), "a".to_string());
        map.insert("c".to_string(), "b".to_string());
        assert_eq!(resolve_source_col("c", &map), "a");
    }

    #[test]
    fn test_resolve_source_col_no_alias() {
        let map = HashMap::new();
        assert_eq!(resolve_source_col("orders.amount", &map), "orders.amount");
    }
}
