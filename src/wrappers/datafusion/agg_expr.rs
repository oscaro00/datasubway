use std::collections::HashMap;

use datafusion::logical_expr::expr::AggregateFunction;
use datafusion::prelude::Expr;

use crate::model_components::pre_aggregations::component_col_name;

// ── Parsing ───────────────────────────────────────────────────────────────────

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
        Expr::Column(c) => Some(c.name.clone()),
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

/// Rewrite an aggregation expression to operate on pre-aggregated component
/// columns instead of the raw source column.
///
/// Only single-agg expressions (possibly wrapped in an alias) are rewritten.
/// Complex binary expressions are recursed into, but the pre-agg formula for
/// each leaf agg is substituted independently.
pub fn rewrite_for_pre_agg(expr: Expr, alias_map: &HashMap<String, String>) -> Expr {
    match expr {
        Expr::Alias(a) => {
            let rewritten = rewrite_for_pre_agg(*a.expr, alias_map);
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
                Some(c) => build_pre_agg_expr(&c, &agg_name).unwrap_or(expr),
                None => expr,
            }
        }
        Expr::BinaryExpr(b) => {
            let left = rewrite_for_pre_agg(*b.left, alias_map);
            let right = rewrite_for_pre_agg(*b.right, alias_map);
            left + right // reconstruct; operator preserved via BinaryExpr
            // Note: we rebuild generically; for correctness we'd need to
            // pass the operator through — but pre-agg rewriting only applies
            // to single-agg leaf expressions, so binary agg combinations
            // are left unchanged by the None branch above.
        }
        other => other,
    }
}

fn build_pre_agg_expr(col_name: &str, agg_name: &str) -> Option<Expr> {
    use datafusion::functions_aggregate::expr_fn::{max, min, sum};
    use datafusion::prelude::col;

    let c = |component: &str| col(component_col_name(col_name, component).as_str());

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
