use std::collections::HashMap;

use datafusion::common::Column;
use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
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
///
/// Uses `TreeNode::apply` rather than hand-enumerating `Expr` variants, for the same
/// reason `rewrite_expr_for_pre_agg` does: a measure is free to wrap an aggregate in
/// anything (`cast(sum(x), Float64)`, `nullif(sum(x), lit(0))`, a `CASE`), and a
/// hand-written recursion silently returns no pair for a shape it doesn't know. That
/// is not a harmless miss — `DataFrameRecorder::aggregate` files every column *not*
/// named here into `non_agg_cols`, so an unrecognised wrapper makes `covers()` fail
/// and quietly costs the measure its pre-aggregation.
///
/// Aggregates cannot nest, so a whole-tree walk cannot double-count.
pub fn extract_agg_exprs(expr: &Expr) -> Vec<(String, String)> {
    let mut pairs = Vec::new();
    // The closure never fails, so the `Result` carries no information. Keeping the
    // signature infallible avoids rippling `Result` into `DataFrameRecorder::aggregate`,
    // which returns `Self`.
    let _ = expr.apply(|e| {
        if let Expr::AggregateFunction(AggregateFunction { func, params, .. }) = e
            && let Some(c) = params.args.first().and_then(extract_col_name)
        {
            pairs.push((c, func.name().to_lowercase()));
        }
        Ok(TreeNodeRecursion::Continue)
    });
    pairs
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
/// Every `AggregateFunction` anywhere in the tree is substituted for its pre-agg
/// formula; everything around it — aliases, arithmetic, casts, scalar functions,
/// `CASE` — is carried through untouched. That matters because the surrounding
/// nodes are load-bearing: a measure written as
/// `cast(sum(a), Float64) / nullif(sum(b), lit(0))` is float division with a null
/// guard, and dropping the cast on the way to the pre-agg would silently turn it
/// back into integer division.
///
/// `transform_up` maps children before the parent and does not re-traverse what the
/// closure returns, so the `sum(...)` calls that `build_pre_agg_expr` emits are not
/// themselves rewritten. `transform_down` would loop.
pub fn rewrite_for_pre_agg(
    expr: Expr,
    alias_map: &HashMap<String, String>,
    pre_agg_name: &str,
) -> datafusion::common::Result<Expr> {
    expr.transform_up(|e| {
        let Expr::AggregateFunction(ref agg) = e else {
            return Ok(Transformed::no(e));
        };
        let agg_name = agg.func.name().to_lowercase();
        let col_name = agg
            .params
            .args
            .first()
            .and_then(|a| extract_col_name(a).map(|n| resolve_source_col(&n, alias_map)));
        match col_name.and_then(|c| build_pre_agg_expr(&c, &agg_name, pre_agg_name)) {
            Some(rewritten) => Ok(Transformed::yes(rewritten)),
            None => Ok(Transformed::no(e)),
        }
    })
    .map(|t| t.data)
}

/// Rewrite every column reference within an arbitrary expression tree (e.g. a
/// filter predicate, or a `distinct_on` selector) to reference the
/// pre-aggregation's physical (dunder-encoded) column, so expressions written
/// against logical qualified names (e.g. `players.player_name`) can be
/// evaluated directly against a pre-agg source.
///
/// Unlike `rewrite_group_for_pre_agg`/`rewrite_for_pre_agg` (which only need to
/// handle the constrained top-level shapes `.aggregate()` produces — a bare
/// `Column`, or a `Column` wrapped in a single `AggregateFunction`), this walks
/// the whole expression tree via DataFusion's own `TreeNode::transform`, so it
/// is correct for arbitrary predicates (`AND`/`OR`, comparisons, `IS NULL`,
/// `BETWEEN`, `IN`, `LIKE`, ...) without needing to hand-enumerate every `Expr`
/// variant. No alias-back is needed (unlike the group-by rewrite) since these
/// expressions don't appear in the output schema — they only gate/select rows.
pub fn rewrite_expr_for_pre_agg(
    expr: Expr,
    alias_map: &HashMap<String, String>,
    pre_agg_name: &str,
) -> datafusion::common::Result<Expr> {
    expr.transform(|e| {
        if let Expr::Column(c) = &e {
            let flat = qualified_name(c);
            let resolved = resolve_source_col(&flat, alias_map);
            let rewritten =
                col(format!("{pre_agg_name}.{}", to_pre_agg_col_name(&resolved)).as_str());
            Ok(Transformed::yes(rewritten))
        } else {
            Ok(Transformed::no(e))
        }
    })
    .map(|t| t.data)
}

/// Rewrite a plain column name (e.g. for `drop_columns`) to the pre-aggregation's
/// physical dunder-encoded name. Companion to `rewrite_expr_for_pre_agg` for ops
/// that carry column references as `String`s rather than `Expr` trees.
pub fn rewrite_col_name_for_pre_agg(
    name: &str,
    alias_map: &HashMap<String, String>,
    pre_agg_name: &str,
) -> String {
    let resolved = resolve_source_col(name, alias_map);
    format!("{pre_agg_name}.{}", to_pre_agg_col_name(&resolved))
}

fn build_pre_agg_expr(col_name: &str, agg_name: &str, pre_agg_name: &str) -> Option<Expr> {
    use datafusion::arrow::datatypes::DataType;
    use datafusion::functions_aggregate::expr_fn::{max, min, sum};
    use datafusion::logical_expr::expr::Cast;

    // col("pre_agg_name.table__col__component") — DataFusion splits at the first dot,
    // yielding Column{relation: Some(pre_agg_name), name: "table__col__component"}.
    let c = |component: &str| {
        col(format!(
            "{pre_agg_name}.{}",
            pre_agg_component_col_name(col_name, component)
        )
        .as_str())
    };

    // Rolling a stored component back up preserves its type: `sum`, `count` and
    // `sumsq` over an integer source column are all Int64. Dividing two Int64
    // expressions is *integer* division, so any mean below 1 collapsed to 0 —
    // a 93/200 win rate came back as 0 rather than 0.465. Every ratio below is
    // therefore computed in floating point.
    //
    // Only the derived statistics are cast. `sum`/`min`/`max` deliberately keep
    // the source type, so summing an integer column still yields an integer.
    let real = |e: Expr| Expr::Cast(Cast::new(Box::new(e), DataType::Float64));

    Some(match agg_name {
        "sum" => sum(c("sum")),
        "count" => sum(c("count")),
        "min" => min(c("min")),
        "max" => max(c("max")),
        "avg" | "mean" => real(sum(c("sum"))) / real(sum(c("count"))),
        "stddev" | "stddev_pop" | "std" => {
            use datafusion::functions::expr_fn::sqrt;
            let n = real(sum(c("count")));
            let mean = real(sum(c("sum"))) / n.clone();
            let variance = real(sum(c("sumsq"))) / n.clone() - mean.clone() * mean;
            sqrt(variance)
        }
        "variance" | "var_pop" | "var" => {
            let n = real(sum(c("count")));
            let mean = real(sum(c("sum"))) / n.clone();
            real(sum(c("sumsq"))) / n - mean.clone() * mean
        }
        _ => return None,
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::functions_aggregate::expr_fn::{avg, sum};
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

    #[test]
    fn test_rewrite_expr_for_pre_agg_bare_column() {
        let map = HashMap::new();
        let expr = col("players.player_name");
        let rewritten = rewrite_expr_for_pre_agg(expr, &map, "goals_by_player").unwrap();
        assert_eq!(
            format!("{rewritten}"),
            "goals_by_player.players__player_name"
        );
    }

    #[test]
    fn test_rewrite_expr_for_pre_agg_comparison() {
        use datafusion::prelude::lit;

        let map = HashMap::new();
        let expr = col("players.player_name").eq(lit("Nwpo"));
        let rewritten = rewrite_expr_for_pre_agg(expr, &map, "goals_by_player").unwrap();
        assert_eq!(
            format!("{rewritten}"),
            "goals_by_player.players__player_name = Utf8(\"Nwpo\")"
        );
    }

    #[test]
    fn test_rewrite_expr_for_pre_agg_compound_and_or() {
        use datafusion::prelude::lit;

        let map = HashMap::new();
        // (players.player_name = 'Nwpo' AND orders.amount > 0) OR orders.region = 'north'
        let expr = col("players.player_name")
            .eq(lit("Nwpo"))
            .and(col("orders.amount").gt(lit(0.0f64)))
            .or(col("orders.region").eq(lit("north")));
        let rewritten = rewrite_expr_for_pre_agg(expr, &map, "pa").unwrap();
        let text = format!("{rewritten}");
        assert!(text.contains("pa.players__player_name"));
        assert!(text.contains("pa.orders__amount"));
        assert!(text.contains("pa.orders__region"));
        // Structure (AND/OR, comparisons) must be preserved, not just column names.
        assert!(text.contains("AND"));
        assert!(text.contains("OR"));
    }

    #[test]
    fn test_rewrite_expr_for_pre_agg_resolves_through_alias() {
        let mut map = HashMap::new();
        map.insert("amt".to_string(), "orders.amount".to_string());
        let expr = col("amt");
        let rewritten = rewrite_expr_for_pre_agg(expr, &map, "pa").unwrap();
        assert_eq!(format!("{rewritten}"), "pa.orders__amount");
    }

    // The wrapper cases below are the ones a hand-written `match` used to miss: a
    // measure that wraps an aggregate in a cast or a scalar function must still
    // report its source column, or `DataFrameRecorder::aggregate` files that column
    // into `non_agg_cols` and the measure silently loses its pre-aggregation.

    #[test]
    fn test_extract_agg_exprs_through_cast() {
        use datafusion::arrow::datatypes::DataType;
        use datafusion::prelude::cast;

        let expr = cast(sum(col("orders.amount")), DataType::Float64);
        assert_eq!(
            extract_agg_exprs(&expr),
            vec![("orders.amount".into(), "sum".into())]
        );
    }

    #[test]
    fn test_extract_agg_exprs_through_scalar_function() {
        use datafusion::prelude::{lit, nullif};

        let expr = nullif(sum(col("orders.qty")), lit(0));
        assert_eq!(
            extract_agg_exprs(&expr),
            vec![("orders.qty".into(), "sum".into())]
        );
    }

    #[test]
    fn test_extract_agg_exprs_ratio_of_wrapped_aggs() {
        use datafusion::arrow::datatypes::DataType;
        use datafusion::prelude::{cast, lit, nullif};

        // The exact shape a float-division measure takes.
        let expr = (cast(
            sum(col("orders.amount")) + sum(col("orders.tax")),
            DataType::Float64,
        ) / nullif(sum(col("orders.qty")), lit(0)))
        .alias("orders.rate");

        assert_eq!(
            extract_agg_exprs(&expr),
            vec![
                ("orders.amount".into(), "sum".into()),
                ("orders.tax".into(), "sum".into()),
                ("orders.qty".into(), "sum".into()),
            ]
        );
    }

    #[test]
    fn test_rewrite_for_pre_agg_preserves_cast_and_nullif() {
        use datafusion::arrow::datatypes::DataType;
        use datafusion::prelude::{cast, lit, nullif};

        let map = HashMap::new();
        let expr = (cast(sum(col("orders.amount")), DataType::Float64)
            / nullif(sum(col("orders.qty")), lit(0)))
        .alias("orders.rate");

        let rewritten = rewrite_for_pre_agg(expr, &map, "pa").unwrap();
        let text = format!("{rewritten}");

        // Both aggregates point at their component columns...
        assert!(text.contains("pa.orders__amount__sum"), "{text}");
        assert!(text.contains("pa.orders__qty__sum"), "{text}");
        // ...and the wrappers that make this float division survive.
        assert!(text.contains("CAST"), "{text}");
        assert!(text.contains("nullif"), "{text}");
        assert!(text.contains("AS orders.rate"), "{text}");
    }

    #[test]
    fn test_rewrite_for_pre_agg_does_not_rewrite_its_own_output() {
        let map = HashMap::new();
        // `avg` expands to sum/count over component columns; those inner `sum`s must
        // not be fed back through the rewrite (which would look for a `__sum__sum`).
        let expr = avg(col("orders.amount")).alias("orders.avg_amount");
        let rewritten = rewrite_for_pre_agg(expr, &map, "pa").unwrap();
        let text = format!("{rewritten}");

        assert!(text.contains("pa.orders__amount__sum"), "{text}");
        assert!(text.contains("pa.orders__amount__count"), "{text}");
        assert!(!text.contains("__sum__sum"), "{text}");
    }

    #[test]
    fn test_rewrite_col_name_for_pre_agg() {
        let map = HashMap::new();
        assert_eq!(
            rewrite_col_name_for_pre_agg("players.player_name", &map, "goals_by_player"),
            "goals_by_player.players__player_name"
        );
    }
}
