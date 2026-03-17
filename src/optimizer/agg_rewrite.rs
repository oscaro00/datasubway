use datafusion_common::Column;
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::Expr;
use datafusion_functions_aggregate::min_max::{max, min};
use datafusion_functions_aggregate::sum::sum;

use crate::model::pre_agg::PreAggregation;

/// Rewrite an aggregate expression to use pre-agg component columns.
///
/// For example:
///   sum(orders.amount) → sum(orders.amount-sum)
///   avg(orders.amount) → sum(orders.amount-sum) / sum(orders.amount-count)
///   count(orders.amount) → sum(orders.amount-count)
///   min(orders.amount) → min(orders.amount-min)
///   max(orders.amount) → max(orders.amount-max)
pub fn rewrite_agg_expr(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::AggregateFunction(agg) => rewrite_aggregate_function(agg),
        Expr::Alias(alias) => {
            let rewritten = rewrite_agg_expr(&alias.expr)?;
            Some(rewritten.alias(&alias.name))
        }
        _ => None,
    }
}

fn col_expr(name: &str) -> Expr {
    Expr::Column(Column::from_name(name))
}

fn rewrite_aggregate_function(agg: &AggregateFunction) -> Option<Expr> {
    let col_name = extract_column_name(&agg.params.args[0])?;
    let func_name = agg.func.name();

    match func_name {
        "sum" => {
            let comp_col = PreAggregation::component_column(&col_name, "sum");
            Some(sum(col_expr(&comp_col)))
        }
        "count" => {
            let comp_col = PreAggregation::component_column(&col_name, "count");
            Some(sum(col_expr(&comp_col)))
        }
        "min" => {
            let comp_col = PreAggregation::component_column(&col_name, "min");
            Some(min(col_expr(&comp_col)))
        }
        "max" => {
            let comp_col = PreAggregation::component_column(&col_name, "max");
            Some(max(col_expr(&comp_col)))
        }
        "avg" => {
            let sum_col = PreAggregation::component_column(&col_name, "sum");
            let count_col = PreAggregation::component_column(&col_name, "count");
            Some(sum(col_expr(&sum_col)) / sum(col_expr(&count_col)))
        }
        _ => None,
    }
}

fn extract_column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(col) => Some(col.name.clone()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion_functions_aggregate::average::avg;
    use datafusion_functions_aggregate::count::count;
    use datafusion_functions_aggregate::min_max::{max as max_fn, min as min_fn};
    use datafusion_functions_aggregate::sum::sum as sum_fn;

    #[test]
    fn test_rewrite_sum() {
        let expr = sum_fn(col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-sum"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_count() {
        let expr = count(col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_avg() {
        let expr = avg(col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-sum"), "Got: {}", s);
        assert!(s.contains("orders.amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_min() {
        let expr = min_fn(col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-min"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_max() {
        let expr = max_fn(col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-max"), "Got: {}", s);
    }

    #[test]
    fn test_non_agg_returns_none() {
        let expr = col_expr("orders.amount");
        assert!(rewrite_agg_expr(&expr).is_none());
    }
}
