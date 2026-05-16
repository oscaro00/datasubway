use polars::prelude::*;

use crate::model_components::pre_aggregations::component_col_name;

use super::agg_expr_parser::extract_agg_exprs;
use super::lazyframe_wrapper::LazyFrameWrapper;

pub struct LazyGroupByWrapper {
    pub lazygroupby: LazyGroupBy,
    pub from_pre_agg: bool,
}

impl LazyGroupByWrapper {
    pub fn having(self, predicate: Expr) -> LazyGroupByWrapper {
        LazyGroupByWrapper {
            lazygroupby: self.lazygroupby.having(predicate),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn agg(self, aggs: Vec<Expr>) -> LazyFrameWrapper {
        let aggs = if self.from_pre_agg {
            aggs.into_iter().map(rewrite_for_pre_agg).collect()
        } else {
            aggs
        };
        LazyFrameWrapper {
            lazyframe: self.lazygroupby.agg(aggs),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn head(self, n: Option<usize>) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazygroupby.head(n),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn tail(self, n: Option<usize>) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazygroupby.tail(n),
            from_pre_agg: self.from_pre_agg,
        }
    }
}

/// Rewrite a single aggregate expression to reference pre-agg component columns.
///
/// e.g. `col("orders.amount").mean().alias("avg")` becomes
///      `(col("orders.amount-sum").sum() / col("orders.amount-count").sum()).alias("avg")`
fn rewrite_for_pre_agg(expr: Expr) -> Expr {
    let (inner, alias) = match expr {
        Expr::Alias(inner_arc, name) => ((*inner_arc).clone(), Some(name)),
        other => (other, None),
    };

    let pairs = extract_agg_exprs(&inner);
    let rewritten = if pairs.len() == 1 {
        let (col_name, agg_name) = &pairs[0];
        build_pre_agg_expr(col_name, agg_name)
    } else {
        None
    };

    let result = rewritten.unwrap_or(inner);
    match alias {
        Some(name) => result.alias(name.as_str()),
        None => result,
    }
}

fn build_pre_agg_expr(col_name: &str, agg_name: &str) -> Option<Expr> {
    let c = |component: &str| col(&component_col_name(col_name, component));

    Some(match agg_name {
        "sum" => c("sum").sum(),
        "count" => c("count").sum(),
        "min" => c("min").min(),
        "max" => c("max").max(),
        "mean" => c("sum").sum() / c("count").sum(),
        "std" => {
            let n = c("count").sum();
            let mean = c("sum").sum() / n.clone();
            let variance = c("sumsq").sum() / n - mean.pow(lit(2.0f64));
            variance.sqrt()
        }
        "var" => {
            let n = c("count").sum();
            let mean = c("sum").sum() / n.clone();
            c("sumsq").sum() / n - mean.pow(lit(2.0f64))
        }
        _ => return None,
    })
}
