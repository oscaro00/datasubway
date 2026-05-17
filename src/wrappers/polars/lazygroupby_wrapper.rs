use polars::prelude::*;

use super::agg_expr_rewriter::rewrite_for_pre_agg;
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
