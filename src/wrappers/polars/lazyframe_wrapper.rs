use polars::prelude::*;

use crate::wrappers::polars::lazygroupby_wrapper::LazyGroupByWrapper;

pub struct LazyFrameWrapper {
    pub lazyframe: LazyFrame,
    pub from_pre_agg: bool,
}

impl LazyFrameWrapper {
    pub fn filter(self, predicate: Option<Expr>) -> LazyFrameWrapper {
        if let Some(pred) = predicate {
            LazyFrameWrapper {
                lazyframe: self.lazyframe.filter(pred),
                from_pre_agg: self.from_pre_agg,
            }
        } else {
            self
        }
    }

    pub fn sort(
        self,
        by: Option<impl IntoVec<PlSmallStr>>,
        sort_options: Option<SortMultipleOptions>,
    ) -> LazyFrameWrapper {
        match (by, sort_options) {
            (Some(by), Some(opts)) => LazyFrameWrapper {
                lazyframe: self.lazyframe.sort(by, opts),
                from_pre_agg: self.from_pre_agg,
            },
            _ => self,
        }
    }

    pub fn group_by(self, by: Vec<Expr>) -> LazyGroupByWrapper {
        LazyGroupByWrapper {
            lazygroupby: self.lazyframe.group_by(&by),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn with_column(self, expr: Expr) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.with_column(expr),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn limit(self, n: u32) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.limit(n),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn collect(self) -> Result<DataFrame, PolarsError> {
        self.lazyframe.collect()
    }

    #[cfg(feature = "async")]
    pub async fn collect_async(self) -> Result<DataFrame, PolarsError> {
        tokio::task::spawn_blocking(move || self.lazyframe.collect())
            .await
            .expect("collect task panicked")
    }
}
