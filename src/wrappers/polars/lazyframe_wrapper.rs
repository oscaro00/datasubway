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

    pub fn remove(self, predicate: Option<Expr>) -> LazyFrameWrapper {
        if let Some(pred) = predicate {
            LazyFrameWrapper {
                lazyframe: self.lazyframe.remove(pred),
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

    pub fn with_columns(self, exprs: Vec<Expr>) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.with_columns(&exprs),
            from_pre_agg: self.from_pre_agg,
        }
    }

    #[cfg(feature = "dynamic_group_by")]
    pub fn group_by_dynamic(
        self,
        index_column: Expr,
        group_by: Vec<Expr>,
        options: DynamicGroupOptions,
    ) -> LazyGroupByWrapper {
        LazyGroupByWrapper {
            lazygroupby: self
                .lazyframe
                .group_by_dynamic(index_column, &group_by, options),
            from_pre_agg: self.from_pre_agg,
        }
    }

    #[cfg(feature = "dynamic_group_by")]
    pub fn rolling(
        self,
        index_column: Expr,
        group_by: Vec<Expr>,
        options: RollingGroupOptions,
    ) -> LazyGroupByWrapper {
        LazyGroupByWrapper {
            lazygroupby: self.lazyframe.rolling(index_column, &group_by, options),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn sort_by_exprs(self, by: Vec<Expr>, sort_options: SortMultipleOptions) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.sort_by_exprs(&by, sort_options),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn top_k(self, k: u32, by: Vec<Expr>, sort_options: SortMultipleOptions) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.top_k(k, &by, sort_options),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn bottom_k(self, k: u32, by: Vec<Expr>, sort_options: SortMultipleOptions) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.bottom_k(k, &by, sort_options),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn reverse(self) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.reverse(),
            from_pre_agg: self.from_pre_agg,
        }
    }

    pub fn slice(self, offset: i64, len: u32) -> LazyFrameWrapper {
        LazyFrameWrapper {
            lazyframe: self.lazyframe.slice(offset, len),
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
