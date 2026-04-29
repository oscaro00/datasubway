use polars::prelude::*;

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

    pub fn collect(self) -> Result<DataFrame, PolarsError> {
        self.lazyframe.collect()
    }
}
