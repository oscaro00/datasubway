use std::collections::HashSet;

use polars::prelude::*;

use super::super::super::data_model::DataModel;

pub enum LazyOp {
    Sort(Vec<PlSmallStr>, SortMultipleOptions),
    // ... etc
}

pub struct LazyFrameRecorder<'a> {
    pub table_name: String,
    pub data_model: &'a DataModel,
    pub lazy_ops: Vec<LazyOp>,
    pub non_agg_cols: Vec<PlSmallStr>,
    pub agg_exprs: Vec<Expr>,
    pub non_base_tables: HashSet<String>,
    pub use_pre_agg: bool,
}

impl<'a> LazyFrameRecorder<'a> {
    pub fn sort(
        mut self,
        by: impl IntoVec<PlSmallStr>,
        sort_options: SortMultipleOptions,
    ) -> LazyFrameRecorder<'a> {
        let cols = by.into_vec();
        self.non_agg_cols.extend(cols.iter().cloned());
        self.lazy_ops.push(LazyOp::Sort(cols, sort_options));
        self
    }

    pub fn build(self) -> LazyFrame {
        let lf = self.data_model.get_table(&self.table_name);
        self.lazy_ops.into_iter().fold(lf, |lf, op| match op {
            LazyOp::Sort(by, opts) => lf.sort(by, opts),
        })
    }
}
