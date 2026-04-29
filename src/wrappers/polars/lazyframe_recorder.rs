use std::collections::{HashMap, HashSet};

use polars::prelude::*;

use super::super::super::data_model::DataModel;
use super::lazyframe_wrapper::LazyFrameWrapper;

pub enum LazyOp {
    Sort(Vec<PlSmallStr>, SortMultipleOptions),
    Filter(Expr),
    // ... etc
}

pub struct LazyFrameRecorder<'a> {
    pub table_name: String,
    pub data_model: &'a DataModel,
    pub lazy_ops: Vec<LazyOp>,
    pub non_agg_cols: HashSet<PlSmallStr>,
    pub agg_cols: HashMap<PlSmallStr, Vec<String>>,
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

    pub fn filter(mut self, predicate: Expr) -> LazyFrameRecorder<'a> {
        let cols = predicate.clone().meta().root_names();
        self.non_agg_cols.extend(cols);
        self.lazy_ops.push(LazyOp::Filter(predicate));
        self
    }

    pub fn build(self) -> LazyFrameWrapper {
        let lfw = self.data_model.get_table(&self.table_name);
        let mut result = self.lazy_ops.into_iter().fold(lfw, |lfw, op| match op {
            LazyOp::Sort(by, opts) => lfw.sort(Some(by), Some(opts)),
            LazyOp::Filter(predicate) => lfw.filter(Some(predicate)),
        });
        result.from_pre_agg = self.use_pre_agg;
        result
    }
}
