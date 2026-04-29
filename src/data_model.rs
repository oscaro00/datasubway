// NB: in the table! macro, disregard filter columns that cannot possibly join to the base table when selecting a pre aggregation
use std::collections::{HashMap, HashSet};

use polars::prelude::{LazyFrame, Schema};

use crate::{
    model_components::{pre_aggregations::PreAggregation, query_context::QueryContext},
    wrappers::polars::{
        lazyframe_recorder::LazyFrameRecorder, lazyframe_wrapper::LazyFrameWrapper,
    },
};

use super::model_components::joins::JoinGraph;

pub struct DataModel {
    tables: HashMap<String, LazyFrame>,
    joins: JoinGraph,
    measures: HashMap<String, fn(&QueryContext) -> LazyFrame>,
    pre_aggs: Option<Vec<PreAggregation>>,
    pre_agg_path: Option<&'static str>,
    table_schemas: HashMap<String, Schema>,
}

impl DataModel {
    pub fn new(
        mut tables: HashMap<String, LazyFrame>,
        joins: JoinGraph,
        measures: HashMap<String, fn(&QueryContext) -> LazyFrame>,
        pre_aggs: Vec<PreAggregation>,
        pre_agg_path: Option<&'static str>,
    ) -> DataModel {
        let table_schemas = tables
            .iter_mut()
            .map(|(name, lf)| {
                let schema = lf
                    .collect_schema()
                    .unwrap_or_else(|e| panic!("failed to collect schema for table '{name}': {e}"));
                (name.clone(), (*schema).clone())
            })
            .collect();

        DataModel {
            tables,
            joins,
            measures,
            pre_aggs: if pre_aggs.is_empty() {
                None
            } else {
                Some(pre_aggs)
            },
            pre_agg_path,
            table_schemas,
        }
    }

    pub fn table(&self, table_name: &str) -> LazyFrameRecorder<'_> {
        assert!(
            self.tables.contains_key(table_name),
            "table '{table_name}' not found in DataModel"
        );
        LazyFrameRecorder {
            table_name: table_name.to_string(),
            data_model: self,
            lazy_ops: Vec::new(),
            non_agg_cols: HashSet::new(),
            agg_cols: HashMap::new(),
            non_base_tables: HashSet::new(),
            use_pre_agg: false,
        }
    }

    pub(crate) fn get_table(&self, table_name: &str) -> LazyFrameWrapper {
        // This function will eventually need to check if a pre aggregation exists

        LazyFrameWrapper {
            lazyframe: self
                .tables
                .get(table_name)
                .unwrap_or_else(|| panic!("table '{table_name}' not found in DataModel"))
                .clone(),
            from_pre_agg: false,
        }
    }
}
