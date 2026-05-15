use std::collections::{HashMap, HashSet};

use polars::prelude::{LazyFrame, PlRefPath, PlSmallStr, Schema};

use crate::{
    model_components::{
        measures::{
            extract_measure_metadata, validate_measure_structure, Measure, MeasureMetadata,
        },
        pre_aggregations::{agg_expansion, find_best_pre_agg, PreAggregation},
        query_context::QueryContext,
    },
    wrappers::polars::{
        lazyframe_recorder::LazyFrameRecorder, lazyframe_wrapper::LazyFrameWrapper,
    },
};

use super::model_components::joins::JoinGraph;

pub struct DataModel {
    tables: HashMap<String, LazyFrame>,
    joins: JoinGraph,
    pub(crate) measures: HashMap<String, Measure>,
    pub measure_metadata: HashMap<String, MeasureMetadata>,
    pre_aggs: Option<Vec<PreAggregation>>,
    pre_agg_path: Option<&'static str>,
    table_schemas: HashMap<String, Schema>,
}

impl DataModel {
    pub fn new(
        mut tables: HashMap<String, LazyFrame>,
        joins: JoinGraph,
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
            measures: HashMap::new(),
            measure_metadata: HashMap::new(),
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
            allow_exclude_records: Vec::new(),
        }
    }

    /// Returns the schema for a registered table.
    pub fn table_schema(&self, table_name: &str) -> &Schema {
        self.table_schemas
            .get(table_name)
            .unwrap_or_else(|| panic!("schema for table '{table_name}' not found"))
    }

    /// Register a measure with the DataModel. Validates structure at registration time.
    pub fn add_measure(&mut self, measure: Measure) -> Result<(), String> {
        if self.measures.contains_key(&measure.name) {
            return Err(format!("measure '{}' already registered", measure.name));
        }
        let name = measure.name.clone();
        let metadata = self.validate_and_extract_measure(&measure)?;
        self.measures.insert(name.clone(), measure);
        self.measure_metadata.insert(name, metadata);
        Ok(())
    }

    fn validate_and_extract_measure(&self, measure: &Measure) -> Result<MeasureMetadata, String> {
        let stub_qc = QueryContext::stub();
        let recorder = measure.call(self, &stub_qc);
        validate_measure_structure(&recorder)?;
        Ok(extract_measure_metadata(&recorder, &measure.name))
    }

    /// Returns the metadata for a registered measure, if it exists.
    pub fn measure_metadata(&self, name: &str) -> Option<&MeasureMetadata> {
        self.measure_metadata.get(name)
    }

    pub(crate) fn get_table(
        &self,
        table_name: &str,
        non_agg_cols: &HashSet<PlSmallStr>,
        agg_cols: &HashMap<PlSmallStr, Vec<String>>,
    ) -> LazyFrameWrapper {
        if let (Some(pre_aggs), Some(path)) = (&self.pre_aggs, &self.pre_agg_path) {
            let non_agg_vec: Vec<String> = non_agg_cols.iter().map(|s| s.to_string()).collect();

            let agg_map: HashMap<String, HashSet<String>> = agg_cols
                .iter()
                .map(|(col, agg_names)| {
                    let components: HashSet<String> = agg_names
                        .iter()
                        .filter_map(|name| agg_expansion(&name.to_lowercase()).ok())
                        .flatten()
                        .map(|s| s.to_string())
                        .collect();
                    (col.to_string(), components)
                })
                .collect();

            if let Some(best) = find_best_pre_agg(pre_aggs, &non_agg_vec, &agg_map) {
                let pre_agg_file = format!("{}/{}.parquet", path, best.name);
                if let Ok(lf) = LazyFrame::scan_parquet(
                    PlRefPath::from(pre_agg_file.as_str()),
                    Default::default(),
                ) {
                    return LazyFrameWrapper {
                        lazyframe: lf,
                        from_pre_agg: true,
                    };
                }
            }
        }

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
