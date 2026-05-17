use crate::column_expressions::column_context::AllowExcludeRecord;
use crate::data_model::DataModel;
use crate::model_components::query_context::QueryContext;
use crate::wrappers::polars::lazyframe_recorder::{LazyFrameRecorder, LazyOp};

#[derive(Debug, Clone)]
pub struct MeasureMetadata {
    pub name: String,
    pub output_columns: Vec<String>,
    pub aggregate_columns: Vec<String>,
    pub allow_exclude_calls: Vec<AllowExcludeRecord>,
}

pub type MeasureFn = for<'a> fn(&'a DataModel, &QueryContext) -> LazyFrameRecorder<'a>;

pub struct Measure {
    pub name: String,
    func: MeasureFn,
}

impl Measure {
    pub fn new(name: &str, func: MeasureFn) -> Self {
        Measure {
            name: name.to_string(),
            func,
        }
    }

    pub(crate) fn call<'a>(&self, dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
        (self.func)(dm, qc)
    }
}

pub fn validate_measure_structure(recorder: &LazyFrameRecorder) -> Result<(), String> {
    let ops = &recorder.lazy_ops;
    if ops.len() < 2 {
        return Err("measure must contain at least group_by + agg".into());
    }
    if !matches!(ops.last(), Some(LazyOp::Agg(_))) {
        return Err("measure must end with agg()".into());
    }
    match &ops[ops.len() - 2] {
        LazyOp::GroupBy(_) => Ok(()),
        #[cfg(feature = "dynamic_group_by")]
        LazyOp::GroupByDynamic(_, _, _) | LazyOp::Rolling(_, _, _) => Ok(()),
        _ => Err(
            "second-to-last operation must be group_by(), group_by_dynamic(), or rolling()".into(),
        ),
    }
}

pub fn extract_measure_metadata(recorder: &LazyFrameRecorder, name: &str) -> MeasureMetadata {
    MeasureMetadata {
        name: name.into(),
        output_columns: recorder.agg_cols.keys().map(|k| k.to_string()).collect(),
        aggregate_columns: recorder.agg_cols.keys().map(|k| k.to_string()).collect(),
        allow_exclude_calls: recorder.allow_exclude_records.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::column_expressions::column_context::{
        allow, ColumnContext, ColumnInclude, ColumnPattern,
    };
    use crate::model_components::joins::JoinGraph;
    use polars::prelude::*;
    use std::collections::HashMap;

    fn make_dm() -> DataModel {
        let orders = df![
            "orders.id"     => [1i64, 2, 3],
            "orders.amount" => [100.0f64, 200.0, 150.0],
            "orders.region" => ["US", "EU", "US"],
        ]
        .unwrap()
        .lazy();

        let tables = HashMap::from([("orders".to_string(), orders)]);
        let joins = JoinGraph::new(&[]).unwrap();
        DataModel::new(tables, joins, vec![], None)
    }

    fn valid_measure<'a>(dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
        dm.table("orders")
            .group_by(allow(
                ColumnPattern::OnePattern("orders.*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::OneString("orders.amount".into()),
                dm.table_schema("orders"),
            ))
            .agg(vec![col("orders.amount").sum().alias("revenue")])
    }

    #[test]
    fn test_valid_measure_passes_validation() {
        let mut dm = make_dm();
        assert!(dm
            .add_measure(Measure::new("revenue", valid_measure))
            .is_ok());
    }

    #[test]
    fn test_missing_agg_fails_validation() {
        fn no_agg<'a>(dm: &'a DataModel, _qc: &QueryContext) -> LazyFrameRecorder<'a> {
            dm.table("orders").group_by(vec![col("orders.region")])
        }
        let mut dm = make_dm();
        let err = dm.add_measure(Measure::new("bad", no_agg)).unwrap_err();
        assert!(err.contains("agg"), "expected agg error, got: {err}");
    }

    #[test]
    fn test_wrong_second_to_last_fails_validation() {
        fn bad_measure<'a>(dm: &'a DataModel, _qc: &QueryContext) -> LazyFrameRecorder<'a> {
            dm.table("orders")
                .filter(col("orders.amount").gt(lit(0.0f64)))
                .agg(vec![col("orders.amount").sum()])
        }
        let dm = make_dm();
        let stub_qc = QueryContext::stub();
        let recorder = bad_measure(&dm, &stub_qc);
        let err = validate_measure_structure(&recorder).unwrap_err();
        assert!(
            err.contains("group_by"),
            "expected group_by error, got: {err}"
        );
    }

    #[test]
    fn test_duplicate_name_fails() {
        let mut dm = make_dm();
        dm.add_measure(Measure::new("revenue", valid_measure))
            .unwrap();
        let err = dm
            .add_measure(Measure::new("revenue", valid_measure))
            .unwrap_err();
        assert!(err.contains("already registered"));
    }

    #[test]
    fn test_allow_in_group_by_populates_records() {
        let mut dm = make_dm();
        dm.add_measure(Measure::new("revenue", valid_measure))
            .unwrap();
        let meta = dm.measure_metadata("revenue").unwrap();
        assert!(!meta.allow_exclude_calls.is_empty());
    }

    #[test]
    fn test_plain_vec_group_by_has_no_records() {
        fn plain_measure<'a>(dm: &'a DataModel, _qc: &QueryContext) -> LazyFrameRecorder<'a> {
            dm.table("orders")
                .group_by(vec![col("orders.region")])
                .agg(vec![col("orders.amount").sum().alias("revenue")])
        }
        let mut dm = make_dm();
        dm.add_measure(Measure::new("plain", plain_measure))
            .unwrap();
        let meta = dm.measure_metadata("plain").unwrap();
        assert!(meta.allow_exclude_calls.is_empty());
    }
}
