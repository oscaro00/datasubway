use datafusion::prelude::DataFrame;

/// Thin wrapper around a DataFusion `DataFrame` that records whether the
/// underlying data came from a pre-aggregation table. The `from_pre_agg` flag
/// is checked by `DataFrameRecorder::build()` to decide whether to rewrite
/// aggregation expressions for pre-agg component columns.
pub struct DataFrameWrapper {
    pub inner: DataFrame,
    pub from_pre_agg: bool,
    /// The pre-aggregation name when `from_pre_agg` is true. It is the relation
    /// of the version's `TableScan`, so column refs resolve as `pre_agg_name.col`.
    pub pre_agg_name: Option<String>,
}
