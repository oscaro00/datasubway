use std::sync::Arc;

use datafusion::prelude::DataFrame;

use crate::model_components::pre_agg_store::PreAggVersion;

/// Thin wrapper around a DataFusion `DataFrame` that records which
/// pre-aggregation version, if any, the underlying data came from.
///
/// `DataFrameRecorder::build()` checks this to decide whether to rewrite
/// aggregation expressions for pre-agg component columns, and the version it
/// carries is what tells the rewriter both where the scan sits (`pre_agg.<name>`)
/// and which physical field holds each logical column.
pub struct DataFrameWrapper {
    pub inner: DataFrame,
    pub pre_agg: Option<Arc<PreAggVersion>>,
}
