use datafusion::logical_expr::LogicalPlan;
use datafusion::prelude::{DataFrame, Expr};

use crate::column_expressions::column_context::{
    AllowExcludeKind, AllowExcludeRecord, ColumnReturn, allow, exclude,
};
use crate::data_model::DataModel;
use crate::model_components::agg_context::AggContext;
use crate::wrappers::datafusion::aggregate_with_metadata::AggregateWithMetadata;

#[derive(Debug, Clone)]
pub struct MeasureMetadata {
    pub name: String,
    pub output_columns: Vec<String>,
    pub aggregate_columns: Vec<String>,
    pub allow_exclude_calls: Vec<AllowExcludeRecord>,
}

// ── DataFusion measure types ──────────────────────────────────────────────────
//
// The new measure API: user functions return a `DataFrame` (built via
// `DataFrameRecorder::build()`). Validation and metadata extraction work
// directly from the logical plan rather than inspecting recorder state.

/// Measure function returning a DataFusion DataFrame. The function body should
/// end with `.build(base)` on a `DataFrameRecorder` chain, and that chain must
/// include exactly one `.aggregate()` call as its terminal operation.
pub type DfMeasureFn = fn(&DataModel, &AggContext) -> datafusion::common::Result<DataFrame>;

pub struct DfMeasure {
    pub name: String,
    func: DfMeasureFn,
}

impl DfMeasure {
    pub fn new(name: &str, func: DfMeasureFn) -> Self {
        DfMeasure {
            name: name.to_string(),
            func,
        }
    }

    pub fn call(&self, dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
        (self.func)(dm, qc)
    }
}

/// A measure is valid iff its logical plan root is an `AggregateWithMetadata`
/// node — i.e. the chain ended with `.aggregate()`.
pub fn validate_df_measure_structure(df: &DataFrame) -> Result<(), String> {
    match df.logical_plan() {
        LogicalPlan::Extension(e)
            if e.node
                .as_any()
                .downcast_ref::<AggregateWithMetadata>()
                .is_some() =>
        {
            Ok(())
        }
        _ => Err(
            "measure must end with aggregate() (no AggregateWithMetadata node at plan root)".into(),
        ),
    }
}

/// Extract `MeasureMetadata` by reading the `AggregateWithMetadata` node from
/// the DataFrame's logical plan. The `allow_exclude` key in the node metadata
/// holds a JSON-serialized `Vec<AllowExcludeRecord>`.
pub fn extract_df_measure_metadata(df: &DataFrame, name: &str) -> Result<MeasureMetadata, String> {
    match df.logical_plan() {
        LogicalPlan::Extension(e) => {
            let node = e
                .node
                .as_any()
                .downcast_ref::<AggregateWithMetadata>()
                .ok_or("plan root is not AggregateWithMetadata")?;

            let allow_exclude_records: Vec<AllowExcludeRecord> = node
                .metadata
                .get("allow_exclude")
                .map(|s| serde_json::from_str(s).unwrap_or_default())
                .unwrap_or_default();

            let output_columns: Vec<String> = node
                .aggr_expr
                .iter()
                .filter_map(|e| {
                    if let Expr::Alias(a) = e {
                        Some(a.name.to_string())
                    } else {
                        None
                    }
                })
                .collect();

            Ok(MeasureMetadata {
                name: name.into(),
                output_columns: output_columns.clone(),
                aggregate_columns: output_columns,
                allow_exclude_calls: allow_exclude_records,
            })
        }
        _ => Err("expected AggregateWithMetadata at plan root".into()),
    }
}

/// Resolve the group-by column names from a freshly-built measure DataFrame.
/// Reads the `allow_exclude` records embedded in the `AggregateWithMetadata` node
/// and re-evaluates the last one; falls back to the literal `group_expr` column
/// names if no allow/exclude calls were recorded.
pub fn resolve_group_by_cols(df: &DataFrame) -> Result<Vec<String>, String> {
    let LogicalPlan::Extension(e) = df.logical_plan() else {
        return Err("measure must end with aggregate()".into());
    };
    let node = e
        .node
        .as_any()
        .downcast_ref::<AggregateWithMetadata>()
        .ok_or("expected AggregateWithMetadata at plan root")?;

    let records: Vec<AllowExcludeRecord> = node
        .metadata
        .get("allow_exclude")
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default();

    match records.last() {
        None => {
            let mut cols: Vec<String> = node
                .group_expr
                .iter()
                .filter_map(|e| {
                    if let Expr::Column(c) = e {
                        let name = match &c.relation {
                            Some(r) => format!("{r}.{}", c.name),
                            None => c.name.clone(),
                        };
                        Some(name)
                    } else {
                        None
                    }
                })
                .collect();
            cols.sort();
            Ok(cols)
        }
        Some(record) => {
            let result = match record.kind {
                AllowExcludeKind::Allow => allow(
                    record.pattern.clone(),
                    record.context.clone(),
                    record.include.clone(),
                ),
                AllowExcludeKind::Exclude => exclude(
                    record.pattern.clone(),
                    record.context.clone(),
                    record.include.clone(),
                ),
            };
            match result.inner {
                ColumnReturn::Strings(cols) => Ok(cols),
                ColumnReturn::Expr(_) => {
                    Err("JSON-context allow/exclude cannot produce group-by column names".into())
                }
            }
        }
    }
}
