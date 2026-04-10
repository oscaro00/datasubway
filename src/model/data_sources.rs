//! Data source registration helpers for DataFusion SessionContext.
//!
//! Each function registers a table with the given SessionContext and returns
//! qualified column names (`table.column`) for schema tracking.

use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::execution::context::SessionContext;

/// Extract qualified column names from an already-registered table.
async fn extract_schema_names(
    ctx: &SessionContext,
    name: &str,
) -> Result<Vec<String>, DataFusionError> {
    let df = ctx.table(name).await?;
    Ok(df
        .schema()
        .fields()
        .iter()
        .map(|f| format!("{}.{}", name, f.name()))
        .collect())
}

/// Register a RecordBatch as an in-memory table.
pub fn register_record_batch(
    ctx: &SessionContext,
    name: &str,
    batch: RecordBatch,
) -> Result<Vec<String>, DataFusionError> {
    let schema_names: Vec<String> = batch
        .schema()
        .fields()
        .iter()
        .map(|f| format!("{}.{}", name, f.name()))
        .collect();

    let schema = batch.schema();
    let mem_table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])?;
    ctx.register_table(name, Arc::new(mem_table))?;
    Ok(schema_names)
}

/// Register a Parquet file as a named table.
pub async fn register_parquet(
    ctx: &SessionContext,
    name: &str,
    path: &str,
) -> Result<Vec<String>, DataFusionError> {
    ctx.register_parquet(name, path, Default::default()).await?;
    extract_schema_names(ctx, name).await
}

/// Register a CSV file as a named table.
pub async fn register_csv(
    ctx: &SessionContext,
    name: &str,
    path: &str,
) -> Result<Vec<String>, DataFusionError> {
    ctx.register_csv(name, path, Default::default()).await?;
    extract_schema_names(ctx, name).await
}

/// Register a newline-delimited JSON file as a named table.
pub async fn register_json(
    ctx: &SessionContext,
    name: &str,
    path: &str,
) -> Result<Vec<String>, DataFusionError> {
    ctx.register_json(name, path, Default::default()).await?;
    extract_schema_names(ctx, name).await
}

/// Register an Arrow IPC file as a named table.
pub async fn register_arrow(
    ctx: &SessionContext,
    name: &str,
    path: &str,
) -> Result<Vec<String>, DataFusionError> {
    ctx.register_arrow(name, path, Default::default()).await?;
    extract_schema_names(ctx, name).await
}
