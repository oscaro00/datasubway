use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_functions_aggregate::sum::sum;
use datasubway::data_model::DataModel;
use datasubway::model::column_context::ColumnInput::*;
use datasubway::model::joins::{Join, JoinDirection, JoinHow};
use datasubway::model::pre_agg::PreAggregation;
use datasubway::model::query_context::QueryContext;
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dm = DataModel::new();

    // Register tables
    let orders = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Int64, false),
            Field::new("customer_id", DataType::Int64, false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["US", "EU", "US", "EU", "US"])),
            Arc::new(Int64Array::from(vec![100, 200, 150, 250, 300])),
            Arc::new(Int64Array::from(vec![1, 2, 1, 2, 1])),
        ],
    )?;

    let customers = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1, 2])),
            Arc::new(StringArray::from(vec!["Alice", "Bob"])),
        ],
    )?;

    dm.register_record_batch("orders", orders)?;
    dm.register_record_batch("customers", customers)?;

    dm.set_joins(&[Join {
        left: "orders".into(),
        right: "customers".into(),
        left_on: vec!["customer_id".into()],
        right_on: vec!["id".into()],
        how: JoinHow::Inner,
        direction: JoinDirection::Right2Left,
    }])?;

    // Register a measure
    dm.register_measure(
        "revenue",
        Arc::new(|qc, dm| {
            Box::pin(async move {
                let filter_expr = dm
                    .allow(&["*".into()], FilterTree(&qc.filters), None)?
                    .into_filter_expr();
                let group_exprs = dm
                    .allow(&["*".into()], Columns(&qc.groups), None)?
                    .into_exprs();
                dm.table("orders")
                    .await?
                    .filter(filter_expr)?
                    .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
            })
        }),
    )
    .await?;

    // Query with cross-table grouping and a filter (eager joins from table())
    let qc = QueryContext::new(
        vec!["revenue".into()],
        Some(json!({"AND": [["orders.region", "=", "US"]]})),
        Some(vec!["customers.name".into()]),
        None,
        Some(vec![("revenue".into(), "desc".into())]),
        None,
        None,
        None,
    )?;

    let results = dm.collect(&qc).await?;
    for batch in &results {
        println!("{:?}", batch);
    }

    // ── Pre-aggregation demo ────────────────────────────────────────────
    //
    // Pre-aggregations store grouped results in local parquet files.
    // The optimizer automatically rewrites queries to use them when they
    // cover the requested group-by columns and aggregations.

    println!("\n=== Pre-Aggregation Demo ===\n");

    // 1. Set the output directory for pre-agg parquet files
    let tmp_dir = std::env::temp_dir().join("datasubway_demo");
    dm.set_pre_agg_path(tmp_dir.to_str().unwrap());

    // 2. Define the pre-aggregation
    let pa = PreAggregation::new(
        "regional_revenue_preagg".into(),
        vec!["orders.region".into()],
        HashMap::from([("orders.amount".into(), vec!["sum".into()])]),
    )
    .map_err(|e| Box::<dyn std::error::Error>::from(e))?;

    // 3. Write it — computes the aggregation, writes parquet, registers optimizers
    dm.write_pre_agg(vec![pa]).await?;

    // Query revenue grouped by region — the optimizer should use the pre-agg
    let qc_preagg = QueryContext::new(
        vec!["revenue".into()],
        None,
        Some(vec!["orders.region".into()]),
        None,
        Some(vec![("orders.region".into(), "asc".into())]),
        None,
        None,
        None, // use_pre_agg defaults to true
    )?;

    // Show the explain plan — should reference the pre-agg table
    println!("Explain plan (pre-agg enabled):");
    let explain_df = dm.explain(&qc_preagg, false, false).await?;
    let explain_batches = explain_df.collect().await?;
    for batch in &explain_batches {
        let plan_col = batch
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        for i in 0..batch.num_rows() {
            println!("  {}", plan_col.value(i));
        }
    }

    // Collect and print results
    println!("\nResults (from pre-agg):");
    let results = dm.collect(&qc_preagg).await?;
    for batch in &results {
        println!("{:?}", batch);
    }

    // Clean up temp file
    let _ = std::fs::remove_dir_all(&tmp_dir);

    Ok(())
}
