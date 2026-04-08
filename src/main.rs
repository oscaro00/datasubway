use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_functions_aggregate::sum::sum;
use datasubway::data_model::DataModel;
use datasubway::model::column_context;
use datasubway::model::joins::Join;
use datasubway::model::query_context::QueryContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dm = DataModel::new()?;

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
        how: "inner".into(),
        direction: "right2left".into(),
    }])?;

    // Register a measure
    dm.register_measure(
        "revenue",
        Arc::new(|qc, dm| {
            let group_exprs = column_context::allow_exprs(&["*".into()], &qc.groups)
                .map_err(|e| datafusion::common::DataFusionError::Plan(e))?;
            dm.table("orders")?
                .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
        }),
    )?;

    // Query with cross-table grouping (AutoJoinRule resolves the join)
    let qc = QueryContext::new(
        vec!["revenue".into()],
        None,
        Some(vec!["customers.name".into()]),
        None,
        Some(vec![("revenue".into(), "desc".into())]),
        None,
        None,
        None,
    )?;

    let results = dm.query(&qc)?;
    for batch in &results {
        println!("{:?}", batch);
    }

    Ok(())
}
