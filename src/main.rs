use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int64Array};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::datasource::MemTable;
use datafusion::prelude::{col, lit};

use datasubway::{data_model::DataModel, model_components::joins::JoinGraph};

#[tokio::main]
async fn main() {
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("customer_id", DataType::Int64, false),
        Field::new("amount", DataType::Float64, false),
    ]));

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from(vec![1, 2, 3])),
            Arc::new(Int64Array::from(vec![10, 20, 10])),
            Arc::new(Float64Array::from(vec![100.0, 200.0, 150.0])),
        ],
    )
    .unwrap();

    let provider = Arc::new(MemTable::try_new(schema, vec![vec![batch]]).unwrap());
    let tables = HashMap::from([(
        "orders".to_string(),
        provider as Arc<dyn datafusion::catalog::TableProvider>,
    )]);
    let dm = DataModel::new(tables, JoinGraph::new(&[]).unwrap(), vec![], None);

    // Basic filter
    let df = dm
        .table("orders", true)
        .filter(col("orders.customer_id").not_eq(lit(20i64)))
        .build()
        .unwrap();
    let results = df.collect().await.unwrap();
    println!(
        "filter result: {} rows",
        results.iter().map(|b| b.num_rows()).sum::<usize>()
    );

    // Full scan
    let df2 = dm.table("orders", true).build().unwrap();
    let results2 = df2.collect().await.unwrap();
    println!(
        "full scan: {} rows",
        results2.iter().map(|b| b.num_rows()).sum::<usize>()
    );

    // Group-by + agg
    use datafusion::functions_aggregate::expr_fn::sum;
    let df3 = dm
        .table("orders", true)
        .aggregate(
            vec![col("orders.customer_id")],
            vec![sum(col("orders.amount")).alias("total")],
        )
        .build()
        .unwrap();
    let results3 = df3.collect().await.unwrap();
    println!(
        "agg result: {} rows",
        results3.iter().map(|b| b.num_rows()).sum::<usize>()
    );
}
