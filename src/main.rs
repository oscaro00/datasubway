use std::collections::HashMap;

use datasubway::{
    data_model::DataModel,
    model_components::{joins::JoinGraph, pre_aggregations::PreAggregation},
};
use polars::prelude::*;

pub fn main() -> Result<(), &'static str> {
    let orders = df![
        "id"          => [1i64, 2, 3],
        "customer_id" => [10i64, 20, 10],
        "amount"      => [100.0f64, 200.0, 150.0],
    ]
    .unwrap()
    .lazy();

    let tables = HashMap::from([("orders".to_string(), orders)]);
    let joins = JoinGraph::new(&[]).unwrap();

    let dm = DataModel::new(tables, joins, vec![], None);

    let test_result = dm
        .table("orders")
        .filter(Some(col("customer_id").neq(lit(20i64))))
        .sort(["amount"], SortMultipleOptions::default())
        .build()
        .collect();

    println!("{:?}", test_result);

    let test_result2 = dm.table("orders").build().collect();

    println!("{:?}", test_result2);

    let _test_result3 = dm
        .table("orders")
        .group_by(vec![col("customer_id")])
        .agg(vec![col("amount").max().alias("amount")])
        .build()
        .collect();

    println!("{:?}", _test_result3);

    // --- Pre-aggregation example ---
    // Build a table with date/region breakdown so the pre-agg is meaningful.
    let orders_detailed = df![
        "date"        => ["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02"],
        "region"      => ["north", "south", "north", "south"],
        "amount"      => [100.0f64, 200.0, 150.0, 250.0],
    ]
    .unwrap()
    .lazy();

    // Define a pre-aggregation that groups by date and region, storing sum and
    // the components needed to reconstruct mean (sum + count).
    let pa = PreAggregation::new(
        "daily_revenue".into(),
        vec!["orders.date".into(), "orders.region".into()],
        HashMap::from([("orders.amount".into(), vec!["sum".into(), "mean".into()])]),
    )
    .unwrap();

    let tmp_dir = std::env::temp_dir();
    let dm2 = DataModel::new(
        HashMap::from([("orders".to_string(), orders_detailed)]),
        JoinGraph::new(&[]).unwrap(),
        vec![pa],
        Some(tmp_dir.to_str().unwrap().to_string()),
    );

    // Write the pre-agg parquet to the temp directory.
    dm2.write_pre_aggs(&["daily_revenue"]).unwrap();

    // This query is covered by the pre-agg (same group_by + sum on amount),
    // so DataModel automatically reads from the pre-agg file instead of
    // recomputing over the base table.
    let pre_agg_result = dm2
        .table("orders")
        .group_by(vec![col("orders.date"), col("orders.region")])
        .agg(vec![col("orders.amount").sum().alias("total")])
        .build()
        .collect();

    println!("pre-agg result: {:?}", pre_agg_result);

    Ok(())
}
