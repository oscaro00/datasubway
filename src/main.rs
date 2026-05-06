use std::collections::HashMap;

use datasubway::{data_model::DataModel, model_components::joins::JoinGraph};
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

    Ok(())
}
