use std::collections::HashMap;

use datasubway::{data_model::DataModel, model_components::joins::JoinGraph, table};
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

    let dm = DataModel::new(tables, joins, HashMap::new(), vec![], None);

    let test_result = table!(dm, "orders")
        .filter(Some(col("customer_id").neq(lit(20i64))))
        .sort(Some(["amount"]), Some(SortMultipleOptions::default()))
        .collect();

    println!("{:?}", test_result);

    let test_result2 = table!(dm, "orders")
        .filter(None)
        .sort(Some(["amount"]), Some(SortMultipleOptions::default()))
        .collect();

    println!("{:?}", test_result2);

    Ok(())
}
