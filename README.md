# Project Plan

The purpose of this library is to define a data model using python focusing on the library datafusion. The data model will act as a central repository of domain specific calculations that can be written very flexibly. Calling these calculations will be as simple as naming the calculations along with a json-like object for the query parameters you want as context. I imagine this library would be implemented in tandem with an api library to make data access easy. This data model is primarily geared towards OLAP use cases (i.e. read heavy workloads of large chunks of data). Data insertion and updates on tables will not be scope of this work.

## Core requirements for the data model:

- Measures are atomic 
  - Measures will not be nested because of the maintenance difficulties this causes
- Zero cost pre-aggregation abstractions make performance a priority
  - Regardless of data source, it will be possible to stored aggregated versions of tables in local parquet files, which can be queried indirectly for quicker results
- Measures are written in a dataframe-like method chaining syntax so arbitrarily complex measures are possible
  - Datafusion syntax is a nice balance of clarity, verbosity, and flexibility
  - In order to allow query parameters to be used or excluded in specific datafusion methods in a calculation, a system of allow() and exclude() at column positions in datafusion methods will manage valid columns

## Key features:

- Using the datafusion lazy frame syntax also gives datafusion optimizations to query executions by default
- Pre-aggregations are declarative (i.e. users define which columns to group by before calculations measures) and optimal pre-aggregations are selected by the data model engine without user input
- Pre-aggregations are effective with the assumption that most queries do not need the fully granularity of the table, so a local parquet file with an aggregated version of the data will be faster (no network calls and less data to crunch)
- Pre-aggregations can span several tables (table joins are expensive, so pre-computing them makes sense)

## Longer term vision (this has been harder than anticipated):

- I believe this combinations of structured yet flexible calculations is a great use case for small, local AI models to take natural language and execute reliable calculations. Passing measure descriptions along with measures will support domain specific calculations and terminology and a small model should be able to create a json-like query context. This will hopefully avoid the problem where models are great at writing simple queries, but are inconsistent on harder queries with domain specific knowledge especially across multiple users.



## Example Usage

```rust
use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use datafusion::prelude::*;
use datafusion_functions_aggregate::sum::sum;
use datasubway::data_model::DataModel;
use datasubway::model::column_context;
use datasubway::model::joins::Join;
use datasubway::model::pre_agg::PreAggregation;
use datasubway::model::query_context::QueryContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut dm = DataModel::new()?;

    // ── 1. Register tables ──────────────────────────────────────────────
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

    // You can also register from files:
    // dm.register_parquet("orders", "data/orders.parquet")?;
    // dm.register_csv("orders", "data/orders.csv")?;

    // ── 2. Define joins ─────────────────────────────────────────────────
    // direction: "right2left" means orders can reach customers (not reverse).
    //            "both" allows traversal in either direction.
    dm.set_joins(&[Join {
        left: "orders".into(),
        right: "customers".into(),
        left_on: vec!["customer_id".into()],
        right_on: vec!["id".into()],
        how: "inner".into(),
        direction: "right2left".into(),
    }])?;

    // ── 3. Define pre-aggregations (optional) ───────────────────────────
    // Pre-aggregations store grouped results in local parquet files.
    // The engine automatically picks the best one when it covers the query.
    dm.set_pre_aggregations(vec![
        PreAggregation::new(
            "regional_revenue".into(),
            vec!["orders.region".into()],
            HashMap::from([
                ("orders.amount".into(), vec!["sum".into(), "mean".into()]),
            ]),
            "_pre_aggregations/regional_revenue.parquet".into(),
        )?,
    ]);

    // ── 4. Register measures ────────────────────────────────────────────
    // Measures are closures that build DataFusion DataFrames.
    // allow("*", ...) includes only the columns present in the query
    // context, so the same measure works with any grouping.
    dm.register_measure(
        "revenue",
        Arc::new(|qc, dm| {
            let group_exprs = column_context::allow(
                &["*".into()], column_context::ColumnInput::Columns(&qc.groups)
            )?.into_exprs();
            dm.table("orders")?
                .aggregate(group_exprs, vec![sum(col("amount")).alias("revenue")])
        }),
    )?;

    // ── 5. Query: call measures with a QueryContext ─────────────────────

    // Simple: total revenue, no grouping
    let qc = QueryContext::new(
        vec!["revenue".into()],       // measures
        None,                          // filters
        None,                          // groups
        None,                          // havings
        None,                          // sorts
        None,                          // limit (default 10000)
        None,                          // offset
        None,                          // use_pre_agg (default true)
    )?;
    let results = dm.collect(&qc)?;

    // Grouped by region, sorted descending
    let qc = QueryContext::new(
        vec!["revenue".into()],
        None,
        Some(vec!["orders.region".into()]),
        None,
        Some(vec![("revenue".into(), "desc".into())]),
        None,
        None,
        None,
    )?;
    let results = dm.collect(&qc)?;

    // Cross-table grouping — the join is resolved automatically
    let qc = QueryContext::new(
        vec!["revenue".into()],
        None,
        Some(vec!["customers.name".into()]),
        None,
        None,
        None,
        None,
        None,
    )?;
    let results = dm.collect(&qc)?;

    // With filters and havings
    let qc = QueryContext::new(
        vec!["revenue".into()],
        Some(serde_json::json!({"AND": [["orders.region", "=", "US"]]})),
        Some(vec!["customers.name".into()]),
        Some(serde_json::json!({"AND": [["revenue", ">", 100]]})),
        Some(vec![("revenue".into(), "desc".into())]),
        Some(5),   // limit
        Some(0),   // offset
        None,
    )?;
    let results = dm.collect(&qc)?;

    for batch in &results {
        println!("{:?}", batch);
    }
    Ok(())
}
```

## Building, Testing, and Running

```bash
# Build the library and binary
cargo build

# Run all tests (unit tests are inline in each module)
cargo test

# Run the example binary (src/main.rs)
cargo run

# Type-check without producing artifacts (faster feedback loop)
cargo check

# Build with optimizations
cargo build --release
```

### Running specific tests

```bash
# Run tests in a specific module
cargo test --lib data_model
cargo test --lib model::joins
cargo test --lib model::pre_agg
cargo test --lib model::query_context
cargo test --lib model::column_context
cargo test --lib optimizer

# Run a single test by name
cargo test test_auto_join

# Show stdout from tests (println! output)
cargo test -- --nocapture

# Show debug logs with tests
DATASUBWAY_DEBUG=1 cargo test -- --nocapture
```

## Using as a Local Dependency

To use `datasubway` as a dependency in another Rust project on your machine, add a path dependency to that project's `Cargo.toml`:

```toml
[dependencies]
datasubway = { path = "/Users/oscarobrien/Developer/Repos/datasubway" }
```

Adjust the path to wherever this repo lives on your system. Relative paths also work (e.g. `path = "../datasubway"`).

### Keeping the dependency up to date

With a `path` dependency, Cargo always builds from the current state of the source files on disk. There is no cached/pinned version — any time you `cargo build` in the consuming project, it picks up whatever is in the `datasubway` directory. So the workflow is:

1. Make changes in this repo (`datasubway`)
2. Run `cargo build` (or `cargo check`) in the consuming project — it automatically recompiles with the latest changes

No extra steps are needed. There is no equivalent of `cargo update` for path dependencies since they always point at live source.

### If you later publish or push to a remote

If you want to switch from a local path to a git dependency (e.g. to share across machines), replace the path entry:

```toml
[dependencies]
datasubway = { git = "https://github.com/your-user/datasubway.git", branch = "datafusion_rust" }
```

With a `git` dependency, Cargo pins to a specific commit in `Cargo.lock`. To pull the latest, run `cargo update -p datasubway` in the consuming project.

## TO DO

- More comprehensive tests, so tests are not all passing when usage is not working

- Look into datafusion-flight-sql-server for serving data or at least how to use flight to avoid serialization

- Logging (there is a datafusion OTEL rust crate called datafusion-tracing)



Two approaches:
- Back to the Dataframe wrapper approach in rust (works for polars and datafusion, maybe can do both?)
- Different logical plan optimizer approach than the current implementation
  - Have a logical plan optimizer that run first to replace table scans with pre aggregations if necessary
  - Walk the logical plan to find necessary columns




- Parameter to only look at pre aggregations within a certain time frame

- Benchmark system
- HTMX UI/TUI for displaying pre agg metadata and rewriting files
  - Could also display logs
  - Probably makes sense to expose methods from data_model.py that will print this info (leave the UI to users)
- Roll based access control?
  - Might make more sense for this to be a user implemented feature because it involves auth
- Add optional AI dependency to get chat bot functionality working
