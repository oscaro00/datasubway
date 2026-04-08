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

- Do more validation in QueryContext __init__
- Parameter to only look at pre aggregations within a certain time frame


- Logging (there is a datafusion OTEL rust crate)
- Benchmark system
- HTMX UI/TUI for displaying pre agg metadata and rewriting files
  - Could also display logs
  - Probably makes sense to expose methods from data_model.py that will print this info (leave the UI to users)
- Roll based access control?
  - Might make more sense for this to be a user implemented feature because it involves auth
- Add optional AI dependency to get chat bot functionality working
