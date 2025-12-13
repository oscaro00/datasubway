# Project Plan

The purpose of this library is to define a data model using python focusing on the library polars. The data model will act as a central repository of domain specific calculations that can be written very flexibly. Calling these calculations will be as simple as naming the calculations along with a json-like object for the query parameters you want as context. I imagine this library would be implemented in tandem with an api library to make data access easy. This data model is primarily geared towards OLAP use cases (i.e. read heavy workloads of large chunks of data). Data insertion and updates on tables will not be scope of this work.

## Core requirements for the data model:

- Measures are atomic 
  - Measures will not be nested because of the maintenance difficulties this causes
- Zero cost pre-aggregation abstractions make performance a priority
  - Regardless of data source, it will be possible to stored aggregated versions of tables in local parquet files, which can be queried indirectly for quicker results
- Measures are written in a polars-like method chaining syntax so arbitrarily complex measures are possible
  - Polars syntax is a nice balance of clarity, verbosity, and flexibility
  - In order to allow query parameters to be used or excluded in specific polars methods in a calculation, a system of allow() and exclude() at column positions in polars methods will manage valid columns

## Key features:

- Using the polars lazy frame syntax also gives polars optimizations to query executions by default
- Pre-aggregations are declarative (i.e. users define which columns to group by before calculations measures) and optimal pre-aggregations are selected by the data model engine without user input
- Pre-aggregations are effective with the assumption that most queries do not need the fully granularity of the table, so a local parquet file with an aggregated version of the data will be faster (no network calls and less data to crunch)
- Pre-aggregations can span several tables (table joins are expensive, so pre-computing them makes sense)

## Longer term vision:

- I believe this combinations of structured yet flexible calculations is a great use case for small, local AI models to take natural language and execute reliable calculations. Passing measure descriptions along with measures will support domain specific calculations and terminology and a small model should be able to create a json-like query context. This will hopefully avoid the problem where models are great a writing simple queries, but are inconsistent on harder queries with domain specific knowledge especially across multiple users.



## Learnings from prior implementations

- The metroframe implementations of group_by() and agg() need to be more intelligent
    - skipping group by if there are no group by columns
    - agg turns into a select if there are no group by columns
    - AggExpr need to be smarter to convert operations like mean() to sum of values / sum of count if using a pre aggregation
    - I think the metro frame wrapper should be the first piece implemented in rust
- Measure parsing
    - No compile time parsing because python can't do anything at compile time
    - Measure granularity probably needs to be stored at a step level because multiple table() calls should still only pull the relevant tables?
        - Maybe granularity steps are determined by the number of table() calls? (think more on this)
        - Adding granularity per step might be too much complexity... Most measures probably don't need several steps
    - Syn crate is useful for validating measure rules
    - Planning the measure with a blank query context is useful for the output columns
- Pre aggregations need a smarter way of handling filters
    - Either the filters of the pre agg need to match the measure or the relevant columns exist to do filtering
- Pre aggregations need to store more information for operations like means -> sum of values and count to make subsequent calculations correct
