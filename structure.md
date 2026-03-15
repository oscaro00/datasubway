The goal of this project is to create a data model interface to return data queries as fast as possible. The main mechanism for speeding up queries is saving pre aggregated versions of tables ahead of time, then reading from those pre aggregated parquet files whenever possible.

The current project builds a frontend on polars because polars is fast and has a nice user interface in python. However, there have been lots of challenges because most of functionality of this project is not a goal use case of polars. Polars does not really intend for people to rewrite queries during lazyframe optimization, so I get around that by creating wrappers for polars methods. This process is not super ergonomic and potentially has pitfalls that I haven't encountered yet.

As a result, I want to utilize a different dataframe api that I came across recently... datafusion. Datafusion is intended to have query planning optimizations and has a much more accessible api to extract information from query plans. Thus, I think it is a natural pivot for this project. Datafusion is still really fast (it can even send logical plans to duckdb if desired) and has a similar syntax to polars although maybe not quite as polished.

Part of this transition is moving from a purely python implemention to using datafusion's rust backend and sticking a python frontend on it with pyo3. Hence, rust decisions must consider that they must work with the python frontend.

Implementation components:

Data Model struct:
  - tables / data sources
  - joins / relationships between tables
    - A process will need to exist to validate there is only one path between each table (excluding bidirectional joins between tables)
    - Cycles in the join network of length 3+ are also not allowed
    - If a table needs to be joined in a query, then this network is used to find how tables join
  - information about available pre aggregations
    - pre aggregations should always be selected to minimize rows needed to parse
    - a pre aggregation "covers" a query if the queries columns and aggregations all are a subset of columns in the pre aggregation table
    - A pre aggregation registry will need to exist to track all pre aggregation files and their metadata
  - measures or allowed calculations / metrics
    - measures should allow arbitrarily complex datafusion dataframe api logic
    - measures need to end with a step that aggregates data, so that measures can be joined on group_by fields
  - There will be other fields like tracking the pre aggregation directory, list of measures, etc...

Datafusion optimizer:
- Replace data sources with pre aggregation tables if possible
  - This means joins are removed because those columns will already exist in the pre aggregation
  - There is some complexity here because measures can have intermediate steps and measures can be joined together (need a robust optimizer approach that still gets the correct result)
- Aggregations will need to be rewritten based on available columns in pre aggregation tables and query correctness
  - If a measure calculates the mean of a column, then a pre aggregation will store the sum of the column and the count, so sum of the column / count of the column is the correct answer

Column context:
- Methods/functions for allow() and exclude() will need to exist to control which tables and columns are allowed with a method of the dataframe api
- This creates a sort of chicken and egg problem because the data source can only be chosen based on subsequent dataframe methods, but those come after the table call
  - Fortunately, datafusion allows the logical plan to be parsed in reverse order

Query context:
- When users submits a request to the data model, there should be a structure that is expected.
- Users should request specific measures, filters, groupings, limts, offset, havings (like a SQL query)

At the end of the day, the python api should be fairly simple where users declare their data model and measure methods/functions and submit query context objects to get results. Users should expect to rewrite measure logic using the datafusion python dataframe api. Queries will be executed/collected async which is native to datafusion.

I expect the migration to mostly be a clean rewrite because this repo is not public yet.
