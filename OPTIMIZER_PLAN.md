There are some fundamental errors with the current approach in the optimzer directory.

1. Datafusion's relation on all pre aggregation columns is being set to the base table
  - This causes dimension columns like dimension_name.column_name to be converted to fact_name.column_name, so downstream logic fails
  - A possible solution: When reading a pre agg parquet file, update the relation of all columns to be the table it is coming from; maybe encode this in the physical parquet file when writing the file by saving columns as table_name.column_name, then change them to the right relation and column_name when reading them. When When reading a pre agg parquet file, use alias_qualified(Some("relation_name"), "column_name") to fix the relations for downstream operations

2. covers() is overcomplicated:
  - It should take parameters for aggregate columns and non aggregate columns
  - It doesn't consider methods like select, with_column, etc...
  - Just walk the logical plan and extract all Columns which will either be in an aggregate function or not

3. A new optimzer approach:
  - table() calls should accept a table name in the data model. They should return a dataframe with no rows and all columns from all tables in the data model. This avoids issues with eager schema checks by datafusion
  - This means that the eliminate joins optimizer rule is no longer necessary
  - The new pre agg rule logical plan optimizer should be the first optimizer that runs (you can specify the order of optimizers when defining them). It should start from all table scans that match the schema that table() returned, and walk the plan collecting all columns (and aggregation functions used) used up until it reaches a join (including the join columns). Once it does that walk, it will know which columns are necessary to satisfy a pre aggregation. If no pre aggregation exists, or a user does not want pre aggregations, then use the base table specified in the table() call and join the other tables according to the join object
