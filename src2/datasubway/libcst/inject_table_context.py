"""
When a measure is registered with the @measure decorator,
add parameters (e.g. any columns referenced) for non_agg_context and agg_context in table() calls.
Additionally, within the allow() and exclude() calls that are added as parameters to table() calls,
need to add the parameter include_tables=True, so the joins can know which tables are necessary.

allow() and exclude() calls from filter() and group_by() are always parameters for non_agg_context.
Within a select(), if a column is not being aggregated, then it is a non_agg_context parameter.
Columns within a select() or agg() that have an aggregation applied go in agg_context along with
the specific aggregation function.
"""

"""
Using libcst to do this complex task feels really brittle and hard to understand/maintain.

What if a radically different approach was taken?

When a DataModel object is created, all table schemas are collected with collect_schema().
Each data model object could have a field with an empty lazyframe where the columns are all of the columns
from the tables with the format table1.col1, table1.col2, ..., table2.col1, etc...

When a measure is run, a first pass is done using the wide, empty table in order to access the .explain()
of a measure, which contains needed columns wrapped in col(). This info could be passed back to the table()
calls to select the optimal pre aggregations and automatically adding necessary joins.

Challenges:
- how to handle measures with several table() calls? (some sort of queue?)
- how to handle table().join(table().filter()).filter()...
- how emit necessary columns back to table calls?
"""


import libcst
import polars as pl

from datasubway.column_context import allow, exclude
from datasubway.data_model import DataModel

# TODO: put libcst code here

if __name__ == "__main__":
    lf = pl.LazyFrame(
        {"col1": ["a", "b", "c", "d"], "col2": [1, 2, 3, 4], "col3": [5, 4, 3, 2]}
    )

    tables = {"table1": lf}

    dm = DataModel(tables, None, None, None)

    # def example_measure1(qc):
    #     """The dm.table() call should be transformed into
    #     dm.table("table1", )"""

    #     return (
    #         dm.table("table1")
    #         .group_by(exclude(pattern="*", include="table1.col1", context=qc["groups"]))
    #         .agg(pl.col("col2").sum())
    #     )

    test1 = lf.filter(pl.col("col2") >= 2).group_by("col1").agg(pl.col("col3").sum())

    print(test1.explain())
