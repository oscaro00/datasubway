import os
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Self, Union

import polars as pl

from src2.datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper


def __init__(
    self,
    tables: Dict[str, pl.LazyFrame],
    joins: List[Dict[str, Any]],
    pre_aggregations: Dict[str, Any],
    pre_agg_directory: Optional[Path],
    logging_directory: Optional[Path] = None,
) -> None:
    """
    Expected join format:
    [
        {
            'left':'table1', 'right':'table2',
            'left_on':['col1', 'col3'], 'right_on':['col1', 'col2'],
            'how':'inner', 'direction':'right2left' # direction can also be 'both'
            # left joins only make sense if direction is right2left
        },
        {} # more join edges
    ]

    Expected pre_aggregations format:
    {
        'pre_agg1_name' : {
            'group_by' : ['tbl1.col10', 'tbl2.col11'],
            'aggregations' : {
                'tbl1.col1' : 'sum',           # Single function
                'tbl1.col2' : ['max', 'min'],  # Multiple functions
                'tbl2.col3' : ['mean']
            }
        }
    }

    Note: Aggregation values can be either:
    - A single function string (e.g., 'sum', 'max', 'mean')
    - A list of function strings (e.g., ['sum', 'max', 'mean'])
    Both formats are supported and will be normalized internally.

    Expected data in pre_agg_metadata:
    - name, file path, last modified timestamp, group by columns, aggregated columns with type of aggregation, row count (sort key)

    The pre_agg_metadata list should be sorted in ascending order of row count
    """

    self.tables = tables
    self.joins = joins
    self.pre_aggregations = pre_aggregations
    self.pre_agg_directory = pre_agg_directory or Path("_pre_aggregations/")

    self.table_schemas = {
        tbl_name: lf.collect_schema().names() for tbl_name, lf in self.tables.items()
    }

    self.measures = {}

    def table(
        self,
        table_name: str,
        non_agg_context: list[str],
        agg_context: dict[str, str],
        *,
        allow_pre_aggs: bool = True,
        query_context: dict,
    ) -> LazyFrameWrapper:
        # Validate inputs
        if table_name not in self.tables:
            raise KeyError(
                f"Table '{table_name}' not found. Available: {list(self.tables.keys())}"
            )

        # TODO: Resolve allow() and exclude() to know needed columns

        # TODO: If allow_pre_aggs, look for a potential pre aggregation
        # Add a lazyframewrapper parameter to say if a pre agg was returned or not

        # TODO: Otherwise, fallback on building from the source tables with joins

        # this return is just a temporary solution to get something working
        # there is more complex logic that is described above
        return LazyFrameWrapper(self.tables[table_name], from_pre_agg=False)
