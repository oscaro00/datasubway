from typing import Self, Dict, List
from pathlib import Path
import polars as pl


# TODO create a measure decorator to add measure to a specific data model
# The measure decorator should eventually be where the measure validation occurs (must end with group_by() and agg())


class DataModel:

    def __init__(self: Self, tables: Dict[pl.LazyFrame], joins: Dict, pre_aggregations: Dict, pre_agg_directory: Path) -> Self:
        """
        Expected join format:
        [
            {
                'left':'table1', 'right':'table2', 
                'left_on':['col1', 'col3'], 'right_on':['col1', 'col2'], 
                'how':'inner', 'direction':'right2left' # direction can also be 'both'
            }, 
            {} # more join edges
        ]

        Expected pre_aggregations format:
        {
            'pre_agg1_name' : {
                'group_by' : ['tbl1.col10', 'col11'],
                'aggregations' : {
                    'tbl1.col1' : 'sum',
                    'tbl1.col2' : 'max',
                    'tbl2.col3' : 'min'
                }
            }
        }

        Expected data in pre_agg_metadata:
        - name, file path, last modified timestamp, group by columns, aggregated columns with type of aggregation, row count (sort key)

        The pre_agg_metadata list should be sorted in ascending order of row count
        """
        
        self.tables = tables
        self.joins = joins,
        self.pre_aggregations = pre_aggregations
        self.pre_agg_directory = pre_agg_directory or Path('_pre_aggregations/')

        self.measures = {}
        self.join_lookup = {}
        self.pre_agg_metadata = []

        self.validate_tables()
        self.build_join_lookup()


    def validate_tables(self: Self) -> None:
        for key, val in self.tables.items():
            if not isinstance(key, str) or key.find('.') != -1:
                raise TypeError('Table keys must be strings and cannot contain periods (.)')
            
            if not isinstance(val, pl.LazyFrame):
                raise TypeError('Table values must be lazy frame objects')
    
    def build_join_lookup(self: Self) -> None:
        """
        This method should parse the self.joins object and build a dictionary of dictionaries to find the joins necessary to join
        a table with another (or error that it's not possible).

        There should also be a check to make sure there are no loops or multiple paths from table A to table B.

        The inner most dictionary's values should be polars join objects and possibly a list of tables involved in the join chain.
        """
        pass

    def write_pre_aggregation(self: Self) -> None:
        """
        This method should allow users to write out one, several, or all pre aggregations to the given pre aggregation directory.

        When writing out tables, the pre_agg_metadata should be updated.
        """
        pass

    def table(self: Self, original_table: str, needed_columns: List[str], allow_pre_aggs: bool = True):
        """
        This method should be inserted into measures using libcst in place of LazyFrames at the beginning of polars method chains.
        The method will return an object cst based on the cases below

        If allow_pre_aggs is true, then search for the smallest pre_aggregation that has the necessary columns.
        This process will involve using libcst to update aggregations to work with pre aggregated columns.

        If a pre aggregation does not exist or allow_pre_aggs is false, then return a lazy frame.
        This lazy frame may potentially need other tables to be joined on
        """
        pass