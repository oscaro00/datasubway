from typing import Self, Set, List, Dict
import re
import polars as pl


def extract_table_columns(column_list: List[str]) -> Set:
    table_columns = set()
    
    for column in column_list:
        table_column = re.findall(r'^([\w_*]+)\.?([\w_*]+)?$', column)

        if table_column[0] == ('*', ''):
            table_columns = {table_column[0]}
            return table_columns

        table_columns.add(table_column[0])
    
    return table_columns


class Allow:
    raw_columns = []
    table_columns = set()
    use_columns = []

    def __init__(self: Self, *columns: str, **kwargs: Dict) -> Self:
        self.raw_columns = list(columns)
        self.table_columns = extract_table_columns(self.raw_columns)
        
        use_raw = []
        for key, val in kwargs.items():
            if key == 'use' and isinstance(val, list):
                use_raw = val
        self.use_columns = list(extract_table_columns(use_raw))

    def get_columns(self: Self) -> Set:
        return self.table_columns
    
    def get_use(self: Self, polars_col: bool = False) -> List[str]:
        return [pl.col(column) if polars_col else column for column in self.use_columns]


class Exclude:
    raw_columns = []
    table_columns = set()
    use_columns = []

    def __init__(self: Self, *columns: str, **kwargs: Dict) -> Self:
        self.raw_columns = list(columns)
        self.table_columns = extract_table_columns(self.raw_columns)
        
        use_raw = []
        for key, val in kwargs.items():
            if key == 'use' and isinstance(val, list):
                use_raw = val
        self.use_columns = list(extract_table_columns(use_raw))
    
    def get_columns(self: Self) -> Set:
        return self.table_columns

    def get_use(self: Self, polars_col: bool = False) -> List[str]:
        return [pl.col(column) if polars_col else column for column in self.use_columns]




if __name__ == '__main__':
    allow_test = Allow('*', 'table2.*', 'table.column123', use=['use_table.use_this_column'])

    print(allow_test.get_columns())
    print(allow_test.get_use())