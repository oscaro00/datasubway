from typing import Self, Set, List, Dict, Union
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


def flatten_list(lst: List) -> List:
    output_list = []
    for item in lst:
        if isinstance(item, list):
            output_list.extend(item)
        else:
            output_list.append(item)
    return output_list


class Allow:
    raw_columns = []
    table_columns = set()
    include_columns = []
    context_columns = []

    def __init__(self: Self, *columns: str, **kwargs: Dict) -> Self:
        self.raw_columns = list(columns)
        self.table_columns = extract_table_columns(self.raw_columns)
        
        include_raw = []
        context_raw = []
        for key, val in kwargs.items():
            if key == 'include' and isinstance(val, list):
                include_raw = flatten_list(val)
            if key == 'include' and isinstance(val, str):
                include_raw = [val]
            if key == 'context' and isinstance(val, list):
                context_raw = flatten_list(val)
        self.include_columns = list(extract_table_columns(include_raw))
        self.context_columns = list(extract_table_columns(context_raw))

    def get_columns(self: Self) -> Set:
        return self.table_columns
    
    def get_include(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.include_columns]

    def get_context(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.context_columns]
    
    def get_relevant_columns(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        relevant_columns = [col for _, col in self.include_columns]

        for tbl_col in self.context_columns:
            tbl, col = tbl_col
            
            if {('*', ''), (tbl, '*'), tbl_col}.intersection(self.table_columns):
                relevant_columns.append(col)
        
        return [pl.col(column) if polars_col else column for column in relevant_columns]


class Exclude:
    raw_columns = []
    table_columns = set()
    include_columns = []

    def __init__(self: Self, *columns: str, **kwargs: Dict) -> Self:
        self.raw_columns = list(columns)
        self.table_columns = extract_table_columns(self.raw_columns)
        
        include_raw = []
        context_raw = []
        for key, val in kwargs.items():
            if key == 'include' and isinstance(val, list):
                include_raw = flatten_list(val)
            if key == 'include' and isinstance(val, str):
                include_raw = [val]
            if key == 'context' and isinstance(val, list):
                context_raw = flatten_list(val)
        self.include_columns = list(extract_table_columns(include_raw))
        self.context_columns = list(extract_table_columns(context_raw))

    def get_columns(self: Self) -> Set:
        return self.table_columns
    
    def get_include(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.include_columns]

    def get_context(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.context_columns]

    def get_relevant_columns(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        relevant_columns = [col for _, col in self.include_columns]

        for tbl_col in self.context_columns:
            tbl, col = tbl_col
            
            if not {('*', ''), (tbl, '*'), (tbl, col)}.intersection(self.table_columns):
                relevant_columns.append(col)
        
        return [pl.col(column) if polars_col else column for column in relevant_columns]




if __name__ == '__main__':
    query_context = {
        'groupings' : ['df.store_id'],
        'orderings' : ['df.store_id']
    }
    
    allow_test = Allow('*', include='df.item_id', context=[query_context['groupings']])
    print(allow_test.get_columns())
    print(allow_test.get_include())
    print(allow_test.get_context())
    print(allow_test.get_relevant_columns())

    exclude_test = Exclude('*', include=['df.item_id'], context=query_context['orderings'])
    print(exclude_test.get_columns())
    print(allow_test.get_include())
    print(allow_test.get_context())
    print(exclude_test.get_relevant_columns())