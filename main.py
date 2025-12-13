import inspect
import textwrap
import libcst as cst
from libcst.display import dump
import polars as pl

from column_context import Allow, Exclude
from cst.visitors.get_column_context import GetColumnContext

def main():
    df = pl.DataFrame({
        'item_id' : [1, 2, 1, 3, 2, 3, 2, 1],
        'store_id' : [3, 2, 2, 1, 3, 2, 3, 1],
        'revenue' : [3, 6, 2, 7, 2, 7, 2, 5]
    })

    def revenue_by_item():
        return (
            df
            .group_by(Allow('*', use=['item_id']))
            .agg(
                pl.col('revenue').sum().alias('total_revenue')
            )
            .order_by(Exclude('*', use=['item_id']))
        )
    
    measure_source = inspect.getsource(revenue_by_item)
    dedent_measure_source = textwrap.dedent(measure_source)

    func_node = cst.parse_statement(dedent_measure_source)

    # print(func_node)
    # print(dump(func_node))

    # Create visitor to get the column context and traverse the tree
    column_context_visitor = GetColumnContext(function_name='revenue_by_item')
    func_node.visit(column_context_visitor)

    print(f"Function: {column_context_visitor.function_name}")
    print(f"\nAllow() calls found: {len(column_context_visitor.allow_calls)}")
    for i, call in enumerate(column_context_visitor.allow_calls, 1):
        print(f"  {i}. Positional args: {call['positional']}")
        print(f"     use= keyword args: {call['use']}")

    print(f"\nExclude() calls found: {len(column_context_visitor.exclude_calls)}")
    for i, call in enumerate(column_context_visitor.exclude_calls, 1):
        print(f"  {i}. Positional args: {call['positional']}")
        print(f"     use= keyword args: {call['use']}")


if __name__ == "__main__":
    main()
