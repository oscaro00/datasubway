import inspect
import textwrap
import libcst as cst
from libcst.display import dump
import polars as pl

from column_context import Allow, Exclude

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

    print(func_node)
    print(dump(func_node))


if __name__ == "__main__":
    main()
