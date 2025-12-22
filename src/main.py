import inspect
import textwrap
import libcst as cst
# from libcst.display import dump
import polars as pl

from column_context import Allow, Exclude
# from cst.visitors.get_column_context import GetColumnContext
from cst.transformers.replace_context_with_table_columns import resolve_table_columns

def main():
    df = pl.DataFrame({
        'item_id' : [1, 2, 1, 3, 2, 3, 2, 1],
        'store_id' : [3, 2, 2, 1, 3, 2, 3, 1],
        'revenue' : [3, 6, 2, 7, 2, 7, 2, 5]
    })

    query_context = {
        'group' : ['df.store_id'],
        'sort' : [('df.store_id', 'desc')]
    }

    def revenue_by_item():
        return (
            df
            .group_by(Allow('*', include='df.item_id', context=[query_context['group']]))
            .agg(
                pl.col('df.revenue').sum().alias('total_revenue')
            )
            .sort(Exclude('*', include=['df.item_id'], context=query_context['sort']), descending=[False])
        )
    
    measure_source = inspect.getsource(revenue_by_item)
    dedent_measure_source = textwrap.dedent(measure_source)

    # Test the transformer
    transformed = resolve_table_columns(
        source_code=dedent_measure_source,
        function_name='revenue_by_item',
        runtime_context={'query_context': query_context},
        output_type='polar_col'
    )
    print("Transformed code:")
    print(transformed)

    query_context2 = {
        'filter' : {
            'AND': [
                ('df.item_id', '=', 3),
                ('df.store_id', 'IN', [1, 2, 3])
            ]
        },
        'group' : ['df.store_id'],
        'sort' : [('df.store_id', 'desc')]
    }
    
    def revenue_from_expensive_items():
        return (
            df
            .filter(Allow('*', include=(pl.col('df.revenue') >= 5), context=query_context2['filter']))
            .group_by(Allow('*', context=query_context2['group']))
            .agg(
                pl.col('df.revenue').sum().alias('total_revenue')
            )
            .sort(Allow('*', context=query_context2['sort']))
        )
    
    measure_source = inspect.getsource(revenue_from_expensive_items)
    dedent_measure_source = textwrap.dedent(measure_source)

    # Test the transformer
    transformed = resolve_table_columns(
        source_code=dedent_measure_source,
        function_name='revenue_from_expensive_items',
        runtime_context={'query_context2': query_context2},
        output_type='polar_col'
    )
    print("Transformed code:")
    print(transformed)


    query_context3 = {
        'filter' : {
            'AND': [
                ('df.item_id', '=', 3),
                ('df.store_id', 'IN', [1, 2, 3])
            ]
        },
        'group' : ['df.store_id']
    }

    def revenue_no_filter():
        return {
            df
            .filter(Exclude('*', context=query_context3['filter']))
            .group_by(Allow('*', context=query_context3['group']))
            .agg(
                pl.col('df.revenue').sum().alias('total_revenue')
            )
        }
    
    measure_source = inspect.getsource(revenue_no_filter)
    dedent_measure_source = textwrap.dedent(measure_source)

    # Test the transformer
    transformed = resolve_table_columns(
        source_code=dedent_measure_source,
        function_name='revenue_no_filter',
        runtime_context={'query_context3': query_context3},
        output_type='polar_col'
    )
    print("Transformed code:")
    print(transformed)

    # func_node = cst.parse_statement(dedent_measure_source)

    # print(func_node)
    # print(dump(func_node))

    

    # # Create namespace with required variables
    # exec_namespace = {'df': df, 'pl': pl}

    # # Execute the transformed code (defines the function in exec_namespace)
    # exec(transformed, exec_namespace)

    # # Call the function
    # result = exec_namespace['revenue_by_item']()

    # print("Result:")
    # print(result)


if __name__ == "__main__":
    main()
