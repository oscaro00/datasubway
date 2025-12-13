import inspect
import textwrap
import libcst as cst
# from libcst.display import dump
import polars as pl

from column_context import Allow, Exclude
from cst.visitors.get_column_context import GetColumnContext
from cst.transformers.replace_context_with_columns import transform_function

def main():
    df = pl.DataFrame({
        'item_id' : [1, 2, 1, 3, 2, 3, 2, 1],
        'store_id' : [3, 2, 2, 1, 3, 2, 3, 1],
        'revenue' : [3, 6, 2, 7, 2, 7, 2, 5]
    })

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

    def revenue_by_item():
        return (
            df
            .group_by(Allow('*', include='df.item_id', context=[query_context['groupings']]))
            .agg(
                pl.col('revenue').sum().alias('total_revenue')
            )
            .sort(Exclude('*', include=['df.item_id'], context=query_context['orderings']))
        )
    
    measure_source = inspect.getsource(revenue_by_item)
    dedent_measure_source = textwrap.dedent(measure_source)

    func_node = cst.parse_statement(dedent_measure_source)


    # print(func_node)
    # print(dump(func_node))

    # Test the transformer
    transformed = transform_function(
        source_code=dedent_measure_source,
        function_name='revenue_by_item',
        runtime_context={'query_context': query_context},
        use_polars_col=False
    )
    print("Transformed code:")
    print(transformed)

    # Create namespace with required variables
    exec_namespace = {'df': df, 'pl': pl}

    # Execute the transformed code (defines the function in exec_namespace)
    exec(transformed, exec_namespace)

    # Call the function
    result = exec_namespace['revenue_by_item']()

    print("Result:")
    print(result)


if __name__ == "__main__":
    main()
