import inspect
import textwrap
# import libcst as cst
# from libcst.display import dump
import polars as pl
from pathlib import Path

from column_context import Allow# , Exclude
# from cst.visitors.get_column_context import GetColumnContext
from cst.transformers.replace_context_with_table_columns import resolve_table_columns
from data_model import DataModel
from decorators import measure

def main():
    
    df_sales = pl.LazyFrame({
        'store_id' : [1, 1, 2, 3, 2, 1, 2, 3, 3, 2],
        'product_id' : [7, 8, 9, 8, 7, 8, 9, 9, 8, 7],
        'revenue' : [45, 34, 76, 23, 87, 34, 65, 23, 56, 78]
    })

    df_stores = pl.LazyFrame({
        'store_id' : [1, 2, 3],
        'store_name' : ['Store A', 'Store B', 'Store C'],
        'geography_id' : [10, 11, 12]
    })

    df_products = pl.LazyFrame({
        'product_id' : [7, 8, 9],
        'product_name' : ['Product X', 'Product Y', 'Product Z']
    })

    df_geography = pl.LazyFrame({
        'geography_id' : [10, 11, 12],
        'geography_name' : ['East', 'West', 'North']
    })

    tables = {
        'sales' : df_sales,
        'stores' : df_stores,
        'products' : df_products,
        'geography' : df_geography
    }

    joins = [
        {
            'left' : 'sales', 'right' : 'products',
            'left_on' : ['product_id'], 'right_on' : ['product_id'],
            'how' : 'inner', 'direction' : 'both'
        },
        {
            'left' : 'sales', 'right' : 'stores',
            'left_on' : ['store_id'], 'right_on' : ['store_id'],
            'how' : 'left', 'direction' : 'right2left'
        },
        {
            'left' : 'stores', 'right' : 'geography',
            'left_on' : ['geography_id'], 'right_on' : ['geography_id'],
            'how' : 'inner', 'direction' : 'both'
        }
    ]

    pre_aggs = {
        'sales_by_product_geo' : {
            'group_by' : ['product.product_name', 'geography.geography_name'],
            'aggregations' : {
                'sales.revenue' : ['sum', 'max', 'mean']
            }
        }
    }

    data_model = DataModel(tables=tables, joins=joins, pre_aggregations=pre_aggs, pre_agg_directory=Path('pre_aggs_test/'))

    @measure(data_model)
    def total_revenue(qc):
        return (
            data_model.table('sales')
            .filter(Allow('*', context=qc['filter']))
            .group_by(Allow('*', context=qc['group']))
            .agg(
                pl.col('sales.revenue').sum().alias('total_revenue')
            )
        )

    @measure(data_model)
    def average_revenue(qc):
        return (
            data_model.table('sales')
            .filter(Allow('*', context=qc['filter']))
            .group_by(Allow('*', context=qc['group']))
            .agg(
                pl.col('sales.revenue').mean().alias('average_revenue')
            )
        )

    qc1 = {
        'measure' : ['total_revenue'],
        'group': ['products.product_name']
    }

    print(data_model.query(qc1, output_type='query'))
    print(data_model.query(qc1, output_type='data'))



















    # # ===================================================================
    # # Demonstration: Measure Source Code Transformations
    # # ===================================================================

    # print("\n" + "="*70)
    # print("=== Demonstration: Measure Source Code Transformations ===")
    # print("="*70 + "\n")

    # # Define example query contexts
    # query_contexts = [
    #     {
    #         'name': 'Simple grouping by product',
    #         'context': {
    #             'group': ['products.product_name'],
    #             'filter': None
    #         }
    #     },
    #     {
    #         'name': 'Simple grouping by geography',
    #         'context': {
    #             'group': ['geography.geography_name'],
    #             'filter': None
    #         }
    #     },
    #     {
    #         'name': 'Multi-dimensional grouping (product + geography)',
    #         'context': {
    #             'group': ['products.product_name', 'geography.geography_name'],
    #             'filter': None
    #         }
    #     },
    #     {
    #         'name': 'Filter only (revenue > 50)',
    #         'context': {
    #             'group': [],
    #             'filter': ('sales.revenue', '>', 50)
    #         }
    #     },
    #     {
    #         'name': 'Complex filter with OR logic',
    #         'context': {
    #             'group': [],
    #             'filter': {
    #                 'OR': [
    #                     ('sales.revenue', '>', 70),
    #                     ('sales.store_id', '=', 1)
    #                 ]
    #             }
    #         }
    #     },
    #     {
    #         'name': 'Combined: group by product + filter high revenue',
    #         'context': {
    #             'group': ['products.product_name'],
    #             'filter': {
    #                 'AND': [
    #                     ('sales.revenue', '>', 50),
    #                     ('sales.revenue', '<=', 80)
    #                 ]
    #             }
    #         }
    #     },
    #     {
    #         'name': 'Empty context (no grouping or filtering)',
    #         'context': {
    #             'group': [],
    #             'filter': None
    #         }
    #     }
    # ]

    # # Extract source code for both measures
    # total_revenue_source = textwrap.dedent(inspect.getsource(total_revenue))
    # average_revenue_source = textwrap.dedent(inspect.getsource(average_revenue))

    # measures = [
    #     ('total_revenue', total_revenue_source),
    #     ('average_revenue', average_revenue_source)
    # ]

    # # Transform and display each combination
    # for qc_info in query_contexts:
    #     qc_name = qc_info['name']
    #     qc = qc_info['context']

    #     print(f"\n{'─'*70}")
    #     print(f"Query Context: {qc_name}")
    #     print(f"{'─'*70}")
    #     print(f"Group by: {qc['group'] if qc['group'] else 'None'}")
    #     print(f"Filter:   {qc['filter'] if qc['filter'] else 'None'}")
    #     print()

    #     for measure_name, measure_source in measures:
    #         print(f"\n📊 Measure: {measure_name}")
    #         print("─" * 70)

    #         # Transform the source code
    #         transformed = resolve_table_columns(
    #             source_code=measure_source,
    #             function_name=measure_name,
    #             runtime_context={'qc': qc},
    #             output_type='polar_col'
    #         )

    #         print(transformed)
    #         print()

    # print("\n" + "="*70)
    # print("=== End of Demonstration ===")
    # print("="*70 + "\n")

    # # func_node = cst.parse_statement(dedent_measure_source)

    # # print(func_node)
    # # print(dump(func_node))

    

    # # # Create namespace with required variables
    # # exec_namespace = {'df': df, 'pl': pl}

    # # # Execute the transformed code (defines the function in exec_namespace)
    # # exec(transformed, exec_namespace)

    # # # Call the function
    # # result = exec_namespace['revenue_by_item']()

    # # print("Result:")
    # # print(result)


if __name__ == "__main__":
    main()
