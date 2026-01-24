import pytest
import polars as pl
from polars.testing import assert_frame_equal
from pathlib import Path
import tempfile
import shutil

from datasubway import DataModel, measure, Allow, Exclude


class TestComplexMeasures:
    """
    Test suite for complex measures validating:
    1. Pre-aggregation usage correctness
    2. Measures return same results as direct polars queries
    3. Complex aggregation patterns (mean, weighted averages, ratios, etc.)
    """

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for pre-agg files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def complex_sales_data(self):
        """Create more complex sales data for thorough testing."""
        return pl.LazyFrame({
            'transaction_id': list(range(1, 101)),
            'item_id': [i % 10 + 1 for i in range(100)],
            'store_id': [i % 5 + 1 for i in range(100)],
            'date': [f'2024-01-{(i % 30) + 1:02d}' for i in range(100)],
            'revenue': [100.0 + (i * 10.5) for i in range(100)],
            'quantity': [1 + (i % 20) for i in range(100)],
            'cost': [60.0 + (i * 6.3) for i in range(100)],
            'discount': [5.0 + (i * 0.5) for i in range(100)],
        },
        schema={
            'transaction_id' : pl.Int64,
            'item_id' : pl.Int64,
            'store_id' : pl.Int64,
            'date' : pl.Date,
            'revenue' : pl.Float64,
            'quantity' : pl.Int64,
            'cost' : pl.Float64,
            'discount' : pl.Float64
        })

    @pytest.fixture
    def products_data(self):
        """Create products dimension table."""
        return pl.LazyFrame({
            'item_id': list(range(1, 11)),
            'product_name': [f'Product_{i}' for i in range(1, 11)],
            'category': ['Electronics' if i % 3 == 0 else 'Clothing' if i % 3 == 1 else 'Food' for i in range(1, 11)],
            'brand': [f'Brand_{i % 3}' for i in range(1, 11)],
        },
        schema={
            'item_id' : pl.Int64,
            'product_name' : pl.String,
            'category' : pl.String,
            'brand' : pl.String
        })

    @pytest.fixture
    def stores_data(self):
        """Create stores dimension table."""
        return pl.LazyFrame({
            'store_id': list(range(1, 6)),
            'store_name': [f'Store_{i}' for i in range(1, 6)],
            'region': ['North' if i <= 2 else 'South' for i in range(1, 6)],
            'size': ['Large' if i % 2 == 0 else 'Small' for i in range(1, 6)],
        },
        schema={
            'store_id' : pl.Int64,
            'store_name' : pl.String,
            'region' : pl.String,
            'size' : pl.String
        })

    @pytest.fixture
    def datamodel_without_pre_aggs(self, complex_sales_data, products_data, stores_data):
        """DataModel with joins but without pre-aggregations for baseline queries."""
        tables = {
            'sales': complex_sales_data,
            'products': products_data,
            'stores': stores_data
        }
        joins = [
            {
                'left': 'sales',
                'right': 'products',
                'left_on': ['item_id'],
                'right_on': ['item_id'],
                'how': 'inner',
                'direction': 'both'
            },
            {
                'left': 'sales',
                'right': 'stores',
                'left_on': ['store_id'],
                'right_on': ['store_id'],
                'how': 'inner',
                'direction': 'both'
            }
        ]
        return DataModel(
            tables=tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

    @pytest.fixture
    def datamodel_with_pre_aggs(self, complex_sales_data, products_data, stores_data, temp_dir):
        """DataModel with joins and comprehensive pre-aggregations."""
        tables = {
            'sales': complex_sales_data,
            'products': products_data,
            'stores': stores_data
        }
        joins = [
            {
                'left': 'sales',
                'right': 'products',
                'left_on': ['item_id'],
                'right_on': ['item_id'],
                'how': 'inner',
                'direction': 'both'
            },
            {
                'left': 'sales',
                'right': 'stores',
                'left_on': ['store_id'],
                'right_on': ['store_id'],
                'how': 'inner',
                'direction': 'both'
            }
        ]

        # Define comprehensive pre-aggregations
        pre_aggs = {
            'sales_by_item': {
                'group_by': ['sales.item_id'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean', 'count'],
                    'sales.quantity': ['sum', 'mean', 'count'],
                    'sales.cost': ['sum', 'mean', 'count'],
                    'sales.discount': ['sum', 'mean', 'count'],
                }
            },
            'sales_by_store': {
                'group_by': ['sales.store_id'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean', 'count'],
                    'sales.quantity': ['sum', 'mean', 'count'],
                    'sales.cost': ['sum', 'mean', 'count'],
                }
            },
            'sales_by_item_store': {
                'group_by': ['sales.item_id', 'sales.store_id'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean', 'count'],
                    'sales.quantity': ['sum', 'mean', 'count'],
                    'sales.cost': ['sum', 'mean'],
                }
            },
            'sales_by_category': {
                'group_by': ['products.category'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean', 'count'],
                    'sales.quantity': 'sum',
                }
            },
            'sales_by_region': {
                'group_by': ['stores.region'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean'],
                    'sales.quantity': 'sum',
                }
            },
            'sales_by_date_store' : {
                'group_by': ['sales.date', 'sales.store_id'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean'],
                    'sales.quantity': 'sum',
                }
            }
        }

        dm = DataModel(
            tables=tables,
            joins=joins,
            pre_aggregations=pre_aggs,
            pre_agg_directory=temp_dir
        )

        # Write all pre-aggregations
        for pre_agg_name in pre_aggs.keys():
            dm.write_pre_aggregation(pre_agg_name)

        return dm

    # ========================================================================
    # BASELINE DIRECT POLARS QUERIES
    # ========================================================================

    def test_direct_polars_sum_by_item(self, complex_sales_data):
        """Baseline: Direct polars query for sum aggregation."""
        result = (
            complex_sales_data
            .group_by('item_id')
            .agg(pl.col('revenue').sum())
            .collect()
            .sort('item_id')
        )

        assert len(result) == 10  # 10 unique items
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns

    def test_direct_polars_mean_by_store(self, complex_sales_data):
        """Baseline: Direct polars query for mean aggregation."""
        result = (
            complex_sales_data
            .group_by('store_id')
            .agg(pl.col('revenue').mean())
            .collect()
            .sort('store_id')
        )

        assert len(result) == 5  # 5 unique stores
        assert 'revenue' in result.columns

    def test_direct_polars_multiple_aggs(self, complex_sales_data):
        """Baseline: Direct polars query with multiple aggregations."""
        result = (
            complex_sales_data
            .group_by('item_id')
            .agg(
                pl.col('revenue').sum().alias('total_revenue'),
                pl.col('quantity').mean().alias('avg_quantity'),
                pl.col('cost').sum().alias('total_cost')
            )
            .collect()
            .sort('item_id')
        )

        assert 'total_revenue' in result.columns
        assert 'avg_quantity' in result.columns
        assert 'total_cost' in result.columns

    # ========================================================================
    # SIMPLE AGGREGATION TESTS (Sum, Count, Mean)
    # ========================================================================

    def test_sum_measure_without_pre_agg(self, datamodel_without_pre_aggs):
        """Test sum measure without pre-aggregation."""
        dm = datamodel_without_pre_aggs

        @measure(dm)
        def total_revenue(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='data'
        ).sort('item_id')

        # Verify structure
        assert len(result) == 10
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns

    def test_sum_measure_with_pre_agg(self, datamodel_with_pre_aggs):
        """Test sum measure using pre-aggregation."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def total_revenue_pre_agg(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('revenue'))
            )

        result = dm.query(
            query_context={'measure': ['total_revenue_pre_agg'], 'group': ['item_id']},
            output_type='data'
        ).sort('item_id')

        # Verify structure
        assert len(result) == 10
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns

    def test_sum_measures_match(self, datamodel_without_pre_aggs, datamodel_with_pre_aggs):
        """Validate sum measure returns same results with and without pre-agg."""
        dm_no_agg = datamodel_without_pre_aggs
        dm_with_agg = datamodel_with_pre_aggs

        @measure(dm_no_agg)
        def total_revenue_no_agg(qc):
            return (
                dm_no_agg.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        @measure(dm_with_agg)
        def total_revenue_with_agg(qc):
            return (
                dm_with_agg.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result_no_agg = dm_no_agg.query(
            query_context={'measure': ['total_revenue_no_agg'], 'group': ['item_id']},
            output_type='data'
        ).sort('item_id')

        result_with_agg = dm_with_agg.query(
            query_context={'measure': ['total_revenue_with_agg'], 'group': ['item_id']},
            output_type='data'
        ).sort('item_id')

        # Results should have identical values (column names may differ)
        # Compare the revenue values
        assert result_no_agg['item_id'].to_list() == result_with_agg['item_id'].to_list()
        # Get revenue column (may have different names like 'revenue' vs 'revenue-sum')
        revenue_col_no_agg = [col for col in result_no_agg.columns if 'revenue' in col.lower()][0]
        revenue_col_with_agg = [col for col in result_with_agg.columns if 'revenue' in col.lower()][0]
        assert result_no_agg[revenue_col_no_agg].to_list() == result_with_agg[revenue_col_with_agg].to_list()

    def test_mean_measure_without_pre_agg(self, datamodel_without_pre_aggs):
        """Test mean measure without pre-aggregation."""
        dm = datamodel_without_pre_aggs

        @measure(dm)
        def avg_revenue(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').mean())
            )

        result = dm.query(
            query_context={'measure': ['avg_revenue'], 'group': ['store_id']},
            output_type='data'
        ).sort('store_id')

        assert len(result) == 5
        assert 'revenue' in result.columns

    def test_mean_measure_with_pre_agg(self, datamodel_with_pre_aggs):
        """Test mean measure using pre-aggregation (should transform to sum/count)."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def avg_revenue_pre_agg(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'mean'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').mean())
            )

        result = dm.query(
            query_context={'measure': ['avg_revenue_pre_agg'], 'group': ['store_id']},
            output_type='data'
        ).sort('store_id')

        assert len(result) == 5
        # Note: column name may differ after transformation

    # ========================================================================
    # MULTI-DIMENSIONAL GROUPING TESTS
    # ========================================================================

    def test_multi_dimension_grouping(self, datamodel_with_pre_aggs):
        """Test measure with multiple group by dimensions."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def revenue_by_item_and_store(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_by_item_and_store'], 'group': ['item_id', 'store_id']},
            output_type='data'
        )

        assert 'item_id' in result.columns
        assert 'store_id' in result.columns
        assert len(result) >= 10  # Should have multiple combinations

    # ========================================================================
    # MULTIPLE MEASURES JOINED TOGETHER
    # ========================================================================

    def test_multiple_measures_joined(self, datamodel_with_pre_aggs):
        """Test multiple measures queried together and joined."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def total_revenue(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('total_revenue'))
            )

        @measure(dm)
        def total_quantity(qc):
            return (
                dm.table('sales', qc.get('group', []), {'quantity': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('quantity').sum().alias('total_quantity'))
            )

        result = dm.query(
            query_context={'measure': ['total_revenue', 'total_quantity'], 'group': ['item_id']},
            output_type='data'
        ).sort('item_id')

        assert 'total_revenue' in result.columns
        assert 'total_quantity' in result.columns
        assert len(result) == 10

    # ========================================================================
    # TESTS WITH JOINED DIMENSION TABLES
    # ========================================================================

    def test_measure_with_product_dimension(self, datamodel_with_pre_aggs):
        """Test measure grouped by product dimension column."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def revenue_by_category(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_by_category'], 'group': ['category']},
            output_type='data'
        )

        assert 'category' in result.columns
        assert 'revenue' in result.columns
        assert len(result) == 3  # Electronics, Clothing, Food

    def test_measure_with_store_dimension(self, datamodel_with_pre_aggs):
        """Test measure grouped by store dimension column."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def revenue_by_region(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_by_region'], 'group': ['region']},
            output_type='data'
        )

        assert 'region' in result.columns
        assert 'revenue' in result.columns
        assert len(result) == 2  # North, South

    def test_measure_with_mixed_dimensions(self, datamodel_with_pre_aggs):
        """Test measure grouped by columns from multiple tables."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def revenue_by_category_and_region(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_by_category_and_region'], 'group': ['category', 'region']},
            output_type='data'
        )

        assert 'category' in result.columns
        assert 'region' in result.columns
        assert 'revenue' in result.columns

    # ========================================================================
    # PRE-AGGREGATION CONTROL TESTS
    # ========================================================================

    def test_pre_agg_disabled_explicitly(self, datamodel_with_pre_aggs):
        """Test that allow_pre_aggs=False prevents pre-agg usage."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def revenue_no_pre_agg(qc):
            allow_pre_aggs = qc.get('allow_pre_aggs', True)
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'}, allow_pre_aggs=allow_pre_aggs)
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={
                'measure': ['revenue_no_pre_agg'],
                'group': ['item_id'],
                'allow_pre_aggs': False
            },
            output_type='data'
        )

        # Should still work, just using source tables
        assert len(result) == 10

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    def test_measure_with_no_grouping(self, datamodel_with_pre_aggs):
        """Test measure that aggregates all data (no group by)."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def total_revenue_all(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['total_revenue_all']},
            output_type='data'
        )

        assert len(result) == 1  # Single aggregated row
        assert 'revenue' in result.columns
    
    # ========================================================================
    # MULTI-STEP MEASURES
    # ========================================================================

    def test_percent_of_total_measure(self, datamodel_with_pre_aggs, complex_sales_data):
        """Test measure that calculates a percentage of total or share"""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def store_share_of_revenue(qc):
            numerator = (
                dm.table('sales')
                .filter(Allow('*', context=qc.get('filter')))
                .group_by(Allow('*', include=['stores.store_id'], context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('numerator_revenue'))
            )
            
            total_denominator = (
                dm.table('sales')
                .filter(Exclude('stores.*', context=qc.get('filter')))
                .group_by(Exclude('stores.*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('total_revenue'))
            )
            
            return (
                numerator
                .join(
                    total_denominator, 
                    on=qc.get('group') if len(qc.get('group', [])) >= 1 else None, 
                    how='inner' if len(qc.get('group', [])) >= 1 else 'cross'
                )
                .group_by(Allow('*', include=['stores.store_id'], context=qc.get('group', [])))
                .agg(
                    (pl.col('numerator_revenue') / pl.col('total_revenue') * 100).round(1).first().alias('revenue_percentage')
                )
            )
        
        result = dm.query(
            query_context={'measure' : ['store_share_of_revenue'], 'sort' : [('stores.store_id', 'asc')]},
            output_type='data'
        )

        polars_num = (
            complex_sales_data
            .group_by('store_id')
            .agg(pl.col('revenue').sum().alias('numerator_revenue'))
        )

        polars_denom = (
            complex_sales_data
            .select(pl.col('revenue').sum().alias('total_revenue'))
        )

        polars_result = (
            polars_num
            .join(polars_denom, how='cross')
            .group_by('store_id')
            .agg(
                (pl.col('numerator_revenue') / pl.col('total_revenue') * 100).round(1).first().alias('revenue_percentage')
            )
            .sort('store_id', descending=False)
            .collect()
        )

        assert_frame_equal(result, polars_result)

        # dm.show_measure_transformation(query_context={'measure' : ['store_share_of_revenue']}, verbose=True)
    
    def test_3_day_rolling_average_revenue_measure(self, datamodel_with_pre_aggs, complex_sales_data, stores_data):
        """Test measure that calculates average revenue over a 3 day rolling window"""
        dm = datamodel_with_pre_aggs
        
        @measure(dm)
        def rolling_3_day_average_revenue(qc):
            return (
                dm.table('sales')
                .filter(Allow('*', context=qc.get('filter')))
                .sort('sales.date')
                .group_by_dynamic('sales.date', every='1d', period='3d', group_by=Allow('*', context=qc.get('group')))
                .agg(
                    pl.col('revenue').mean().alias('average_3_day_rolling_revenue')
                )
            )

        result = dm.query(
            query_context={
                'measure' : ['rolling_3_day_average_revenue'], 
                'filter' : ('stores.store_name', '!=', 'Store_1'), 
                'group' : ['stores.store_name'], 
                'sort' : [('stores.store_name', 'desc'), ('sales.date', 'asc')]
            },
            output_type='data'
        )

        polars_result = (
            complex_sales_data
            .join(stores_data, on='store_id', how='left')
            .filter(pl.col('store_name') != 'Store_1')
            .sort('date')
            .group_by_dynamic('date', every='1d', period='3d', group_by='store_name')
            .agg(pl.col('revenue').mean().alias('average_3_day_rolling_revenue'))
            .sort(['store_name', 'date'], descending=[True, False])
            .collect()
        )

        assert_frame_equal(result, polars_result)
    
    def test_prior_day_revenue_measure(self, datamodel_with_pre_aggs, complex_sales_data, stores_data):
        """Test measure that calculates prior day revenue"""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def prior_day_revenue(qc):
            return (
                dm.table('sales', qc.get('group', []), {}, allow_pre_aggs=False)
                .filter(Allow('*', context=qc.get('filter')))
                .with_columns(
                    (pl.col('sales.date') + pl.duration(days=1)).alias('date')
                )
                .group_by(Allow('*', include=['sales.date'], context=qc.get('group', [])))
                .agg(
                    pl.col('revenue').sum().alias('prior_day_revenue')
                )
            )

        result = dm.query(
            query_context={
                'measure': ['prior_day_revenue'],
                'group': ['stores.store_name', 'sales.date'],
                'sort': [('sales.date', 'asc'), ('stores.store_name', 'asc')]
            },
            output_type='data'
        )

        print(result)

        polars_result = (
            complex_sales_data
            .join(stores_data, on='store_id', how='left')
            .with_columns(
                (pl.col('date') + pl.duration(days=1)).alias('date')
            )
            .group_by(['date', 'store_name'])
            .agg(pl.col('revenue').sum().alias('prior_day_revenue'))
            .sort(['date', 'store_name'])
            .collect()
        )

        assert_frame_equal(result, polars_result, check_column_order=False)

        # Measure transformation testing example for presentation
        # @measure(dm)
        # def prior_day_revenue2(qc):
        #     return (
        #         dm.table('sales')
        #         .filter(Allow('*', context=qc.get('filter')))
        #         .with_columns(
        #             (pl.col('sales.date') + pl.duration(days=1)).alias('date')
        #         )
        #         .group_by(Allow('*', include=['sales.date'], context=qc.get('group', [])))
        #         .agg(
        #             pl.col('revenue').sum().alias('prior_day_revenue')
        #         )
        #     )

        # result = dm.show_measure_transformation(
        #     query_context={
        #         'measure': ['prior_day_revenue2'],
        #         'group': ['sales.store_id'],
        #         'sort': [('sales.store_id', 'asc'), ('sales.date', 'asc')],
        #         'allow_pre_aggs': True
        #     },
        #     verbose=True
        # )

    def test_week_to_date_revenue_measure(self, datamodel_with_pre_aggs, complex_sales_data, products_data):
        """Test measure that calculates week-to-date revenue (Sunday-based weeks)"""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def wtd_revenue(qc):
            return (
                dm.table('sales', qc.get('group', []), {}, allow_pre_aggs=False)
                .filter(Allow('*', context=qc.get('filter')))
                .with_columns([
                    ((pl.col('date') + pl.duration(days=1)).dt.truncate('1w') - pl.duration(days=1)).alias('week_start')
                ])
                .sort(Allow('*', context=qc.get('group', [])))
                .with_columns([
                    pl.col('revenue').cum_sum().over(Exclude('sales.date', include=['week_start'], context=qc.get('group', []))).alias('wtd_revenue')
                ])
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(
                    pl.col('wtd_revenue').max()
                )
            )

        result = dm.query(
            query_context={
                'measure': ['wtd_revenue'],
                'filter': ('sales.quantity', '>', 5),
                'group': ['products.category', 'sales.date'],
                'sort': [('products.category', 'asc'), ('sales.date', 'asc')]
            },
            output_type='data'
        )

        print(result.head(30))

        # Polars baseline uses hardcoded columns for validation (this is expected)
        polars_result = (
            complex_sales_data
            .join(products_data, on='item_id', how='left')
            .filter(pl.col('quantity') > 5)
            .with_columns([
                ((pl.col('date') + pl.duration(days=1)).dt.truncate('1w') - pl.duration(days=1)).alias('week_start')
            ])
            .sort(['category', 'date'])
            .with_columns([
                pl.col('revenue').cum_sum().over(['category', 'week_start']).alias('wtd_revenue')
            ])
            .group_by(['category', 'date'])
            .agg(pl.col('wtd_revenue').max())
            .sort(['category', 'date'])
            .collect()
        )

        assert_frame_equal(result, polars_result, check_column_order=False)
    
    def test_product_rank_by_revenue_measure(self, datamodel_with_pre_aggs, complex_sales_data, products_data):
        """Test measure that calculates the product rank by revenue"""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def product_rank_by_revenue(qc):
            # Step 1: Calculate TOTAL revenue per product, THEN rank those totals
            # Specify pre-agg columns to enable pre-agg usage
            product_rank = (
                dm.table('sales', ['sales.item_id'], {'revenue': 'sum'})
                .filter(Allow('*', context=qc.get('filter')))
                .group_by(Allow('*', include=['sales.item_id'], context=[]))  # Use Allow for proper transformation
                .agg(pl.col('revenue').sum().alias('total_revenue'))
                .with_columns(
                    pl.col('total_revenue').rank('min', descending=True).alias('product_rank')
                )
                .select(pl.col('item_id'), pl.col('product_rank'))
            )

            # Step 2: Join ranks back and aggregate (use pre-agg with correct spec)
            return (
                dm.table('sales', ['sales.item_id'], {'revenue': 'sum'})
                .filter(Allow('*', context=qc.get('filter')))
                .join(product_rank, on='item_id', how='inner')
                .group_by(Allow('*', include=['sales.item_id'], context=qc.get('group')))
                .agg(
                    pl.col('product_rank').min().alias('product_rank'),
                    pl.col('revenue').sum().alias('revenue')
                )
            )
        result = dm.query(
            query_context={
                'measure': ['product_rank_by_revenue'],
                # 'filter': ('sales.quantity', '>', 5),
                # 'group': ['products.category', 'sales.date'],
                'sort': [('sales.item_id', 'asc')]
            },
            output_type='data'
        )

        print(result)

        # Polars baseline: mirror the measure logic
        polars_product_rank = (
            complex_sales_data
            .group_by('item_id')
            .agg(pl.col('revenue').sum().alias('total_revenue'))
            .with_columns(
                pl.col('total_revenue').rank('min', descending=True).alias('product_rank')
            )
            .select('item_id', 'product_rank')
        )

        polars_result = (
            complex_sales_data
            .join(polars_product_rank, on='item_id', how='inner')
            .group_by('item_id')
            .agg(
                pl.col('product_rank').min().alias('product_rank'),
                pl.col('revenue').sum().alias('revenue')
            )
            .sort('item_id')
            .collect()
        )

        assert_frame_equal(result, polars_result)
