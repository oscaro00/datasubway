import pytest
import polars as pl
from pathlib import Path
import tempfile
import shutil

from datasubway import DataModel, measure, Allow, Exclude


class TestIntegrationQueryPipeline:
    """Integration tests for the complete query pipeline."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for pre-agg files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sales_data(self):
        """Create sample sales data."""
        return pl.LazyFrame({
            'item_id': [1, 1, 1, 2, 2, 2, 3, 3],
            'store_id': [1, 1, 2, 1, 2, 2, 1, 2],
            'date': ['2024-01-01', '2024-01-02', '2024-01-01', '2024-01-01', '2024-01-02', '2024-01-03', '2024-01-01', '2024-01-02'],
            'revenue': [100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0, 450.0],
            'quantity': [10, 15, 20, 25, 30, 35, 40, 45],
            'cost': [60.0, 90.0, 120.0, 150.0, 180.0, 210.0, 240.0, 270.0]
        })

    @pytest.fixture
    def products_data(self):
        """Create sample products data."""
        return pl.LazyFrame({
            'item_id': [1, 2, 3],
            'product_name': ['Widget', 'Gadget', 'Doohickey'],
            'category': ['A', 'B', 'A']
        })

    @pytest.fixture
    def datamodel_with_joins(self, sales_data, products_data):
        """Create DataModel with joined tables."""
        tables = {
            'sales': sales_data,
            'products': products_data
        }
        joins = [
            {
                'left': 'sales',
                'right': 'products',
                'left_on': ['item_id'],
                'right_on': ['item_id'],
                'how': 'inner',
                'direction': 'both'
            }
        ]
        dm = DataModel(
            tables=tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )
        return dm

    @pytest.fixture
    def datamodel_with_pre_aggs(self, sales_data, temp_dir):
        """Create DataModel with pre-aggregations."""
        tables = {'sales': sales_data}

        # Define pre-aggregations
        pre_aggs = {
            'sales_by_item': {
                'group_by': ['sales.item_id'],
                'aggregations': {
                    'sales.revenue': ['sum', 'mean'],
                    'sales.quantity': 'sum',
                    'sales.cost': 'sum'
                }
            }
        }

        dm = DataModel(
            tables=tables,
            joins=[],
            pre_aggregations=pre_aggs,
            pre_agg_directory=temp_dir
        )

        # Write the pre-aggregation
        dm.write_pre_aggregation('sales_by_item')

        return dm

    # ========================================================================
    # END-TO-END PIPELINE TESTS
    # ========================================================================

    def test_full_pipeline_single_measure(self, datamodel_with_joins):
        """Test complete pipeline with single measure."""
        dm = datamodel_with_joins

        @measure(dm)
        def revenue_by_item(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_by_item'], 'group': ['item_id']},
            output_type='data'
        )

        assert len(result) == 3
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns

    def test_full_pipeline_multiple_measures_with_join(self, datamodel_with_joins):
        """Test pipeline with multiple measures joined together."""
        dm = datamodel_with_joins

        @measure(dm)
        def revenue_by_item(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('total_revenue'))
            )

        @measure(dm)
        def quantity_by_item(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('quantity').sum().alias('total_quantity'))
            )

        result = dm.query(
            query_context={
                'measure': ['revenue_by_item', 'quantity_by_item'],
                'group': ['item_id']
            },
            output_type='data'
        )

        assert len(result) == 3
        assert 'item_id' in result.columns
        assert 'total_revenue' in result.columns
        assert 'total_quantity' in result.columns

    def test_pipeline_with_filter_context(self, datamodel_with_joins):
        """Test pipeline with filter context."""
        dm = datamodel_with_joins

        @measure(dm)
        def filtered_revenue(qc):
            return (
                dm.table('sales')
                .filter(Allow('*', context=qc.get('filter')))
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={
                'measure': ['filtered_revenue'],
                'filter': ('sales.item_id', '=', 1),
                'group': ['item_id']
            },
            output_type='data'
        )

        assert len(result) == 1
        assert result['item_id'][0] == 1

    def test_pipeline_with_exclude_context(self, datamodel_with_joins):
        """Test pipeline with Exclude context."""
        dm = datamodel_with_joins

        @measure(dm)
        def revenue_excluding_store(qc):
            return (
                dm.table('sales')
                .group_by(Exclude('store_id', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={
                'measure': ['revenue_excluding_store'],
                'group': ['item_id', 'store_id']
            },
            output_type='data'
        )

        # Should only group by item_id (store_id excluded)
        assert 'item_id' in result.columns
        assert 'store_id' not in result.columns

    def test_pipeline_removes_empty_methods(self, datamodel_with_joins):
        """Test that pipeline removes empty polars methods."""
        dm = datamodel_with_joins

        @measure(dm)
        def revenue_with_empty_filter(qc):
            return (
                dm.table('sales')
                .filter([])  # Empty filter
                .drop([])    # Empty drop
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_with_empty_filter'], 'group': ['item_id']},
            output_type='data'
        )

        # Should work correctly despite empty methods
        assert len(result) == 3
        assert 'revenue' in result.columns

    def test_pipeline_converts_empty_group_by_to_select(self, datamodel_with_joins):
        """Test that empty group_by with agg converts to select."""
        dm = datamodel_with_joins

        @measure(dm)
        def total_revenue(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        # Query without group by
        result = dm.query(
            query_context={'measure': ['total_revenue']},
            output_type='data'
        )

        # Should aggregate all data into single row
        assert len(result) == 1
        assert result['revenue'][0] == 2200.0  # Sum of all revenue

    # ========================================================================
    # PRE-AGGREGATION INTEGRATION TESTS
    # ========================================================================

    def test_pipeline_with_pre_agg_simple_aggregation(self, datamodel_with_pre_aggs):
        """Test pipeline using pre-aggregation with simple aggregation."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def revenue_from_pre_agg(qc):
            # This should use the pre-aggregation
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'sum'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_from_pre_agg'], 'group': ['item_id']},
            output_type='data'
        )

        # Should get results from pre-agg
        assert len(result) == 3
        assert 'item_id' in result.columns

    def test_pipeline_transforms_pre_agg_mean_to_formula(self, datamodel_with_pre_aggs):
        """Test that mean is transformed to sum/count formula when using pre-agg."""
        dm = datamodel_with_pre_aggs

        @measure(dm)
        def avg_revenue_from_pre_agg(qc):
            return (
                dm.table('sales', qc.get('group', []), {'revenue': 'mean'})
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').mean())
            )

        # Get the transformed query
        transformed = dm.query(
            query_context={'measure': ['avg_revenue_from_pre_agg'], 'group': ['item_id']},
            output_type='query'
        )

        # Should transform mean to sum/count formula
        assert 'revenue-mean-sum' in transformed
        assert 'revenue-mean-count' in transformed
        assert '/' in transformed

    def test_pipeline_with_allow_pre_aggs_false(self, datamodel_with_pre_aggs):
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

        # Should still work, just using source tables instead of pre-agg
        assert len(result) == 3

    # ========================================================================
    # COMPLEX SCENARIO TESTS
    # ========================================================================

    def test_pipeline_three_measures_different_aggregations(self, datamodel_with_joins):
        """Test complex pipeline with three different measures."""
        dm = datamodel_with_joins

        @measure(dm)
        def total_revenue(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('sum_revenue'))
            )

        @measure(dm)
        def avg_quantity(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('quantity').mean().alias('avg_qty'))
            )

        @measure(dm)
        def max_cost(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('cost').max().alias('max_cost'))
            )

        result = dm.query(
            query_context={
                'measure': ['total_revenue', 'avg_quantity', 'max_cost'],
                'group': ['item_id']
            },
            output_type='data'
        )

        assert len(result) == 3
        assert 'sum_revenue' in result.columns
        assert 'avg_qty' in result.columns
        assert 'max_cost' in result.columns

    def test_pipeline_with_complex_filter(self, datamodel_with_joins):
        """Test pipeline with complex AND/OR filter."""
        dm = datamodel_with_joins

        @measure(dm)
        def filtered_revenue(qc):
            return (
                dm.table('sales')
                .filter(Allow('*', context=qc.get('filter')))
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={
                'measure': ['filtered_revenue'],
                'filter': {
                    'OR': [
                        ('sales.item_id', '=', 1),
                        ('sales.item_id', '=', 2)
                    ]
                },
                'group': ['item_id']
            },
            output_type='data'
        )

        # Should only have item_id 1 and 2
        assert len(result) == 2
        assert set(result['item_id'].to_list()) == {1, 2}

    def test_pipeline_cross_join_multiple_measures_no_group(self, datamodel_with_joins):
        """Test that multiple measures without group by use cross join."""
        dm = datamodel_with_joins

        @measure(dm)
        def total_revenue(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum().alias('total_rev'))
            )

        @measure(dm)
        def total_quantity(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('quantity').sum().alias('total_qty'))
            )

        result = dm.query(
            query_context={
                'measure': ['total_revenue', 'total_quantity']
                # No group by - should cross join
            },
            output_type='data'
        )

        # Cross join of two single-row results = 1 row
        assert len(result) == 1
        assert 'total_rev' in result.columns
        assert 'total_qty' in result.columns
        assert result['total_rev'][0] == 2200.0
        assert result['total_qty'][0] == 220

    def test_pipeline_query_code_inspection(self, datamodel_with_joins):
        """Test that 'query' output shows all transformations applied."""
        dm = datamodel_with_joins

        @measure(dm)
        def complex_measure(qc):
            return (
                dm.table('sales')
                .filter([])  # Should be removed
                .filter(Allow('*', context=qc.get('filter')))
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={
                'measure': ['complex_measure'],
                'filter': ('sales.item_id', '=', 1),
                'group': ['item_id']
            },
            output_type='query'
        )

        # Should not have Allow/Exclude
        assert 'Allow' not in result
        assert 'Exclude' not in result

        # Should not have empty filter
        assert result.count('.filter([])') == 0

        # Should have transformed filter expression
        assert 'pl.col(' in result

    def test_pipeline_explain_output(self, datamodel_with_joins):
        """Test that 'explain' output returns query plan."""
        dm = datamodel_with_joins

        @measure(dm)
        def simple_measure(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['simple_measure'], 'group': ['item_id']},
            output_type='explain'
        )

        assert isinstance(result, str)
        assert len(result) > 0
        # Polars explain output should contain plan information

    def test_pipeline_preserves_data_accuracy(self, datamodel_with_joins):
        """Test that pipeline transformations preserve data accuracy."""
        dm = datamodel_with_joins

        @measure(dm)
        def revenue_by_item(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['revenue_by_item'], 'group': ['item_id']},
            output_type='data'
        )

        # Verify actual values
        result_sorted = result.sort('item_id')
        assert result_sorted['revenue'][0] == 450.0  # item_id 1
        assert result_sorted['revenue'][1] == 900.0  # item_id 2
        assert result_sorted['revenue'][2] == 850.0  # item_id 3
