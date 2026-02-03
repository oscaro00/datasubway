import pytest
import polars as pl
from pathlib import Path
import tempfile
import shutil

from datasubway import DataModel, measure, Allow


class TestDataModelQueryMethod:
    """Test suite for DataModel.query() method."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for pre-agg files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def simple_datamodel(self):
        """Create a simple DataModel for testing."""
        tables = {
            'sales': pl.LazyFrame({
                'item_id': [1, 1, 2, 2, 3],
                'store_id': [1, 2, 1, 2, 1],
                'revenue': [100, 150, 200, 250, 300],
                'quantity': [10, 15, 20, 25, 30]
            })
        }
        dm = DataModel(
            tables=tables,
            joins=[],
            pre_aggregations={},
            pre_agg_directory=None
        )
        return dm

    @pytest.fixture
    def datamodel_with_measures(self, simple_datamodel):
        """Create DataModel with registered measures."""
        dm = simple_datamodel

        @measure(dm)
        def total_revenue(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        @measure(dm)
        def total_quantity(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('quantity').sum())
            )

        @measure(dm)
        def item_count(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.len().alias('count'))
            )

        return dm

    # ========================================================================
    # BASIC QUERY EXECUTION TESTS
    # ========================================================================

    def test_query_single_measure_returns_dataframe(self, datamodel_with_measures):
        """Test that query with single measure returns DataFrame."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns

    def test_query_with_group_by(self, datamodel_with_measures):
        """Test query with group by columns."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='data'
        )
        # Should have 3 rows (item_id 1, 2, 3)
        assert len(result) == 3
        assert set(result['item_id'].to_list()) == {1, 2, 3}

    def test_query_without_group_by(self, datamodel_with_measures):
        """Test query without group by (aggregate all)."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue']},
            output_type='data'
        )
        # Should have 1 row (total)
        assert len(result) == 1
        # Total revenue should be sum of all
        assert result['revenue'][0] == 1000  # 100+150+200+250+300

    def test_query_multiple_group_by_columns(self, datamodel_with_measures):
        """Test query with multiple group by columns."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id', 'store_id']},
            output_type='data'
        )
        # Should have 5 rows (one per combination)
        assert len(result) == 5
        assert 'item_id' in result.columns
        assert 'store_id' in result.columns

    # ========================================================================
    # MULTIPLE MEASURES TESTS
    # ========================================================================

    def test_query_multiple_measures_with_group_by(self, datamodel_with_measures):
        """Test query with multiple measures joined on group by columns."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue', 'total_quantity'],
                'group': ['item_id']
            },
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns
        assert 'quantity' in result.columns
        assert len(result) == 3  # 3 items

    def test_query_multiple_measures_without_group_by(self, datamodel_with_measures):
        """Test query with multiple measures and no group by (cross join)."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue', 'total_quantity']
            },
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)
        # Cross join of two single-row results = 1 row
        assert len(result) == 1
        assert 'revenue' in result.columns
        assert 'quantity' in result.columns

    def test_query_three_measures_joined(self, datamodel_with_measures):
        """Test query with three measures joined together."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue', 'total_quantity', 'item_count'],
                'group': ['item_id']
            },
            output_type='data'
        )
        assert len(result) == 3
        assert 'item_id' in result.columns
        assert 'revenue' in result.columns
        assert 'quantity' in result.columns
        assert 'count' in result.columns

    # ========================================================================
    # OUTPUT TYPE TESTS
    # ========================================================================

    def test_query_output_type_data(self, datamodel_with_measures):
        """Test output_type='data' returns DataFrame."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)

    def test_query_output_type_query_single_measure(self, datamodel_with_measures):
        """Test output_type='query' returns transformed source code."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='query'
        )
        assert isinstance(result, str)
        assert 'def total_revenue' in result
        # Should have transformed Allow to actual columns
        assert 'pl.col(' in result

    def test_query_output_type_query_multiple_measures(self, datamodel_with_measures):
        """Test output_type='query' with multiple measures returns dict."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue', 'total_quantity'],
                'group': ['item_id']
            },
            output_type='query'
        )
        assert isinstance(result, dict)
        assert 'total_revenue' in result
        assert 'total_quantity' in result
        assert isinstance(result['total_revenue'], str)
        assert isinstance(result['total_quantity'], str)

    def test_query_output_type_explain(self, datamodel_with_measures):
        """Test output_type='explain' returns query plan."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='explain'
        )
        assert isinstance(result, str)
        # Should contain polars query plan information
        # Polars explain output typically contains keywords like these
        assert len(result) > 0

    # ========================================================================
    # VALIDATION TESTS
    # ========================================================================

    def test_query_invalid_measure_name(self, datamodel_with_measures):
        """Test that invalid measure name raises KeyError."""
        with pytest.raises(KeyError, match='not registered'):
            datamodel_with_measures.query(
                query_context={'measure': ['nonexistent_measure']},
                output_type='data'
            )

    def test_query_invalid_output_type(self, datamodel_with_measures):
        """Test that invalid output_type raises ValueError."""
        with pytest.raises(ValueError, match='output_type must be'):
            datamodel_with_measures.query(
                query_context={'measure': ['total_revenue']},
                output_type='invalid'
            )

    def test_query_missing_measure_key(self, datamodel_with_measures):
        """Test that missing 'measure' key raises TypeError."""
        with pytest.raises(TypeError):
            datamodel_with_measures.query(
                query_context={'group': ['item_id']},
                output_type='data'
            )

    def test_query_empty_context(self, datamodel_with_measures):
        """Test that empty context raises Exception."""
        with pytest.raises(Exception, match='cannot be empty'):
            datamodel_with_measures.query(
                query_context={},
                output_type='data'
            )

    def test_query_with_invalid_context_key(self, datamodel_with_measures):
        """Test that invalid context key raises KeyError."""
        with pytest.raises(KeyError, match='not in'):
            datamodel_with_measures.query(
                query_context={
                    'measure': ['total_revenue'],
                    'invalid_key': 'value'
                },
                output_type='data'
            )

    # ========================================================================
    # TRANSFORMATION PIPELINE TESTS
    # ========================================================================

    def test_query_applies_allow_transformation(self, datamodel_with_measures):
        """Test that Allow() is properly transformed."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='query'
        )
        # Allow should be replaced with actual column list
        assert 'Allow' not in result
        assert 'pl.col(' in result

    def test_query_removes_empty_methods(self, simple_datamodel):
        """Test that empty methods are removed in transformation."""
        dm = simple_datamodel

        @measure(dm)
        def test_measure(qc):
            return (
                dm.table('sales')
                .filter([])
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['test_measure'], 'group': ['item_id']},
            output_type='query'
        )
        # Empty filter should be removed
        assert '.filter([])' not in result

    def test_query_converts_empty_group_by_agg_to_select(self, simple_datamodel):
        """Test that empty group_by with agg converts to select."""
        dm = simple_datamodel

        @measure(dm)
        def test_measure(qc):
            return (
                dm.table('sales')
                .group_by(Allow('*', context=qc.get('group', [])))
                .agg(pl.col('revenue').sum())
            )

        result = dm.query(
            query_context={'measure': ['test_measure']},  # No group
            output_type='query'
        )
        # Should convert to .select()
        assert '.select(' in result
        assert '.agg(' not in result or result.count('.agg(') == 0

    # ========================================================================
    # DEFAULT VALUES TESTS
    # ========================================================================

    def test_query_default_output_type_is_data(self, datamodel_with_measures):
        """Test that default output_type is 'data'."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']}
        )
        # Default should be DataFrame
        assert isinstance(result, pl.DataFrame)

    def test_query_with_limit_and_offset(self, datamodel_with_measures):
        """Test query with limit and offset in context."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue'],
                'group': ['item_id'],
                'limit': 2,
                'offset': 1
            },
            output_type='data'
        )
        # Note: limit/offset might not be applied automatically in measures
        # This tests that they don't cause errors
        assert isinstance(result, pl.DataFrame)

    # ========================================================================
    # EDGE CASES
    # ========================================================================

    def test_query_single_measure_list_with_one_item(self, datamodel_with_measures):
        """Test that single measure in list works correctly."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)

    def test_query_with_allow_pre_aggs_false(self, datamodel_with_measures):
        """Test query with allow_pre_aggs=False in context."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue'],
                'group': ['item_id'],
                'allow_pre_aggs': False
            },
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)

    def test_query_with_allow_pre_aggs_true(self, datamodel_with_measures):
        """Test query with allow_pre_aggs=True in context."""
        result = datamodel_with_measures.query(
            query_context={
                'measure': ['total_revenue'],
                'group': ['item_id'],
                'allow_pre_aggs': True
            },
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)

    def test_query_measure_returns_lazy_frame(self, simple_datamodel):
        """Test that measure returning LazyFrame works correctly."""
        dm = simple_datamodel

        @measure(dm)
        def lazy_measure(qc):
            # Returns LazyFrame
            return dm.table('sales').group_by('item_id').agg(pl.col('revenue').sum())

        result = dm.query(
            query_context={'measure': ['lazy_measure']},
            output_type='data'
        )
        assert isinstance(result, pl.DataFrame)

    def test_query_measure_not_returning_lazyframe_raises_error(self, simple_datamodel):
        """Test that measure not returning LazyFrame raises ValueError."""
        dm = simple_datamodel

        # Manually register measure to bypass decorator validation
        # (decorator would reject this at decoration time)
        def bad_measure(qc):
            # Returns DataFrame instead of LazyFrame
            return dm.tables['sales'].collect()

        dm.measures['bad_measure'] = bad_measure

        with pytest.raises(ValueError, match='must return pl.LazyFrame'):
            dm.query(
                query_context={'measure': ['bad_measure']},
                output_type='data'
            )

    def test_query_preserves_column_types(self, datamodel_with_measures):
        """Test that query preserves column data types."""
        result = datamodel_with_measures.query(
            query_context={'measure': ['total_revenue'], 'group': ['item_id']},
            output_type='data'
        )
        assert result['item_id'].dtype == pl.Int64
        assert result['revenue'].dtype == pl.Int64

    def test_query_with_filter_in_context(self, simple_datamodel):
        """Test query with filter in context."""
        dm = simple_datamodel

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
        # Should only have item_id=1
        assert len(result) == 1
        assert result['item_id'][0] == 1

    # ========================================================================
    # PARALLEL PROCESSING TESTS (commented out - PARALLEL_THRESHOLD raised to 50)
    # ========================================================================

    # @pytest.fixture
    # def datamodel_with_many_measures(self, simple_datamodel):
    #     """Create DataModel with 6 measures to trigger parallel processing."""
    #     dm = simple_datamodel

    #     @measure(dm)
    #     def measure_revenue_sum(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('revenue').sum().alias('revenue_sum'))
    #         )

    #     @measure(dm)
    #     def measure_revenue_mean(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('revenue').mean().alias('revenue_mean'))
    #         )

    #     @measure(dm)
    #     def measure_quantity_sum(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('quantity').sum().alias('quantity_sum'))
    #         )

    #     @measure(dm)
    #     def measure_quantity_mean(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('quantity').mean().alias('quantity_mean'))
    #         )

    #     @measure(dm)
    #     def measure_row_count(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.len().alias('row_count'))
    #         )

    #     @measure(dm)
    #     def measure_revenue_max(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('revenue').max().alias('revenue_max'))
    #         )

    #     @measure(dm)
    #     def measure_revenue_min(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('revenue').min().alias('revenue_min'))
    #         )

    #     @measure(dm)
    #     def measure_quantity_max(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('quantity').max().alias('quantity_max'))
    #         )

    #     @measure(dm)
    #     def measure_quantity_min(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('quantity').min().alias('quantity_min'))
    #         )

    #     @measure(dm)
    #     def measure_revenue_std(qc):
    #         return (
    #             dm.table('sales')
    #             .group_by(Allow('*', context=qc.get('group', [])))
    #             .agg(pl.col('revenue').std().alias('revenue_std'))
    #         )

    #     return dm

    # def test_parallel_processing_with_ten_measures(self, datamodel_with_many_measures):
    #     """Test that 10 measures triggers parallel processing and returns correct results."""
    #     from datasubway.data_model import PARALLEL_THRESHOLD

    #     dm = datamodel_with_many_measures
    #     measure_names = [
    #         'measure_revenue_sum',
    #         'measure_revenue_mean',
    #         'measure_quantity_sum',
    #         'measure_quantity_mean',
    #         'measure_row_count',
    #         'measure_revenue_max',
    #         'measure_revenue_min',
    #         'measure_quantity_max',
    #         'measure_quantity_min',
    #         'measure_revenue_std'
    #     ]

    #     # Verify we're testing the parallel path
    #     assert len(measure_names) >= PARALLEL_THRESHOLD

    #     result = dm.query(
    #         query_context={
    #             'measure': measure_names,
    #             'group': ['item_id']
    #         },
    #         output_type='data'
    #     )

    #     assert isinstance(result, pl.DataFrame)
    #     # Should have 3 rows (item_id 1, 2, 3)
    #     assert len(result) == 3
    #     # Should have all measure columns
    #     assert 'revenue_sum' in result.columns
    #     assert 'revenue_mean' in result.columns
    #     assert 'quantity_sum' in result.columns
    #     assert 'quantity_mean' in result.columns
    #     assert 'row_count' in result.columns
    #     assert 'revenue_max' in result.columns
    #     assert 'revenue_min' in result.columns
    #     assert 'quantity_max' in result.columns
    #     assert 'quantity_min' in result.columns
    #     assert 'revenue_std' in result.columns

    # def test_parallel_results_match_sequential(self, datamodel_with_many_measures):
    #     """Test that parallel processing produces same results as sequential."""
    #     from datasubway.data_model import PARALLEL_THRESHOLD

    #     dm = datamodel_with_many_measures

    #     # Run with 9 measures (sequential path)
    #     sequential_measures = [
    #         'measure_revenue_sum',
    #         'measure_revenue_mean',
    #         'measure_quantity_sum',
    #         'measure_quantity_mean',
    #         'measure_row_count',
    #         'measure_revenue_max',
    #         'measure_revenue_min',
    #         'measure_quantity_max',
    #         'measure_quantity_min'
    #     ]
    #     assert len(sequential_measures) < PARALLEL_THRESHOLD

    #     sequential_result = dm.query(
    #         query_context={
    #             'measure': sequential_measures,
    #             'group': ['item_id']
    #         },
    #         output_type='data'
    #     ).sort('item_id')

    #     # Run with 10 measures (parallel path) but compare overlapping columns
    #     parallel_measures = [
    #         'measure_revenue_sum',
    #         'measure_revenue_mean',
    #         'measure_quantity_sum',
    #         'measure_quantity_mean',
    #         'measure_row_count',
    #         'measure_revenue_max',
    #         'measure_revenue_min',
    #         'measure_quantity_max',
    #         'measure_quantity_min',
    #         'measure_revenue_std'
    #     ]
    #     assert len(parallel_measures) >= PARALLEL_THRESHOLD

    #     parallel_result = dm.query(
    #         query_context={
    #             'measure': parallel_measures,
    #             'group': ['item_id']
    #         },
    #         output_type='data'
    #     ).sort('item_id')

    #     # Compare the overlapping columns
    #     for col in ['item_id', 'revenue_sum', 'revenue_mean', 'quantity_sum', 'quantity_mean',
    #                 'row_count', 'revenue_max', 'revenue_min', 'quantity_max', 'quantity_min']:
    #         assert sequential_result[col].to_list() == parallel_result[col].to_list(), \
    #             f"Column {col} differs between sequential and parallel processing"

    # def test_parallel_processing_output_type_query(self, datamodel_with_many_measures):
    #     """Test that parallel processing works with output_type='query'."""
    #     dm = datamodel_with_many_measures
    #     measure_names = [
    #         'measure_revenue_sum',
    #         'measure_revenue_mean',
    #         'measure_quantity_sum',
    #         'measure_quantity_mean',
    #         'measure_row_count',
    #         'measure_revenue_max',
    #         'measure_revenue_min',
    #         'measure_quantity_max',
    #         'measure_quantity_min',
    #         'measure_revenue_std'
    #     ]

    #     result = dm.query(
    #         query_context={
    #             'measure': measure_names,
    #             'group': ['item_id']
    #         },
    #         output_type='query'
    #     )

    #     # With multiple measures, should return dict
    #     assert isinstance(result, dict)
    #     assert len(result) == 10
    #     for name in measure_names:
    #         assert name in result
    #         assert isinstance(result[name], str)

    # def test_parallel_processing_without_group_by(self, datamodel_with_many_measures):
    #     """Test parallel processing without group_by columns."""
    #     dm = datamodel_with_many_measures
    #     measure_names = [
    #         'measure_revenue_sum',
    #         'measure_revenue_mean',
    #         'measure_quantity_sum',
    #         'measure_quantity_mean',
    #         'measure_row_count',
    #         'measure_revenue_max',
    #         'measure_revenue_min',
    #         'measure_quantity_max',
    #         'measure_quantity_min',
    #         'measure_revenue_std'
    #     ]

    #     result = dm.query(
    #         query_context={'measure': measure_names},
    #         output_type='data'
    #     )

    #     assert isinstance(result, pl.DataFrame)
    #     # Without group by, should have 1 row with aggregated totals
    #     assert len(result) == 1
