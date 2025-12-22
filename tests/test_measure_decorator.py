import pytest
import polars as pl
from pathlib import Path
import sys

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_model import DataModel
from decorators import measure


class TestMeasureDecorator:
    """Test suite for the @measure decorator."""

    @pytest.fixture
    def simple_datamodel(self):
        """Create a simple DataModel for testing."""
        tables = {
            'sales': pl.LazyFrame({
                'item_id': [1, 2, 3],
                'revenue': [100, 200, 300],
                'quantity': [10, 20, 30]
            })
        }
        return DataModel(
            tables=tables,
            joins=[],
            pre_aggregations={},
            pre_agg_directory=None
        )

    # ========================================================================
    # VALID MEASURES (should succeed)
    # ========================================================================

    def test_measure_with_group_by_agg(self, simple_datamodel):
        """Test that valid measure with .group_by().agg() is registered."""

        @measure(simple_datamodel)
        def total_revenue():
            return (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

        assert 'total_revenue' in simple_datamodel.measures
        assert simple_datamodel.measures['total_revenue'] == total_revenue

    def test_measure_with_dynamic_group_by_agg(self, simple_datamodel):
        """Test that .dynamic_group_by().agg() is accepted."""

        @measure(simple_datamodel)
        def dynamic_revenue():
            return (
                simple_datamodel.tables['sales']
                .dynamic_group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

        assert 'dynamic_revenue' in simple_datamodel.measures

    def test_measure_with_rolling_group_by_agg(self, simple_datamodel):
        """Test that .rolling_group_by().agg() is accepted."""

        @measure(simple_datamodel)
        def rolling_revenue():
            return (
                simple_datamodel.tables['sales']
                .rolling_group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

        assert 'rolling_revenue' in simple_datamodel.measures

    def test_measure_registration(self, simple_datamodel):
        """Test that measure is properly stored in data_model.measures."""

        @measure(simple_datamodel)
        def test_measure():
            return (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(pl.col('revenue').max())
            )

        # Verify it's stored correctly
        assert 'test_measure' in simple_datamodel.measures
        assert callable(simple_datamodel.measures['test_measure'])
        assert simple_datamodel.measures['test_measure'].__name__ == 'test_measure'

    def test_multiple_measures_same_datamodel(self, simple_datamodel):
        """Test registering multiple different measures in the same DataModel."""

        @measure(simple_datamodel)
        def measure1():
            return (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

        @measure(simple_datamodel)
        def measure2():
            return (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(pl.col('quantity').mean())
            )

        assert 'measure1' in simple_datamodel.measures
        assert 'measure2' in simple_datamodel.measures
        assert len(simple_datamodel.measures) == 2

    def test_measure_with_chained_methods_before_group_by(self, simple_datamodel):
        """Test that methods chained BEFORE group_by are allowed."""

        @measure(simple_datamodel)
        def filtered_revenue():
            return (
                simple_datamodel.tables['sales']
                .filter(pl.col('revenue') > 100)
                .select('item_id', 'revenue')
                .group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

        assert 'filtered_revenue' in simple_datamodel.measures

    def test_measure_with_multiple_chains_last_valid(self, simple_datamodel):
        """Test that multiple chains are allowed if last one ends correctly."""

        @measure(simple_datamodel)
        def multiple_chains():
            # First chain - can have any structure
            temp = simple_datamodel.tables['sales'].select('item_id', 'revenue')

            # Last chain - must end with group_by/agg
            return temp.group_by('item_id').agg(pl.col('revenue').sum())

        assert 'multiple_chains' in simple_datamodel.measures

    def test_measure_with_early_invalid_chain(self, simple_datamodel):
        """Test that early chains can have methods after .agg()."""

        @measure(simple_datamodel)
        def early_invalid():
            # First chain - has .sort() after .agg(), but that's OK
            temp = (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(pl.col('revenue').sum())
                .sort('item_id')  # Methods after agg - OK because not last chain
            )

            # Last chain - valid
            return temp.filter(pl.col('revenue-sum') > 100).group_by('item_id').agg(pl.col('revenue-sum').max())

        assert 'early_invalid' in simple_datamodel.measures

    def test_measure_with_multiline_agg(self, simple_datamodel):
        """Test that multiline .agg() with multiple arguments works."""

        @measure(simple_datamodel)
        def multiline_agg():
            return (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(
                    pl.col('revenue').sum().alias('total_revenue'),
                    pl.col('quantity').mean().alias('avg_quantity')
                )
            )

        assert 'multiline_agg' in simple_datamodel.measures

    # ========================================================================
    # DUPLICATE NAMES (should fail)
    # ========================================================================

    def test_duplicate_measure_name(self, simple_datamodel):
        """Test that duplicate measure names raise ValueError."""

        @measure(simple_datamodel)
        def revenue():
            return (
                simple_datamodel.tables['sales']
                .group_by('item_id')
                .agg(pl.col('revenue').sum())
            )

        # Try to register another measure with same name
        with pytest.raises(ValueError, match="already exists"):
            @measure(simple_datamodel)
            def revenue():  # Same name
                return (
                    simple_datamodel.tables['sales']
                    .group_by('item_id')
                    .agg(pl.col('revenue').max())
                )

    # ========================================================================
    # INVALID METHOD CHAINS (should fail)
    # ========================================================================

    def test_measure_without_group_by(self, simple_datamodel):
        """Test that last chain with .agg() but no group_by variant fails."""

        with pytest.raises(ValueError, match="must end with one of"):
            @measure(simple_datamodel)
            def no_group_by():
                return (
                    simple_datamodel.tables['sales']
                    .select('item_id', 'revenue')
                    .agg(pl.col('revenue').sum())  # Missing group_by before agg
                )

    def test_measure_without_agg(self, simple_datamodel):
        """Test that last chain with .group_by() but no .agg() fails."""

        with pytest.raises(ValueError, match="must end with .agg"):
            @measure(simple_datamodel)
            def no_agg():
                return (
                    simple_datamodel.tables['sales']
                    .group_by('item_id')
                    # Missing .agg()
                )

    def test_measure_last_chain_with_method_after_agg(self, simple_datamodel):
        """Test that methods after .agg() in the last chain fail."""

        with pytest.raises(ValueError, match="must not have methods after .agg"):
            @measure(simple_datamodel)
            def method_after_agg():
                return (
                    simple_datamodel.tables['sales']
                    .group_by('item_id')
                    .agg(pl.col('revenue').sum())
                    .sort('item_id')  # This should fail
                )

    def test_measure_last_chain_missing_agg(self, simple_datamodel):
        """Test that if last chain doesn't end with group_by/agg, it fails."""

        with pytest.raises(ValueError, match="must end with"):
            @measure(simple_datamodel)
            def last_chain_invalid():
                # First chain - valid
                temp = (
                    simple_datamodel.tables['sales']
                    .group_by('item_id')
                    .agg(pl.col('revenue').sum())
                )

                # Last chain - doesn't end with group_by/agg
                return temp.select('item_id')

    def test_measure_with_select_after_agg(self, simple_datamodel):
        """Test that .select() after .agg() in last chain fails."""

        with pytest.raises(ValueError, match="must not have methods after .agg"):
            @measure(simple_datamodel)
            def select_after_agg():
                return (
                    simple_datamodel.tables['sales']
                    .group_by('item_id')
                    .agg(pl.col('revenue').sum().alias('total'))
                    .select('item_id', 'total')  # This should fail
                )

    def test_measure_with_filter_after_agg(self, simple_datamodel):
        """Test that .filter() after .agg() in last chain fails."""

        with pytest.raises(ValueError, match="must not have methods after .agg"):
            @measure(simple_datamodel)
            def filter_after_agg():
                return (
                    simple_datamodel.tables['sales']
                    .group_by('item_id')
                    .agg(pl.col('revenue').sum().alias('total'))
                    .filter(pl.col('total') > 100)  # This should fail
                )

    def test_measure_no_polars_chain(self, simple_datamodel):
        """Test that a measure with no polars method chains fails."""

        with pytest.raises(ValueError, match="must contain at least one polars method chain"):
            @measure(simple_datamodel)
            def no_polars():
                # Just some non-polars code
                x = 5
                y = 10
                return x + y
