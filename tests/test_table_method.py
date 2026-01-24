import pytest
import polars as pl
import libcst as cst
from pathlib import Path
import tempfile
import shutil

from datasubway import DataModel


class TestTableMethod:
    """Test suite for DataModel.table() method."""

    @pytest.fixture
    def simple_tables(self):
        """Create simple test tables."""
        return {
            'sales': pl.LazyFrame({
                'item_id': [1, 2, 3],
                'store_id': [1, 1, 2],
                'revenue': [100, 200, 150],
                'quantity': [10, 20, 15]
            }),
            'products': pl.LazyFrame({
                'item_id': [1, 2, 3],
                'category': ['A', 'B', 'A'],
                'product_name': ['Widget', 'Gadget', 'Tool']
            })
        }

    @pytest.fixture
    def simple_joins(self):
        """Create simple join definitions."""
        return [
            {
                'left': 'sales',
                'right': 'products',
                'left_on': ['item_id'],
                'right_on': ['item_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

    @pytest.fixture
    def temp_pre_agg_dir(self):
        """Create temporary directory for pre-aggregations."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def datamodel_with_pre_agg(self, simple_tables, simple_joins, temp_pre_agg_dir):
        """Create DataModel with a pre-aggregation."""
        # Create DataModel
        dm = DataModel(
            tables=simple_tables,
            joins=simple_joins,
            pre_aggregations={},
            pre_agg_directory=temp_pre_agg_dir
        )

        # Manually create pre-agg data and metadata (simpler than write_pre_aggregation)
        pre_agg_data = pl.DataFrame({
            'item_id': [1, 2, 3],
            'revenue-sum': [300, 400, 200],
            'quantity-sum': [30, 40, 20]
        })

        # Write pre-agg to file
        pre_agg_path = temp_pre_agg_dir / 'sales_by_item.parquet'
        pre_agg_data.write_parquet(pre_agg_path)

        # Manually populate metadata
        dm.pre_agg_metadata = [{
            'name': 'sales_by_item',
            'path': str(pre_agg_path),
            'group_by': ['item_id'],
            'aggregations': {'revenue': 'sum', 'quantity': 'sum'},
            'row_count': 3
        }]

        return dm

    @pytest.fixture
    def datamodel_no_pre_agg(self, simple_tables, simple_joins):
        """Create DataModel without pre-aggregations."""
        return DataModel(
            tables=simple_tables,
            joins=simple_joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

    def test_table_returns_pre_agg_cst_exact_match(self, datamodel_with_pre_agg):
        """Test that table() returns correct CST for pre-agg with exact match."""
        result_cst = datamodel_with_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={'revenue': 'sum', 'quantity': 'sum'}
        )

        # Convert CST to code string
        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Verify it matches expected pattern
        assert 'pl.scan_parquet' in code
        assert 'self.pre_agg_directory' in code
        assert 'sales_by_item.parquet' in code

    def test_table_returns_pre_agg_cst_partial_columns(self, datamodel_with_pre_agg):
        """Test that table() returns pre-agg CST when only using subset of agg columns."""
        result_cst = datamodel_with_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={'revenue': 'sum'}  # Only revenue, not quantity
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should still use pre-agg
        assert 'pl.scan_parquet' in code
        assert 'sales_by_item.parquet' in code

    def test_table_falls_back_on_wrong_function(self, datamodel_with_pre_agg):
        """Test that table() falls back when aggregation function doesn't match."""
        result_cst = datamodel_with_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={'revenue': 'mean'}  # Pre-agg has 'sum', not 'mean'
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should NOT use pre-agg
        assert 'pl.scan_parquet' not in code
        assert "self.tables['sales']" in code

    def test_table_falls_back_on_missing_column(self, datamodel_with_pre_agg):
        """Test that table() falls back when pre-agg doesn't have required column."""
        result_cst = datamodel_with_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={'revenue': 'sum', 'nonexistent_col': 'sum'}
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should NOT use pre-agg
        assert 'pl.scan_parquet' not in code
        assert "self.tables['sales']" in code

    def test_table_returns_single_table_cst(self, datamodel_no_pre_agg):
        """Test that table() returns correct CST for single table access."""
        result_cst = datamodel_no_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={'revenue': 'sum'}
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should be simple table access
        assert "self.tables['sales']" in code
        assert '.join(' not in code

    def test_table_returns_join_chain_cst(self, datamodel_no_pre_agg):
        """Test that table() returns correct CST for join chain."""
        result_cst = datamodel_no_pre_agg.table(
            'sales',
            group_by_cols=['sales.item_id', 'products.category'],
            agg_cols={'sales.revenue': 'sum'},
            allow_pre_aggs=False
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should have join chain
        assert "self.tables['sales']" in code
        assert '.join(' in code
        assert "self.tables['products']" in code
        assert 'left_on' in code  # Relaxed to allow spacing variations
        assert 'right_on' in code
        assert "'inner'" in code

    def test_table_handles_empty_group_by(self, datamodel_no_pre_agg):
        """Test that table() handles global aggregation (empty group_by)."""
        result_cst = datamodel_no_pre_agg.table(
            'sales',
            group_by_cols=[],
            agg_cols={'revenue': 'sum'}
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should return single table access
        assert "self.tables['sales']" in code

    def test_table_raises_on_nonexistent_table(self, datamodel_no_pre_agg):
        """Test that table() raises KeyError for non-existent table."""
        with pytest.raises(KeyError, match="Table 'nonexistent' not found"):
            datamodel_no_pre_agg.table(
                'nonexistent',
                group_by_cols=['item_id'],
                agg_cols={'revenue': 'sum'}
            )

    def test_table_allows_empty_agg_cols(self, datamodel_no_pre_agg):
        """Test that table() allows empty agg_cols (e.g., for pl.len())."""
        result = datamodel_no_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={}
        )
        # Should return a CST node without raising
        assert result is not None
        assert isinstance(result, cst.BaseExpression)

    def test_table_raises_on_missing_join_path(self, simple_tables):
        """Test that table() raises ValueError when join path doesn't exist."""
        # Create DataModel with no joins
        dm = DataModel(
            tables=simple_tables,
            joins=[],
            pre_aggregations={},
            pre_agg_directory=None
        )

        with pytest.raises(ValueError, match="No joins defined"):
            dm.table(
                'sales',
                group_by_cols=['sales.item_id', 'products.category'],
                agg_cols={'sales.revenue': 'sum'}
            )

    def test_table_with_column_normalization(self, datamodel_no_pre_agg):
        """Test that table() handles columns with and without table prefixes."""
        # Without prefix - should assume base table
        result_cst = datamodel_no_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],  # No prefix
            agg_cols={'revenue': 'sum'}  # No prefix
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()
        assert "self.tables['sales']" in code

    def test_table_with_missing_pre_agg_file(self, simple_tables, simple_joins, temp_pre_agg_dir):
        """Test that table() falls back when pre-agg file doesn't exist."""
        # Create DataModel
        dm = DataModel(
            tables=simple_tables,
            joins=simple_joins,
            pre_aggregations={},
            pre_agg_directory=temp_pre_agg_dir
        )

        # Manually create metadata pointing to non-existent file
        pre_agg_path = temp_pre_agg_dir / 'missing_pre_agg.parquet'
        dm.pre_agg_metadata = [{
            'name': 'missing_pre_agg',
            'path': str(pre_agg_path),
            'group_by': ['item_id'],
            'aggregations': {'revenue': 'sum'},
            'row_count': 100
        }]

        # The table() method should fall back to source tables with a warning
        with pytest.warns(UserWarning, match="Pre-agg file not found"):
            result_cst = dm.table(
                'sales',
                group_by_cols=['item_id'],
                agg_cols={'revenue': 'sum'}
            )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should fall back to source table
        assert "self.tables['sales']" in code

    def test_table_with_allow_pre_aggs_false(self, datamodel_with_pre_agg):
        """Test that table() skips pre-agg when allow_pre_aggs=False."""
        result_cst = datamodel_with_pre_agg.table(
            'sales',
            group_by_cols=['item_id'],
            agg_cols={'revenue': 'sum'},
            allow_pre_aggs=False
        )

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        # Should NOT use pre-agg
        assert 'pl.scan_parquet' not in code
        assert "self.tables['sales']" in code

    def test_cst_builder_pre_agg(self, datamodel_no_pre_agg):
        """Test _build_pre_agg_cst helper method."""
        result_cst = datamodel_no_pre_agg._build_pre_agg_cst('test_pre_agg')

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        assert code == "pl.scan_parquet(self.pre_agg_directory / 'test_pre_agg.parquet')"

    def test_cst_builder_table_access(self, datamodel_no_pre_agg):
        """Test _build_table_access_cst helper method."""
        result_cst = datamodel_no_pre_agg._build_table_access_cst('test_table')

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        assert code == "self.tables['test_table']"

    def test_cst_builder_join_chain(self, datamodel_no_pre_agg):
        """Test _build_join_chain_cst helper method."""
        join_specs = [
            {
                'right': 'products',
                'left_on': ['item_id'],
                'right_on': ['item_id'],
                'how': 'inner'
            }
        ]

        result_cst = datamodel_no_pre_agg._build_join_chain_cst('sales', join_specs)

        code = cst.Module(body=[cst.Expr(value=result_cst)]).code.strip()

        assert "self.tables['sales']" in code
        assert '.join(' in code
        assert "self.tables['products']" in code
        assert 'left_on' in code  # Relaxed to allow spacing variations
        assert "['item_id']" in code
        assert 'right_on' in code
        assert "'inner'" in code

    def test_column_normalization(self, datamodel_no_pre_agg):
        """Test _normalize_column_name helper method."""
        # Column with prefix - should remain unchanged
        assert datamodel_no_pre_agg._normalize_column_name('sales.revenue', 'sales') == 'sales.revenue'

        # Column without prefix - should add table prefix
        assert datamodel_no_pre_agg._normalize_column_name('revenue', 'sales') == 'sales.revenue'

    def test_columns_match(self, datamodel_no_pre_agg):
        """Test _columns_match helper method."""
        # Same column, different prefixes
        assert datamodel_no_pre_agg._columns_match('sales.revenue', 'revenue')
        assert datamodel_no_pre_agg._columns_match('table1.col', 'table2.col')

        # Different columns
        assert not datamodel_no_pre_agg._columns_match('revenue', 'quantity')
