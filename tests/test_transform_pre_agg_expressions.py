import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions


class TestTransformPreAggExpressions:
    """Test suite for TransformPreAggExpressions transformer."""

    def test_no_transformation_without_pre_agg(self):
        """Test that code without pre-agg usage is not transformed."""
        code = """
def my_measure():
    return df.agg(pl.col('revenue').sum())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Should be unchanged (no self.pre_agg_directory)
        assert result == code

    def test_sum_aggregation_transform(self):
        """Test that .sum() is transformed to use pre-agg column name."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('revenue').sum())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'revenue-sum\').sum()' in result

    def test_min_aggregation_transform(self):
        """Test that .min() is transformed to use pre-agg column name."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('price').min())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'price-min\').min()' in result

    def test_max_aggregation_transform(self):
        """Test that .max() is transformed to use pre-agg column name."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('price').max())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'price-max\').max()' in result

    def test_count_aggregation_transform(self):
        """Test that .count() is transformed to use pre-agg column name."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('id').count())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'id-count\').count()' in result

    def test_mean_aggregation_decomposed(self):
        """Test that .mean() is decomposed into sum/count formula."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('revenue').mean())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Should be: pl.col('revenue-mean-sum').sum() / pl.col('revenue-mean-count').sum()
        assert 'pl.col(\'revenue-mean-sum\').sum()' in result
        assert 'pl.col(\'revenue-mean-count\').sum()' in result
        assert '/' in result

    def test_std_aggregation_decomposed(self):
        """Test that .std() is decomposed into formula."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('revenue').std())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Should include components: sumsq, sum, count
        assert 'revenue-std-sumsq' in result
        assert 'revenue-std-sum' in result
        assert 'revenue-std-count' in result
        assert '.sqrt()' in result

    def test_var_aggregation_decomposed(self):
        """Test that .var() is decomposed into formula."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('revenue').var())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Should include components: sumsq, sum, count
        assert 'revenue-var-sumsq' in result
        assert 'revenue-var-sum' in result
        assert 'revenue-var-count' in result

    def test_only_transforms_inside_agg(self):
        """Test that only expressions inside .agg() are transformed."""
        code = """
def my_measure():
    return (
        pl.scan_parquet(self.pre_agg_directory / 'test.parquet')
        .filter(pl.col('revenue').sum() > 0)
        .agg(pl.col('revenue').sum())
    )
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Filter should NOT be transformed (outside .agg())
        lines = result.split('\n')
        filter_line = [l for l in lines if '.filter(' in l][0]
        assert 'revenue-sum' not in filter_line
        # But agg should be transformed
        agg_line = [l for l in lines if '.agg(' in l][0]
        assert 'revenue-sum' in agg_line

    def test_column_with_table_prefix_stripped(self):
        """Test that table prefixes are stripped from column names."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('sales.revenue').sum())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Should use 'revenue-sum' not 'sales.revenue-sum'
        assert 'pl.col(\'revenue-sum\').sum()' in result

    def test_multiple_aggregations_in_same_agg(self):
        """Test multiple aggregations in one .agg() call."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(
        pl.col('revenue').sum(),
        pl.col('quantity').mean()
    )
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'revenue-sum\').sum()' in result
        assert 'pl.col(\'quantity-mean-sum\').sum()' in result
        assert 'pl.col(\'quantity-mean-count\').sum()' in result

    def test_aliased_aggregation(self):
        """Test that aliased aggregations are transformed correctly."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(
        pl.col('revenue').sum().alias('total_revenue')
    )
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'revenue-sum\').sum()' in result
        assert '.alias(\'total_revenue\')' in result

    def test_only_transforms_target_function(self):
        """Test that only the target function is transformed."""
        code = """
def other_function():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('x').sum())

def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('y').sum())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # other_function should be unchanged
        assert 'pl.col(\'x\').sum()' in result
        # my_measure should be transformed
        assert 'pl.col(\'y-sum\').sum()' in result

    def test_unsupported_aggregation_unchanged(self):
        """Test that unsupported aggregations are left unchanged."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(
        pl.col('revenue').median()
    )
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # median is not in the supported list, should be unchanged
        assert '.median()' in result

    def test_first_aggregation_transform(self):
        """Test that .first() is transformed."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('name').first())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'name-first\').first()' in result

    def test_last_aggregation_transform(self):
        """Test that .last() is transformed."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('name').last())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'name-last\').last()' in result

    def test_len_aggregation_transform(self):
        """Test that .len() is transformed to count."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('id').len())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        assert 'pl.col(\'id-count\').len()' in result

    def test_complex_chain_only_agg_transformed(self):
        """Test that complex chains only transform .agg() content."""
        code = """
def my_measure():
    return (
        pl.scan_parquet(self.pre_agg_directory / 'test.parquet')
        .select(pl.col('revenue'))
        .group_by('store_id')
        .agg(
            pl.col('revenue').sum(),
            pl.col('revenue').mean()
        )
        .filter(pl.col('revenue-sum') > 1000)
    )
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # .select and .filter should not be transformed
        select_line = [l for l in result.split('\n') if '.select' in l][0]
        assert 'revenue-sum' not in select_line
        # .agg() content should be transformed
        assert 'pl.col(\'revenue-sum\').sum()' in result
        assert 'pl.col(\'revenue-mean-sum\').sum()' in result

    def test_nested_function_calls_in_agg(self):
        """Test complex expressions inside .agg() are handled."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(
        (pl.col('revenue').sum() * 1.1).alias('revenue_with_tax')
    )
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # The inner .sum() should be transformed
        assert 'pl.col(\'revenue-sum\').sum()' in result
        assert '.alias(\'revenue_with_tax\')' in result

    def test_mean_formula_structure(self):
        """Test that mean formula has correct structure."""
        code = """
def my_measure():
    return pl.scan_parquet(self.pre_agg_directory / 'test.parquet').agg(pl.col('x').mean())
"""
        result = transform_pre_agg_expressions(code, 'my_measure')
        # Should be: sum(x-mean-sum) / sum(x-mean-count)
        assert '/' in result
        # Extract just the .agg() arguments to check the formula
        agg_line = [l for l in result.split('\n') if '.agg(' in l][0]
        # Extract content inside .agg(...)
        agg_start = agg_line.index('.agg(') + 5
        agg_content = agg_line[agg_start:].rstrip(')')
        # Check order: numerator / denominator
        div_pos = agg_content.index('/')
        assert agg_content[:div_pos].count('x-mean-sum') == 1
        assert agg_content[div_pos:].count('x-mean-count') == 1
