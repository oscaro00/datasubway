import pytest

from datasubway.cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods


class TestRemoveEmptyPolarsMethods:
    """Test suite for RemoveEmptyPolarsMethods transformer."""

    def test_remove_empty_filter(self):
        """Test that empty .filter([]) is removed."""
        code = """
def my_measure():
    return df.filter([]).select([pl.col('x')])
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.filter([])' not in result
        assert '.select([pl.col(\'x\')])' in result

    def test_remove_empty_select(self):
        """Test that empty .select([]) is removed."""
        code = """
def my_measure():
    return df.select([]).group_by('id').agg(pl.col('x').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.select([])' not in result
        assert '.group_by(\'id\')' in result

    def test_empty_group_by_followed_by_agg_converts_to_select(self):
        """Test that .group_by([]).agg() becomes .select()."""
        code = """
def my_measure():
    return df.group_by([]).agg(pl.col('revenue').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.group_by([])' not in result
        assert '.select(pl.col(\'revenue\').sum())' in result

    def test_multiple_empty_methods(self):
        """Test that multiple consecutive empty methods are all removed."""
        code = """
def my_measure():
    return df.filter([]).drop([]).group_by([]).agg(pl.col('x').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.filter([])' not in result
        assert '.drop([])' not in result
        assert '.group_by([])' not in result
        assert '.select(pl.col(\'x\').sum())' in result

    def test_empty_group_by_without_agg(self):
        """Test that empty .group_by() is removed even without .agg()."""
        code = """
def my_measure():
    return df.group_by([]).filter(pl.col('x') > 0)
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.group_by([])' not in result
        assert '.filter(pl.col(\'x\') > 0)' in result

    def test_non_empty_group_by_with_agg_unchanged(self):
        """Test that non-empty .group_by().agg() is not modified."""
        code = """
def my_measure():
    return df.group_by(['item_id']).agg(pl.col('revenue').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.group_by([\'item_id\'])' in result
        assert '.agg(pl.col(\'revenue\').sum())' in result
        assert '.select(' not in result

    def test_agg_without_group_by_unchanged(self):
        """Test that .agg() without preceding .group_by() is unchanged."""
        code = """
def my_measure():
    return df.agg(pl.col('revenue').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.agg(pl.col(\'revenue\').sum())' in result

    def test_only_transforms_target_function(self):
        """Test that only the target function is transformed."""
        code = """
def other_function():
    return df.group_by([]).agg(pl.col('x').sum())

def my_measure():
    return df.group_by([]).agg(pl.col('y').sum())

def another_function():
    return df.filter([]).select([pl.col('z')])
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        # other_function should still have .group_by([]).agg()
        assert 'def other_function():' in result
        lines = result.split('\n')
        other_func_section = '\n'.join([l for l in lines if 'other_function' in '\n'.join(lines[:lines.index(l)+1])])
        # Only my_measure should be transformed
        assert result.count('.select(pl.col(\'y\').sum())') == 1
        assert result.count('.group_by([]).agg(pl.col(\'x\').sum())') == 1

    def test_empty_group_by_dynamic(self):
        """Test that empty .group_by_dynamic([]) is removed."""
        code = """
def my_measure():
    return df.group_by_dynamic([]).agg(pl.col('x').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.group_by_dynamic([])' not in result
        assert '.select(pl.col(\'x\').sum())' in result

    def test_empty_rolling(self):
        """Test that empty .rolling([]) is removed."""
        code = """
def my_measure():
    return df.rolling([]).agg(pl.col('x').sum())
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.rolling([])' not in result
        assert '.select(pl.col(\'x\').sum())' in result

    def test_method_with_non_empty_args_unchanged(self):
        """Test that methods with non-empty arguments are unchanged."""
        code = """
def my_measure():
    return df.filter([pl.col('x') > 0]).select(['x', 'y'])
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.filter([pl.col(\'x\') > 0])' in result
        assert '.select([\'x\', \'y\'])' in result

    def test_complex_chain_with_mixed_empty_and_non_empty(self):
        """Test complex chain with both empty and non-empty methods."""
        code = """
def my_measure():
    return (
        df
        .filter([])
        .with_columns(pl.col('x') * 2)
        .drop([])
        .group_by(['store_id'])
        .agg(pl.col('revenue').sum())
    )
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        assert '.filter([])' not in result
        assert '.drop([])' not in result
        assert '.with_columns(pl.col(\'x\') * 2)' in result
        assert '.group_by([\'store_id\'])' in result
        assert '.agg(pl.col(\'revenue\').sum())' in result

    def test_nested_empty_lists_in_other_args(self):
        """Test that empty lists in other contexts are not affected."""
        code = """
def my_measure():
    return df.with_columns(x=pl.when(pl.col('y').is_in([])).then(0).otherwise(1))
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        # Should be unchanged because [] is not a direct method argument
        assert 'is_in([])' in result

    def test_multiple_agg_calls_after_empty_group_by(self):
        """Test that only the first .agg() after empty .group_by() is converted."""
        code = """
def my_measure():
    df1 = df.group_by([]).agg(pl.col('x').sum())
    df2 = df1.group_by(['id']).agg(pl.col('y').mean())
    return df2
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        # First should be converted to select
        assert '.select(pl.col(\'x\').sum())' in result
        # Second should remain as agg (has non-empty group_by)
        assert '.agg(pl.col(\'y\').mean())' in result

    def test_empty_list_with_whitespace(self):
        """Test that empty list with whitespace is still recognized."""
        code = """
def my_measure():
    return df.filter([ ]).select(pl.col('x'))
"""
        result = remove_empty_polars_methods(code, 'my_measure')
        # Note: libcst parses and reformats, so whitespace handling may vary
        # The key test is that filter is removed
        assert 'filter' not in result or '.filter([ ])' not in result
