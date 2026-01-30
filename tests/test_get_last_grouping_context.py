"""
Unit tests for get_last_grouping_context visitor.

Tests the functionality of extracting Allow/Exclude calls from the last grouping
method (group_by, group_by_dynamic, rolling), with index_column merged into include.
"""

import pytest

from datasubway.cst.visitors.get_last_grouping_context import get_last_grouping_context


class TestGetLastGroupingContext:
    """Test suite for get_last_grouping_context function."""

    def test_simple_group_by_with_allow(self):
        """Test extraction from simple group_by with Allow."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by(Allow('*', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Allow('*', context=qc.get('group', []))" in result

    def test_simple_group_by_with_exclude(self):
        """Test extraction from simple group_by with Exclude."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by(Exclude('stores.*', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Exclude('stores.*', context=qc.get('group', []))" in result

    def test_group_by_with_allow_and_include(self):
        """Test extraction from group_by with Allow and include parameter."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by(Allow('*', include=['sales.item_id'], context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Allow" in result
        assert "include=['sales.item_id']" in result

    def test_group_by_dynamic_merges_index_column_into_include(self):
        """Test that group_by_dynamic merges index_column into include parameter."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by_dynamic('sales.date', every='1d', period='3d', group_by=Allow('*', context=qc.get('group')))
        .agg(pl.col('revenue').mean())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Allow" in result
        assert "'sales.date'" in result
        assert "include" in result

    def test_group_by_dynamic_with_existing_include_appends(self):
        """Test that index_column is appended to existing include list."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by_dynamic('sales.date', every='1d', group_by=Allow('*', include=['other_col'], context=qc.get('group')))
        .agg(pl.col('revenue').mean())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Allow" in result
        assert "'other_col'" in result
        assert "'sales.date'" in result

    def test_group_by_dynamic_with_index_column_kwarg(self):
        """Test extraction from group_by_dynamic with index_column as keyword arg."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by_dynamic(index_column='date', every='1d', group_by=Allow('*', context=qc.get('group')))
        .agg(pl.col('revenue').mean())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Allow" in result
        assert "'date'" in result
        assert "include" in result

    def test_rolling_merges_index_column_into_include(self):
        """Test that rolling merges index_column into include parameter."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .rolling('date', period='3d', group_by=Allow('*', context=qc.get('group', [])))
        .agg(pl.col('revenue').mean())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert "Allow" in result
        assert "'date'" in result
        assert "include" in result

    def test_multiple_grouping_calls_returns_last(self):
        """Test that multiple grouping calls return the last one by line number."""
        code = """
def store_share_of_revenue(qc):
    numerator = (
        dm.table('sales')
        .group_by(Allow('*', include=['stores.store_id'], context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )

    total_denominator = (
        dm.table('sales')
        .group_by(Exclude('stores.*', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )

    return (
        numerator
        .join(total_denominator, on='item_id', how='cross')
        .group_by(Allow('*', include=['final.col'], context=qc.get('group', [])))
        .agg((pl.col('numerator') / pl.col('total')).alias('share'))
    )
"""
        result = get_last_grouping_context(code, 'store_share_of_revenue')
        assert result is not None
        # Should be the LAST group_by (the return statement one)
        assert "Allow('*', include=['final.col']" in result

    def test_no_grouping_returns_none(self):
        """Test that measure without grouping returns None."""
        code = """
def my_measure(qc):
    return dm.table('sales').select('item_id')
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is None

    def test_grouping_without_allow_exclude_returns_none(self):
        """Test that grouping with literal columns returns None."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by('item_id')
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is None

    def test_allow_in_filter_but_not_grouping(self):
        """Test that Allow in filter() doesn't count if group_by has no Allow/Exclude."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .filter(Allow('*', context=qc.get('filter')))
        .group_by('item_id')
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        # Should return None since the group_by doesn't have Allow/Exclude
        assert result is None

    def test_function_not_found_returns_none(self):
        """Test that searching for non-existent function returns None."""
        code = """
def existing_function(qc):
    return (
        dm.table('sales')
        .group_by(Allow('*', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'non_existent_function')
        assert result is None

    def test_invalid_syntax_returns_none(self):
        """Test that invalid Python syntax returns None gracefully."""
        code = """
def broken_syntax(qc
    return None
"""
        result = get_last_grouping_context(code, 'broken_syntax')
        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty source code returns None."""
        code = ""
        result = get_last_grouping_context(code, 'any_function')
        assert result is None

    def test_group_by_with_list_returns_none(self):
        """Test that group_by with list of columns returns None."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by(['item_id', 'store_id'])
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is None

    def test_real_world_rolling_measure(self):
        """Test with a realistic rolling window measure."""
        code = """
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
"""
        result = get_last_grouping_context(code, 'rolling_3_day_average_revenue')
        assert result is not None
        assert "Allow" in result
        assert "'sales.date'" in result
        assert "include" in result

    def test_nested_function_only_extracts_target(self):
        """Test that only the target function is analyzed."""
        code = """
def helper_function(qc):
    return (
        dm.table('sales')
        .group_by(Allow('helper_pattern', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )

def main_measure(qc):
    return (
        dm.table('sales')
        .group_by(Allow('main_pattern', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )
"""
        result = get_last_grouping_context(code, 'main_measure')
        assert result is not None
        assert 'main_pattern' in result
        assert 'helper_pattern' not in result

    def test_result_is_string_not_tuple(self):
        """Test that the result is a string, not a tuple."""
        code = """
def my_measure(qc):
    return (
        dm.table('sales')
        .group_by_dynamic('sales.date', every='1d', group_by=Allow('*', context=qc.get('group')))
        .agg(pl.col('revenue').mean())
    )
"""
        result = get_last_grouping_context(code, 'my_measure')
        assert result is not None
        assert isinstance(result, str)
        assert not isinstance(result, tuple)
