"""
Unit tests for inject_table_parameters transformer.

Tests the functionality of injecting parameters into table() calls,
including support for custom variable names.
"""

import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from cst.transformers.inject_table_parameters import inject_table_parameters


class TestInjectTableParameters:
    """Test suite for inject_table_parameters function."""

    # ==========================================
    # Default Variable Name Tests
    # ==========================================

    def test_recognizes_default_dm_variable(self):
        """Test that the default 'dm' variable name is recognized."""
        code = "def f():\n    return dm.table('t').group_by([]).agg([])"
        result = inject_table_parameters(code, 'f')
        assert "dm.table('t', [], {}," in result

    def test_recognizes_default_self_variable(self):
        """Test that the default 'self' variable name is recognized."""
        code = "def f(self):\n    return self.table('t').group_by([]).agg([])"
        result = inject_table_parameters(code, 'f')
        assert "self.table('t', [], {}," in result

    def test_recognizes_default_data_model_variable(self):
        """Test that the default 'data_model' variable name is recognized."""
        code = "def f():\n    return data_model.table('t').group_by([]).agg([])"
        result = inject_table_parameters(code, 'f')
        assert "data_model.table('t', [], {}," in result

    # ==========================================
    # Custom Variable Name Tests
    # ==========================================

    def test_recognizes_custom_variable_name(self):
        """Test that custom variable names are recognized via runtime_context."""
        code = "def f():\n    return my_model.table('t').group_by([]).agg([])"
        result = inject_table_parameters(code, 'f', {'valid_var_names': ['my_model']})
        assert "my_model.table('t', [], {}," in result

    def test_ignores_unknown_variable_name(self):
        """Test that unknown variable names are NOT transformed."""
        code = "def f():\n    return unknown.table('t').group_by([]).agg([])"
        result = inject_table_parameters(code, 'f')  # No valid_var_names override
        # Should be unchanged - no parameters injected
        assert "unknown.table('t')" in result
        assert "unknown.table('t', [], {}," not in result

    def test_multiple_custom_variable_names(self):
        """Test that multiple custom variable names can be specified."""
        code = "def f():\n    return custom_dm.table('t').agg([])"
        result = inject_table_parameters(code, 'f', {'valid_var_names': ['custom_dm', 'other_dm']})
        assert "custom_dm.table('t', [], {}," in result

    def test_custom_variable_overrides_defaults(self):
        """Test that providing valid_var_names replaces defaults, not appends."""
        code = "def f():\n    return dm.table('t').group_by([]).agg([])"
        # Only recognize 'custom_dm', NOT the default 'dm'
        result = inject_table_parameters(code, 'f', {'valid_var_names': ['custom_dm']})
        # Should NOT transform because 'dm' is not in valid_var_names
        assert "dm.table('t')" in result
        assert "dm.table('t', [], {}," not in result

    # ==========================================
    # Column Extraction Tests
    # ==========================================

    def test_extracts_group_by_columns(self):
        """Test that group_by columns are correctly extracted."""
        code = """
def f():
    return dm.table('sales').group_by([pl.col('product_id')]).agg([])
"""
        result = inject_table_parameters(code, 'f')
        assert "'product_id'" in result

    def test_extracts_agg_columns(self):
        """Test that aggregation columns are correctly extracted."""
        code = """
def f():
    return dm.table('sales').group_by([]).agg([pl.col('revenue').sum()])
"""
        result = inject_table_parameters(code, 'f')
        assert "'revenue': 'sum'" in result

    # ==========================================
    # Edge Cases
    # ==========================================

    def test_only_transforms_target_function(self):
        """Test that only the specified function is transformed."""
        code = """
def f():
    return dm.table('t').group_by([]).agg([])

def g():
    return dm.table('t').group_by([]).agg([])
"""
        result = inject_table_parameters(code, 'f')
        # f should be transformed
        assert "dm.table('t', [], {}," in result
        # Count occurrences - should only have one transformation
        assert result.count("dm.table('t', [], {},") == 1

    def test_handles_method_chain_with_filter(self):
        """Test that method chains with filter are handled."""
        code = """
def f():
    return dm.table('sales').filter(pl.col('active')).group_by([]).agg([])
"""
        result = inject_table_parameters(code, 'f')
        assert "dm.table('sales', [], {}," in result

    def test_handles_empty_groups_and_aggs(self):
        """Test transformation with empty group_by and agg."""
        code = "def f():\n    return dm.table('t').group_by([]).agg([])"
        result = inject_table_parameters(code, 'f')
        assert "dm.table('t', [], {}," in result
