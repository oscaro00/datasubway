"""
Unit tests for extract_decorator_variable extractor.

Tests the functionality of extracting variable names from @measure(variable_name) decorators.
"""

import pytest

from datasubway.cst.extractors.extract_decorator_variable import extract_decorator_variable_name


class TestExtractDecoratorVariable:
    """Test suite for extract_decorator_variable_name function."""

    def test_basic_extraction(self):
        """Test basic extraction of decorator variable name."""
        code = """
@measure(dm_no_agg)
def total_revenue(qc):
    return dm_no_agg.table('sales')
"""
        result = extract_decorator_variable_name(code, 'total_revenue')
        assert result == 'dm_no_agg'

    def test_extraction_with_dm(self):
        """Test extraction with traditional 'dm' variable name."""
        code = """
@measure(dm)
def revenue_measure(qc):
    return dm.table('sales')
"""
        result = extract_decorator_variable_name(code, 'revenue_measure')
        assert result == 'dm'

    def test_extraction_custom_name(self):
        """Test extraction with custom variable name."""
        code = """
@measure(my_custom_datamodel)
def my_measure(qc):
    return my_custom_datamodel.table('sales')
"""
        result = extract_decorator_variable_name(code, 'my_measure')
        assert result == 'my_custom_datamodel'

    def test_no_decorator(self):
        """Test that function without decorator returns None."""
        code = """
def plain_function(qc):
    return None
"""
        result = extract_decorator_variable_name(code, 'plain_function')
        assert result is None

    def test_no_arguments(self):
        """Test that @measure() without arguments returns None."""
        code = """
@measure()
def empty_decorator(qc):
    return None
"""
        result = extract_decorator_variable_name(code, 'empty_decorator')
        assert result is None

    def test_multiple_decorators(self):
        """Test extraction when function has multiple decorators."""
        code = """
@other_decorator
@measure(dm_custom)
def multi_decorated(qc):
    return dm_custom.table('sales')
"""
        result = extract_decorator_variable_name(code, 'multi_decorated')
        assert result == 'dm_custom'

    def test_measure_decorator_first(self):
        """Test extraction when @measure is the first decorator."""
        code = """
@measure(dm_first)
@other_decorator
def measure_first(qc):
    return dm_first.table('sales')
"""
        result = extract_decorator_variable_name(code, 'measure_first')
        assert result == 'dm_first'

    def test_complex_expression_returns_none(self):
        """Test that complex expressions (not simple names) return None."""
        code = """
@measure(factory.get_dm())
def complex_arg(qc):
    return None
"""
        result = extract_decorator_variable_name(code, 'complex_arg')
        assert result is None

    def test_attribute_access_returns_none(self):
        """Test that attribute access in decorator returns None."""
        code = """
@measure(module.dm)
def attribute_arg(qc):
    return None
"""
        result = extract_decorator_variable_name(code, 'attribute_arg')
        assert result is None

    def test_function_not_found(self):
        """Test that searching for non-existent function returns None."""
        code = """
@measure(dm)
def existing_function(qc):
    return dm.table('sales')
"""
        result = extract_decorator_variable_name(code, 'non_existent_function')
        assert result is None

    def test_different_decorator_name(self):
        """Test that non-@measure decorators return None."""
        code = """
@property
def some_property(self):
    return self._value
"""
        result = extract_decorator_variable_name(code, 'some_property')
        assert result is None

    def test_multiple_functions_extract_specific(self):
        """Test extraction from specific function when multiple exist."""
        code = """
@measure(dm1)
def function_one(qc):
    return dm1.table('sales')

@measure(dm2)
def function_two(qc):
    return dm2.table('products')

def function_three(qc):
    return None
"""
        result1 = extract_decorator_variable_name(code, 'function_one')
        result2 = extract_decorator_variable_name(code, 'function_two')
        result3 = extract_decorator_variable_name(code, 'function_three')

        assert result1 == 'dm1'
        assert result2 == 'dm2'
        assert result3 is None

    def test_indented_function(self):
        """Test extraction from indented function (class method returns None)."""
        # Note: In practice, inspect.getsource() extracts just the function,
        # not the class context, so this scenario is unlikely.
        # Class methods are not supported by the extractor as measures are module-level functions.
        code = """
class MyClass:
    @measure(dm_class)
    def class_measure(self, qc):
        return dm_class.table('sales')
"""
        result = extract_decorator_variable_name(code, 'class_measure')
        # Class methods are not supported - extractor only searches module-level functions
        assert result is None

    def test_with_comments(self):
        """Test extraction with comments in code."""
        code = """
# This is a comment
@measure(dm_commented)  # Inline comment
def commented_function(qc):
    # Another comment
    return dm_commented.table('sales')
"""
        result = extract_decorator_variable_name(code, 'commented_function')
        assert result == 'dm_commented'

    def test_multiple_arguments_extracts_first(self):
        """Test that only first argument is extracted when multiple exist."""
        code = """
@measure(dm_first, some_other_arg)
def multi_arg(qc):
    return dm_first.table('sales')
"""
        result = extract_decorator_variable_name(code, 'multi_arg')
        assert result == 'dm_first'

    def test_invalid_syntax_returns_none(self):
        """Test that invalid Python syntax returns None gracefully."""
        code = """
@measure(dm_invalid
def broken_syntax(qc):
    return None
"""
        result = extract_decorator_variable_name(code, 'broken_syntax')
        assert result is None

    def test_empty_string_returns_none(self):
        """Test that empty source code returns None."""
        code = ""
        result = extract_decorator_variable_name(code, 'any_function')
        assert result is None

    def test_decorator_with_kwargs(self):
        """Test extraction when decorator has keyword arguments (extract positional)."""
        code = """
@measure(dm_kwargs, option=True)
def kwargs_decorator(qc):
    return dm_kwargs.table('sales')
"""
        result = extract_decorator_variable_name(code, 'kwargs_decorator')
        assert result == 'dm_kwargs'

    def test_real_world_example(self):
        """Test with a realistic measure function."""
        code = """
@measure(datamodel_sales)
def total_revenue_by_item(qc):
    return (
        datamodel_sales.table('sales', qc.get('group', []), {'revenue': 'sum'})
        .group_by(Allow('*', context=qc.get('group', [])))
        .agg(pl.col('revenue').sum())
    )
"""
        result = extract_decorator_variable_name(code, 'total_revenue_by_item')
        assert result == 'datamodel_sales'
