"""
Extractor to get the variable name from @measure(variable_name) decorator.

This module provides a function that:
1. Parses the source code with libcst
2. Finds the target function definition
3. Extracts the first argument from @measure(...) decorator
4. Returns the variable name as a string

Example:
    >>> code = '''
    ... @measure(dm_no_agg)
    ... def total_revenue(qc):
    ...     return dm_no_agg.table('sales')
    ... '''
    >>> var_name = extract_decorator_variable_name(code, 'total_revenue')
    >>> print(var_name)  # 'dm_no_agg'
"""

from typing import Optional
import libcst as cst


def extract_decorator_variable_name(
    source_code: str,
    function_name: str
) -> Optional[str]:
    """
    Extract the variable name from @measure(variable_name) decorator.

    This function parses the source code and looks for a decorator pattern
    matching @measure(variable_name) on the target function.

    Args:
        source_code: Full source code including decorator
        function_name: Name of the function to extract from

    Returns:
        Variable name used in @measure(variable_name), or None if:
        - No @measure decorator found
        - Decorator has no arguments
        - First argument is not a simple Name node
        - Parsing fails

    Examples:
        >>> code = '@measure(dm_no_agg)\\ndef func(qc): pass'
        >>> extract_decorator_variable_name(code, 'func')
        'dm_no_agg'

        >>> code = 'def func(qc): pass'  # No decorator
        >>> extract_decorator_variable_name(code, 'func')
        None

        >>> code = '@measure()\\ndef func(qc): pass'  # No arguments
        >>> extract_decorator_variable_name(code, 'func')
        None
    """
    try:
        module = cst.parse_module(source_code)

        # Find the target function
        for stmt in module.body:
            if not isinstance(stmt, cst.FunctionDef):
                continue

            if stmt.name.value != function_name:
                continue

            # Look through decorators for @measure(...)
            for decorator in stmt.decorators:
                call_node = decorator.decorator

                # Check if decorator is a Call node (e.g., @measure(...))
                if not isinstance(call_node, cst.Call):
                    continue

                # Check if the function being called is 'measure'
                if not (isinstance(call_node.func, cst.Name) and
                       call_node.func.value == 'measure'):
                    continue

                # Extract first argument if it's a simple name
                if len(call_node.args) > 0:
                    first_arg = call_node.args[0]
                    if isinstance(first_arg.value, cst.Name):
                        return first_arg.value.value

                # Found @measure decorator but couldn't extract variable
                return None

        # Function not found or no @measure decorator
        return None

    except Exception:
        # Parsing failed - return None to maintain backward compatibility
        return None
