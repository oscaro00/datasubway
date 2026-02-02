"""Security validation utilities for query context inputs.

This module provides functions to validate that user-provided query context
values are safe literals and do not contain code injection attempts.
"""

import re
from typing import Any


# Safe primitive types that cannot execute code
SAFE_TYPES = (str, int, float, bool, type(None))

# Pattern for safe identifiers: letters, numbers, underscores, with optional dot-separated parts
# Examples: "column", "table.column", "sales.item_id"
SAFE_IDENTIFIER_PATTERN = re.compile(r'^[a-zA-Z0-9_]+(\.[a-zA-Z0-9_]+)*$')

# Allowed filter/sort operators and direction values
# These are known safe values that don't match the identifier pattern but are valid
ALLOWED_OPERATORS = frozenset({
    # Comparison operators
    '=', '!=', '>', '<', '>=', '<=',
    # Logical operators (used as dict keys)
    'AND', 'OR',
    # Special operators
    'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL',
    'LIKE', 'CONTAINS', 'STARTS_WITH', 'ENDS_WITH',
    # Sort directions
    'asc', 'desc', 'ASC', 'DESC',
})


def is_safe_literal(value: Any) -> bool:
    """Check if value is a safe literal (no code objects, callables, etc.).

    Safe literals are:
    - Primitive types: str, int, float, bool, None
    - Lists/tuples containing only safe literals
    - Dicts with string keys and safe literal values

    Args:
        value: Any Python value to check

    Returns:
        True if the value is a safe literal, False otherwise
    """
    if isinstance(value, SAFE_TYPES):
        return True
    if isinstance(value, (list, tuple)):
        return all(is_safe_literal(v) for v in value)
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and is_safe_literal(v)
            for k, v in value.items()
        )
    return False


def is_safe_identifier(value: str) -> bool:
    """Check if string is a safe identifier.

    Safe identifiers contain only:
    - Letters (a-z, A-Z)
    - Numbers (0-9)
    - Underscores (_)
    - Dots (.) for table.column notation

    Args:
        value: String to validate

    Returns:
        True if the string is a safe identifier, False otherwise
    """
    if not isinstance(value, str):
        return False
    return bool(SAFE_IDENTIFIER_PATTERN.match(value))


def is_safe_string(value: str) -> bool:
    """Check if string is safe (either a safe identifier or an allowed operator).

    Args:
        value: String to validate

    Returns:
        True if the string is safe, False otherwise
    """
    if not isinstance(value, str):
        return False
    # Check if it's an allowed operator first (faster for common cases)
    if value in ALLOWED_OPERATORS:
        return True
    # Then check if it's a valid identifier
    return bool(SAFE_IDENTIFIER_PATTERN.match(value))


def validate_safe_context(context: dict, path: str = "root") -> None:
    """Recursively validate that all values in context are safe literals.

    This ensures no callables, code objects, or other potentially dangerous
    types are present in the context.

    Args:
        context: Dictionary to validate
        path: Current path for error messages (used in recursion)

    Raises:
        ValueError: If an unsafe value is found
    """
    for key, value in context.items():
        current_path = f"{path}.{key}"
        if not is_safe_literal(value):
            raise ValueError(
                f"Unsafe value at {current_path}: "
                f"expected literal type, got {type(value).__name__}"
            )


def validate_all_strings_are_safe(value: Any, path: str = "root") -> None:
    """Recursively validate that ALL string values are safe.

    Safe strings are either:
    - Safe identifiers (letters, numbers, underscores, dots)
    - Allowed operators (=, !=, >, <, AND, OR, etc.)

    This applies strict validation to all strings, including:
    - Column names
    - Table names
    - Filter operators and values
    - Any other string in the context

    Args:
        value: Value to validate (will recurse into lists/dicts)
        path: Current path for error messages (used in recursion)

    Raises:
        ValueError: If an unsafe string is found
    """
    if isinstance(value, str):
        if not is_safe_string(value):
            raise ValueError(
                f"Invalid string at {path}: '{value}' - "
                "must contain only letters, numbers, underscores, dots, or be a valid operator"
            )
    elif isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            validate_all_strings_are_safe(item, f"{path}[{i}]")
    elif isinstance(value, dict):
        for k, v in value.items():
            # Validate dict keys (allow operators like 'AND', 'OR' as keys)
            if isinstance(k, str) and not is_safe_string(k):
                raise ValueError(
                    f"Invalid key at {path}: '{k}' - "
                    "must contain only letters, numbers, underscores, dots, or be a valid operator"
                )
            # Validate dict values
            validate_all_strings_are_safe(v, f"{path}.{k}")
    # int, float, bool, None are safe and don't need string validation
