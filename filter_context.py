from typing import TypedDict, Union, Literal, List, Tuple, Any, TypeAlias

# Operator type definitions
ComparisonOperator: TypeAlias = Literal['=', '!=', '>', '<', '>=', '<=']
StringOperator: TypeAlias = Literal['LIKE', 'CONTAINS', 'STARTS_WITH', 'ENDS_WITH']
SetOperator: TypeAlias = Literal['IN', 'NOT IN']
NullOperator: TypeAlias = Literal['IS NULL', 'IS NOT NULL']
FilterOperator: TypeAlias = Union[ComparisonOperator, StringOperator, SetOperator, NullOperator]

# Core filter types
FilterCondition: TypeAlias = Tuple[str, FilterOperator, Any]


class AndFilter(TypedDict):
    """
    Represents an AND logical operation on multiple filter conditions.

    All conditions within the 'AND' list must be true for the filter to match.

    Example:
        {
            'AND': [
                ('geography.country', '=', 'US'),
                ('revenue', '>', 1000)
            ]
        }
    """
    AND: List['FilterExpression']


class OrFilter(TypedDict):
    """
    Represents an OR logical operation on multiple filter conditions.

    At least one condition within the 'OR' list must be true for the filter to match.

    Example:
        {
            'OR': [
                ('geography.country', '=', 'US'),
                ('geography.country', '=', 'CA')
            ]
        }
    """
    OR: List['FilterExpression']


# Recursive union type - the core filter expression type
FilterExpression: TypeAlias = Union[FilterCondition, AndFilter, OrFilter]


class Filter:
    """
    Encapsulates filter validation and conversion logic.

    This class:
    1. Validates filter structure recursively
    2. Validates operators and values
    3. Validates column names are in 'table_name.column_name' format

    Example:
        filter = Filter({
            'OR': [
                ('geography.country', '=', 'US'),
                {
                    'AND': [
                        ('revenue', '>', 1000),
                        ('status', '!=', 'deleted')
                    ]
                }
            ]
        })
    """

    def __init__(self, filter_expr: FilterExpression) -> None:
        """Initialize and validate the filter expression."""
        self.filter_expr = filter_expr
        self.validate()

    def validate(self) -> None:
        """Recursively validate the filter structure."""
        self._validate_expression(self.filter_expr)

    def _validate_expression(self, expr: FilterExpression) -> None:
        """
        Recursively validate a filter expression.

        Handles three cases:
        1. Tuple (FilterCondition) - validate condition
        2. Dict with 'AND' key - validate AND filter
        3. Dict with 'OR' key - validate OR filter

        Raises:
            TypeError: If expression is not a tuple or dict
            ValueError: If dict has invalid keys or empty lists
        """
        # Case 1: FilterCondition (tuple)
        if isinstance(expr, tuple):
            self._validate_condition(expr)
            return

        # Case 2 & 3: AndFilter or OrFilter (dict)
        if isinstance(expr, dict):
            # Ensure exactly one key: 'AND' or 'OR'
            if len(expr) != 1:
                raise ValueError(
                    f"Filter dict must have exactly one key ('AND' or 'OR'), got: {list(expr.keys())}"
                )

            key = next(iter(expr.keys()))

            if key not in ('AND', 'OR'):
                raise ValueError(
                    f"Filter dict key must be 'AND' or 'OR', got: '{key}'"
                )

            conditions = expr[key]

            # Validate list of conditions
            if not isinstance(conditions, list):
                raise TypeError(
                    f"{key} value must be a list of filter expressions, got: {type(conditions).__name__}"
                )

            if len(conditions) == 0:
                raise ValueError(
                    f"{key} list cannot be empty"
                )

            # Recursively validate each sub-expression
            for i, sub_expr in enumerate(conditions):
                try:
                    self._validate_expression(sub_expr)
                except (TypeError, ValueError) as e:
                    raise ValueError(
                        f"Invalid filter expression at {key}[{i}]: {e}"
                    ) from e

            return

        # Invalid type
        raise TypeError(
            f"Filter expression must be a tuple or dict, got: {type(expr).__name__}"
        )

    def _validate_condition(self, condition: FilterCondition) -> None:
        """
        Validate a single filter condition tuple.

        Validates:
        1. Tuple has exactly 3 elements
        2. Column name is a string in 'table_name.column_name' format
        3. Operator is valid
        4. Value is appropriate for the operator

        Raises:
            TypeError: If condition is not a tuple or has wrong types
            ValueError: If tuple has wrong length, invalid column format, or invalid operator
        """
        if not isinstance(condition, tuple):
            raise TypeError(
                f"Filter condition must be a tuple, got: {type(condition).__name__}"
            )

        if len(condition) != 3:
            raise ValueError(
                f"Filter condition must have exactly 3 elements (column, operator, value), got {len(condition)}"
            )

        column, operator, value = condition

        # Validate column name
        if not isinstance(column, str):
            raise TypeError(
                f"Column name must be a string, got: {type(column).__name__}"
            )

        if not column:
            raise ValueError("Column name cannot be empty")

        # Validate column format: table_name.column_name
        if '.' not in column:
            raise ValueError(
                f"Column '{column}' must be in 'table_name.column_name' format"
            )

        parts = column.split('.')
        if len(parts) != 2:
            raise ValueError(
                f"Column '{column}' must have exactly one dot (table_name.column_name)"
            )

        if not parts[0] or not parts[1]:
            raise ValueError(
                f"Column '{column}' has empty table or column name"
            )

        # Validate operator
        valid_operators = {
            '=', '!=', '>', '<', '>=', '<=',  # Comparison
            'LIKE', 'CONTAINS', 'STARTS_WITH', 'ENDS_WITH',  # String
            'IN', 'NOT IN',  # Set
            'IS NULL', 'IS NOT NULL'  # Null
        }

        if operator not in valid_operators:
            raise ValueError(
                f"Invalid operator '{operator}'. Valid operators: {sorted(valid_operators)}"
            )

        # Validate value based on operator
        self._validate_operator_value(operator, value, column)

    def _validate_operator_value(self, operator: FilterOperator, value: Any, column: str) -> None:
        """
        Validate that the value is appropriate for the given operator.

        Raises:
            TypeError: If value type doesn't match operator requirements
            ValueError: If value is invalid for the operator
        """
        # Null operators - value should be None
        if operator in ('IS NULL', 'IS NOT NULL'):
            if value is not None:
                raise ValueError(
                    f"Column '{column}': {operator} operator should have None as value, got: {value}"
                )
            return

        # Set operators - value should be a list/tuple
        if operator in ('IN', 'NOT IN'):
            if not isinstance(value, (list, tuple)):
                raise TypeError(
                    f"Column '{column}': {operator} operator requires a list or tuple, got: {type(value).__name__}"
                )
            if len(value) == 0:
                raise ValueError(
                    f"Column '{column}': {operator} operator requires a non-empty list"
                )
            return

        # String operators - value should be a string
        if operator in ('LIKE', 'CONTAINS', 'STARTS_WITH', 'ENDS_WITH'):
            if not isinstance(value, str):
                raise TypeError(
                    f"Column '{column}': {operator} operator requires a string value, got: {type(value).__name__}"
                )
            return

        # Comparison operators - value can be many types, but not None
        if operator in ('=', '!=', '>', '<', '>=', '<='):
            if value is None:
                raise ValueError(
                    f"Column '{column}': {operator} operator cannot compare with None (use 'IS NULL' or 'IS NOT NULL')"
                )
            # Allow numbers, strings, dates, etc.
            return

    def get_columns(self) -> List[str]:
        """
        Extract all column references from the filter.

        Returns:
            List of column names in 'table_name.column_name' format
        """
        columns = []
        self._collect_columns(self.filter_expr, columns)
        return columns

    def _collect_columns(self, expr: FilterExpression, columns: List[str]) -> None:
        """
        Recursively collect column names from the filter expression.

        Args:
            expr: The filter expression to process
            columns: List to accumulate column names
        """
        # Case 1: FilterCondition (tuple)
        if isinstance(expr, tuple):
            column = expr[0]
            if column not in columns:
                columns.append(column)
            return

        # Case 2 & 3: AndFilter or OrFilter (dict)
        if isinstance(expr, dict):
            key = next(iter(expr.keys()))
            conditions = expr[key]

            for sub_expr in conditions:
                self._collect_columns(sub_expr, columns)
