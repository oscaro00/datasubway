"""Utility functions for column name normalization and resolution.

These are pure functions that handle column naming conventions:
- Adding/stripping table prefixes
- Matching columns ignoring prefixes
- Resolving columns to their source tables
- Determining required joins for column sets
"""
from typing import List, Dict, Any
import polars as pl


def normalize_column_name(col: str, table: str) -> str:
    """Add table prefix to column if not present.

    Args:
        col: Column name (may or may not have table prefix)
        table: Table name to use as prefix if not present

    Returns:
        Column name with table prefix (e.g., 'table.column')

    Example:
        >>> normalize_column_name('revenue', 'sales')
        'sales.revenue'
        >>> normalize_column_name('sales.revenue', 'orders')
        'sales.revenue'  # Already has prefix, unchanged
    """
    if '.' in col:
        return col
    return f"{table}.{col}"


def columns_match(col1: str, col2: str) -> bool:
    """Check if two columns refer to same column (ignoring table prefix).

    Args:
        col1: First column name (may have table prefix)
        col2: Second column name (may have table prefix)

    Returns:
        True if column names match after stripping table prefixes

    Example:
        >>> columns_match('sales.revenue', 'orders.revenue')
        True
        >>> columns_match('revenue', 'sales.revenue')
        True
    """
    clean1 = col1.split('.')[-1]
    clean2 = col2.split('.')[-1]
    return clean1 == clean2


def resolve_column_table(
    col: str,
    base_table: str,
    table_schemas: Dict[str, List[str]]
) -> str:
    """
    Add table prefix to column if missing by looking up table schemas.

    When columns are provided without table prefixes (e.g., 'category' instead of
    'products.category'), this function searches through table schemas to find which
    table contains the column.

    Args:
        col: Column name (may or may not have table prefix)
        base_table: Base table to check first (optimization)
        table_schemas: Dict mapping table names to lists of column names

    Returns:
        Column name with table prefix (e.g., 'products.category')

    Example:
        >>> schemas = {'sales': ['id', 'amount'], 'products': ['id', 'category']}
        >>> resolve_column_table('category', 'sales', schemas)
        'products.category'  # Found in products table schema
    """
    # Already has table prefix - return as-is
    if '.' in col:
        return col

    # Check if column exists in base table first (common case)
    if base_table in table_schemas:
        if col in table_schemas[base_table]:
            return f"{base_table}.{col}"

    # Search other tables for the column
    for table_name, schema in table_schemas.items():
        if col in schema:
            return f"{table_name}.{col}"

    # Column not found in any schema - assume it belongs to base table
    # This allows for dynamic columns or columns not in schema
    return f"{base_table}.{col}"


def get_join_specs_for_columns(
    columns: List[str],
    base_table: str,
    join_lookup: Dict[str, Dict[str, Any]],
    tables: Dict[str, pl.LazyFrame]
) -> List[Dict]:
    """
    Determine join specs needed for given columns.

    Analyzes column prefixes to determine which tables are needed,
    then collects join specifications to connect them.

    Args:
        columns: List of columns (may include table prefixes)
        base_table: Primary table to start from
        join_lookup: Pre-computed join paths between tables
        tables: Dict of available tables (for validation)

    Returns:
        List of join specifications to build join chain, or empty list if single table

    Raises:
        ValueError: If joins don't exist or tables not found

    Example:
        >>> specs = get_join_specs_for_columns(
        ...     ['sales.amount', 'products.category'],
        ...     'sales',
        ...     join_lookup,
        ...     tables
        ... )
    """
    # Extract unique table names
    tables_needed = {base_table}

    for col in columns:
        if '.' in col:
            table_name = col.split('.')[0]
            if table_name in tables:
                tables_needed.add(table_name)
            else:
                raise ValueError(f"Column '{col}' references unknown table '{table_name}'")

    # Single table - no joins needed
    if len(tables_needed) == 1:
        return []

    # Multiple tables - collect all join specs
    all_join_specs = []
    remaining_tables = tables_needed - {base_table}

    for target_table in remaining_tables:
        if base_table not in join_lookup:
            raise ValueError(f"No joins defined from '{base_table}'")

        if target_table not in join_lookup[base_table]:
            raise ValueError(
                f"No join path from '{base_table}' to '{target_table}'. "
                f"Available: {list(join_lookup[base_table].keys())}"
            )

        join_info = join_lookup[base_table][target_table]
        all_join_specs.extend(join_info['join_specs'])

    return all_join_specs
