from typing import Self, Set, List, Dict, Union, Literal, Tuple, Any, Optional
import re
import polars as pl


def extract_table_columns(column_list: List[str]) -> Set:
    table_columns = set()
    
    for column in column_list:
        table_column = re.findall(r'^([\w_*]+)\.?([\w_*]+)?$', column)

        if table_column[0] == ('*', ''):
            table_columns = {table_column[0]}
            return table_columns

        table_columns.add(table_column[0])
    return table_columns


def flatten_list(lst: List) -> List:
    output_list = []
    for item in lst:
        if isinstance(item, list):
            output_list.extend(item)
        else:
            output_list.append(item)
    return output_list


def _detect_context_type(context_value: Any) -> Literal['group', 'sort', 'filter', 'unknown']:
    """
    Detect context type by examining structure.

    Returns:
        'filter': dict with 'AND'/'OR' key, or tuple (column, operator, value)
        'sort': list of tuples where second element is 'asc'/'desc'
        'group': list of strings (table.column format)
        'unknown': fallback
    """
    # Filter: dict or FilterExpression tuple
    if isinstance(context_value, dict):
        if 'AND' in context_value or 'OR' in context_value:
            return 'filter'

    if isinstance(context_value, tuple) and len(context_value) == 3:
        return 'filter'

    # List-based contexts
    if isinstance(context_value, list):
        if len(context_value) == 0:
            return 'unknown'

        first = context_value[0]

        # Sort: [('df.col', 'asc'), ...]
        if isinstance(first, tuple) and len(first) == 2:
            if isinstance(first[1], str) and first[1].lower() in ('asc', 'desc'):
                return 'sort'

        # Group: ['df.col1', 'df.col2', ...]
        if isinstance(first, str):
            return 'group'

    return 'unknown'


# Operator mapping for filter expressions to Polars
OPERATOR_MAP = {
    '=': lambda col, val: col == val,
    '!=': lambda col, val: col != val,
    '>': lambda col, val: col > val,
    '<': lambda col, val: col < val,
    '>=': lambda col, val: col >= val,
    '<=': lambda col, val: col <= val,
    'IN': lambda col, val: col.is_in(val),
    'NOT IN': lambda col, val: ~col.is_in(val),
    'LIKE': lambda col, val: col.str.contains(val.replace('%', '.*').replace('_', '.')),
    'CONTAINS': lambda col, val: col.str.contains(val),
    'STARTS_WITH': lambda col, val: col.str.starts_with(val),
    'ENDS_WITH': lambda col, val: col.str.ends_with(val),
    'IS NULL': lambda col, val: col.is_null(),
    'IS NOT NULL': lambda col, val: col.is_not_null(),
}


class Allow:
    raw_columns = []
    table_columns = set()
    include_columns = []
    context_columns = []
    raw_context = None
    context_type = 'unknown'

    def __init__(self: Self, *columns: str, **kwargs: Dict) -> Self:
        self.raw_columns = list(columns)
        self.table_columns = extract_table_columns(self.raw_columns)

        include_raw = []
        context_raw = None

        for key, val in kwargs.items():
            if key == 'include':
                if isinstance(val, list):
                    include_raw = flatten_list(val)
                elif isinstance(val, str):
                    include_raw = [val]
                elif isinstance(val, pl.Expr):
                    # Store pl.Expr as-is (for filter context)
                    include_raw = [val]

            if key == 'context':
                # Store raw context without modification
                context_raw = val

        # Extract table columns from strings only, preserve pl.Expr objects
        string_columns = [item for item in include_raw if isinstance(item, str)]
        expr_objects = [item for item in include_raw if isinstance(item, pl.Expr)]
        self.include_columns = list(extract_table_columns(string_columns)) + expr_objects

        # Store raw context and detect type
        if context_raw is not None:
            # Unwrap single-element lists (backward compatibility)
            if isinstance(context_raw, list) and len(context_raw) == 1 and isinstance(context_raw[0], (list, dict)):
                context_raw = context_raw[0]

            self.raw_context = context_raw
            self.context_type = _detect_context_type(context_raw)

            # Legacy: still extract columns for context_columns (backward compatibility)
            if isinstance(context_raw, list) and len(context_raw) > 0 and isinstance(context_raw[0], str):
                # Group context - simple list of strings
                self.context_columns = list(extract_table_columns(context_raw))
            else:
                # For other types, initialize empty for now (will be populated by filtering methods)
                self.context_columns = []
        else:
            self.raw_context = None
            self.context_type = 'unknown'
            self.context_columns = []

    def _parse_column(self, column: str) -> Tuple[str, str]:
        """
        Parse 'table.column' into (table, column) tuple.

        Format matches extract_table_columns():
        - 'table.column' → ('table', 'column')
        - 'column' → ('column', '')
        """
        if '.' in column:
            parts = column.split('.', 1)
            return (parts[0], parts[1])
        return (column, '')

    def _should_include_column(self, tbl_col: Tuple[str, str]) -> bool:
        """
        Check if column should be included based on Allow patterns.

        For Allow: include if matches patterns OR if '*' is present
        """
        tbl, col = tbl_col

        # Check if column matches any pattern
        matches = {('*', ''), (tbl, '*'), (tbl, col)}.intersection(self.table_columns)

        # Allow: include if matches OR if '*' is present
        return bool(matches) or ('*', '') in self.table_columns

    def _filter_group_context(self, group_list: List[str]) -> List[str]:
        """Apply Allow patterns to group context."""
        filtered = []
        for column in group_list:
            tbl_col = self._parse_column(column)
            if self._should_include_column(tbl_col):
                filtered.append(column)
        return filtered

    def _filter_sort_context(self, sort_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Apply Allow patterns to sort context."""
        filtered = []
        for column, direction in sort_list:
            tbl_col = self._parse_column(column)
            if self._should_include_column(tbl_col):
                filtered.append((column, direction))
        return filtered

    def _filter_filter_context(self, filter_expr: Any) -> Optional[Any]:
        """
        Apply Allow patterns to filter context recursively.

        Returns filtered FilterExpression or None if all filtered out.
        Simplifies single-element AND/OR.
        """
        # Simple condition (tuple)
        if isinstance(filter_expr, tuple):
            column, operator, value = filter_expr
            tbl_col = self._parse_column(column)
            return filter_expr if self._should_include_column(tbl_col) else None

        # AND/OR dict
        if isinstance(filter_expr, dict):
            key = next(iter(filter_expr.keys()))
            conditions = filter_expr[key]

            # Recursively filter each condition
            filtered_conditions = []
            for cond in conditions:
                filtered = self._filter_filter_context(cond)
                if filtered is not None:
                    filtered_conditions.append(filtered)

            # Handle results
            if len(filtered_conditions) == 0:
                return None
            elif len(filtered_conditions) == 1:
                # Simplify single-element AND/OR
                return filtered_conditions[0]
            else:
                return {key: filtered_conditions}

        return None

    def _group_to_polars(self, group_list: List[str]) -> List[pl.Expr]:
        """Convert ['df.col1', 'df.col2'] → [pl.col('df.col1'), pl.col('df.col2')] (preserves table prefix)"""
        return [pl.col(column) for column in group_list]

    def _sort_to_polars(self, sort_list: List[Tuple[str, str]]) -> List[pl.Expr]:
        """
        Convert [('df.col', 'desc')] → [pl.col('df.col')] (preserves table prefix)

        Note: Polars .sort() doesn't support .desc()/.asc() on expressions.
        Direction info is lost - use with separate descending parameter.
        """
        return [pl.col(column) for column, direction in sort_list]

    def _filter_to_polars(self, filter_expr: Any) -> pl.Expr:
        """
        Convert FilterExpression to Polars boolean expression.

        Examples:
            ('df.col', '=', 1) → pl.col('col') == 1
            {'AND': [cond1, cond2]} → cond1_expr & cond2_expr
            {'OR': [cond1, cond2]} → cond1_expr | cond2_expr
        """
        # Simple condition (tuple)
        if isinstance(filter_expr, tuple):
            column, operator, value = filter_expr
            # Strip table prefix from column name (e.g., 'sales.item_id' → 'item_id')
            column_name = column.split('.')[-1] if '.' in column else column
            col_expr = pl.col(column_name)

            if operator not in OPERATOR_MAP:
                raise ValueError(f"Unsupported operator: {operator}")

            return OPERATOR_MAP[operator](col_expr, value)

        # AND/OR dict
        if isinstance(filter_expr, dict):
            key = next(iter(filter_expr.keys()))
            conditions = filter_expr[key]

            if key == 'AND':
                # Combine with &
                result = self._filter_to_polars(conditions[0])
                for cond in conditions[1:]:
                    result = result & self._filter_to_polars(cond)
                return result

            elif key == 'OR':
                # Combine with |
                result = self._filter_to_polars(conditions[0])
                for cond in conditions[1:]:
                    result = result | self._filter_to_polars(cond)
                return result

        raise ValueError(f"Invalid filter expression: {filter_expr}")

    def get_columns(self: Self) -> Set:
        return self.table_columns

    def get_include(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.include_columns]

    def get_context(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.context_columns]

    def get_relevant_columns(self: Self, output_type: Literal['tbl_col', 'col', 'polar_col', 'polar_expr'] = 'tbl_col') -> Union[List[str], List[pl.Expr], pl.Expr]:
        """
        Get filtered columns/expressions from context.

        Args:
            output_type:
                'tbl_col': ['df.col1', ...] (default, backward compatible)
                'col': ['col1', ...] (strip table prefix)
                'polar_col': [pl.col('df.col1'), ...] (for group/sort)
                'polar_expr': pl.Expr (for filter, combined expression)

        Returns:
            Filtered and converted columns/expressions based on context type
        """
        # No context - handle include-only case
        if self.raw_context is None:
            return self._handle_include_only(output_type)

        # Dispatch based on context type
        match self.context_type:
            case 'group':
                return self._handle_group_context(output_type)
            case 'sort':
                return self._handle_sort_context(output_type)
            case 'filter':
                return self._handle_filter_context(output_type)
            case _:
                # Fallback to legacy behavior for backward compatibility
                return self._handle_legacy_context(output_type)

    def _handle_include_only(self, output_type: str) -> Union[List[str], List[pl.Expr], pl.Expr]:
        """Handle case where there is no context, only include columns."""
        match output_type:
            case 'tbl_col':
                return [f'{tbl}.{col}' for tbl, col in self.include_columns if isinstance((tbl, col), tuple)]
            case 'col':
                return [col for tbl, col in self.include_columns if isinstance((tbl, col), tuple)]
            case 'polar_col':
                return [pl.col(f'{tbl}.{col}') for tbl, col in self.include_columns if isinstance((tbl, col), tuple)]
            case 'polar_expr':
                # For include expressions (pl.Expr), return them directly
                include_exprs = [inc for inc in self.include_columns if isinstance(inc, pl.Expr)]
                if include_exprs:
                    # Combine multiple include expressions with &
                    result = include_exprs[0]
                    for expr in include_exprs[1:]:
                        result = result & expr
                    return result
                # No expressions, return True (match all)
                return pl.lit(True)

    def _handle_group_context(self, output_type: str) -> Union[List[str], List[pl.Expr]]:
        """Process group context: ['df.col1', 'df.col2']"""
        # Filter based on Allow patterns
        filtered = self._filter_group_context(self.raw_context)

        # Add include columns
        for tbl_col in self.include_columns:
            if isinstance(tbl_col, tuple):
                col_str = f"{tbl_col[0]}.{tbl_col[1]}"
                if col_str not in filtered:
                    filtered.append(col_str)

        # Convert to output format
        match output_type:
            case 'tbl_col':
                return filtered
            case 'col':
                return [col.split('.', 1)[1] if '.' in col else col for col in filtered]
            case 'polar_col' | 'polar_expr':
                return self._group_to_polars(filtered)

    def _handle_sort_context(self, output_type: str) -> Union[List[str], List[pl.Expr]]:
        """Process sort context: [('df.col1', 'desc'), ('df.col2', 'asc')]"""
        # Filter based on Allow patterns
        filtered = self._filter_sort_context(self.raw_context)

        # Add include columns (default to 'asc')
        for tbl_col in self.include_columns:
            if isinstance(tbl_col, tuple):
                col_str = f"{tbl_col[0]}.{tbl_col[1]}"
                if not any(col == col_str for col, _ in filtered):
                    filtered.append((col_str, 'asc'))

        # Convert to output format
        match output_type:
            case 'tbl_col':
                return [col for col, _ in filtered]
            case 'col':
                return [col.split('.', 1)[1] if '.' in col else col for col, _ in filtered]
            case 'polar_col' | 'polar_expr':
                return self._sort_to_polars(filtered)

    def _handle_filter_context(self, output_type: str) -> Union[List[str], pl.Expr]:
        """Process filter context: FilterExpression (dict/tuple)"""
        # Filter based on Allow patterns
        filtered_expr = self._filter_filter_context(self.raw_context)

        # Get pl.Expr from include (if any)
        include_exprs = [inc for inc in self.include_columns if isinstance(inc, pl.Expr)]

        if output_type in ('polar_col', 'polar_expr'):
            # Convert to Polars expression
            result_expr = None

            if filtered_expr is not None:
                result_expr = self._filter_to_polars(filtered_expr)

            # Combine with include expressions using &
            for inc_expr in include_exprs:
                if result_expr is None:
                    result_expr = inc_expr
                else:
                    result_expr = result_expr & inc_expr

            # If no filters, return True (match all)
            if result_expr is None:
                return pl.lit(True)

            return result_expr

        else:
            # Extract column names for tbl_col/col output
            if filtered_expr is None:
                return []

            # Recursively extract columns from filter expression
            from filter_context import Filter
            filter_obj = Filter(filtered_expr)
            columns = filter_obj.get_columns()

            match output_type:
                case 'tbl_col':
                    return columns
                case 'col':
                    return [col.split('.', 1)[1] if '.' in col else col for col in columns]

    def _handle_legacy_context(self, output_type: str) -> Union[List[str], List[pl.Expr]]:
        """Handle legacy context (for backward compatibility)."""
        relevant_columns = self.include_columns

        for tbl_col in self.context_columns:
            tbl, col = tbl_col

            if not {('*', ''), (tbl, '*'), (tbl, col)}.intersection(self.table_columns):
                relevant_columns.append((tbl, col))

        match output_type:
            case 'tbl_col':
                return [f'{tbl}.{col}' for tbl, col in relevant_columns]
            case 'col':
                return [col for tbl, col in relevant_columns]
            case 'polar_col' | 'polar_expr':
                return [pl.col(f'{tbl}.{col}') for tbl, col in relevant_columns]


class Exclude:
    raw_columns = []
    table_columns = set()
    include_columns = []
    context_columns = []
    raw_context = None
    context_type = 'unknown'

    def __init__(self: Self, *columns: str, **kwargs: Dict) -> Self:
        self.raw_columns = list(columns)
        self.table_columns = extract_table_columns(self.raw_columns)

        include_raw = []
        context_raw = None

        for key, val in kwargs.items():
            if key == 'include':
                if isinstance(val, list):
                    include_raw = flatten_list(val)
                elif isinstance(val, str):
                    include_raw = [val]
                elif isinstance(val, pl.Expr):
                    # Store pl.Expr as-is (for filter context)
                    include_raw = [val]

            if key == 'context':
                # Store raw context without modification
                context_raw = val

        # Extract table columns from strings only, preserve pl.Expr objects
        string_columns = [item for item in include_raw if isinstance(item, str)]
        expr_objects = [item for item in include_raw if isinstance(item, pl.Expr)]
        self.include_columns = list(extract_table_columns(string_columns)) + expr_objects

        # Store raw context and detect type
        if context_raw is not None:
            # Unwrap single-element lists (backward compatibility)
            if isinstance(context_raw, list) and len(context_raw) == 1 and isinstance(context_raw[0], (list, dict)):
                context_raw = context_raw[0]

            self.raw_context = context_raw
            self.context_type = _detect_context_type(context_raw)

            # Legacy: still extract columns for context_columns (backward compatibility)
            if isinstance(context_raw, list) and len(context_raw) > 0 and isinstance(context_raw[0], str):
                # Group context - simple list of strings
                self.context_columns = list(extract_table_columns(context_raw))
            else:
                # For other types, initialize empty for now (will be populated by filtering methods)
                self.context_columns = []
        else:
            self.raw_context = None
            self.context_type = 'unknown'
            self.context_columns = []

    def _parse_column(self, column: str) -> Tuple[str, str]:
        """
        Parse 'table.column' into (table, column) tuple.

        Format matches extract_table_columns():
        - 'table.column' → ('table', 'column')
        - 'column' → ('column', '')
        """
        if '.' in column:
            parts = column.split('.', 1)
            return (parts[0], parts[1])
        return (column, '')

    def _should_include_column(self, tbl_col: Tuple[str, str]) -> bool:
        """
        Check if column should be included based on Exclude patterns.

        For Exclude: include if NOT matching patterns
        """
        tbl, col = tbl_col

        # Check if column matches any pattern
        matches = {('*', ''), (tbl, '*'), (tbl, col)}.intersection(self.table_columns)

        # Exclude: include if NOT matching
        return not bool(matches)

    def _filter_group_context(self, group_list: List[str]) -> List[str]:
        """Apply Exclude patterns to group context."""
        filtered = []
        for column in group_list:
            tbl_col = self._parse_column(column)
            if self._should_include_column(tbl_col):
                filtered.append(column)
        return filtered

    def _filter_sort_context(self, sort_list: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Apply Exclude patterns to sort context."""
        filtered = []
        for column, direction in sort_list:
            tbl_col = self._parse_column(column)
            if self._should_include_column(tbl_col):
                filtered.append((column, direction))
        return filtered

    def _filter_filter_context(self, filter_expr: Any) -> Optional[Any]:
        """
        Apply Exclude patterns to filter context recursively.

        Returns filtered FilterExpression or None if all filtered out.
        Simplifies single-element AND/OR.
        """
        # Simple condition (tuple)
        if isinstance(filter_expr, tuple):
            column, operator, value = filter_expr
            tbl_col = self._parse_column(column)
            return filter_expr if self._should_include_column(tbl_col) else None

        # AND/OR dict
        if isinstance(filter_expr, dict):
            key = next(iter(filter_expr.keys()))
            conditions = filter_expr[key]

            # Recursively filter each condition
            filtered_conditions = []
            for cond in conditions:
                filtered = self._filter_filter_context(cond)
                if filtered is not None:
                    filtered_conditions.append(filtered)

            # Handle results
            if len(filtered_conditions) == 0:
                return None
            elif len(filtered_conditions) == 1:
                # Simplify single-element AND/OR
                return filtered_conditions[0]
            else:
                return {key: filtered_conditions}

        return None

    def _group_to_polars(self, group_list: List[str]) -> List[pl.Expr]:
        """Convert ['df.col1', 'df.col2'] → [pl.col('df.col1'), pl.col('df.col2')] (preserves table prefix)"""
        return [pl.col(column) for column in group_list]

    def _sort_to_polars(self, sort_list: List[Tuple[str, str]]) -> List[pl.Expr]:
        """
        Convert [('df.col', 'desc')] → [pl.col('df.col')] (preserves table prefix)

        Note: Polars .sort() doesn't support .desc()/.asc() on expressions.
        Direction info is lost - use with separate descending parameter.
        """
        return [pl.col(column) for column, direction in sort_list]

    def _filter_to_polars(self, filter_expr: Any) -> pl.Expr:
        """
        Convert FilterExpression to Polars boolean expression.

        Examples:
            ('df.col', '=', 1) → pl.col('col') == 1
            {'AND': [cond1, cond2]} → cond1_expr & cond2_expr
            {'OR': [cond1, cond2]} → cond1_expr | cond2_expr
        """
        # Simple condition (tuple)
        if isinstance(filter_expr, tuple):
            column, operator, value = filter_expr
            # Strip table prefix from column name (e.g., 'sales.item_id' → 'item_id')
            column_name = column.split('.')[-1] if '.' in column else column
            col_expr = pl.col(column_name)

            if operator not in OPERATOR_MAP:
                raise ValueError(f"Unsupported operator: {operator}")

            return OPERATOR_MAP[operator](col_expr, value)

        # AND/OR dict
        if isinstance(filter_expr, dict):
            key = next(iter(filter_expr.keys()))
            conditions = filter_expr[key]

            if key == 'AND':
                # Combine with &
                result = self._filter_to_polars(conditions[0])
                for cond in conditions[1:]:
                    result = result & self._filter_to_polars(cond)
                return result

            elif key == 'OR':
                # Combine with |
                result = self._filter_to_polars(conditions[0])
                for cond in conditions[1:]:
                    result = result | self._filter_to_polars(cond)
                return result

        raise ValueError(f"Invalid filter expression: {filter_expr}")

    def get_columns(self: Self) -> Set:
        return self.table_columns

    def get_include(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.include_columns]

    def get_context(self: Self, polars_col: bool = False) -> Union[List[str], List[pl.Expr]]:
        return [pl.col(column) if polars_col else column for column in self.context_columns]

    def get_relevant_columns(self: Self, output_type: Literal['tbl_col', 'col', 'polar_col', 'polar_expr'] = 'tbl_col') -> Union[List[str], List[pl.Expr], pl.Expr]:
        """
        Get filtered columns/expressions from context.

        Args:
            output_type:
                'tbl_col': ['df.col1', ...] (default, backward compatible)
                'col': ['col1', ...] (strip table prefix)
                'polar_col': [pl.col('df.col1'), ...] (for group/sort)
                'polar_expr': pl.Expr (for filter, combined expression)

        Returns:
            Filtered and converted columns/expressions based on context type
        """
        # No context - handle include-only case
        if self.raw_context is None:
            return self._handle_include_only(output_type)

        # Dispatch based on context type
        match self.context_type:
            case 'group':
                return self._handle_group_context(output_type)
            case 'sort':
                return self._handle_sort_context(output_type)
            case 'filter':
                return self._handle_filter_context(output_type)
            case _:
                # Fallback to legacy behavior for backward compatibility
                return self._handle_legacy_context(output_type)

    def _handle_include_only(self, output_type: str) -> Union[List[str], List[pl.Expr], pl.Expr]:
        """Handle case where there is no context, only include columns."""
        match output_type:
            case 'tbl_col':
                return [f'{tbl}.{col}' for tbl, col in self.include_columns if isinstance((tbl, col), tuple)]
            case 'col':
                return [col for tbl, col in self.include_columns if isinstance((tbl, col), tuple)]
            case 'polar_col':
                return [pl.col(f'{tbl}.{col}') for tbl, col in self.include_columns if isinstance((tbl, col), tuple)]
            case 'polar_expr':
                # For include expressions (pl.Expr), return them directly
                include_exprs = [inc for inc in self.include_columns if isinstance(inc, pl.Expr)]
                if include_exprs:
                    # Combine multiple include expressions with &
                    result = include_exprs[0]
                    for expr in include_exprs[1:]:
                        result = result & expr
                    return result
                # No expressions, return False (exclude all)
                return pl.lit(False)

    def _handle_group_context(self, output_type: str) -> Union[List[str], List[pl.Expr]]:
        """Process group context: ['df.col1', 'df.col2']"""
        # Filter based on Exclude patterns
        filtered = self._filter_group_context(self.raw_context)

        # Add include columns
        for tbl_col in self.include_columns:
            if isinstance(tbl_col, tuple):
                col_str = f"{tbl_col[0]}.{tbl_col[1]}"
                if col_str not in filtered:
                    filtered.append(col_str)

        # Convert to output format
        match output_type:
            case 'tbl_col':
                return filtered
            case 'col':
                return [col.split('.', 1)[1] if '.' in col else col for col in filtered]
            case 'polar_col' | 'polar_expr':
                return self._group_to_polars(filtered)

    def _handle_sort_context(self, output_type: str) -> Union[List[str], List[pl.Expr]]:
        """Process sort context: [('df.col1', 'desc'), ('df.col2', 'asc')]"""
        # Filter based on Exclude patterns
        filtered = self._filter_sort_context(self.raw_context)

        # Add include columns (default to 'asc')
        for tbl_col in self.include_columns:
            if isinstance(tbl_col, tuple):
                col_str = f"{tbl_col[0]}.{tbl_col[1]}"
                if not any(col == col_str for col, _ in filtered):
                    filtered.append((col_str, 'asc'))

        # Convert to output format
        match output_type:
            case 'tbl_col':
                return [col for col, _ in filtered]
            case 'col':
                return [col.split('.', 1)[1] if '.' in col else col for col, _ in filtered]
            case 'polar_col' | 'polar_expr':
                return self._sort_to_polars(filtered)

    def _handle_filter_context(self, output_type: str) -> Union[List[str], pl.Expr]:
        """Process filter context: FilterExpression (dict/tuple)"""
        # Filter based on Exclude patterns
        filtered_expr = self._filter_filter_context(self.raw_context)

        # Get pl.Expr from include (if any)
        include_exprs = [inc for inc in self.include_columns if isinstance(inc, pl.Expr)]

        if output_type in ('polar_col', 'polar_expr'):
            # Convert to Polars expression
            result_expr = None

            if filtered_expr is not None:
                result_expr = self._filter_to_polars(filtered_expr)

            # Combine with include expressions using &
            for inc_expr in include_exprs:
                if result_expr is None:
                    result_expr = inc_expr
                else:
                    result_expr = result_expr & inc_expr

            # If no filters, return False (exclude all) for Exclude
            if result_expr is None:
                return pl.lit(False)

            return result_expr

        else:
            # Extract column names for tbl_col/col output
            if filtered_expr is None:
                return []

            # Recursively extract columns from filter expression
            from filter_context import Filter
            filter_obj = Filter(filtered_expr)
            columns = filter_obj.get_columns()

            match output_type:
                case 'tbl_col':
                    return columns
                case 'col':
                    return [col.split('.', 1)[1] if '.' in col else col for col in columns]

    def _handle_legacy_context(self, output_type: str) -> Union[List[str], List[pl.Expr]]:
        """Handle legacy context (for backward compatibility)."""
        relevant_columns = self.include_columns

        for tbl_col in self.context_columns:
            tbl, col = tbl_col

            if not {('*', ''), (tbl, '*'), (tbl, col)}.intersection(self.table_columns):
                relevant_columns.append((tbl, col))

        match output_type:
            case 'tbl_col':
                return [f'{tbl}.{col}' for tbl, col in relevant_columns]
            case 'col':
                return [col for tbl, col in relevant_columns]
            case 'polar_col' | 'polar_expr':
                return [pl.col(f'{tbl}.{col}') for tbl, col in relevant_columns]




if __name__ == '__main__':
    query_context = {
        'groupings' : ['df.store_id'],
        'orderings' : ['df.store_id']
    }
    
    allow_test = Allow('*', include='df.item_id', context=[query_context['groupings']])
    print(allow_test.get_columns())
    print(allow_test.get_include())
    print(allow_test.get_context())
    print(allow_test.get_relevant_columns())

    exclude_test = Exclude('*', include=['df.item_id'], context=query_context['orderings'])
    print(exclude_test.get_columns())
    print(allow_test.get_include())
    print(allow_test.get_context())
    print(exclude_test.get_relevant_columns())