"""
Transformer to replace Allow() and Exclude() calls with resolved column lists.

This module provides a libcst transformer that:
1. Finds Allow() and Exclude() object instantiations within a target function
2. Evaluates them with runtime context to resolve variable references
3. Calls .get_relevant_columns() to get the resolved column list
4. Replaces the entire call with a list literal

Example:
    >>> from cst.transformers.replace_context_with_columns import transform_function
    >>>
    >>> code = '''
    ... def revenue_by_item():
    ...     return df.group_by(Allow('*', include='df.item_id'))
    ... '''
    >>>
    >>> result = transform_function(code, 'revenue_by_item')
    >>> print(result)
    def revenue_by_item():
        return df.group_by(['item_id'])
"""

from typing import Dict, Any, Union, Optional, Literal, Tuple
import ast
import warnings

import libcst as cst
import libcst.matchers as m

from datasubway.column_context import Allow, Exclude


class ReplaceContextWithTableColumns(cst.CSTTransformer):
    """
    Transforms Allow() and Exclude() calls into resolved column lists.

    This transformer:
    1. Finds all Allow() and Exclude() instantiations within a target function
    2. Evaluates them with runtime context to resolve variable references
    3. Calls .get_relevant_columns() to get the resolved column list
    4. Replaces the entire Call node with a List literal
    """

    def __init__(
        self,
        function_name: str,
        runtime_context: Optional[Dict[str, Any]] = None,
        output_type: Literal['tbl_col', 'col', 'polar_col'] = 'tbl_col'
    ):
        """
        Initialize the transformer.

        Args:
            function_name: Name of the function to transform (other functions unchanged)
            runtime_context: Dict of runtime variables (e.g., {'query_context': {...}})
            output_type: tbl_col for table.column, col for column name, polar_col for pl.col(column)
        """
        super().__init__()
        self.function_name = function_name
        self.runtime_context = runtime_context or {}
        self.output_type = output_type
        # Note: We use manual function tracking rather than @m.call_if_inside()
        # because function_name is a dynamic instance attribute, and decorator-based
        # matchers are evaluated at class definition time.
        self.current_function: Optional[str] = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool:
        """Track which function we're currently visiting."""
        self.current_function = node.name.value
        return True  # Continue visiting children

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        """Reset function tracking when leaving."""
        self.current_function = None
        return updated_node

    def _safe_construct_allow_exclude(self, call_node: cst.Call) -> Optional[Union[Allow, Exclude]]:
        """Safely construct an Allow or Exclude instance from a CST Call node.

        Parses arguments from CST and constructs the instance without using eval().
        For complex expressions referencing runtime_context, uses safe lookup.

        Args:
            call_node: CST Call node for Allow() or Exclude()

        Returns:
            Allow or Exclude instance, or None if construction fails
        """
        # Determine which class to instantiate
        func_name = cst.ensure_type(call_node.func, cst.Name).value
        cls = Allow if func_name == 'Allow' else Exclude

        # Extract arguments
        # Allow/Exclude signature: def __init__(self, *columns, **kwargs)
        # Positional args go into *columns, keyword args go into **kwargs
        args_list = []  # For positional args (*columns)
        kwargs: Dict[str, Any] = {}  # For keyword args (include=, context=)

        for arg in call_node.args:
            if arg.keyword is not None:
                # Keyword argument
                arg_name = arg.keyword.value
                value = self._safe_extract_arg_value(arg.value)
                kwargs[arg_name] = value
            else:
                # Positional argument - goes into *columns
                value = self._safe_extract_arg_value(arg.value)
                if value is None:
                    return None
                args_list.append(value)

        try:
            return cls(*args_list, **kwargs)
        except Exception:
            return None

    def _safe_extract_arg_value(self, node: cst.BaseExpression) -> Optional[Any]:
        """Safely extract a Python value from a CST expression node.

        Handles:
        - String literals
        - List literals
        - Dict literals
        - Boolean/None literals
        - var['key'] subscripts (looks up in runtime_context)
        - var.get('key', default) calls (looks up in runtime_context)

        Args:
            node: CST expression node

        Returns:
            Extracted Python value, or None if extraction fails
        """
        # String literal
        if isinstance(node, cst.SimpleString):
            return node.value.strip('\'"')

        # Concatenated string
        if isinstance(node, cst.ConcatenatedString):
            parts = []
            for part in node.left, node.right:
                if isinstance(part, cst.SimpleString):
                    parts.append(part.value.strip('\'"'))
            return ''.join(parts) if parts else None

        # List literal
        if isinstance(node, cst.List):
            result = []
            for elem in node.elements:
                if isinstance(elem, cst.Element):
                    val = self._safe_extract_arg_value(elem.value)
                    if val is not None:
                        result.append(val)
            return result

        # Tuple literal
        if isinstance(node, cst.Tuple):
            result = []
            for elem in node.elements:
                if isinstance(elem, cst.Element):
                    val = self._safe_extract_arg_value(elem.value)
                    if val is not None:
                        result.append(val)
            return tuple(result)

        # Dict literal
        if isinstance(node, cst.Dict):
            result = {}
            for elem in node.elements:
                if isinstance(elem, cst.DictElement):
                    key = self._safe_extract_arg_value(elem.key)
                    val = self._safe_extract_arg_value(elem.value)
                    if key is not None:
                        result[key] = val
            return result

        # Boolean/None literals
        if isinstance(node, cst.Name):
            if node.value == 'True':
                return True
            elif node.value == 'False':
                return False
            elif node.value == 'None':
                return None

        # Integer literal
        if isinstance(node, cst.Integer):
            return int(node.value)

        # Float literal
        if isinstance(node, cst.Float):
            return float(node.value)

        # var['key'] subscript - lookup in runtime_context
        if m.matches(node, m.Subscript(value=m.Name())):
            return self._safe_context_lookup_subscript(node)

        # var.get('key', default) call - lookup in runtime_context
        if m.matches(node, m.Call(func=m.Attribute(value=m.Name(), attr=m.Name('get')))):
            return self._safe_context_lookup_get(node)

        # Try ast.literal_eval as fallback for other literals
        try:
            code = cst.Module(body=[cst.Expr(value=node)]).code.strip()
            return ast.literal_eval(code)
        except (ValueError, SyntaxError):
            pass

        return None

    def _safe_context_lookup_subscript(self, node: cst.Subscript) -> Optional[Any]:
        """Safely lookup var['key'] in runtime_context."""
        var_name = cst.ensure_type(node.value, cst.Name).value
        context_var = self.runtime_context.get(var_name)
        if context_var is None or not isinstance(context_var, dict):
            return None

        if len(node.slice) != 1:
            return None

        slice_elem = node.slice[0]
        if not isinstance(slice_elem, cst.SubscriptElement):
            return None

        slice_val = slice_elem.slice
        if not isinstance(slice_val, cst.Index):
            return None

        key_node = slice_val.value
        if isinstance(key_node, cst.SimpleString):
            key = key_node.value.strip('\'"')
            return context_var.get(key)

        return None

    def _safe_context_lookup_get(self, node: cst.Call) -> Optional[Any]:
        """Safely lookup var.get('key', default) in runtime_context."""
        attr_node = cst.ensure_type(node.func, cst.Attribute)
        var_name = cst.ensure_type(attr_node.value, cst.Name).value

        context_var = self.runtime_context.get(var_name)
        if context_var is None or not isinstance(context_var, dict):
            return None

        if len(node.args) < 1:
            return None

        key_arg = node.args[0].value
        if not isinstance(key_arg, cst.SimpleString):
            return None
        key = key_arg.value.strip('\'"')

        # Extract default value if present
        default = None
        if len(node.args) >= 2:
            default = self._safe_extract_arg_value(node.args[1].value)

        return context_var.get(key, default)

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> Union[cst.Call, cst.List]:
        """
        Replace Allow() and Exclude() calls with resolved column lists.
        For .sort() calls with sort context, add the descending parameter.

        Steps:
        1. Check if we're in the target function
        2. Check if call is .sort() with Allow/Exclude argument -> transform entire call
        3. Check if call is Allow() or Exclude() -> transform to list
        4. Convert CST node to code string
        5. Evaluate to get Allow/Exclude instance
        6. Call get_relevant_columns() to resolve
        7. Create CST List node with results
        8. Return transformed node
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Check if this is a .sort() method call with Allow/Exclude argument
        if self._is_sort_method_call(updated_node):
            sort_result = self._transform_sort_call(updated_node)
            if sort_result is not None:
                return sort_result

        # Only transform Allow() or Exclude() calls
        if not (m.matches(updated_node, m.Call(func=m.Name('Allow'))) or
                m.matches(updated_node, m.Call(func=m.Name('Exclude')))):
            return updated_node

        try:
            # Safely construct Allow/Exclude instance from CST
            instance = self._safe_construct_allow_exclude(updated_node)
            if instance is None:
                return updated_node

            # Get resolved columns and create appropriate output
            if self.output_type == 'polar_col':
                # Check if this is a filter context (returns single expression)
                if instance.context_type == 'filter':
                    return self._create_filter_expr(instance, updated_node)
                # Check if this is a sort context (needs direction info)
                elif instance.context_type == 'sort':
                    # Don't transform here - let the parent .sort() call handle it
                    # This preserves the Allow/Exclude node for _transform_sort_call
                    return updated_node
                else:
                    return self._create_polars_list(instance)
            else:
                return self._create_string_list(instance, self.output_type)

        except (NameError, KeyError, AttributeError):
            # Missing runtime context variable - leave unchanged
            return updated_node
        except (SyntaxError, TypeError, ValueError) as e:
            # Eval or type-related error - leave unchanged
            warnings.warn(
                f"Failed to transform Allow/Exclude call: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )
            return updated_node

    def _is_sort_method_call(self, node: cst.Call) -> bool:
        """
        Check if this is a .sort() method call.

        Args:
            node: Call node to check

        Returns:
            True if this is a .sort() method call
        """
        return m.matches(
            node,
            m.Call(func=m.Attribute(attr=m.Name('sort')))
        )

    def _transform_sort_call(self, node: cst.Call) -> Optional[cst.Call]:
        """
        Transform a .sort() call to add the descending parameter.

        Args:
            node: The .sort() Call node

        Returns:
            Transformed Call node with descending parameter, or None if no transformation needed
        """
        if self.output_type != 'polar_col':
            return None

        # Find the Allow/Exclude argument
        allow_exclude_arg = None
        allow_exclude_arg_index = None

        for i, arg in enumerate(node.args):
            if isinstance(arg.value, cst.Call):
                if (m.matches(arg.value, m.Call(func=m.Name('Allow'))) or
                    m.matches(arg.value, m.Call(func=m.Name('Exclude')))):
                    allow_exclude_arg = arg.value
                    allow_exclude_arg_index = i
                    break

        if allow_exclude_arg is None:
            return None

        try:
            # Safely construct Allow/Exclude instance from CST
            instance = self._safe_construct_allow_exclude(allow_exclude_arg)
            if instance is None:
                return None

            # Only handle sort contexts
            if instance.context_type != 'sort':
                return None

            # Get sort columns with direction using shared helper
            sort_columns = self._get_sort_columns_with_direction(instance)

            # Build column list (without .desc())
            col_elements = []
            descending_values = []

            for column, direction in sort_columns:
                col_call = self._build_pl_col(column)
                col_elements.append(cst.Element(value=col_call))
                descending_values.append(direction.lower() == 'desc')

            # Create the column list
            col_list = cst.List(elements=col_elements)

            # Create the descending list
            desc_elements = [
                cst.Element(value=cst.Name('True' if desc else 'False'))
                for desc in descending_values
            ]
            desc_list = cst.List(elements=desc_elements)

            # Build new arguments list
            new_args = list(node.args)
            new_args[allow_exclude_arg_index] = cst.Arg(value=col_list)

            # Add descending parameter if not already present
            has_descending = any(
                arg.keyword and arg.keyword.value == 'descending'
                for arg in new_args
            )

            if not has_descending:
                new_args.append(
                    cst.Arg(
                        keyword=cst.Name('descending'),
                        value=desc_list,
                        equal=cst.AssignEqual(
                            whitespace_before=cst.SimpleWhitespace(''),
                            whitespace_after=cst.SimpleWhitespace('')
                        )
                    )
                )

            # Return updated call
            return node.with_changes(args=new_args)

        except (NameError, KeyError, AttributeError):
            # Missing runtime context variable - fall back to default behavior
            return None
        except (SyntaxError, TypeError, ValueError) as e:
            # Eval or type-related error - fall back with warning
            warnings.warn(
                f"Failed to transform sort call: {type(e).__name__}: {e}",
                RuntimeWarning,
                stacklevel=2
            )
            return None

    def _create_string_list(self, instance: Union[Allow, Exclude], output_type: Literal['tbl_col', 'col', 'polar_col']) -> cst.List:
        """
        Create a List node with string literals.

        Example output: ['item_id', 'store_id']

        Args:
            instance: Allow or Exclude instance to get columns from

        Returns:
            CST List node with string elements
        """
        columns = instance.get_relevant_columns(output_type=output_type)

        elements = [
            cst.Element(value=cst.SimpleString(repr(col)))
            for col in columns
        ]

        return cst.List(elements=elements)

    def _create_filter_expr(self, instance: Union[Allow, Exclude], original_call: cst.Call) -> Optional[cst.BaseExpression]:
        """
        Create a single Polars expression node for filter contexts.

        Builds CST directly from the filter structure, combining context filters
        with include expressions using & operator.

        Args:
            instance: Allow or Exclude instance with filter context
            original_call: The original Allow/Exclude call CST node

        Returns:
            CST expression node representing the combined filter, or None if empty
        """
        # Step 1: Extract include parameter from original CST call
        include_expr = None
        for arg in original_call.args:
            if arg.keyword and arg.keyword.value == 'include':
                include_expr = arg.value
                break

        # Step 2: Get filtered context from instance
        filtered_context = None
        if instance.raw_context is not None:
            filtered_context = instance._filter_filter_context(instance.raw_context)

        # Step 3: Build CST from filtered context
        context_cst = None
        if filtered_context is not None:
            context_cst = self._build_filter_cst_from_structure(filtered_context)

        # Step 4: Combine context and include
        # Both context and include present - combine with &
        if context_cst is not None and include_expr is not None:
            return self._combine_with_and(context_cst, include_expr)

        # Only context
        if context_cst is not None:
            return context_cst

        # Only include
        if include_expr is not None:
            return include_expr

        # Neither - return None literal as the parameter (e.g., .filter(None))
        return cst.Name('None')

    def _get_sort_columns_with_direction(
        self,
        instance: Union[Allow, Exclude]
    ) -> list[tuple[str, str]]:
        """
        Get sort columns with their direction from an Allow/Exclude instance.

        Combines filtered sort context with include columns (defaulting to 'asc').

        Args:
            instance: Allow or Exclude instance with sort context

        Returns:
            List of (column, direction) tuples where direction is 'asc' or 'desc'
        """
        filtered_sort = instance._filter_sort_context(instance.raw_context)

        # Add include columns (default to 'asc')
        for tbl_col in instance.include_columns:
            if isinstance(tbl_col, tuple):
                col_str = f"{tbl_col[0]}.{tbl_col[1]}"
                if not any(col == col_str for col, _ in filtered_sort):
                    filtered_sort.append((col_str, 'asc'))

        return filtered_sort

    def _create_sort_list(self, instance: Union[Allow, Exclude]) -> cst.List:
        """
        Create a List node with pl.col() calls for sort context.

        Note: This is used when transforming Allow/Exclude with sort context
        outside of a .sort() call. The direction information is lost in this case.
        For .sort() calls, use _transform_sort_call instead.

        Example output: [pl.col('item_id'), pl.col('store_id')]

        Args:
            instance: Allow or Exclude instance with sort context

        Returns:
            CST List node with column expressions (no direction info)
        """
        sort_columns = self._get_sort_columns_with_direction(instance)

        elements = []
        for column, direction in sort_columns:
            col_call = self._build_pl_col(column)
            elements.append(cst.Element(value=col_call))

        return cst.List(elements=elements)

    def _create_polars_list(self, instance: Union[Allow, Exclude]) -> cst.List:
        """
        Create a List node with pl.col() calls.

        Example output: [pl.col('item_id'), pl.col('store_id')]

        Args:
            instance: Allow or Exclude instance to get columns from

        Returns:
            CST List node with pl.col() call elements

        Note:
            This relies on the string representation of polars Expr objects
            which has the format: col("column_name")
        """
        exprs = instance.get_relevant_columns(output_type='polar_col')

        elements = []
        for expr in exprs:
            # Extract column name from polars Expr string representation
            # Format: col("column_name")
            col_name = str(expr).split('"')[1]

            # Build pl.col('column_name') CST node
            col_call = cst.Call(
                func=cst.Attribute(
                    value=cst.Name('pl'),
                    attr=cst.Name('col')
                ),
                args=[cst.Arg(value=cst.SimpleString(repr(col_name)))]
            )
            elements.append(cst.Element(value=col_call))

        return cst.List(elements=elements)

    def _build_pl_col(self, column: str) -> cst.Call:
        """
        Build pl.col('column') CST node.

        Args:
            column: Column name (e.g., 'df.item_id' or 'item_id')

        Returns:
            CST Call node representing pl.col('item_id') (table prefix stripped)
        """
        # Strip table prefix if present (e.g., 'sales.item_id' → 'item_id')
        column_name = column.split('.')[-1] if '.' in column else column
        return cst.Call(
            func=cst.Attribute(
                value=cst.Name('pl'),
                attr=cst.Name('col')
            ),
            args=[cst.Arg(value=cst.SimpleString(repr(column_name)))]
        )

    def _build_value_cst(self, value: Any) -> cst.BaseExpression:
        """
        Convert Python value to CST node.

        Args:
            value: Python value (int, float, str, list, bool, None)

        Returns:
            Appropriate CST node for the value
        """
        if isinstance(value, bool):
            return cst.Name('True' if value else 'False')
        elif isinstance(value, int):
            return cst.Integer(str(value))
        elif isinstance(value, float):
            return cst.Float(str(value))
        elif isinstance(value, str):
            return cst.SimpleString(repr(value))
        elif isinstance(value, list):
            elements = [cst.Element(value=self._build_value_cst(item)) for item in value]
            return cst.List(elements=elements)
        elif value is None:
            return cst.Name('None')
        else:
            # Fallback for unsupported types
            return cst.SimpleString(repr(value))

    def _combine_with_and(self, left: cst.BaseExpression, right: cst.BaseExpression) -> cst.BinaryOperation:
        """
        Combine two expressions with & operator.

        Args:
            left: Left expression
            right: Right expression

        Returns:
            BinaryOperation node with & operator
        """
        return cst.BinaryOperation(
            left=left,
            operator=cst.BitAnd(),
            right=right
        )

    def _wrap_in_parens(self, node: cst.BaseExpression) -> cst.BaseExpression:
        """
        Wrap a CST expression in parentheses for proper precedence.

        Args:
            node: CST expression to wrap

        Returns:
            Same expression wrapped with lpar/rpar
        """
        return node.with_changes(
            lpar=[cst.LeftParen()],
            rpar=[cst.RightParen()]
        )

    def _build_comparison(self, column: str, operator: str, value: Any) -> cst.Comparison:
        """
        Build comparison expression for =, !=, >, <, >=, <= operators.

        Args:
            column: Column name (e.g., 'df.item_id')
            operator: Comparison operator
            value: Value to compare against

        Returns:
            CST Comparison node with parentheses for proper precedence
        """
        operator_map = {
            '=': cst.Equal,
            '!=': cst.NotEqual,
            '>': cst.GreaterThan,
            '<': cst.LessThan,
            '>=': cst.GreaterThanEqual,
            '<=': cst.LessThanEqual,
        }

        comparison = cst.Comparison(
            left=self._build_pl_col(column),
            comparisons=[
                cst.ComparisonTarget(
                    operator=operator_map[operator](),
                    comparator=self._build_value_cst(value)
                )
            ]
        )

        return self._wrap_in_parens(comparison)

    def _build_is_in(self, column: str, value_list: list) -> cst.Call:
        """
        Build .is_in([values]) method call.

        Args:
            column: Column name
            value_list: List of values

        Returns:
            CST Call node for pl.col(column).is_in([values]) with parentheses
        """
        call = cst.Call(
            func=cst.Attribute(
                value=self._build_pl_col(column),
                attr=cst.Name('is_in')
            ),
            args=[cst.Arg(value=self._build_value_cst(value_list))]
        )

        return self._wrap_in_parens(call)

    def _build_not_in(self, column: str, value_list: list) -> cst.UnaryOperation:
        """
        Build ~.is_in([values]) expression.

        Args:
            column: Column name
            value_list: List of values

        Returns:
            CST UnaryOperation node for ~pl.col(column).is_in([values])
        """
        # Note: _build_is_in already wraps in parentheses, so the ~ will be outside
        # Result: ~(pl.col(column).is_in([values]))
        return cst.UnaryOperation(
            operator=cst.BitInvert(),
            expression=self._build_is_in(column, value_list)
        )

    def _build_string_method(self, column: str, method: str, value: str) -> cst.Call:
        """
        Build .str.method(value) call for string operators.

        Args:
            column: Column name
            method: String method name (contains, starts_with, ends_with)
            value: String value

        Returns:
            CST Call node for pl.col(column).str.method(value) with parentheses
        """
        call = cst.Call(
            func=cst.Attribute(
                value=cst.Attribute(
                    value=self._build_pl_col(column),
                    attr=cst.Name('str')
                ),
                attr=cst.Name(method)
            ),
            args=[cst.Arg(value=cst.SimpleString(repr(value)))]
        )

        return self._wrap_in_parens(call)

    def _build_null_check(self, column: str, method: str) -> cst.Call:
        """
        Build .is_null() or .is_not_null() call.

        Args:
            column: Column name
            method: Null check method name (is_null or is_not_null)

        Returns:
            CST Call node for pl.col(column).method() with parentheses
        """
        call = cst.Call(
            func=cst.Attribute(
                value=self._build_pl_col(column),
                attr=cst.Name(method)
            ),
            args=[]
        )

        return self._wrap_in_parens(call)

    def _build_filter_cst_from_structure(self, filter_expr: Union[Tuple, Dict]) -> cst.BaseExpression:
        """
        Recursively build CST from filter structure.

        Args:
            filter_expr: Filter expression as tuple or dict
                - Tuple: (column, operator, value) - FilterCondition
                - Dict: {'AND': [cond1, cond2]} or {'OR': [cond1, cond2]}

        Returns:
            CST expression node

        Examples:
            ('df.item_id', '=', 3) → pl.col('df.item_id') == 3
            {'AND': [cond1, cond2]} → cond1 & cond2
        """
        # Handle FilterCondition tuple
        if isinstance(filter_expr, tuple):
            column, operator, value = filter_expr

            # Comparison operators
            if operator in ('=', '!=', '>', '<', '>=', '<='):
                return self._build_comparison(column, operator, value)

            # IN operator
            elif operator == 'IN':
                return self._build_is_in(column, value)

            # NOT IN operator
            elif operator == 'NOT IN':
                return self._build_not_in(column, value)

            # String operators
            elif operator == 'CONTAINS':
                return self._build_string_method(column, 'contains', value)

            elif operator == 'STARTS_WITH':
                return self._build_string_method(column, 'starts_with', value)

            elif operator == 'ENDS_WITH':
                return self._build_string_method(column, 'ends_with', value)

            elif operator == 'LIKE':
                # Convert SQL LIKE wildcards to regex
                pattern = value.replace('%', '.*').replace('_', '.')
                return self._build_string_method(column, 'contains', pattern)

            # Null operators
            elif operator == 'IS NULL':
                return self._build_null_check(column, 'is_null')

            elif operator == 'IS NOT NULL':
                return self._build_null_check(column, 'is_not_null')

            else:
                raise ValueError(f"Unsupported operator: {operator}")

        # Handle AND/OR dictionary
        elif isinstance(filter_expr, dict):
            key = next(iter(filter_expr.keys()))
            conditions = filter_expr[key]

            # Recursively build each condition
            cst_conditions = [
                self._build_filter_cst_from_structure(cond)
                for cond in conditions
            ]

            # Combine conditions
            if len(cst_conditions) == 1:
                return cst_conditions[0]

            # Chain with appropriate operator
            result = cst_conditions[0]
            operator_class = cst.BitAnd if key == 'AND' else cst.BitOr

            for cond in cst_conditions[1:]:
                result = cst.BinaryOperation(
                    left=result,
                    operator=operator_class(),
                    right=cond
                )

            return result

        else:
            raise ValueError(f"Invalid filter expression type: {type(filter_expr)}")


def resolve_table_columns(
    source_code: str,
    function_name: str,
    runtime_context: Optional[Dict[str, Any]] = None,
    output_type: Literal['tbl_col', 'col', 'polar_col'] = 'tbl_col'
) -> str:
    """
    Transform a function's Allow/Exclude calls to column lists.

    Args:
        source_code: Python source code containing the function
        function_name: Name of function to transform
        runtime_context: Dict of runtime variables accessible to Allow/Exclude
        output_type: tbl_col for table.column, col for column name, polar_col for pl.col(column)

    Returns:
        Transformed source code as string

    Example:
        >>> code = '''
        ... def revenue_by_item():
        ...     query_context = {'groupings': ['df.store_id']}
        ...     return df.group_by(
        ...         Allow('*', include='df.item_id', context=[query_context['groupings']])
        ...     )
        ... '''
        >>>
        >>> context = {'query_context': {'groupings': ['df.store_id']}}
        >>> result = transform_function(code, 'revenue_by_item', runtime_context=context)
        >>>
        >>> print(result)
        def revenue_by_item():
            query_context = {'groupings': ['df.store_id']}
            return df.group_by(['item_id', 'store_id'])
    """
    module = cst.parse_module(source_code)
    transformer = ReplaceContextWithTableColumns(
        function_name=function_name,
        runtime_context=runtime_context,
        output_type=output_type
    )
    new_module = module.visit(transformer)
    return new_module.code
