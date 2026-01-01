"""
Transformer to inject parameters into dm.table() calls.

This transformer:
1. Finds dm.table() or data_model.table() calls with only the table name argument
2. Scans the method chain to extract columns from .group_by() and .agg() calls
3. Extracts aggregation functions from .agg() calls
4. Injects group_by_cols and agg_cols as parameters to the table() call

Example:
    >>> code = '''
    ... def total_revenue(qc):
    ...     return (
    ...         data_model.table('sales')
    ...         .group_by([pl.col('products.product_name')])
    ...         .agg(pl.col('sales.revenue').sum().alias('total_revenue'))
    ...     )
    ... '''
    >>> # After transformation:
    >>> # data_model.table('sales', ['products.product_name'], {'sales.revenue': 'sum'})
"""

from typing import Dict, Any, Optional, List, Set, Tuple
import libcst as cst
import libcst.matchers as m


class InjectTableParameters(cst.CSTTransformer):
    """
    Injects parameters into table() calls based on method chain analysis.

    This transformer scans the entire return statement to extract:
    - Columns from .group_by() calls
    - Columns and aggregations from .agg() calls

    Then injects these as parameters to the table() call.
    """

    def __init__(
        self,
        function_name: str,
        runtime_context: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the transformer.

        Args:
            function_name: Name of the function to transform
            runtime_context: Dict containing runtime variables (e.g., 'qc')
        """
        super().__init__()
        self.function_name = function_name
        self.runtime_context = runtime_context or {}
        self.current_function: Optional[str] = None

    def visit_FunctionDef(self, node: cst.FunctionDef) -> None:
        """Track which function we're currently visiting."""
        self.current_function = node.name.value

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        """Reset function tracking when leaving."""
        self.current_function = None
        return updated_node

    def leave_Assign(
        self,
        original_node: cst.Assign,
        updated_node: cst.Assign
    ) -> cst.Assign:
        """
        Process assignment statements to inject table() parameters.
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Try to transform the assigned value
        transformed_value = self._transform_expression(updated_node.value)

        if transformed_value is not updated_node.value:
            return updated_node.with_changes(value=transformed_value)

        return updated_node

    def leave_AnnAssign(
        self,
        original_node: cst.AnnAssign,
        updated_node: cst.AnnAssign
    ) -> cst.AnnAssign:
        """
        Process annotated assignment statements to inject table() parameters.
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        if updated_node.value is None:
            return updated_node

        # Try to transform the assigned value
        transformed_value = self._transform_expression(updated_node.value)

        if transformed_value is not updated_node.value:
            return updated_node.with_changes(value=transformed_value)

        return updated_node

    def leave_Return(
        self,
        original_node: cst.Return,
        updated_node: cst.Return
    ) -> cst.Return:
        """
        Process return statements to inject table() parameters.

        This is where we analyze the entire method chain and inject parameters.
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        if updated_node.value is None:
            return updated_node

        # Try to transform the return value
        transformed_value = self._transform_expression(updated_node.value)

        if transformed_value is not updated_node.value:
            return updated_node.with_changes(value=transformed_value)

        return updated_node

    def _transform_expression(self, node: cst.BaseExpression) -> cst.BaseExpression:
        """
        Recursively transform expressions to inject table() parameters.
        """
        # Handle method chains (Call nodes)
        if isinstance(node, cst.Call):
            # Check if this is the root of a table() call chain
            root_table_call = self._find_table_call_root(node)
            if root_table_call is not None:
                # Extract columns from the entire chain
                group_by_cols = self._extract_group_by_cols(node)
                agg_cols = self._extract_agg_cols(node)

                # Always inject parameters, even if empty (dm.table needs them)
                # This handles cases like .filter([]).group_by([]).agg(...) where no columns are extracted
                return self._inject_parameters(node, root_table_call, group_by_cols, agg_cols)

        return node

    def _find_table_call_root(self, node: cst.BaseExpression) -> Optional[cst.Call]:
        """
        Find the root table() call in a method chain.

        Traverses backwards through the method chain to find dm.table() or data_model.table().

        Args:
            node: Current expression node

        Returns:
            The table() Call node if found, None otherwise
        """
        current = node

        while True:
            if isinstance(current, cst.Call):
                # Check if this is a table() call
                if self._is_table_call(current) and self._has_only_table_name(current):
                    return current

                # If it's a method call, traverse to the object it's called on
                if isinstance(current.func, cst.Attribute):
                    current = current.func.value
                else:
                    break
            elif isinstance(current, cst.Attribute):
                current = current.value
            else:
                break

        return None

    def _is_table_call(self, node: cst.Call) -> bool:
        """Check if this is a dm.table() or data_model.table() call."""
        if not isinstance(node.func, cst.Attribute):
            return False

        if node.func.attr.value != 'table':
            return False

        if isinstance(node.func.value, cst.Name):
            # Get valid variable names from runtime context, default to standard names
            valid_var_names = self.runtime_context.get('valid_var_names', ['dm', 'self', 'data_model'])
            return node.func.value.value in valid_var_names

        return False

    def _has_only_table_name(self, node: cst.Call) -> bool:
        """Check if table() call has only 1 argument (the table name)."""
        # Count non-keyword arguments
        positional_args = [arg for arg in node.args if arg.keyword is None]
        return len(positional_args) == 1

    def _extract_group_by_cols(self, node: cst.BaseExpression) -> List[str]:
        """
        Extract column names from .group_by(), .group_by_dynamic(), and .rolling() calls in the method chain.

        Returns:
            List of column names with table prefixes (e.g., ['products.product_name'])
        """
        columns: Set[str] = set()

        # Find all .group_by() calls
        group_by_calls = self._find_method_calls(node, 'group_by')
        for call in group_by_calls:
            # Extract columns from the list argument
            cols = self._extract_columns_from_call_args(call)
            columns.update(cols)

        # Find all .group_by_dynamic() calls
        group_by_dynamic_calls = self._find_method_calls(node, 'group_by_dynamic')
        for call in group_by_dynamic_calls:
            # Extract first positional arg (index/time column)
            cols = self._extract_columns_from_call_args(call)
            columns.update(cols)

            # Extract columns from group_by= keyword argument
            cols = self._extract_columns_from_keyword_arg(call, 'group_by')
            columns.update(cols)

        # Find all .rolling() calls
        rolling_calls = self._find_method_calls(node, 'rolling')
        for call in rolling_calls:
            # Extract columns from index_column= keyword argument
            cols = self._extract_columns_from_keyword_arg(call, 'index_column')
            columns.update(cols)

            # Extract columns from group_by= keyword argument (if present)
            cols = self._extract_columns_from_keyword_arg(call, 'group_by')
            columns.update(cols)

        # Also extract non-aggregated columns from .select() calls
        select_group_cols, _ = self._extract_select_params(node)
        columns.update(select_group_cols)

        return sorted(list(columns))

    def _extract_agg_cols(self, node: cst.BaseExpression) -> Dict[str, str]:
        """
        Extract column names and aggregation functions from .agg() calls.

        Only includes columns that exist in the base table's schema.
        Columns from joins or calculations are excluded.

        Returns:
            Dict mapping column -> agg function (e.g., {'sales.revenue': 'sum'})
        """
        # Find the base table name from dm.table() call
        base_table = self._extract_table_name_from_chain(node)

        agg_cols: Dict[str, str] = {}

        # Find all .agg() calls
        agg_calls = self._find_method_calls(node, 'agg')

        for call in agg_calls:
            # Extract aggregations and filter by base_table schema
            aggs = self._extract_aggregations_from_call_args(call)

            # Filter: only include columns that exist in base table
            if base_table:
                for col, agg_func in aggs.items():
                    if self._column_exists_in_table(col, base_table):
                        agg_cols[col] = agg_func
                    # else: skip columns not in base table (from joins/calculations)
            else:
                # No table context, include all (backward compatibility)
                agg_cols.update(aggs)

        # Also extract aggregations from .select() calls
        _, select_agg_cols = self._extract_select_params(node)

        # Filter select agg cols same way
        if base_table:
            filtered_select_aggs = {
                col: func for col, func in select_agg_cols.items()
                if self._column_exists_in_table(col, base_table)
            }
            agg_cols.update(filtered_select_aggs)
        else:
            agg_cols.update(select_agg_cols)

        return agg_cols

    def _extract_select_params(
        self,
        node: cst.BaseExpression
    ) -> Tuple[List[str], Dict[str, str]]:
        """
        Extract group_by columns and aggregations from .select() calls.

        In .select() calls:
        - Columns WITHOUT aggregation methods → group_by_cols
        - Columns WITH aggregation methods → agg_cols

        Args:
            node: Expression node to analyze

        Returns:
            Tuple of (group_by_cols, agg_cols)

        Example:
            .select(
                pl.col('item_id'),           # → group_by_cols
                pl.col('revenue').rank()     # → agg_cols
            )
        """
        group_by_cols = []
        agg_cols = {}

        # Find all .select() calls in the method chain
        select_calls = self._find_method_calls(node, 'select')

        for select_call in select_calls:
            # Process each argument in the .select() call
            for arg in select_call.args:
                # Try to extract column name and aggregation function
                result = self._extract_column_and_agg(arg.value)

                if result:
                    col_name, agg_func = result

                    if agg_func:
                        # Has aggregation → add to agg_cols
                        # Strip table prefix for consistency
                        clean_col = col_name.split('.')[-1] if '.' in col_name else col_name
                        agg_cols[clean_col] = agg_func
                    else:
                        # No aggregation → add to group_by_cols
                        if col_name not in group_by_cols:
                            group_by_cols.append(col_name)

        return group_by_cols, agg_cols

    def _find_method_calls(self, node: cst.BaseExpression, method_name: str) -> List[cst.Call]:
        """
        Find all calls to a specific method in the expression tree.

        Args:
            node: Expression to search
            method_name: Method name to find (e.g., 'group_by', 'agg')

        Returns:
            List of Call nodes matching the method name
        """
        calls = []

        # Use a visitor to collect method calls
        class MethodCallCollector(cst.CSTVisitor):
            def visit_Call(self, node: cst.Call) -> None:
                if isinstance(node.func, cst.Attribute):
                    if node.func.attr.value == method_name:
                        calls.append(node)

        # Create a temporary module to visit
        # Wrap the expression in a statement so we can visit it
        try:
            stmt = cst.SimpleStatementLine(body=[cst.Expr(value=node)])
            collector = MethodCallCollector()
            stmt.visit(collector)
        except Exception:
            # If wrapping fails, return empty list
            pass

        return calls

    def _extract_table_name_from_chain(self, node: cst.BaseExpression) -> Optional[str]:
        """
        Extract table name from dm.table('table_name') call in chain.

        Returns:
            Table name string or None if not found
        """
        # Find table() call in the chain
        table_call = self._find_table_call_root(node)

        if table_call and len(table_call.args) > 0:
            # Extract table name from first argument
            first_arg = table_call.args[0].value
            if isinstance(first_arg, cst.SimpleString):
                return first_arg.value.strip('\'"')

        return None

    def _column_exists_in_table(self, col_name: str, table_name: str) -> bool:
        """
        Check if column exists in the specified table's schema.

        Args:
            col_name: Column name (with or without table prefix)
            table_name: Table name to check

        Returns:
            True if column exists in table schema, False otherwise
        """
        table_schemas = self.runtime_context.get('table_schemas', {})

        # No schema context available - include all columns (backward compatibility)
        if not table_schemas or table_name not in table_schemas:
            return True

        schema_cols = table_schemas[table_name]

        # Strip table prefix if present
        clean_col = col_name.split('.')[-1] if '.' in col_name else col_name

        return clean_col in schema_cols

    def _extract_columns_from_call_args(self, call: cst.Call) -> List[str]:
        """
        Extract column names from call arguments.

        Handles: .group_by([pl.col('products.product_name'), pl.col('sales.revenue')])

        Returns:
            List of column names (e.g., ['products.product_name', 'sales.revenue'])
        """
        columns = []

        for arg in call.args:
            if arg.keyword is None:  # Positional argument
                cols = self._extract_columns_from_expression(arg.value)
                columns.extend(cols)

        return columns

    def _extract_columns_from_keyword_arg(self, call: cst.Call, keyword_name: str) -> List[str]:
        """
        Extract column names from a specific keyword argument.

        Handles: .group_by_dynamic(..., group_by=[pl.col('store_name')])
                .rolling(index_column='date', ...)

        Args:
            call: The Call node
            keyword_name: Name of the keyword argument (e.g., 'group_by', 'index_column')

        Returns:
            List of column names
        """
        columns = []

        for arg in call.args:
            if arg.keyword is not None and arg.keyword.value == keyword_name:
                # Found the keyword argument, extract columns from its value
                cols = self._extract_columns_from_expression(arg.value)
                columns.extend(cols)

        return columns

    def _extract_columns_from_expression(self, expr: cst.BaseExpression) -> List[str]:
        """
        Extract column names from an expression.

        Handles:
        - List: [pl.col('col1'), pl.col('col2')]
        - Single pl.col() call
        - String literal: 'column_name'
        """
        columns = []

        if isinstance(expr, cst.List):
            for element in expr.elements:
                if isinstance(element, cst.Element):
                    cols = self._extract_columns_from_expression(element.value)
                    columns.extend(cols)
        elif isinstance(expr, cst.Call):
            # Check if this is pl.col('column_name')
            col_name = self._extract_column_from_pl_col(expr)
            if col_name:
                columns.append(col_name)
        elif isinstance(expr, cst.SimpleString):
            # Handle raw string column names like 'sales.date'
            col_name = expr.value.strip('\'"')
            if col_name:
                columns.append(col_name)

        return columns

    def _extract_column_from_pl_col(self, call: cst.Call) -> Optional[str]:
        """
        Extract column name from pl.col('column_name') call.

        Returns:
            Column name as string (e.g., 'products.product_name') or None
        """
        # Check if this is pl.col() call
        if not isinstance(call.func, cst.Attribute):
            return None

        if call.func.attr.value != 'col':
            return None

        if not isinstance(call.func.value, cst.Name) or call.func.value.value != 'pl':
            return None

        # Extract the string argument
        if len(call.args) > 0:
            arg = call.args[0]
            if isinstance(arg.value, cst.SimpleString):
                # Remove quotes
                return arg.value.value.strip('\'"')

        return None

    def _extract_aggregations_from_call_args(self, call: cst.Call) -> Dict[str, str]:
        """
        Extract aggregations from .agg() call arguments.

        Handles: .agg(pl.col('sales.revenue').sum().alias('total_revenue'))

        Returns:
            Dict mapping column -> agg function (e.g., {'sales.revenue': 'sum'})
        """
        agg_cols = {}

        for arg in call.args:
            if arg.keyword is None:  # Positional argument
                aggs = self._extract_aggregations_from_expression(arg.value)
                agg_cols.update(aggs)

        return agg_cols

    def _extract_aggregations_from_expression(self, expr: cst.BaseExpression) -> Dict[str, str]:
        """
        Extract aggregations from an expression.

        Handles:
        - Single: pl.col('sales.revenue').sum()
        - Multiple: [pl.col('col1').sum(), pl.col('col2').max()]
        """
        aggs = {}

        if isinstance(expr, cst.List):
            for element in expr.elements:
                if isinstance(element, cst.Element):
                    extracted = self._extract_aggregations_from_expression(element.value)
                    aggs.update(extracted)
        elif isinstance(expr, cst.Call):
            # This might be pl.col('col').sum().alias('name')
            # We need to find the pl.col() and the aggregation function
            col_name, agg_func = self._extract_column_and_agg(expr)
            if col_name and agg_func:
                aggs[col_name] = agg_func

        return aggs

    def _extract_column_and_agg(self, expr: cst.BaseExpression) -> tuple[Optional[str], Optional[str]]:
        """
        Extract column name and aggregation function from method chain.

        Handles: pl.col('sales.revenue').sum().alias('total_revenue')

        Returns:
            Tuple of (column_name, agg_function) or (None, None)
        """
        col_name = None
        agg_func = None

        # Traverse the method chain backwards
        current = expr
        while isinstance(current, cst.Call):
            # Check if this is pl.col() - if so, extract and stop
            col_name_from_current = self._extract_column_from_pl_col(current)
            if col_name_from_current is not None:
                col_name = col_name_from_current
                break

            # Check if this is an aggregation method
            if isinstance(current.func, cst.Attribute):
                method_name = current.func.attr.value

                # Check if this is an aggregation method
                if method_name in ['sum', 'mean', 'max', 'min', 'count', 'first', 'last', 'std', 'var', 'rank']:
                    agg_func = method_name

                # Traverse to the object being called on
                current = current.func.value
            else:
                break

        return col_name, agg_func

    def _inject_parameters(
        self,
        chain_root: cst.BaseExpression,
        table_call: cst.Call,
        group_by_cols: List[str],
        agg_cols: Dict[str, str]
    ) -> cst.BaseExpression:
        """
        Inject parameters into the table() call within the method chain.

        Args:
            chain_root: Root of the method chain
            table_call: The table() call to inject parameters into
            group_by_cols: List of group by columns
            agg_cols: Dict of aggregations

        Returns:
            Updated expression with injected parameters
        """
        # Build new arguments for table() call
        new_args = list(table_call.args)  # Keep existing table name argument

        # Add group_by_cols as second argument
        group_by_list = cst.List([
            cst.Element(value=cst.SimpleString(f"'{col}'"))
            for col in group_by_cols
        ])
        new_args.append(cst.Arg(value=group_by_list))

        # Add agg_cols as third argument
        agg_dict = cst.Dict([
            cst.DictElement(
                key=cst.SimpleString(f"'{col}'"),
                value=cst.SimpleString(f"'{func}'")
            )
            for col, func in agg_cols.items()
        ])
        new_args.append(cst.Arg(value=agg_dict))

        # Add allow_pre_aggs as fourth argument (from query context)
        qc = self.runtime_context.get('qc', {})
        allow_pre_aggs = qc.get('allow_pre_aggs', True)
        allow_pre_aggs_node = cst.Name('True') if allow_pre_aggs else cst.Name('False')
        new_args.append(cst.Arg(value=allow_pre_aggs_node))

        # Create updated table() call
        updated_table_call = table_call.with_changes(args=new_args)

        # Replace the table() call in the chain
        return self._replace_node_in_tree(chain_root, table_call, updated_table_call)

    def _replace_node_in_tree(
        self,
        tree: cst.BaseExpression,
        old_node: cst.CSTNode,
        new_node: cst.CSTNode
    ) -> cst.BaseExpression:
        """
        Replace a node in the expression tree.

        This is a simplified implementation that handles the common case
        of replacing a table() call in a method chain.
        """
        if tree is old_node:
            return new_node

        if isinstance(tree, cst.Call):
            # Check if the function being called contains the old node
            if isinstance(tree.func, cst.Attribute):
                updated_value = self._replace_node_in_tree(tree.func.value, old_node, new_node)
                if updated_value is not tree.func.value:
                    return tree.with_changes(
                        func=tree.func.with_changes(value=updated_value)
                    )

        return tree


def inject_table_parameters(
    source_code: str,
    function_name: str,
    runtime_context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Inject parameters into table() calls based on method chain analysis.

    Args:
        source_code: Python source code containing the function
        function_name: Name of function to transform
        runtime_context: Dict containing runtime variables (e.g., 'qc')

    Returns:
        Transformed source code with table() parameters injected

    Example:
        >>> code = '''
        ... def total_revenue(qc):
        ...     return data_model.table('sales').group_by([pl.col('products.product_name')])
        ... '''
        >>> result = inject_table_parameters(code, 'total_revenue', {'qc': {...}})
        >>> # Result: data_model.table('sales', ['products.product_name'], {}, True)...
    """
    module = cst.parse_module(source_code)
    transformer = InjectTableParameters(function_name, runtime_context)
    new_module = module.visit(transformer)
    return new_module.code
