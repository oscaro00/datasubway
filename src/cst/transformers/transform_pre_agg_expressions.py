"""
Transformer to adjust aggregation expressions when using pre-aggregations.

This module provides a libcst transformer that:
1. Detects if code uses pre-aggregations (by looking for self.pre_agg_directory)
2. Transforms column names in .agg() expressions to match pre-agg column naming
3. Decomposes complex aggregations (mean, std, var) into formulas using stored components
4. Transforms window functions (rank) to operate on pre-aggregated columns

Example transformations:
    >>> # Simple aggregation
    >>> pl.col('revenue').sum()  →  pl.col('revenue-sum').sum()

    >>> # Window function
    >>> pl.col('revenue').rank('min', descending=True)  →  pl.col('revenue-sum').rank('min', descending=True)

    >>> # Decomposed aggregation
    >>> pl.col('revenue').mean()  →  pl.col('revenue-mean-sum').sum() / pl.col('revenue-mean-count').sum()
"""

from typing import Optional, Dict, Any, Union
import libcst as cst
import libcst.matchers as m


class TransformPreAggExpressions(cst.CSTTransformer):
    """
    Transform aggregation expressions to use pre-agg column names.

    This transformer modifies .agg() method arguments to use the correct
    column names from pre-aggregations and decomposes complex aggregations
    (like mean, std, var) into formulas that can be correctly re-aggregated.
    """

    def __init__(
        self,
        function_name: str,
        pre_agg_metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the transformer.

        Args:
            function_name: Name of the function to transform
            pre_agg_metadata: Optional metadata about which pre-agg is being used
        """
        super().__init__()
        self.function_name = function_name
        self.pre_agg_metadata = pre_agg_metadata
        self.current_function: Optional[str] = None
        self.inside_agg: int = 0  # Track nesting level

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

    def _is_part_of_pre_agg_chain(self, node: cst.Call) -> bool:
        """
        Check if this node is part of a chain that starts with a pre-agg scan.

        Walks up the tree looking for: pl.scan_parquet(...pre_agg_directory...)

        IMPORTANT: Stops at .join() calls. If we encounter a join before reaching
        the pre-agg scan, returns False because expressions after joins may operate
        on joined data, not pre-agg columns.

        Args:
            node: Call node to check

        Returns:
            True if part of pre-agg chain (before any joins), False otherwise
        """
        current = node

        while True:
            # Check if current node is pl.scan_parquet with pre_agg_directory
            if self._is_pre_agg_scan(current):
                return True

            # Check if current node is a .join() call - stop here!
            # Expressions after joins may operate on joined data, not pre-agg data
            if self._is_join_call(current):
                return False

            # Try to go up one level in the chain
            if isinstance(current.func, cst.Attribute) and isinstance(current.func.value, cst.Call):
                current = current.func.value
            else:
                # Reached the top of the chain
                return False

    def _is_join_call(self, node: cst.Call) -> bool:
        """
        Check if this is a .join() method call.

        Args:
            node: Call node to check

        Returns:
            True if this is a .join() call
        """
        if not isinstance(node.func, cst.Attribute):
            return False
        return node.func.attr.value == 'join'

    def _is_pre_agg_scan(self, node: cst.Call) -> bool:
        """
        Check if this is a pl.scan_parquet call with pre_agg_directory path.

        Pattern: pl.scan_parquet(self.pre_agg_directory / 'xxx.parquet')

        Args:
            node: Call node to check

        Returns:
            True if this is a pre-agg scan
        """
        # Check if it's pl.scan_parquet
        if not isinstance(node.func, cst.Attribute):
            return False
        if node.func.attr.value != 'scan_parquet':
            return False
        if not isinstance(node.func.value, cst.Name) or node.func.value.value != 'pl':
            return False

        # Check if argument contains pre_agg_directory
        if len(node.args) == 0:
            return False

        # Look for BinaryOperation with pre_agg_directory (path / 'file.parquet')
        arg = node.args[0].value
        return self._contains_pre_agg_directory(arg)

    def _contains_pre_agg_directory(self, node: cst.BaseExpression) -> bool:
        """
        Recursively check if expression contains self.pre_agg_directory.

        Args:
            node: Expression node to check

        Returns:
            True if contains self.pre_agg_directory
        """
        if isinstance(node, cst.Attribute):
            return (node.attr.value == 'pre_agg_directory' and
                    isinstance(node.value, cst.Name) and
                    node.value.value == 'self')
        elif isinstance(node, cst.BinaryOperation):
            return (self._contains_pre_agg_directory(node.left) or
                    self._contains_pre_agg_directory(node.right))
        return False

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> Union[cst.Call, cst.BaseExpression]:
        """
        Transform aggregation expressions in .agg() calls when using pre-agg.

        Only transforms expressions that are part of a pre-aggregation chain.

        Returns:
            Transformed call node or expression
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Only transform if this specific call is part of a pre-agg chain
        if self._is_agg_method(updated_node) or self._is_select_method(updated_node):
            if self._is_part_of_pre_agg_chain(updated_node):
                return self._transform_agg_call(updated_node)

        return updated_node

    def _transform_agg_call(self, node: cst.Call) -> cst.Call:
        """
        Transform all aggregation expressions in .agg() arguments.

        Args:
            node: The .agg() call node

        Returns:
            .agg() call with transformed arguments
        """
        new_args = []

        for arg in node.args:
            # Transform each argument expression
            new_expr = self._transform_expression(arg.value)
            new_args.append(arg.with_changes(value=new_expr))

        return node.with_changes(args=new_args)

    def _is_agg_method(self, node: cst.Call) -> bool:
        """Check if this is an .agg() method call."""
        if not isinstance(node.func, cst.Attribute):
            return False
        return node.func.attr.value == 'agg'

    def _is_select_method(self, node: cst.Call) -> bool:
        """Check if this is a .select() method call."""
        if not isinstance(node.func, cst.Attribute):
            return False
        return node.func.attr.value == 'select'

    def _transform_expression(
        self,
        node: Union[cst.Call, cst.BaseExpression]
    ) -> cst.BaseExpression:
        """
        Transform a single aggregation expression recursively.

        Handles:
        - pl.col('col').sum() → pl.col('col-sum').sum()
        - pl.col('col').mean() → pl.col('col-mean-sum').sum() / pl.col('col-mean-count').sum()
        - Expressions with .alias()
        - Nested expressions like (pl.col('x').sum() * 2).alias('y')
        """
        if not isinstance(node, cst.Call):
            # For non-call nodes, check if they have children to transform
            if isinstance(node, cst.BinaryOperation):
                return node.with_changes(
                    left=self._transform_expression(node.left),
                    right=self._transform_expression(node.right)
                )
            return node

        # Check if this matches pl.col('name').agg_func()
        if self._is_simple_agg_expression(node):
            return self._transform_simple_agg(node)

        # Check if this matches pl.col('name').agg_func().alias('name')
        if self._is_aliased_agg_expression(node):
            return self._transform_aliased_agg(node)

        # For other calls (like .alias() with complex expressions),
        # recursively transform their arguments
        if isinstance(node.func, cst.Attribute):
            # Transform the value (chain before this call)
            if isinstance(node.func.value, (cst.Call, cst.BinaryOperation)):
                new_value = self._transform_expression(node.func.value)
                new_func = node.func.with_changes(value=new_value)
                return node.with_changes(func=new_func)

        return node

    def _is_simple_agg_expression(self, node: cst.Call) -> bool:
        """
        Check if this is pl.col('name').agg_func() pattern.

        Returns:
            True if matches the pattern
        """
        if not isinstance(node.func, cst.Attribute):
            return False

        # Check if func is an aggregation method
        agg_func = node.func.attr.value
        if agg_func not in ['sum', 'min', 'max', 'count', 'len', 'mean', 'std', 'var', 'first', 'last', 'rank']:
            return False

        # Check if the value is a pl.col() call
        if not isinstance(node.func.value, cst.Call):
            return False

        col_call = node.func.value
        if not isinstance(col_call.func, cst.Attribute):
            return False

        # Check if it's pl.col
        if not (isinstance(col_call.func.value, cst.Name) and
                col_call.func.value.value == 'pl' and
                col_call.func.attr.value == 'col'):
            return False

        return True

    def _is_aliased_agg_expression(self, node: cst.Call) -> bool:
        """Check if this is pl.col('name').agg_func().alias('name') pattern."""
        if not isinstance(node.func, cst.Attribute):
            return False

        if node.func.attr.value != 'alias':
            return False

        # Check if the value is a simple agg expression
        if not isinstance(node.func.value, cst.Call):
            return False

        return self._is_simple_agg_expression(node.func.value)

    def _transform_simple_agg(self, node: cst.Call, add_alias: bool = True) -> cst.BaseExpression:
        """
        Transform simple aggregation: pl.col('col').sum()

        Maps based on _get_pre_agg_calculation logic and optionally adds .alias() to normalize column names.

        Args:
            node: The aggregation call node
            add_alias: If True, add .alias() to restore original column name. Set to False when
                      there's already an explicit user-provided alias.
        """
        col_name = self._extract_column_name(node)
        agg_func = self._extract_agg_function(node)

        # Strip table prefix from column name (pre-aggs don't have it)
        clean_col = col_name.split('.')[-1] if '.' in col_name else col_name

        # Check if column already has a pre-agg suffix - if so, don't transform again
        if self._has_pre_agg_suffix(clean_col):
            return node

        # Check if this column is in the pre-agg metadata
        # Only transform columns that are actually in the pre-aggregation
        if self.pre_agg_metadata:
            pre_agg_cols = self.pre_agg_metadata.get('aggregations', {})
            if clean_col not in pre_agg_cols:
                # Column not in pre-agg (e.g., from join) - don't transform
                return node

        match agg_func:
            case 'sum':
                transformed = self._build_simple_transform(node, clean_col, 'sum')
            case 'min':
                transformed = self._build_simple_transform(node, clean_col, 'min')
            case 'max':
                transformed = self._build_simple_transform(node, clean_col, 'max')
            case 'count' | 'len':
                transformed = self._build_simple_transform(node, clean_col, 'count')
            case 'first':
                transformed = self._build_simple_transform(node, clean_col, 'first')
            case 'last':
                transformed = self._build_simple_transform(node, clean_col, 'last')
            case 'rank':
                transformed = self._build_rank_transform(node, clean_col)
            case 'mean':
                transformed = self._build_mean_transform(clean_col)
            case 'std':
                transformed = self._build_std_transform(clean_col)
            case 'var':
                transformed = self._build_var_transform(clean_col)
            case _:
                # Unsupported aggregation - leave unchanged
                return node

        # Add alias to normalize column name if requested
        if add_alias:
            return self._add_alias_to_expression(transformed, clean_col)
        else:
            return transformed

    def _has_pre_agg_suffix(self, col_name: str) -> bool:
        """
        Check if column name already has a pre-agg suffix.

        Args:
            col_name: Column name to check

        Returns:
            True if column has a pre-agg suffix
        """
        pre_agg_suffixes = [
            '-sum', '-min', '-max', '-count', '-first', '-last',
            '-null_count', '-unique-set',
            '-mean-sum', '-mean-count',
            '-std-sum', '-std-sumsq', '-std-count',
            '-var-sum', '-var-sumsq', '-var-count'
        ]
        return any(col_name.endswith(suffix) for suffix in pre_agg_suffixes)

    def _add_alias_to_expression(
        self,
        expr: cst.BaseExpression,
        alias_name: str
    ) -> cst.Call:
        """
        Wrap expression with .alias('name') to normalize column names.

        Transforms pre-agg column names back to their original names for user-facing results.
        For example, pl.col('revenue-sum').sum() becomes pl.col('revenue-sum').sum().alias('revenue')

        Args:
            expr: Expression to wrap (e.g., pl.col('revenue-sum').sum())
            alias_name: Alias name (original column name without pre-agg suffix)

        Returns:
            Call node for expr.alias('alias_name')
        """
        return cst.Call(
            func=cst.Attribute(
                value=expr,
                attr=cst.Name('alias')
            ),
            args=[cst.Arg(value=cst.SimpleString(f"'{alias_name}'"))]
        )

    def _transform_aliased_agg(self, node: cst.Call) -> cst.Call:
        """
        Transform aliased aggregation: pl.col('col').sum().alias('total')

        Transforms the inner aggregation and preserves the user's explicit alias.
        Does not add automatic column normalization since user has provided their own alias.
        """
        # Transform the inner aggregation without adding automatic alias
        # (user's explicit alias takes precedence)
        inner_agg = node.func.value
        transformed_inner = self._transform_simple_agg(inner_agg, add_alias=False)

        # Rebuild with transformed inner and same user-provided alias
        return node.with_changes(
            func=node.func.with_changes(
                value=transformed_inner
            )
        )

    def _build_simple_transform(
        self,
        original_node: cst.Call,
        col_name: str,
        suffix: str
    ) -> cst.Call:
        """
        Build simple transformation: pl.col('col').func() → pl.col('col-suffix').func()

        Args:
            original_node: Original aggregation call
            col_name: Clean column name (without table prefix)
            suffix: Suffix to add (e.g., 'sum', 'min', 'max')

        Returns:
            Transformed call with new column name
        """
        new_col_name = f'{col_name}-{suffix}'

        # Build pl.col('col-suffix')
        new_col_call = cst.Call(
            func=cst.Attribute(
                value=cst.Name('pl'),
                attr=cst.Name('col')
            ),
            args=[cst.Arg(value=cst.SimpleString(f"'{new_col_name}'"))]
        )

        # Build pl.col('col-suffix').func()
        agg_func = self._extract_agg_function(original_node)
        return cst.Call(
            func=cst.Attribute(
                value=new_col_call,
                attr=cst.Name(agg_func)
            ),
            args=[]
        )

    def _build_rank_transform(
        self,
        original_node: cst.Call,
        col_name: str
    ) -> cst.Call:
        """
        Build rank transformation: pl.col('col').rank(...) → pl.col('col-sum').rank(...)

        Unlike simple aggregations, rank() is a window function that operates on
        pre-aggregated values. We transform the column name but preserve all
        rank() arguments (method, descending, seed).

        Args:
            original_node: Original rank() call node
            col_name: Clean column name (without table prefix)

        Returns:
            Transformed call with new column name and preserved rank() arguments
        """
        # Use 'sum' as the default pre-agg suffix for rank operations
        new_col_name = f'{col_name}-sum'

        # Build pl.col('col-sum')
        new_col_call = cst.Call(
            func=cst.Attribute(
                value=cst.Name('pl'),
                attr=cst.Name('col')
            ),
            args=[cst.Arg(value=cst.SimpleString(f"'{new_col_name}'"))]
        )

        # Build pl.col('col-sum').rank(...) with preserved arguments
        return cst.Call(
            func=cst.Attribute(
                value=new_col_call,
                attr=cst.Name('rank')
            ),
            args=original_node.args  # KEY: Preserve all rank() arguments
        )

    def _build_mean_transform(self, col_name: str) -> cst.BinaryOperation:
        """
        Build mean transformation.

        Formula: pl.col('col-mean-sum').sum() / pl.col('col-mean-count').sum()
        """
        numerator = self._build_pl_col_agg(f'{col_name}-mean-sum', 'sum')
        denominator = self._build_pl_col_agg(f'{col_name}-mean-count', 'sum')

        return cst.BinaryOperation(
            left=numerator,
            operator=cst.Divide(),
            right=denominator
        )

    def _build_std_transform(self, col_name: str) -> cst.Call:
        """
        Build standard deviation transformation.

        Formula: sqrt((sum(sumsq) - sum(sum)^2/n) / (n-1))
        """
        # sum(sumsq)
        sum_sumsq = self._build_pl_col_agg(f'{col_name}-std-sumsq', 'sum')

        # sum(sum)
        sum_sum = self._build_pl_col_agg(f'{col_name}-std-sum', 'sum')

        # n = sum(count)
        n = self._build_pl_col_agg(f'{col_name}-std-count', 'sum')

        # sum(sum)^2
        sum_squared = cst.Call(
            func=cst.Attribute(
                value=sum_sum,
                attr=cst.Name('pow')
            ),
            args=[cst.Arg(value=cst.Integer('2'))]
        )

        # sum(sum)^2 / n
        sum_squared_over_n = cst.BinaryOperation(
            left=sum_squared,
            operator=cst.Divide(),
            right=n
        )

        # sum(sumsq) - sum(sum)^2/n
        variance_numerator = cst.BinaryOperation(
            left=sum_sumsq,
            operator=cst.Subtract(),
            right=sum_squared_over_n
        )

        # n - 1
        n_minus_1 = cst.BinaryOperation(
            left=n,
            operator=cst.Subtract(),
            right=cst.Integer('1')
        )

        # (sum(sumsq) - sum(sum)^2/n) / (n-1)
        variance = cst.BinaryOperation(
            left=variance_numerator,
            operator=cst.Divide(),
            right=n_minus_1
        )

        # sqrt(variance)
        return cst.Call(
            func=cst.Attribute(
                value=variance,
                attr=cst.Name('sqrt')
            ),
            args=[]
        )

    def _build_var_transform(self, col_name: str) -> cst.BinaryOperation:
        """
        Build variance transformation.

        Formula: (sum(sumsq) - sum(sum)^2/n) / (n-1)
        """
        # sum(sumsq)
        sum_sumsq = self._build_pl_col_agg(f'{col_name}-var-sumsq', 'sum')

        # sum(sum)
        sum_sum = self._build_pl_col_agg(f'{col_name}-var-sum', 'sum')

        # n = sum(count)
        n = self._build_pl_col_agg(f'{col_name}-var-count', 'sum')

        # sum(sum)^2
        sum_squared = cst.Call(
            func=cst.Attribute(
                value=sum_sum,
                attr=cst.Name('pow')
            ),
            args=[cst.Arg(value=cst.Integer('2'))]
        )

        # sum(sum)^2 / n
        sum_squared_over_n = cst.BinaryOperation(
            left=sum_squared,
            operator=cst.Divide(),
            right=n
        )

        # sum(sumsq) - sum(sum)^2/n
        variance_numerator = cst.BinaryOperation(
            left=sum_sumsq,
            operator=cst.Subtract(),
            right=sum_squared_over_n
        )

        # n - 1
        n_minus_1 = cst.BinaryOperation(
            left=n,
            operator=cst.Subtract(),
            right=cst.Integer('1')
        )

        # (sum(sumsq) - sum(sum)^2/n) / (n-1)
        return cst.BinaryOperation(
            left=variance_numerator,
            operator=cst.Divide(),
            right=n_minus_1
        )

    def _build_pl_col_agg(self, col_name: str, agg_func: str) -> cst.Call:
        """
        Build: pl.col('col_name').agg_func()

        Args:
            col_name: Column name (with pre-agg suffix)
            agg_func: Aggregation function name

        Returns:
            CST Call node
        """
        col_call = cst.Call(
            func=cst.Attribute(
                value=cst.Name('pl'),
                attr=cst.Name('col')
            ),
            args=[cst.Arg(value=cst.SimpleString(f"'{col_name}'"))]
        )

        return cst.Call(
            func=cst.Attribute(
                value=col_call,
                attr=cst.Name(agg_func)
            ),
            args=[]
        )

    def _extract_column_name(self, node: cst.Call) -> str:
        """
        Extract column name from pl.col('name').method() chain.

        Args:
            node: Call node for the aggregation

        Returns:
            Column name as string
        """
        # Navigate: node.func.value is the pl.col(...) part
        col_call = node.func.value
        # Get the first argument (the column name string)
        col_arg = col_call.args[0].value

        # Extract the string value
        if isinstance(col_arg, cst.SimpleString):
            return col_arg.value.strip('\'"')
        elif isinstance(col_arg, cst.ConcatenatedString):
            # Handle f-strings or concatenated strings
            return col_arg.left.value.strip('\'"') if isinstance(col_arg.left, cst.SimpleString) else ''

        return ''

    def _extract_agg_function(self, node: cst.Call) -> str:
        """
        Extract aggregation function name from pl.col('name').func().

        Args:
            node: Call node for the aggregation

        Returns:
            Function name as string (e.g., 'sum', 'mean')
        """
        if isinstance(node.func, cst.Attribute):
            return node.func.attr.value
        return ''


def transform_pre_agg_expressions(
    source_code: str,
    function_name: str,
    pre_agg_metadata: Optional[Dict[str, Any]] = None
) -> str:
    """
    Transform aggregation expressions to use pre-agg column names.

    This function applies the TransformPreAggExpressions transformer to
    adjust column names and decompose complex aggregations when using
    pre-aggregated data.

    Args:
        source_code: Python source code containing the function
        function_name: Name of function to transform
        pre_agg_metadata: Optional metadata about which pre-agg is being used

    Returns:
        Transformed source code with adjusted aggregation expressions

    Example:
        >>> code = '''
        ... def my_measure():
        ...     return df.agg(pl.col('revenue').mean())
        ... '''
        >>> result = transform_pre_agg_expressions(code, 'my_measure')
        >>> # Result: df.agg(pl.col('revenue-mean-sum').sum() / pl.col('revenue-mean-count').sum())
    """
    module = cst.parse_module(source_code)
    transformer = TransformPreAggExpressions(function_name, pre_agg_metadata)
    new_module = module.visit(transformer)
    return new_module.code
