"""
Transformer to adjust aggregation expressions when using pre-aggregations.

This module provides a libcst transformer that:
1. Detects if code uses pre-aggregations (by looking for self.pre_agg_directory)
2. Transforms column names in .agg() expressions to match pre-agg column naming
3. Decomposes complex aggregations (mean, std, var) into formulas using stored components

Example transformations:
    >>> # Simple aggregation
    >>> pl.col('revenue').sum()  →  pl.col('revenue-sum').sum()

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
        self.uses_pre_agg: Optional[bool] = None

    def visit_Module(self, node: cst.Module) -> None:
        """
        Scan entire module to detect pre-agg usage.

        Looks for self.pre_agg_directory to determine if this code
        is using a pre-aggregation.
        """
        # Use libcst matcher to find: self.pre_agg_directory
        pre_agg_pattern = m.Attribute(
            value=m.Name('self'),
            attr=m.Name('pre_agg_directory')
        )

        matches = m.findall(node, pre_agg_pattern)
        self.uses_pre_agg = len(matches) > 0

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

    def visit_Call(self, node: cst.Call) -> None:
        """Track when entering .agg() method."""
        if self._is_agg_method(node):
            self.inside_agg += 1

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> Union[cst.Call, cst.BaseExpression]:
        """
        Transform aggregation expressions if inside .agg() and using pre-agg.

        Returns:
            Transformed call node or expression
        """
        # Check if leaving an agg method
        was_inside_agg = self.inside_agg > 0
        if self._is_agg_method(original_node):
            self.inside_agg -= 1

        # Only transform if in target function, using pre-agg, and inside agg
        if (self.current_function != self.function_name or
            not self.uses_pre_agg or
            self.inside_agg == 0):
            return updated_node

        # Don't transform the .agg() call itself, only its contents
        if self._is_agg_method(original_node):
            return updated_node

        # Transform aggregation expressions
        return self._transform_expression(updated_node)

    def _is_agg_method(self, node: cst.Call) -> bool:
        """Check if this is an .agg() method call."""
        if not isinstance(node.func, cst.Attribute):
            return False
        return node.func.attr.value == 'agg'

    def _transform_expression(
        self,
        node: Union[cst.Call, cst.BaseExpression]
    ) -> cst.BaseExpression:
        """
        Transform a single aggregation expression.

        Handles:
        - pl.col('col').sum() → pl.col('col-sum').sum()
        - pl.col('col').mean() → pl.col('col-mean-sum').sum() / pl.col('col-mean-count').sum()
        - Expressions with .alias()
        """
        if not isinstance(node, cst.Call):
            return node

        # Check if this matches pl.col('name').agg_func()
        if self._is_simple_agg_expression(node):
            return self._transform_simple_agg(node)

        # Check if this matches pl.col('name').agg_func().alias('name')
        if self._is_aliased_agg_expression(node):
            return self._transform_aliased_agg(node)

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
        if agg_func not in ['sum', 'min', 'max', 'count', 'len', 'mean', 'std', 'var', 'first', 'last']:
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

    def _transform_simple_agg(self, node: cst.Call) -> cst.BaseExpression:
        """
        Transform simple aggregation: pl.col('col').sum()

        Maps based on _get_pre_agg_calculation logic.
        """
        col_name = self._extract_column_name(node)
        agg_func = self._extract_agg_function(node)

        # Strip table prefix from column name (pre-aggs don't have it)
        clean_col = col_name.split('.')[-1] if '.' in col_name else col_name

        match agg_func:
            case 'sum':
                return self._build_simple_transform(node, clean_col, 'sum')
            case 'min':
                return self._build_simple_transform(node, clean_col, 'min')
            case 'max':
                return self._build_simple_transform(node, clean_col, 'max')
            case 'count' | 'len':
                return self._build_simple_transform(node, clean_col, 'count')
            case 'first':
                return self._build_simple_transform(node, clean_col, 'first')
            case 'last':
                return self._build_simple_transform(node, clean_col, 'last')
            case 'mean':
                return self._build_mean_transform(clean_col)
            case 'std':
                return self._build_std_transform(clean_col)
            case 'var':
                return self._build_var_transform(clean_col)
            case _:
                # Unsupported aggregation - leave unchanged
                return node

    def _transform_aliased_agg(self, node: cst.Call) -> cst.Call:
        """
        Transform aliased aggregation: pl.col('col').sum().alias('total')

        Transforms the inner aggregation and preserves the alias.
        """
        # Transform the inner aggregation
        inner_agg = node.func.value
        transformed_inner = self._transform_simple_agg(inner_agg)

        # Rebuild with transformed inner and same alias
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
