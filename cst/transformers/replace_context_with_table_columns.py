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

from typing import Dict, Any, Union, Optional, Literal
import libcst as cst
import libcst.matchers as m
from column_context import Allow, Exclude


class ReplaceContextWithColumns(cst.CSTTransformer):
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

    def leave_Call(
        self,
        original_node: cst.Call,
        updated_node: cst.Call
    ) -> Union[cst.Call, cst.List]:
        """
        Replace Allow() and Exclude() calls with resolved column lists.

        Steps:
        1. Check if we're in the target function
        2. Check if call is Allow() or Exclude()
        3. Convert CST node to code string
        4. Evaluate to get Allow/Exclude instance
        5. Call get_relevant_columns() to resolve
        6. Create CST List node with results
        7. Return List node to replace Call node
        """
        # Only transform in target function
        if self.current_function != self.function_name:
            return updated_node

        # Only transform Allow() or Exclude() calls
        if not (m.matches(updated_node, m.Call(func=m.Name('Allow'))) or
                m.matches(updated_node, m.Call(func=m.Name('Exclude')))):
            return updated_node

        try:
            # Convert CST node to executable code string
            temp_module = cst.Module(body=[cst.Expr(value=updated_node)])
            call_code = temp_module.code.strip()

            # Prepare restricted eval globals
            eval_globals = {
                'Allow': Allow,
                'Exclude': Exclude,
                **self.runtime_context
            }

            # Evaluate to get instance
            instance = eval(call_code, eval_globals)

            # Get resolved columns and create appropriate list
            if self.output_type == 'polar_col':
                return self._create_polars_list(instance)
            else:
                return self._create_string_list(instance, self.output_type)

        except (NameError, KeyError, AttributeError) as e:
            # Missing runtime context variable - leave unchanged
            # Could add logging here for debugging
            return updated_node
        except Exception as e:
            # Unexpected error - fail safe by leaving unchanged
            # Could add logging here for debugging
            return updated_node

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


def transform_function(
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
    transformer = ReplaceContextWithColumns(
        function_name=function_name,
        runtime_context=runtime_context,
        output_type='tbl_col'
    )
    new_module = module.visit(transformer)
    return new_module.code
