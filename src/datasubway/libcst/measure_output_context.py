"""
When a measure is registered with the @measure decorator,
make sure the measure ends with .group_by()/.group_by_dynamic()/.rolling()
followed by .agg().

Additionally, get the allow() or exclude() call from the final grouping method (including the parameters),
and save it to the data model object upon @measure decorator validation.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import TypedDict, cast

import libcst as cst
from libcst.metadata import CodeRange, MetadataWrapper, PositionProvider

GROUP_BY_VARIANTS = {"group_by", "group_by_dynamic", "rolling"}
KEYWORD_GROUP_BY_VARIANTS = {"group_by_dynamic", "rolling"}  # use group_by= kwarg


class GroupingContext(TypedDict):
    type: str  # "allow" or "exclude"
    pattern: str  # source code of the pattern arg
    context: str  # source code of the context arg
    include: (
        str | None
    )  # source code of include arg (with index_column merged), or None


@dataclass
class GroupingCallInfo:
    method_name: str
    line: int
    allow_exclude_node: cst.Call | None
    index_column_node: cst.BaseExpression | None


@dataclass
class ChainInfo:
    methods: list[str]
    end_line: int
    grouping_info: GroupingCallInfo | None
    agg_node: cst.Call | None


@dataclass
class ValidatedMeasureInfo:
    grouping_info: GroupingCallInfo
    agg_node: cst.Call


class MeasureGroupingValidator(cst.CSTVisitor):
    METADATA_DEPENDENCIES = (PositionProvider,)

    def __init__(self, function_name: str):
        self.function_name = function_name
        self.in_target_function = False
        self.chains: list[ChainInfo] = []
        self._visited_calls: set[int] = set()

    def visit_FunctionDef(self, node: cst.FunctionDef) -> bool | None:
        if node.name.value == self.function_name:
            self.in_target_function = True
        return None

    def leave_FunctionDef(self, original_node: cst.FunctionDef) -> None:
        if original_node.name.value == self.function_name:
            self.in_target_function = False

    def visit_Call(self, node: cst.Call) -> bool | None:
        if not self.in_target_function:
            return None
        if id(node) in self._visited_calls:
            return None

        # Only process method calls (func is Attribute)
        if not isinstance(node.func, cst.Attribute):
            return None

        # Extract the full method chain walking inward
        methods = []
        grouping_info = None
        agg_call_node: cst.Call | None = None
        current = node

        while isinstance(current, cst.Call) and isinstance(current.func, cst.Attribute):
            method_name = current.func.attr.value
            methods.append(method_name)
            self._visited_calls.add(id(current))

            if method_name == "agg":
                agg_call_node = current

            if method_name in GROUP_BY_VARIANTS:
                allow_exclude = _extract_allow_exclude(current, method_name)
                index_column = _extract_index_column(current, method_name)
                pos = cast(CodeRange, self.get_metadata(PositionProvider, current))
                grouping_info = GroupingCallInfo(
                    method_name=method_name,
                    line=pos.start.line,
                    allow_exclude_node=allow_exclude,
                    index_column_node=index_column,
                )

            current = current.func.value

        methods.reverse()

        if methods:
            pos = cast(CodeRange, self.get_metadata(PositionProvider, node))
            self.chains.append(
                ChainInfo(
                    methods=methods,
                    end_line=pos.end.line,
                    grouping_info=grouping_info,
                    agg_node=agg_call_node,
                )
            )

        return None

    def validate(self) -> ValidatedMeasureInfo:
        if not self.chains:
            raise ValueError(
                f"No method chains found in function '{self.function_name}'."
            )

        # Find the last chain by end line
        last_chain = max(self.chains, key=lambda c: c.end_line)
        methods = last_chain.methods

        if (
            len(methods) < 2
            or methods[-1] != "agg"
            or methods[-2] not in GROUP_BY_VARIANTS
        ):
            raise ValueError(
                f"Measure function '{self.function_name}' must end with "
                f".group_by()/.group_by_dynamic()/.rolling() followed by .agg(). "
                f"Found chain ending with: .{'().'.join(methods)}()"
            )

        grouping_info = last_chain.grouping_info
        if grouping_info is None or grouping_info.allow_exclude_node is None:
            raise ValueError(
                f"Measure function '{self.function_name}' must have an allow() or exclude() "
                f"call in the grouping method."
            )

        agg_node = last_chain.agg_node
        assert agg_node is not None  # guaranteed since methods[-1] == "agg"

        return ValidatedMeasureInfo(
            grouping_info=grouping_info,
            agg_node=agg_node,
        )


def _extract_allow_exclude(call_node: cst.Call, method_name: str) -> cst.Call | None:
    if method_name not in KEYWORD_GROUP_BY_VARIANTS:
        # First positional arg should be allow()/exclude()
        for arg in call_node.args:
            if arg.keyword is None:
                if isinstance(arg.value, cst.Call) and isinstance(
                    arg.value.func, cst.Name
                ):
                    if arg.value.func.value in ("allow", "exclude"):
                        return arg.value
                return None
    elif method_name in KEYWORD_GROUP_BY_VARIANTS:
        # group_by= keyword arg
        for arg in call_node.args:
            if arg.keyword and arg.keyword.value == "group_by":
                if isinstance(arg.value, cst.Call) and isinstance(
                    arg.value.func, cst.Name
                ):
                    if arg.value.func.value in ("allow", "exclude"):
                        return arg.value
                return None
    return None


def _extract_index_column(
    call_node: cst.Call, method_name: str
) -> cst.BaseExpression | None:
    if method_name not in KEYWORD_GROUP_BY_VARIANTS:
        return None

    # Check for index_column= keyword arg
    for arg in call_node.args:
        if arg.keyword and arg.keyword.value == "index_column":
            return arg.value

    # For group_by_dynamic, first positional arg is index_column
    if method_name == "group_by_dynamic":
        for arg in call_node.args:
            if arg.keyword is None:
                return arg.value

    return None


def _extract_grouping_context(
    tree: cst.Module,
    ae_node: cst.Call,
    index_col_node: cst.BaseExpression | None,
    method_name: str,
) -> GroupingContext:
    """Extract a structured GroupingContext from the allow()/exclude() call node."""
    # type from the function name
    assert isinstance(ae_node.func, cst.Name)
    ctx_type = ae_node.func.value  # "allow" or "exclude"

    pattern: str | None = None
    context: str | None = None
    include: str | None = None

    for arg in ae_node.args:
        if arg.keyword and arg.keyword.value == "pattern":
            pattern = tree.code_for_node(arg.value)
        elif arg.keyword and arg.keyword.value == "context":
            context = tree.code_for_node(arg.value)
        elif arg.keyword and arg.keyword.value == "include":
            include = tree.code_for_node(arg.value)

    assert pattern is not None, "allow()/exclude() missing pattern= argument"
    assert context is not None, "allow()/exclude() missing context= argument"

    # Merge index_column into include for keyword group_by variants
    if method_name in KEYWORD_GROUP_BY_VARIANTS and index_col_node is not None:
        index_col_src = tree.code_for_node(index_col_node)
        if include is not None:
            include = f"[{include}, {index_col_src}]"
        else:
            include = index_col_src

    return GroupingContext(
        type=ctx_type,
        pattern=pattern,
        context=context,
        include=include,
    )


def _extract_string_value(node: cst.BaseExpression) -> str:
    """Extract a plain string value from a CST node.

    Raises ValueError for f-strings, concatenated strings, or non-strings.
    """
    if not isinstance(node, cst.SimpleString):
        raise ValueError(
            f"Expected a simple string literal, got: {type(node).__name__}"
        )
    val = ast.literal_eval(node.value)
    if not isinstance(val, str):
        raise ValueError(f"Expected a string, got: {type(val).__name__}")
    return val


def _extract_column_name_from_expr(expr: cst.BaseExpression) -> list[str]:
    """Extract output column name(s) from a single agg expression.

    Walks the method chain outer to inner. If .alias() is found, uses its arg.
    Otherwise walks to innermost pl.col(...) and uses column names.
    """
    if not isinstance(expr, cst.Call):
        raise ValueError(
            f"Expected a Call expression in agg argument, got: {type(expr).__name__}"
        )

    # First pass: walk outer to inner looking for .alias()
    current: cst.BaseExpression = expr
    while isinstance(current, cst.Call) and isinstance(current.func, cst.Attribute):
        method_name = current.func.attr.value
        if method_name == "alias":
            positional_args = [arg for arg in current.args if arg.keyword is None]
            if not positional_args:
                raise ValueError("alias() called with no arguments")
            return [_extract_string_value(positional_args[0].value)]
        current = current.func.value

    # No alias found — walk expr to innermost call (while inner value is still a Call)
    current = expr
    while (
        isinstance(current, cst.Call)
        and isinstance(current.func, cst.Attribute)
        and isinstance(current.func.value, cst.Call)
    ):
        current = current.func.value

    # current should now be pl.col(...) or pl.all()
    if not isinstance(current, cst.Call):
        raise ValueError(
            f"Expected a function call at innermost level, got: {type(current).__name__}"
        )

    func = current.func
    if not isinstance(func, cst.Attribute):
        raise ValueError(
            f"Unresolvable innermost expression type: {type(func).__name__}"
        )

    func_name = func.attr.value
    if func_name == "all":
        raise ValueError("pl.all() is not resolvable to specific column names")
    if func_name == "col":
        result = []
        for arg in current.args:
            if arg.keyword is None:
                val = _extract_string_value(arg.value)
                if val == "*":
                    raise ValueError(
                        "pl.col('*') is not resolvable to specific column names"
                    )
                result.append(val)
        if not result:
            raise ValueError("pl.col() called with no positional string arguments")
        return result

    raise ValueError(f"Unresolvable innermost expression: pl.{func_name}()")


def _extract_agg_output_columns(agg_node: cst.Call) -> list[str]:
    """Extract all output column names from the positional args of a .agg() call node."""
    result = []
    for arg in agg_node.args:
        if arg.keyword is None:
            cols = _extract_column_name_from_expr(arg.value)
            result.extend(cols)
    return result


def _run_validator(
    source_code: str, function_name: str
) -> tuple[cst.Module, ValidatedMeasureInfo]:
    tree = cst.parse_module(source_code)
    wrapper = MetadataWrapper(tree)
    validator = MeasureGroupingValidator(function_name)
    wrapper.visit(validator)
    return tree, validator.validate()


def extract_grouping_context(source_code: str, function_name: str) -> GroupingContext:
    """Validate that a measure function follows the correct polars grouping pattern.

    Raises ValueError if invalid. Returns a GroupingContext dict if valid.
    For group_by_dynamic/rolling, merges index_column into include.
    """
    tree, validated = _run_validator(source_code, function_name)
    ae_node = validated.grouping_info.allow_exclude_node
    assert ae_node is not None
    return _extract_grouping_context(
        tree,
        ae_node,
        validated.grouping_info.index_column_node,
        validated.grouping_info.method_name,
    )


def extract_agg_output_columns(source_code: str, function_name: str) -> list[str]:
    """Extract the output column names produced by the .agg() call of a measure function."""
    _, validated = _run_validator(source_code, function_name)
    return _extract_agg_output_columns(validated.agg_node)


