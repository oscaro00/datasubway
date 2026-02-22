"""
When a measure is registered with the @measure decorator,
make sure the measure ends with .group_by()/.group_by_dynamic()/.rolling()
followed by .agg().

Additionally, get the allow() or exclude() call from the final grouping method (including the parameters),
and save it to the data model object upon @measure decorator validation.
"""

from __future__ import annotations

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
        current = node

        while isinstance(current, cst.Call) and isinstance(current.func, cst.Attribute):
            method_name = current.func.attr.value
            methods.append(method_name)
            self._visited_calls.add(id(current))

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
                )
            )

        return None

    def validate(self) -> GroupingCallInfo:
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

        return grouping_info


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


def validate_and_extract_grouping_context(
    source_code: str, function_name: str
) -> GroupingContext:
    """Validate that a measure function follows the correct polars grouping pattern.

    Raises ValueError if invalid. Returns a GroupingContext dict if valid.
    For group_by_dynamic/rolling, merges index_column into include.
    """
    tree = cst.parse_module(source_code)
    wrapper = MetadataWrapper(tree)
    validator = MeasureGroupingValidator(function_name)
    wrapper.visit(validator)
    grouping_info = validator.validate()

    ae_node = grouping_info.allow_exclude_node
    assert ae_node is not None  # guaranteed by validate()

    return _extract_grouping_context(
        tree, ae_node, grouping_info.index_column_node, grouping_info.method_name
    )


if __name__ == "__main__":
    import polars as pl

    from src.datasubway.column_context import allow, exclude

    lf = pl.LazyFrame(
        {
            "a": [1, 2, 3, 4, 5],
            "b": ["cat", "dog", "dog", "cat", "dog"],
            "c": [True, False, True, False, True],
        }
    )

    qc = {"groups": ["lf.b"]}

    source_code = open(__file__).read()

    # --- Valid examples ---

    def valid_measure1(qc):
        return (
            lf.filter(pl.col("c"))
            .group_by(allow(pattern="*", context=qc["groups"]))
            .agg(pl.col("a").sum().alias("sum_a"))
        )

    def valid_measure2(qc):
        return lf.rolling(
            index_column="a",
            period="1d",
            group_by=exclude(pattern="*", context=qc["groups"]),
        ).agg(pl.col("a").sum().alias("sum_a"))

    # --- Invalid examples ---

    def invalid_measure1(qc):
        return lf.select("b", "c")

    def invalid_measure2(qc):
        return lf.filter(pl.col("c")).select(pl.col("a").sum().alias("sum_a"))

    def invalid_measure3(qc):
        first = (
            lf.filter(pl.col("c"))
            .group_by(allow(pattern="*", context=qc["groups"]))
            .agg(pl.col("a").sum().alias("sum_a"))
        )

        return first.filter(pl.col("sum_a") > 1)

    # --- Run tests ---

    print("=== Testing valid_measure1 ===")
    result = validate_and_extract_grouping_context(source_code, "valid_measure1")
    print(f"  Result: {dict(result)}")

    print("\n=== Testing valid_measure2 ===")
    result = validate_and_extract_grouping_context(source_code, "valid_measure2")
    print(f"  Result: {dict(result)}")

    for name in ["invalid_measure1", "invalid_measure2", "invalid_measure3"]:
        print(f"\n=== Testing {name} ===")
        try:
            validate_and_extract_grouping_context(source_code, name)
            print("  ERROR: Should have raised ValueError!")
        except ValueError as e:
            print(f"  Correctly raised: {e}")
