"""The idea of these functions is to pass valid columns (with or without table prefixes)
to polars methods. It is assumed that QueryContext objects have already validated that the table.column
inputs are members of the DataModel"""

import re
from functools import reduce


def _filter_spec_by_pattern(spec, parsed_patterns, keep_matching: bool):
    """
    Traverse a filter spec tree and collect leaf conditions whose column
    matches (keep_matching=True) or does not match (keep_matching=False) the parsed patterns.
    Returns a pl.Expr with the original AND/OR structure preserved, or None if nothing matches.
    """
    from datasubway.polars_wrappers.filter_expr import build_filter_expr

    if isinstance(spec, tuple):
        col = spec[0]
        table, column = parse_table_column(col)
        matches = any(
            (pat_table == "*" or table == pat_table) and (pat_col == "*" or column == pat_col)
            for pat_table, pat_col in parsed_patterns
        )
        if matches == keep_matching:
            return build_filter_expr(spec, strip_prefixes=False)
        return None

    if "AND" in spec:
        children = [_filter_spec_by_pattern(s, parsed_patterns, keep_matching) for s in spec["AND"]]
        children = [c for c in children if c is not None]
        if not children:
            return None
        return reduce(lambda a, b: a & b, children)

    if "OR" in spec:
        children = [_filter_spec_by_pattern(s, parsed_patterns, keep_matching) for s in spec["OR"]]
        children = [c for c in children if c is not None]
        if not children:
            return None
        return reduce(lambda a, b: a | b, children)

    raise ValueError(f"Invalid filter spec: {spec!r}")


def parse_table_column(table_column_str: str) -> tuple[str, str]:
    table_column_pattern = r"^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$"
    regex_find = re.findall(table_column_pattern, table_column_str)

    if len(regex_find) == 0:
        raise Exception(f"Invalid table column string: {table_column_str}")
    # regex_find should be an object like: [('table_name', 'column_name')]
    return regex_find[0][0], regex_find[0][1]


def parse_table_columns(table_column_list: list[str] | dict) -> list[tuple[str, str]]:
    return [
        parse_table_column(table_column_str) for table_column_str in table_column_list
    ]


def parse_pattern(pattern_str: str) -> tuple[str, str]:
    if pattern_str == "*":
        return "*", "*"

    pattern_regex = r"^([a-zA-Z0-9_*]+)\.([a-zA-Z0-9_*]+)$"
    regex_find = re.findall(pattern_regex, pattern_str)

    if len(regex_find) == 0:
        raise Exception(
            f"Invalid table column pattern in allow()/exclude(): {pattern_str}"
        )
    # regex_find should be an object like: [('table_name', 'column_name')]
    return regex_find[0][0], regex_find[0][1]


def parse_patterns(pattern_list: list[str]) -> list[tuple[str, str]]:
    return [parse_pattern(pattern_str) for pattern_str in pattern_list]


def _process_include_columns(include: list[str], include_tables: bool) -> list[str]:
    """Handle both 'table.col' schema columns and raw derived column names."""
    result = []
    for col in include:
        if "." in col:
            table, column = parse_table_column(col)
            result.append(f"{table}.{column}" if include_tables else column)
        else:
            result.append(col)  # Raw derived column - pass through as-is
    return result


class _AllowExcludeResult(list):
    """List subclass returned by allow()/exclude() carrying call metadata."""

    def __init__(self, items, *, ae_type, pattern, include):
        super().__init__(items)
        self.ae_type = ae_type
        self.pattern = pattern
        self.include = include


def allow(
    pattern: str | list[str],
    context: str | list[str] | dict,
    include: str | list[str] = [],
    *,
    include_tables: bool = True,
) -> list[str]:
    original_pattern = pattern
    original_include = include

    if isinstance(pattern, str):
        pattern = [pattern]
    if isinstance(include, str):
        include = [include]

    result_columns = _process_include_columns(include, include_tables)

    def _sentinel(items):
        return _AllowExcludeResult(items, ae_type="allow", pattern=original_pattern, include=original_include)

    if isinstance(context, dict):
        if not context:
            return _sentinel(list(set(result_columns)))
        expr = _filter_spec_by_pattern(context, parse_patterns(pattern), keep_matching=True)
        if expr is None:
            return _sentinel(list(set(result_columns)))
        return _sentinel(list(set(result_columns)) + [expr])

    if isinstance(context, str):
        context = [context]

    if "*" in pattern:
        allowed_context = context
    else:
        allowed_context = []
        for table, column in parse_table_columns(context):
            for pattern_table, pattern_column in parse_patterns(pattern):
                if table == pattern_table and (
                    column == pattern_column or pattern_column == "*"
                ):
                    if include_tables:
                        allowed_context.append(f"{table}.{column}")
                    else:
                        allowed_context.append(column)
                    break

    result_columns.extend(allowed_context)

    # remove duplicates
    return _sentinel(list(set(result_columns)))


def exclude(
    pattern: str | list[str],
    context: str | list[str] | dict,
    include: str | list[str] = [],
    *,
    include_tables: bool = True,
) -> list[str]:
    original_pattern = pattern
    original_include = include

    if isinstance(pattern, str):
        pattern = [pattern]
    if isinstance(include, str):
        include = [include]

    result_columns = _process_include_columns(include, include_tables)

    def _sentinel(items):
        return _AllowExcludeResult(items, ae_type="exclude", pattern=original_pattern, include=original_include)

    if isinstance(context, dict):
        if not context:
            return _sentinel(list(set(result_columns)))
        expr = _filter_spec_by_pattern(context, parse_patterns(pattern), keep_matching=False)
        if expr is None:
            return _sentinel(list(set(result_columns)))
        return _sentinel(list(set(result_columns)) + [expr])

    if isinstance(context, str):
        context = [context]

    if "*" in pattern:
        allowed_context = []
    else:
        allowed_context = []
        for table, column in parse_table_columns(context):
            for pattern_table, pattern_column in parse_patterns(pattern):
                if table == pattern_table and (
                    column == pattern_column or pattern_column == "*"
                ):
                    break
            else:
                if include_tables:
                    allowed_context.append(f"{table}.{column}")
                else:
                    allowed_context.append(column)

    result_columns.extend(allowed_context)

    # remove duplicates
    return _sentinel(list(set(result_columns)))
