"""The idea of these functions is to pass valid columns (with or without table prefixes)
to polars methods. It is assumed that QueryContext objects have already validated that the table.column
inputs are members of the DataModel"""

import re


def parse_table_column(table_column_str: str) -> tuple[str, str]:
    table_column_pattern = r"^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$"
    regex_find = re.findall(table_column_pattern, table_column_str)

    if len(regex_find) == 0:
        raise Exception(f"Invalid table column string: {table_column_str}")
    # regex_find should be an object like: [('table_name', 'column_name')]
    return regex_find[0][0], regex_find[0][1]


def parse_table_columns(table_column_list: list[str]) -> list[tuple[str, str]]:
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


def allow(
    pattern: str | list[str],
    context: str | list[str],
    include: str | list[str] = [],
    *,
    include_tables: bool = True,
) -> list[str]:
    if isinstance(pattern, str):
        pattern = [pattern]
    if isinstance(context, str):
        context = [context]
    if isinstance(include, str):
        include = [include]

    if include_tables:
        result_columns = [
            f"{table}.{column}" for table, column in parse_table_columns(include)
        ]
    else:
        result_columns = [f"{column}" for table, column in parse_table_columns(include)]

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
    return list(set(result_columns))


def exclude(
    pattern: str | list[str],
    context: str | list[str],
    include: str | list[str] = [],
    *,
    include_tables: bool = True,
) -> list[str]:
    if isinstance(pattern, str):
        pattern = [pattern]
    if isinstance(context, str):
        context = [context]
    if isinstance(include, str):
        include = [include]

    if include_tables:
        result_columns = [
            f"{table}.{column}" for table, column in parse_table_columns(include)
        ]
    else:
        result_columns = [f"{column}" for table, column in parse_table_columns(include)]

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
    return list(set(result_columns))
