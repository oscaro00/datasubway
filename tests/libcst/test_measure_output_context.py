import pytest

from datasubway.libcst.measure_output_context import (
    extract_agg_output_columns,
    extract_grouping_context,
)

# ---------------------------------------------------------------------------
# Source code fixtures — grouping context (valid)
# ---------------------------------------------------------------------------

SOURCE_GROUP_BY_ALLOW = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("a").sum().alias("sum_a"),
        pl.col("b").first(),
    )
"""

SOURCE_GROUP_BY_EXCLUDE = """
def my_measure(qc):
    return lf.group_by(exclude(pattern="*", context=qc["groups"])).agg(
        pl.col("a").sum().alias("sum_a")
    )
"""

SOURCE_GROUP_BY_WITH_INCLUDE = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"], include="[col_a]")).agg(
        pl.col("a").sum().alias("sum_a")
    )
"""

SOURCE_GROUP_BY_DYNAMIC_KWARG_INDEX = """
def my_measure(qc):
    return lf.group_by_dynamic(
        index_column="ts",
        every="1d",
        group_by=allow(pattern="*", context=qc["groups"]),
    ).agg(pl.col("a").sum().alias("sum_a"))
"""

SOURCE_GROUP_BY_DYNAMIC_POSITIONAL_INDEX = """
def my_measure(qc):
    return lf.group_by_dynamic(
        "ts",
        every="1d",
        group_by=allow(pattern="*", context=qc["groups"]),
    ).agg(pl.col("a").sum().alias("sum_a"))
"""

SOURCE_ROLLING_EXCLUDE = """
def my_measure(qc):
    return lf.rolling(
        index_column="a",
        period="1d",
        group_by=exclude(pattern="*", context=qc["groups"]),
    ).agg(pl.col("a").sum().alias("sum_a"))
"""

SOURCE_GROUP_BY_DYNAMIC_EXISTING_INCLUDE = """
def my_measure(qc):
    return lf.group_by_dynamic(
        index_column="ts",
        every="1d",
        group_by=allow(pattern="*", context=qc["groups"], include="[col_a]"),
    ).agg(pl.col("a").sum().alias("sum_a"))
"""

# ---------------------------------------------------------------------------
# Source code fixtures — grouping context (invalid)
# ---------------------------------------------------------------------------

SOURCE_NO_METHOD_CHAIN = """
def my_measure(qc):
    x = 1
    return x
"""

SOURCE_CHAIN_NO_AGG = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).filter(pl.col("a") > 0)
"""

SOURCE_SELECT_AGG = """
def my_measure(qc):
    return lf.select(pl.col("a")).agg(pl.col("b").first())
"""

SOURCE_GROUP_BY_NO_ALLOW_EXCLUDE = """
def my_measure(qc):
    return lf.group_by("col_name").agg(pl.col("a").sum().alias("sum_a"))
"""

SOURCE_INTERMEDIATE_CHAIN_WITH_GROUP_BY = """
def my_measure(qc):
    first = (
        lf.filter(pl.col("c"))
        .group_by(allow(pattern="*", context=qc["groups"]))
        .agg(pl.col("a").sum().alias("sum_a"))
    )
    return first.filter(pl.col("sum_a") > 1)
"""

# ---------------------------------------------------------------------------
# Source code fixtures — agg output columns
# ---------------------------------------------------------------------------

SOURCE_AGG_SINGLE_ALIAS = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("a").sum().alias("sum_a")
    )
"""

SOURCE_AGG_SINGLE_NO_ALIAS = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("b").first()
    )
"""

SOURCE_AGG_MULTIPLE_COLS = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("a").sum().alias("s"),
        pl.col("b").first(),
    )
"""

SOURCE_AGG_MULTI_COL_EXPR = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("a", "b").sum()
    )
"""

SOURCE_AGG_ALL = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.all().sum()
    )
"""

SOURCE_AGG_WILDCARD_COL = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("*").sum()
    )
"""

SOURCE_AGG_ALIAS_NO_ARGS = """
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col("a").sum().alias()
    )
"""

SOURCE_AGG_FSTRING_COL = '''
def my_measure(qc):
    return lf.group_by(allow(pattern="*", context=qc["groups"])).agg(
        pl.col(f"a_{x}").sum()
    )
'''

# ---------------------------------------------------------------------------
# Group 1 — extract_grouping_context: valid cases
# ---------------------------------------------------------------------------


def test_group_by_allow_type():
    result = extract_grouping_context(SOURCE_GROUP_BY_ALLOW, "my_measure")
    assert result["type"] == "allow"


def test_group_by_allow_pattern_and_context():
    result = extract_grouping_context(SOURCE_GROUP_BY_ALLOW, "my_measure")
    assert result["pattern"] == '"*"'
    assert result["context"] == 'qc["groups"]'


def test_group_by_allow_include_is_none():
    result = extract_grouping_context(SOURCE_GROUP_BY_ALLOW, "my_measure")
    assert result["include"] is None


def test_group_by_exclude_type():
    result = extract_grouping_context(SOURCE_GROUP_BY_EXCLUDE, "my_measure")
    assert result["type"] == "exclude"


def test_group_by_exclude_include_is_none():
    result = extract_grouping_context(SOURCE_GROUP_BY_EXCLUDE, "my_measure")
    assert result["include"] is None


def test_group_by_with_include_captured():
    result = extract_grouping_context(SOURCE_GROUP_BY_WITH_INCLUDE, "my_measure")
    assert result["include"] == '"[col_a]"'


def test_group_by_dynamic_kwarg_index_column_merged():
    result = extract_grouping_context(SOURCE_GROUP_BY_DYNAMIC_KWARG_INDEX, "my_measure")
    assert result["type"] == "allow"
    assert result["include"] == '"ts"'


def test_group_by_dynamic_positional_index_column_merged():
    result = extract_grouping_context(
        SOURCE_GROUP_BY_DYNAMIC_POSITIONAL_INDEX, "my_measure"
    )
    assert result["include"] == '"ts"'


def test_rolling_exclude_index_column_merged():
    result = extract_grouping_context(SOURCE_ROLLING_EXCLUDE, "my_measure")
    assert result["type"] == "exclude"
    assert result["include"] == '"a"'


def test_group_by_dynamic_existing_include_merged_with_index_column():
    result = extract_grouping_context(
        SOURCE_GROUP_BY_DYNAMIC_EXISTING_INCLUDE, "my_measure"
    )
    assert result["include"] == '["[col_a]", "ts"]'


# ---------------------------------------------------------------------------
# Group 2 — extract_grouping_context: invalid / error cases
# ---------------------------------------------------------------------------


def test_no_method_chain_raises():
    with pytest.raises(ValueError, match="No method chains"):
        extract_grouping_context(SOURCE_NO_METHOD_CHAIN, "my_measure")


def test_chain_ends_without_agg_raises():
    with pytest.raises(ValueError, match="must end with"):
        extract_grouping_context(SOURCE_CHAIN_NO_AGG, "my_measure")


def test_no_group_by_before_agg_raises():
    with pytest.raises(ValueError, match="must end with"):
        extract_grouping_context(SOURCE_SELECT_AGG, "my_measure")


def test_group_by_without_allow_exclude_raises():
    with pytest.raises(ValueError, match=r"must have an allow\(\) or exclude\(\)"):
        extract_grouping_context(SOURCE_GROUP_BY_NO_ALLOW_EXCLUDE, "my_measure")


def test_last_chain_not_group_by_chain_raises():
    with pytest.raises(ValueError, match="must end with"):
        extract_grouping_context(SOURCE_INTERMEDIATE_CHAIN_WITH_GROUP_BY, "my_measure")


# ---------------------------------------------------------------------------
# Group 3 — extract_agg_output_columns: valid cases
# ---------------------------------------------------------------------------


def test_agg_single_alias():
    result = extract_agg_output_columns(SOURCE_AGG_SINGLE_ALIAS, "my_measure")
    assert result == ["sum_a"]


def test_agg_single_col_no_alias():
    result = extract_agg_output_columns(SOURCE_AGG_SINGLE_NO_ALIAS, "my_measure")
    assert result == ["b"]


def test_agg_multiple_positional_args():
    result = extract_agg_output_columns(SOURCE_AGG_MULTIPLE_COLS, "my_measure")
    assert result == ["s", "b"]


def test_agg_multi_col_expr():
    result = extract_agg_output_columns(SOURCE_AGG_MULTI_COL_EXPR, "my_measure")
    assert result == ["a", "b"]


# ---------------------------------------------------------------------------
# Group 4 — extract_agg_output_columns: error cases
# ---------------------------------------------------------------------------


def test_agg_pl_all_raises():
    with pytest.raises(ValueError, match=r"pl\.all\(\) is not resolvable"):
        extract_agg_output_columns(SOURCE_AGG_ALL, "my_measure")


def test_agg_wildcard_col_raises():
    with pytest.raises(ValueError, match=r"pl\.col\('\*'\)"):
        extract_agg_output_columns(SOURCE_AGG_WILDCARD_COL, "my_measure")


def test_agg_alias_no_args_raises():
    with pytest.raises(ValueError, match=r"alias\(\) called with no arguments"):
        extract_agg_output_columns(SOURCE_AGG_ALIAS_NO_ARGS, "my_measure")


def test_agg_fstring_col_raises():
    with pytest.raises(ValueError, match="Expected a simple string"):
        extract_agg_output_columns(SOURCE_AGG_FSTRING_COL, "my_measure")
