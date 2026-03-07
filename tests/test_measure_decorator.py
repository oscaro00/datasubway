import polars as pl
import pytest

from datasubway import allow, exclude
from datasubway.data_model import DataModel
from datasubway.measure_decorator import measure

# Module-level LazyFrame used by the dm fixture
lf = pl.LazyFrame({"a": [1, 2, 3], "b": [4, 5, 6]})


@pytest.fixture
def dm():
    return DataModel(tables={"test": lf})


# ---------------------------------------------------------------------------
# Group 1 — Basic registration
# ---------------------------------------------------------------------------


def test_measure_stored_in_measures(dm):
    @measure(dm)
    def my_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert "my_measure" in dm.measures


def test_decorator_returns_original_function(dm):
    def my_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    original = my_measure
    decorated = measure(dm)(my_measure)
    assert decorated is original


def test_measure_grouping_context_populated(dm):
    @measure(dm)
    def my_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert "my_measure" in dm.measure_grouping_contexts


def test_measure_output_cols_populated(dm):
    @measure(dm)
    def my_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert "my_measure" in dm.measure_output_cols


# ---------------------------------------------------------------------------
# Group 2 — GroupingContext content
# ---------------------------------------------------------------------------


def test_grouping_context_allow_type(dm):
    @measure(dm)
    def allow_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert dm.measure_grouping_contexts["allow_measure"]["type"] == "allow"


def test_grouping_context_exclude_type(dm):
    @measure(dm)
    def exclude_measure(qc):
        return dm.table("test").group_by(exclude(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert dm.measure_grouping_contexts["exclude_measure"]["type"] == "exclude"


def test_grouping_context_pattern_extracted(dm):
    @measure(dm)
    def pattern_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert "*" in dm.measure_grouping_contexts["pattern_measure"]["pattern"]


# ---------------------------------------------------------------------------
# Group 3 — Output column extraction
# ---------------------------------------------------------------------------


def test_output_cols_with_alias(dm):
    @measure(dm)
    def alias_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    assert dm.measure_output_cols["alias_measure"] == ["sum_a"]


def test_output_cols_without_alias(dm):
    @measure(dm)
    def no_alias_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.b").first()
        )

    assert dm.measure_output_cols["no_alias_measure"] == ["test.b"]


def test_output_cols_multiple_columns(dm):
    @measure(dm)
    def multi_col_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a"),
            pl.col("test.b").first(),
        )

    assert dm.measure_output_cols["multi_col_measure"] == ["sum_a", "test.b"]


# ---------------------------------------------------------------------------
# Group 4 — Error cases
# ---------------------------------------------------------------------------


def test_duplicate_measure_name_raises(dm):
    """This is a bit messy, but it avoids linting errors for functions with duplicate names"""

    def dup_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    measure(dm)(dup_measure)

    def dup_measure_copy(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    dup_measure_copy.__name__ = "dup_measure"

    with pytest.raises(ValueError, match="already exists"):
        measure(dm)(dup_measure_copy)


def test_invalid_measure_no_agg_raises(dm):
    def invalid_no_agg(qc):
        return dm.table("test").select("test.a", "test.b")

    with pytest.raises(ValueError):
        measure(dm)(invalid_no_agg)


def test_invalid_measure_no_allow_exclude_raises(dm):
    def invalid_no_allow(qc):
        return dm.table("test").group_by("test.a").agg(pl.col("test.b").sum().alias("sum_b"))

    with pytest.raises(ValueError):
        measure(dm)(invalid_no_allow)


# ---------------------------------------------------------------------------
# Group 5 — Multiple measures
# ---------------------------------------------------------------------------


def test_multiple_measures_on_same_data_model(dm):
    @measure(dm)
    def first_measure(qc):
        return dm.table("test").group_by(allow(pattern="*", context=qc.groups)).agg(
            pl.col("test.a").sum().alias("sum_a")
        )

    @measure(dm)
    def second_measure(qc):
        return dm.table("test").group_by(exclude(pattern="*", context=qc.groups)).agg(
            pl.col("test.b").first()
        )

    for name in ("first_measure", "second_measure"):
        assert name in dm.measures
        assert name in dm.measure_grouping_contexts
        assert name in dm.measure_output_cols
