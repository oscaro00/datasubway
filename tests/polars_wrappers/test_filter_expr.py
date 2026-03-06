import polars as pl
import pytest

from datasubway.polars_wrappers.filter_expr import (
    _strip_table_prefix,
    build_filter_expr,
    extract_table_columns_from_filter_dict,
)

SAMPLE_DF = pl.DataFrame(
    {
        "country": ["US", "CA", "UK", "CA"],
        "revenue": [500, 1500, 200, 800],
        "score": [None, 1.0, None, 2.0],
    }
)


# ---------------------------------------------------------------------------
# build_filter_expr — leaf operators
# ---------------------------------------------------------------------------


def test_equality():
    result = SAMPLE_DF.filter(build_filter_expr(("country", "=", "US")))
    assert result.to_dicts() == [{"country": "US", "revenue": 500, "score": None}]


def test_inequality():
    result = SAMPLE_DF.filter(build_filter_expr(("country", "!=", "US")))
    countries = [r["country"] for r in result.to_dicts()]
    assert "US" not in countries
    assert len(countries) == 3


def test_greater_than():
    result = SAMPLE_DF.filter(build_filter_expr(("revenue", ">", 1000)))
    assert all(r["revenue"] > 1000 for r in result.to_dicts())


def test_greater_than_or_equal():
    result = SAMPLE_DF.filter(build_filter_expr(("revenue", ">=", 500)))
    assert all(r["revenue"] >= 500 for r in result.to_dicts())
    assert len(result) == 3


def test_less_than():
    result = SAMPLE_DF.filter(build_filter_expr(("revenue", "<", 500)))
    assert all(r["revenue"] < 500 for r in result.to_dicts())


def test_less_than_or_equal():
    result = SAMPLE_DF.filter(build_filter_expr(("revenue", "<=", 500)))
    assert all(r["revenue"] <= 500 for r in result.to_dicts())
    assert len(result) == 2


def test_in():
    result = SAMPLE_DF.filter(build_filter_expr(("country", "in", ["US", "CA"])))
    countries = [r["country"] for r in result.to_dicts()]
    assert set(countries) == {"US", "CA"}
    assert len(countries) == 3


def test_not_in():
    result = SAMPLE_DF.filter(build_filter_expr(("country", "not in", ["US", "CA"])))
    assert result.to_dicts() == [{"country": "UK", "revenue": 200, "score": None}]


def test_is_null():
    result = SAMPLE_DF.filter(build_filter_expr(("score", "is null", None)))
    assert all(r["score"] is None for r in result.to_dicts())
    assert len(result) == 2


def test_is_not_null():
    result = SAMPLE_DF.filter(build_filter_expr(("score", "is not null", None)))
    assert all(r["score"] is not None for r in result.to_dicts())
    assert len(result) == 2


def test_unknown_operator_raises():
    with pytest.raises(ValueError, match="Unsupported filter operator"):
        build_filter_expr(("country", "~=", "US"))


# ---------------------------------------------------------------------------
# build_filter_expr — logical combinators
# ---------------------------------------------------------------------------


def test_and_combinator():
    spec = {"AND": [("country", "=", "CA"), ("revenue", ">", 1000)]}
    result = SAMPLE_DF.filter(build_filter_expr(spec))
    assert result.to_dicts() == [{"country": "CA", "revenue": 1500, "score": 1.0}]


def test_or_combinator():
    spec = {"OR": [("country", "=", "US"), ("country", "=", "UK")]}
    result = SAMPLE_DF.filter(build_filter_expr(spec))
    countries = [r["country"] for r in result.to_dicts()]
    assert set(countries) == {"US", "UK"}


def test_nested_or_with_inner_and():
    spec = {
        "OR": [
            ("country", "=", "US"),
            {"AND": [("country", "=", "CA"), ("revenue", ">", 1000)]},
        ]
    }
    result = SAMPLE_DF.filter(build_filter_expr(spec))
    rows = result.to_dicts()
    assert len(rows) == 2
    countries = [r["country"] for r in rows]
    assert "US" in countries
    assert "CA" in countries


# ---------------------------------------------------------------------------
# build_filter_expr — prefix stripping
# ---------------------------------------------------------------------------


def test_strip_prefixes_true():
    # "geography.country" should resolve to "country" column
    result = SAMPLE_DF.filter(
        build_filter_expr(("geography.country", "=", "US"), strip_prefixes=True)
    )
    assert result.to_dicts() == [{"country": "US", "revenue": 500, "score": None}]


def test_strip_prefixes_false():
    # Without stripping, "geography.country" is used verbatim — column not found
    with pytest.raises(Exception):
        SAMPLE_DF.filter(
            build_filter_expr(("geography.country", "=", "US"), strip_prefixes=False)
        )


def test_invalid_spec_raises():
    with pytest.raises(ValueError, match="Invalid filter spec"):
        build_filter_expr({"INVALID": []})


# ---------------------------------------------------------------------------
# extract_table_columns_from_filter_dict
# ---------------------------------------------------------------------------


def test_extract_single_tuple():
    result = extract_table_columns_from_filter_dict(("geography.country", "=", "US"))
    assert result == ["geography.country"]


def test_extract_and_spec():
    spec = {"AND": [("geography.country", "=", "US"), ("facts.revenue", ">", 1000)]}
    result = extract_table_columns_from_filter_dict(spec)
    assert result == ["geography.country", "facts.revenue"]


def test_extract_or_spec():
    spec = {"OR": [("geography.country", "=", "US"), ("geography.country", "=", "CA")]}
    result = extract_table_columns_from_filter_dict(spec)
    assert result == ["geography.country", "geography.country"]


def test_extract_nested_spec():
    spec = {
        "OR": [
            ("geography.country", "=", "US"),
            {"AND": [("geography.country", "=", "CA"), ("facts.revenue", ">", 1000)]},
        ]
    }
    result = extract_table_columns_from_filter_dict(spec)
    assert result == ["geography.country", "geography.country", "facts.revenue"]


def test_extract_invalid_spec_raises():
    with pytest.raises(ValueError, match="Invalid filter spec"):
        extract_table_columns_from_filter_dict({"INVALID": []})


# ---------------------------------------------------------------------------
# _strip_table_prefix
# ---------------------------------------------------------------------------


def test_strip_with_prefix():
    assert _strip_table_prefix("geography.country") == "country"


def test_strip_without_prefix():
    assert _strip_table_prefix("country") == "country"
