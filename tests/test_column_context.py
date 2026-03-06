import pytest

from datasubway import allow, exclude
from datasubway.column_context import (
    parse_pattern,
    parse_patterns,
    parse_table_column,
    parse_table_columns,
)


def test_parse_table_column_valid():
    assert parse_table_column("orders.amount") == ("orders", "amount")
    assert parse_table_column("fact_sales2020.total_revenue") == (
        "fact_sales2020",
        "total_revenue",
    )


def test_parse_table_column_invalid():
    with pytest.raises(Exception):
        parse_table_column("not_a_valid_string")
    with pytest.raises(Exception):
        parse_table_column("good.start5badend()")
    with pytest.raises(Exception):
        parse_table_column("table2.column3 ")


def test_parse_table_columns_valid():
    assert parse_table_columns(["orders.amount"]) == [("orders", "amount")]
    assert parse_table_columns(
        ["fact_sales2020.total_revenue", "dim_2020cal.year"]
    ) == [
        (
            "fact_sales2020",
            "total_revenue",
        ),
        ("dim_2020cal", "year"),
    ]


def test_parse_table_columns_invalid():
    with pytest.raises(Exception):
        parse_table_columns(["orders.amount", "not_valid"])
    with pytest.raises(Exception):
        parse_table_columns(["table.col()"])


def test_parse_pattern_valid():
    assert parse_pattern("*") == ("*", "*")
    assert parse_pattern("orders.*") == ("orders", "*")
    assert parse_pattern("*.amount") == ("*", "amount")
    assert parse_pattern("orders.amount") == ("orders", "amount")


def test_parse_pattern_invalid():
    with pytest.raises(Exception):
        parse_pattern("not_valid")
    with pytest.raises(Exception):
        parse_pattern("table.column ")
    with pytest.raises(Exception):
        parse_pattern("table.bad()")


def test_parse_patterns_valid():
    assert parse_patterns(["*"]) == [("*", "*")]
    assert parse_patterns(["orders.*", "customers.name"]) == [
        ("orders", "*"),
        ("customers", "name"),
    ]


def test_parse_patterns_invalid():
    with pytest.raises(Exception):
        parse_patterns(["orders.*", "not_valid"])


CONTEXT = ["orders.amount", "orders.quantity", "customers.name", "customers.email"]


def test_allow_wildcard():
    result = allow("*", CONTEXT)
    assert set(result) == {"orders.amount", "orders.quantity", "customers.name", "customers.email"}


def test_allow_table_wildcard():
    result = allow("orders.*", CONTEXT)
    assert set(result) == {"orders.amount", "orders.quantity"}


def test_allow_exact_match():
    result = allow("orders.amount", CONTEXT)
    assert set(result) == {"orders.amount"}


def test_allow_multiple_patterns():
    result = allow(["orders.amount", "customers.name"], CONTEXT)
    assert set(result) == {"orders.amount", "customers.name"}


def test_allow_no_match():
    result = allow("nonexistent.*", CONTEXT)
    assert set(result) == set()


def test_allow_with_include():
    result = allow("orders.*", CONTEXT, include="customers.email")
    assert set(result) == {"orders.amount", "orders.quantity", "customers.email"}


def test_allow_exclude_tables():
    result = allow("orders.*", CONTEXT, include_tables=False)
    assert set(result) == {"amount", "quantity"}


def test_exclude_wildcard():
    result = exclude("*", CONTEXT)
    assert result == []


def test_exclude_table_wildcard():
    result = exclude("orders.*", CONTEXT)
    assert set(result) == {"customers.name", "customers.email"}


def test_exclude_exact_match():
    result = exclude("orders.amount", CONTEXT)
    assert set(result) == {"orders.quantity", "customers.name", "customers.email"}


def test_exclude_multiple_patterns():
    result = exclude(["orders.amount", "customers.name"], CONTEXT)
    assert set(result) == {"orders.quantity", "customers.email"}


def test_exclude_no_match():
    result = exclude("nonexistent.*", CONTEXT)
    assert set(result) == {"orders.amount", "orders.quantity", "customers.name", "customers.email"}


def test_exclude_with_include():
    result = exclude("orders.*", CONTEXT, include="customers.email")
    assert set(result) == {"customers.name", "customers.email"}


def test_exclude_exclude_tables():
    result = exclude("orders.*", CONTEXT, include_tables=False)
    assert set(result) == {"name", "email"}
