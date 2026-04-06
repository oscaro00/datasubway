"""Tests for allow() and exclude() column context resolution."""

from datasubway.column_context import allow, exclude

CONTEXT = ["orders.region", "orders.amount", "customers.name"]


class TestAllow:
    def test_wildcard(self):
        result = allow("*", CONTEXT)
        assert list(result) == CONTEXT

    def test_table_wildcard(self):
        result = allow("orders.*", CONTEXT)
        assert list(result) == ["orders.region", "orders.amount"]

    def test_column_wildcard(self):
        result = allow("*.region", CONTEXT)
        assert list(result) == ["orders.region"]

    def test_exact(self):
        result = allow("orders.amount", CONTEXT)
        assert list(result) == ["orders.amount"]

    def test_include(self):
        result = allow("orders.*", CONTEXT, include="customers.name")
        assert "orders.region" in result
        assert "orders.amount" in result
        assert "customers.name" in result

    def test_no_match(self):
        result = allow("products.*", CONTEXT)
        assert list(result) == []

    def test_filter_dict_context(self):
        filter_ctx = {
            "AND": [("orders.region", "=", "US"), ("orders.amount", ">", 100)]
        }
        result = allow("orders.*", filter_ctx)
        assert "orders.region" in result
        assert "orders.amount" in result

    def test_multiple_patterns(self):
        result = allow(["orders.region", "customers.name"], CONTEXT)
        assert list(result) == ["orders.region", "customers.name"]


class TestExclude:
    def test_wildcard(self):
        result = exclude("*", CONTEXT)
        assert list(result) == []

    def test_table_wildcard(self):
        result = exclude("orders.*", CONTEXT)
        assert list(result) == ["customers.name"]

    def test_exact(self):
        result = exclude("orders.amount", CONTEXT)
        assert list(result) == ["orders.region", "customers.name"]
