"""Security tests for safe_literals validation module."""

import pytest

from datasubway.validation.safe_literals import (
    is_safe_literal,
    is_safe_identifier,
    is_safe_string,
    validate_safe_context,
    validate_all_strings_are_safe,
    ALLOWED_OPERATORS,
)


class TestIsSafeLiteral:
    """Tests for is_safe_literal function."""

    def test_accepts_string(self):
        assert is_safe_literal("hello") is True

    def test_accepts_int(self):
        assert is_safe_literal(42) is True

    def test_accepts_float(self):
        assert is_safe_literal(3.14) is True

    def test_accepts_bool(self):
        assert is_safe_literal(True) is True
        assert is_safe_literal(False) is True

    def test_accepts_none(self):
        assert is_safe_literal(None) is True

    def test_accepts_list_of_literals(self):
        assert is_safe_literal(["a", "b", "c"]) is True
        assert is_safe_literal([1, 2, 3]) is True
        assert is_safe_literal([1, "two", 3.0, True, None]) is True

    def test_accepts_nested_list(self):
        assert is_safe_literal([["a", "b"], ["c", "d"]]) is True

    def test_accepts_dict_with_string_keys(self):
        assert is_safe_literal({"key": "value"}) is True
        assert is_safe_literal({"a": 1, "b": 2}) is True

    def test_accepts_nested_dict(self):
        assert is_safe_literal({"outer": {"inner": "value"}}) is True

    def test_accepts_tuple(self):
        assert is_safe_literal(("a", "b")) is True

    def test_rejects_callable(self):
        assert is_safe_literal(lambda: None) is False

    def test_rejects_function(self):
        def my_func():
            pass
        assert is_safe_literal(my_func) is False

    def test_rejects_class(self):
        class MyClass:
            pass
        assert is_safe_literal(MyClass) is False

    def test_rejects_object_instance(self):
        class MyClass:
            pass
        assert is_safe_literal(MyClass()) is False

    def test_rejects_list_containing_callable(self):
        assert is_safe_literal([1, 2, lambda: None]) is False

    def test_rejects_dict_containing_callable(self):
        assert is_safe_literal({"key": lambda: None}) is False


class TestIsSafeIdentifier:
    """Tests for is_safe_identifier function."""

    def test_accepts_simple_identifier(self):
        assert is_safe_identifier("column") is True
        assert is_safe_identifier("my_column") is True
        assert is_safe_identifier("Column1") is True

    def test_accepts_dotted_identifier(self):
        assert is_safe_identifier("table.column") is True
        assert is_safe_identifier("sales.item_id") is True
        assert is_safe_identifier("a.b.c") is True

    def test_accepts_numbers(self):
        assert is_safe_identifier("col1") is True
        assert is_safe_identifier("123") is True

    def test_rejects_special_characters(self):
        assert is_safe_identifier("user@example.com") is False
        assert is_safe_identifier("hello world") is False
        assert is_safe_identifier("name's") is False
        assert is_safe_identifier("a+b") is False
        assert is_safe_identifier("a-b") is False

    def test_rejects_code_injection_attempts(self):
        assert is_safe_identifier("__import__('os')") is False
        assert is_safe_identifier("eval('code')") is False
        assert is_safe_identifier("print('pwned')") is False
        assert is_safe_identifier("'; DROP TABLE users; --") is False

    def test_rejects_empty_string(self):
        assert is_safe_identifier("") is False

    def test_rejects_only_dots(self):
        assert is_safe_identifier(".") is False
        assert is_safe_identifier("..") is False

    def test_rejects_leading_dot(self):
        assert is_safe_identifier(".column") is False

    def test_rejects_trailing_dot(self):
        assert is_safe_identifier("table.") is False


class TestIsSafeString:
    """Tests for is_safe_string function."""

    def test_accepts_safe_identifier(self):
        assert is_safe_string("column") is True
        assert is_safe_string("table.column") is True

    def test_accepts_allowed_operators(self):
        for op in ALLOWED_OPERATORS:
            assert is_safe_string(op) is True, f"Operator '{op}' should be safe"

    def test_accepts_comparison_operators(self):
        assert is_safe_string("=") is True
        assert is_safe_string("!=") is True
        assert is_safe_string(">") is True
        assert is_safe_string("<") is True
        assert is_safe_string(">=") is True
        assert is_safe_string("<=") is True

    def test_accepts_logical_operators(self):
        assert is_safe_string("AND") is True
        assert is_safe_string("OR") is True

    def test_accepts_sort_directions(self):
        assert is_safe_string("asc") is True
        assert is_safe_string("desc") is True
        assert is_safe_string("ASC") is True
        assert is_safe_string("DESC") is True

    def test_rejects_code_injection(self):
        assert is_safe_string("__import__('os')") is False
        assert is_safe_string("eval('code')") is False


class TestValidateSafeContext:
    """Tests for validate_safe_context function."""

    def test_accepts_valid_context(self):
        # Should not raise
        validate_safe_context({
            "measure": ["revenue"],
            "group": ["store_id"],
            "limit": 100,
            "filter": None
        })

    def test_rejects_callable_in_context(self):
        with pytest.raises(ValueError, match="Unsafe value"):
            validate_safe_context({"key": lambda: None})

    def test_rejects_nested_callable(self):
        with pytest.raises(ValueError, match="Unsafe value"):
            validate_safe_context({"outer": {"inner": lambda: None}})

    def test_error_message_includes_path(self):
        with pytest.raises(ValueError, match="root.bad_key"):
            validate_safe_context({"bad_key": lambda: None})


class TestValidateAllStringsAreSafe:
    """Tests for validate_all_strings_are_safe function."""

    def test_accepts_valid_identifiers(self):
        # Should not raise
        validate_all_strings_are_safe({
            "measure": ["revenue", "total_sales"],
            "group": ["store.region", "store.city"],
        })

    def test_rejects_special_characters_in_string(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({"email": "user@example.com"})

    def test_rejects_spaces_in_string(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({"name": "John Doe"})

    def test_rejects_apostrophe_in_string(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({"name": "O'Brien"})

    def test_rejects_code_injection_in_group(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({
                "measure": ["test"],
                "group": ["__import__('os').system('ls')"]
            })

    def test_rejects_sql_injection_attempt(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({
                "filter": {"column": "'; DROP TABLE users; --"}
            })

    def test_accepts_numeric_values(self):
        # Should not raise - numeric values don't need string validation
        validate_all_strings_are_safe({
            "limit": 100,
            "offset": 0,
            "ratio": 0.5
        })

    def test_accepts_boolean_values(self):
        # Should not raise
        validate_all_strings_are_safe({
            "allow_pre_aggs": True,
            "enabled": False
        })

    def test_accepts_none_values(self):
        # Should not raise
        validate_all_strings_are_safe({
            "filter": None,
            "having": None
        })

    def test_validates_nested_structures(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({
                "outer": {
                    "inner": ["valid", "also valid", "not valid!"]
                }
            })

    def test_validates_list_items(self):
        with pytest.raises(ValueError, match="Invalid string"):
            validate_all_strings_are_safe({
                "columns": ["col1", "col2", "bad column!"]
            })

    def test_error_message_includes_path(self):
        with pytest.raises(ValueError, match=r"root\.group\[1\]"):
            validate_all_strings_are_safe({
                "group": ["valid", "not valid!"]
            })


class TestSecurityScenarios:
    """End-to-end security scenario tests."""

    def test_prevents_code_execution_via_measure_name(self):
        """Ensure malicious measure names are rejected."""
        with pytest.raises(ValueError):
            validate_all_strings_are_safe({
                "measure": ["__import__('os').system('ls')"]
            })

    def test_prevents_code_execution_via_filter_value(self):
        """Ensure malicious filter values are rejected."""
        with pytest.raises(ValueError):
            validate_all_strings_are_safe({
                "filter": {
                    "column": "sales.region",
                    "operator": "=",
                    "value": "eval('malicious')"
                }
            })

    def test_prevents_code_execution_via_sort_column(self):
        """Ensure malicious sort columns are rejected."""
        with pytest.raises(ValueError):
            validate_all_strings_are_safe({
                "sort": [("exec('bad')", "asc")]
            })

    def test_accepts_legitimate_query_context(self):
        """Ensure legitimate query contexts pass validation."""
        # Should not raise - using operators in filter
        validate_safe_context({
            "measure": ["total_revenue", "average_order_value"],
            "group": ["time.month", "geography.region"],
            "filter": {
                "column": "sales.amount",
                "operator": ">",
                "value": 100
            },
            "sort": [("time.month", "desc")],
            "limit": 1000,
            "offset": 0,
            "allow_pre_aggs": True
        })
        # Also test with validate_all_strings_are_safe (operators are now allowed)
        validate_all_strings_are_safe({
            "measure": ["total_revenue", "average_order_value"],
            "group": ["time.month", "geography.region"],
            "filter": {
                "column": "sales.amount",
                "operator": ">",  # Operators like '>' are now allowed
                "value": 100
            },
            "sort": [("time.month", "desc")],
            "limit": 1000,
            "offset": 0,
            "allow_pre_aggs": True
        })

    def test_accepts_complex_filter_with_operators(self):
        """Ensure complex filters with AND/OR operators pass validation."""
        validate_all_strings_are_safe({
            "filter": {
                "OR": [
                    ("geography.country", "=", "US"),
                    {
                        "AND": [
                            ("geography.country", "=", "CA"),
                            ("sales.revenue", ">", 1000)
                        ]
                    }
                ]
            }
        })

    def test_accepts_filter_tuple_format(self):
        """Ensure filter tuples (column, operator, value) pass validation."""
        validate_all_strings_are_safe({
            "filter": [
                ("sales.region", "=", "east"),
                ("sales.amount", ">=", 100),
                ("sales.status", "!=", "cancelled"),
            ]
        })
