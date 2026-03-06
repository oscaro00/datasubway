import pytest

from datasubway.query_context import QueryContext

VALID_FULL = {
    "measures": ["orders.total", "orders.count"],
    "filters": {"AND": [("orders.status", "=", "complete")]},
    "groups": ["orders.region", "orders.category"],
    "havings": {"AND": [("orders.total", ">", 100)]},
    "sorts": [("orders.total", "asc"), ("orders.region", "desc")],
    "limit": 500,
    "offset": 10,
    "use_pre_agg": False,
}

# --- Happy path ---


def test_minimal_valid_dict_uses_defaults():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.measures == ["orders.total"]
    assert qc.filters == {}
    assert qc.groups == []
    assert qc.havings == {}
    assert qc.sorts == []
    assert qc.limit == 10000
    assert qc.offset == 0
    assert qc.use_pre_agg is True


def test_full_valid_dict_stores_all_fields():
    qc = QueryContext(VALID_FULL)
    assert qc.measures == ["orders.total", "orders.count"]
    assert qc.filters == {"AND": [("orders.status", "=", "complete")]}
    assert qc.groups == ["orders.region", "orders.category"]
    assert qc.havings == {"AND": [("orders.total", ">", 100)]}
    assert qc.sorts == [("orders.total", "asc"), ("orders.region", "desc")]
    assert qc.limit == 500
    assert qc.offset == 10
    assert qc.use_pre_agg is False


# --- Measures validation ---


def test_measures_missing_raises_key_error():
    with pytest.raises(KeyError):
        QueryContext({})


def test_measures_not_a_list_raises_value_error():
    with pytest.raises(ValueError, match="measures"):
        QueryContext({"measures": "orders.total"})


def test_measures_contains_non_string_raises_value_error():
    with pytest.raises(ValueError, match="measures"):
        QueryContext({"measures": [123]})


def test_measures_valid_list_stored():
    qc = QueryContext({"measures": ["orders.total", "orders.count"]})
    assert qc.measures == ["orders.total", "orders.count"]


# --- Filters validation ---


def test_filters_not_a_dict_raises_value_error():
    with pytest.raises(ValueError, match="filters"):
        QueryContext({"measures": ["orders.total"], "filters": ["not", "a", "dict"]})


def test_filters_omitted_defaults_to_empty_dict():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.filters == {}


def test_filters_valid_dict_stored():
    filters = {"AND": [("orders.status", "=", "complete")]}
    qc = QueryContext({"measures": ["orders.total"], "filters": filters})
    assert qc.filters == filters


# --- Groups validation ---


def test_groups_not_a_list_raises_value_error():
    with pytest.raises(ValueError, match="groups"):
        QueryContext({"measures": ["orders.total"], "groups": "orders.region"})


def test_groups_contains_non_string_raises_value_error():
    with pytest.raises(ValueError, match="groups"):
        QueryContext({"measures": ["orders.total"], "groups": [42]})


def test_groups_omitted_defaults_to_empty_list():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.groups == []


def test_groups_valid_list_stored():
    qc = QueryContext({"measures": ["orders.total"], "groups": ["orders.region"]})
    assert qc.groups == ["orders.region"]


# --- Havings validation ---


def test_havings_not_a_dict_raises_value_error():
    with pytest.raises(ValueError, match="havings"):
        QueryContext({"measures": ["orders.total"], "havings": [1, 2, 3]})


def test_havings_omitted_defaults_to_empty_dict():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.havings == {}


def test_havings_valid_dict_stored():
    havings = {"AND": [("orders.total", ">", 100)]}
    qc = QueryContext({"measures": ["orders.total"], "havings": havings})
    assert qc.havings == havings


# --- Sorts validation ---


def test_sorts_not_a_list_raises_value_error():
    with pytest.raises(ValueError, match="sorts"):
        QueryContext({"measures": ["orders.total"], "sorts": "orders.total"})


def test_sorts_contains_non_string_raises_value_error():
    with pytest.raises(ValueError, match="sorts"):
        QueryContext({"measures": ["orders.total"], "sorts": [99]})


def test_sorts_omitted_defaults_to_empty_list():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.sorts == []


def test_sorts_valid_list_stored():
    qc = QueryContext({"measures": ["orders.total"], "sorts": [("orders.total", "asc")]})
    assert qc.sorts == [("orders.total", "asc")]


# --- Limit validation ---


def test_limit_not_an_int_raises_value_error():
    with pytest.raises(ValueError, match="limit"):
        QueryContext({"measures": ["orders.total"], "limit": "500"})


def test_limit_zero_raises_value_error():
    with pytest.raises(ValueError, match="limit"):
        QueryContext({"measures": ["orders.total"], "limit": 0})


def test_limit_negative_raises_value_error():
    with pytest.raises(ValueError, match="limit"):
        QueryContext({"measures": ["orders.total"], "limit": -1})


def test_limit_omitted_defaults_to_10000():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.limit == 10000


def test_limit_valid_positive_int_stored():
    qc = QueryContext({"measures": ["orders.total"], "limit": 250})
    assert qc.limit == 250


# --- Offset validation ---


def test_offset_not_an_int_raises_value_error():
    with pytest.raises(ValueError, match="offset"):
        QueryContext({"measures": ["orders.total"], "offset": "10"})


def test_offset_negative_raises_value_error():
    with pytest.raises(ValueError, match="offset"):
        QueryContext({"measures": ["orders.total"], "offset": -1})


def test_offset_omitted_defaults_to_zero():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.offset == 0


def test_offset_zero_is_valid():
    qc = QueryContext({"measures": ["orders.total"], "offset": 0})
    assert qc.offset == 0


def test_offset_valid_positive_int_stored():
    qc = QueryContext({"measures": ["orders.total"], "offset": 50})
    assert qc.offset == 50


# --- Use pre-agg validation ---


def test_use_pre_agg_not_a_bool_raises_value_error():
    with pytest.raises(ValueError, match="use_pre_agg"):
        QueryContext({"measures": ["orders.total"], "use_pre_agg": "true"})


def test_use_pre_agg_omitted_defaults_to_true():
    qc = QueryContext({"measures": ["orders.total"]})
    assert qc.use_pre_agg is True


def test_use_pre_agg_false_stored():
    qc = QueryContext({"measures": ["orders.total"], "use_pre_agg": False})
    assert qc.use_pre_agg is False
