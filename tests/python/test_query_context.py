"""Tests for QueryContext validation."""

import pytest
from datasubway.query_context import QueryContext


class TestQueryContext:
    def test_minimal(self):
        qc = QueryContext({"measures": ["revenue"]})
        assert qc.measures == ["revenue"]
        assert qc.filters == {}
        assert qc.groups == []
        assert qc.havings == {}
        assert qc.sorts == []
        assert qc.limit == 10000
        assert qc.offset == 0
        assert qc.use_pre_agg is True

    def test_full(self):
        qc = QueryContext(
            {
                "measures": ["revenue"],
                "filters": {"AND": [("orders.region", "=", "US")]},
                "groups": ["orders.date"],
                "havings": {"AND": [("revenue", ">", 1000)]},
                "sorts": [("revenue", "desc")],
                "limit": 500,
                "offset": 10,
                "use_pre_agg": False,
            }
        )
        assert qc.measures == ["revenue"]
        assert qc.limit == 500
        assert qc.offset == 10
        assert qc.use_pre_agg is False

    def test_missing_measures(self):
        with pytest.raises(ValueError, match="measures"):
            QueryContext({})

    def test_empty_measures(self):
        with pytest.raises(ValueError, match="measures"):
            QueryContext({"measures": []})

    def test_zero_limit(self):
        with pytest.raises(ValueError, match="limit"):
            QueryContext({"measures": ["revenue"], "limit": 0})

    def test_negative_offset(self):
        with pytest.raises(ValueError, match="offset"):
            QueryContext({"measures": ["revenue"], "offset": -1})

    def test_invalid_use_pre_agg(self):
        with pytest.raises(ValueError, match="use_pre_agg"):
            QueryContext({"measures": ["revenue"], "use_pre_agg": "yes"})

    def test_multiple_measures(self):
        qc = QueryContext({"measures": ["revenue", "quantity"]})
        assert qc.measures == ["revenue", "quantity"]
