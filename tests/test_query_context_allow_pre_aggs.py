import pytest

from datasubway.query_context.query_context import QueryContext


class TestQueryContextAllowPreAggs:
    """Test suite for allow_pre_aggs functionality in QueryContext."""

    def test_allow_pre_aggs_with_true(self):
        """Test that allow_pre_aggs=True is accepted and validated."""
        qc = QueryContext({
            'measure': ['test_measure'],
            'allow_pre_aggs': True
        })
        assert qc.context['allow_pre_aggs'] is True

    def test_allow_pre_aggs_with_false(self):
        """Test that allow_pre_aggs=False is accepted and validated."""
        qc = QueryContext({
            'measure': ['test_measure'],
            'allow_pre_aggs': False
        })
        assert qc.context['allow_pre_aggs'] is False

    def test_allow_pre_aggs_defaults_to_true(self):
        """Test that allow_pre_aggs defaults to True when not specified."""
        qc = QueryContext({
            'measure': ['test_measure']
        })
        # Should not be in context yet
        assert 'allow_pre_aggs' not in qc.context
        # But getter should return default True
        assert qc.get_allow_pre_aggs() is True

    def test_get_allow_pre_aggs_when_true(self):
        """Test get_allow_pre_aggs() method when explicitly set to True."""
        qc = QueryContext({
            'measure': ['test_measure'],
            'allow_pre_aggs': True
        })
        assert qc.get_allow_pre_aggs() is True

    def test_get_allow_pre_aggs_when_false(self):
        """Test get_allow_pre_aggs() method when explicitly set to False."""
        qc = QueryContext({
            'measure': ['test_measure'],
            'allow_pre_aggs': False
        })
        assert qc.get_allow_pre_aggs() is False

    def test_get_allow_pre_aggs_with_default(self):
        """Test get_allow_pre_aggs() returns True by default."""
        qc = QueryContext({
            'measure': ['test_measure']
        })
        assert qc.get_allow_pre_aggs() is True

    def test_allow_pre_aggs_invalid_type_string(self):
        """Test that non-boolean allow_pre_aggs raises TypeError."""
        with pytest.raises(TypeError, match='allow_pre_aggs must be a boolean'):
            QueryContext({
                'measure': ['test_measure'],
                'allow_pre_aggs': 'true'  # String instead of boolean
            })

    def test_allow_pre_aggs_invalid_type_int(self):
        """Test that integer allow_pre_aggs raises TypeError."""
        with pytest.raises(TypeError, match='allow_pre_aggs must be a boolean'):
            QueryContext({
                'measure': ['test_measure'],
                'allow_pre_aggs': 1  # Integer instead of boolean
            })

    def test_allow_pre_aggs_invalid_type_none(self):
        """Test that None allow_pre_aggs raises TypeError."""
        with pytest.raises(TypeError, match='allow_pre_aggs must be a boolean'):
            QueryContext({
                'measure': ['test_measure'],
                'allow_pre_aggs': None
            })

    def test_allow_pre_aggs_with_other_valid_keys(self):
        """Test allow_pre_aggs works with other valid query context keys."""
        qc = QueryContext({
            'measure': ['test_measure'],
            'group': ['item_id', 'store_id'],
            'filter': ('sales.item_id', '=', 1),  # Must use table.column format
            'sort': [('item_id', 'asc')],
            'limit': 100,
            'offset': 10,
            'allow_pre_aggs': False
        })
        assert qc.get_allow_pre_aggs() is False
        assert qc.context['group'] == ['item_id', 'store_id']
        assert qc.context['limit'] == 100

    def test_backward_compatibility_without_allow_pre_aggs(self):
        """Test that existing code without allow_pre_aggs still works."""
        # This simulates old code that doesn't know about allow_pre_aggs
        qc = QueryContext({
            'measure': ['revenue_total'],
            'group': ['store_id']
        })
        # Should work fine and default to True
        assert qc.get_allow_pre_aggs() is True
        assert 'measure' in qc.context
        assert 'group' in qc.context
