import pytest
import polars as pl

from datasubway import DataModel


class TestDataModelJoins:
    """Test suite for DataModel join functionality."""

    @pytest.fixture
    def simple_tables(self):
        """Create simple test tables."""
        return {
            'users': pl.LazyFrame({
                'user_id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie']
            }),
            'orders': pl.LazyFrame({
                'order_id': [101, 102, 103],
                'user_id': [1, 2, 1],
                'amount': [100, 200, 150]
            }),
            'products': pl.LazyFrame({
                'product_id': [1, 2, 3],
                'order_id': [101, 102, 103],
                'product_name': ['Widget', 'Gadget', 'Tool']
            }),
            'categories': pl.LazyFrame({
                'category_id': [1, 2],
                'product_id': [1, 2],
                'category_name': ['Electronics', 'Tools']
            })
        }

    def test_transitive_joins_simple_chain(self, simple_tables):
        """Test that transitive joins work for a simple chain: users -> orders -> products."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'orders',
                'right': 'products',
                'left_on': ['order_id'],
                'right_on': ['order_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Check direct join exists
        assert 'users' in dm.join_lookup
        assert 'orders' in dm.join_lookup['users']
        assert dm.join_lookup['users']['orders']['path'] == ['users', 'orders']
        assert len(dm.join_lookup['users']['orders']['join_specs']) == 1

        # Check transitive join exists (users -> orders -> products)
        assert 'products' in dm.join_lookup['users']
        assert dm.join_lookup['users']['products']['path'] == ['users', 'orders', 'products']
        assert len(dm.join_lookup['users']['products']['join_specs']) == 2

        # Verify join specs are correct
        specs = dm.join_lookup['users']['products']['join_specs']
        assert specs[0]['left'] == 'users'
        assert specs[0]['right'] == 'orders'
        assert specs[1]['left'] == 'orders'
        assert specs[1]['right'] == 'products'

    def test_transitive_joins_long_chain(self, simple_tables):
        """Test transitive joins across a longer chain: users -> orders -> products -> categories."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'orders',
                'right': 'products',
                'left_on': ['order_id'],
                'right_on': ['order_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'products',
                'right': 'categories',
                'left_on': ['product_id'],
                'right_on': ['product_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Check all intermediate paths exist
        assert dm.join_lookup['users']['orders']['path'] == ['users', 'orders']
        assert dm.join_lookup['users']['products']['path'] == ['users', 'orders', 'products']
        assert dm.join_lookup['users']['categories']['path'] == ['users', 'orders', 'products', 'categories']

        # Verify longest path has 3 join specs
        assert len(dm.join_lookup['users']['categories']['join_specs']) == 3

    def test_cycle_detection_simple_loop(self, simple_tables):
        """Test that 2-node cycles are allowed (bidirectional edges).

        Note: Two separate right2left joins forming A -> B -> A is functionally
        equivalent to a single bidirectional join and should be allowed.
        """
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'orders',
                'right': 'users',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        # This should NOT raise an error - 2-node cycles are allowed
        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Verify both directions exist in join_lookup
        assert 'users' in dm.join_lookup
        assert 'orders' in dm.join_lookup['users']
        assert 'orders' in dm.join_lookup
        assert 'users' in dm.join_lookup['orders']

    def test_cycle_detection_three_node_loop(self, simple_tables):
        """Test that cycles with three nodes (A -> B -> C -> A) are detected."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'orders',
                'right': 'products',
                'left_on': ['order_id'],
                'right_on': ['order_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'products',
                'right': 'users',
                'left_on': ['product_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        with pytest.raises(ValueError, match="Cycle detected in join graph"):
            DataModel(
                tables=simple_tables,
                joins=joins,
                pre_aggregations={},
                pre_agg_directory=None
            )

    def test_cycle_detection_self_loop(self):
        """Test that self-loops (A -> A) are detected."""
        tables = {
            'users': pl.LazyFrame({
                'user_id': [1, 2],
                'parent_id': [2, 1]
            })
        }

        joins = [
            {
                'left': 'users',
                'right': 'users',
                'left_on': ['user_id'],
                'right_on': ['parent_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        with pytest.raises(ValueError, match="Cycle detected in join graph"):
            DataModel(
                tables=tables,
                joins=joins,
                pre_aggregations={},
                pre_agg_directory=None
            )

    def test_multiple_paths_detection(self, simple_tables):
        """Test that multiple paths between two tables are detected and raise an error."""
        joins = [
            # Path 1: users -> orders
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            # Path 2: users -> products -> orders (creates second path to orders)
            {
                'left': 'users',
                'right': 'products',
                'left_on': ['user_id'],
                'right_on': ['product_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'products',
                'right': 'orders',
                'left_on': ['order_id'],
                'right_on': ['order_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        with pytest.raises(ValueError, match="Multiple paths from users to orders"):
            DataModel(
                tables=simple_tables,
                joins=joins,
                pre_aggregations={},
                pre_agg_directory=None
            )

    def test_bidirectional_cycle_not_multiple_paths(self):
        """Test that paths differing only by bidirectional cycles are not flagged as multiple paths.

        This reproduces the main.py scenario where:
        - sales -> stores (direct)
        - sales -> stores -> geography (transitive)
        - stores <-> geography (bidirectional, creates potential cycle)

        The path sales -> stores -> geography -> stores should be skipped,
        not treated as a second path to stores.
        """
        tables = {
            'sales': pl.LazyFrame({'sale_id': [1], 'store_id': [1], 'product_id': [1]}),
            'stores': pl.LazyFrame({'store_id': [1], 'geography_id': [1]}),
            'geography': pl.LazyFrame({'geography_id': [1], 'name': ['East']}),
            'products': pl.LazyFrame({'product_id': [1], 'name': ['Widget']})
        }

        joins = [
            {
                'left': 'sales', 'right': 'products',
                'left_on': ['product_id'], 'right_on': ['product_id'],
                'how': 'inner', 'direction': 'both'
            },
            {
                'left': 'sales', 'right': 'stores',
                'left_on': ['store_id'], 'right_on': ['store_id'],
                'how': 'left', 'direction': 'right2left'
            },
            {
                'left': 'stores', 'right': 'geography',
                'left_on': ['geography_id'], 'right_on': ['geography_id'],
                'how': 'inner', 'direction': 'both'
            }
        ]

        # Should NOT raise error (previously raised "Multiple paths from sales to stores")
        dm = DataModel(tables=tables, joins=joins, pre_aggregations={}, pre_agg_directory=None)

        # Verify paths exist
        assert 'sales' in dm.join_lookup
        assert 'stores' in dm.join_lookup['sales']
        assert 'geography' in dm.join_lookup['sales']

        # Path to stores should be direct (not through geography cycle)
        assert dm.join_lookup['sales']['stores']['path'] == ['sales', 'stores']

        # Path to geography should be through stores
        assert dm.join_lookup['sales']['geography']['path'] == ['sales', 'stores', 'geography']

    def test_direction_right2left_only(self, simple_tables):
        """Test that right2left direction only allows one-way traversal."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Should allow users -> orders
        assert 'users' in dm.join_lookup
        assert 'orders' in dm.join_lookup['users']

        # Should NOT allow orders -> users (reverse direction)
        assert 'orders' not in dm.join_lookup or 'users' not in dm.join_lookup.get('orders', {})

    def test_direction_both_allows_bidirectional(self, simple_tables):
        """Test that direction='both' allows traversal in both directions."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'both'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Should allow users -> orders
        assert 'users' in dm.join_lookup
        assert 'orders' in dm.join_lookup['users']
        assert dm.join_lookup['users']['orders']['path'] == ['users', 'orders']

        # Should also allow orders -> users
        assert 'orders' in dm.join_lookup
        assert 'users' in dm.join_lookup['orders']
        assert dm.join_lookup['orders']['users']['path'] == ['orders', 'users']

        # Verify the join specs are correctly swapped for reverse direction
        forward_spec = dm.join_lookup['users']['orders']['join_specs'][0]
        reverse_spec = dm.join_lookup['orders']['users']['join_specs'][0]

        assert forward_spec['left'] == 'users'
        assert forward_spec['right'] == 'orders'
        assert reverse_spec['left'] == 'orders'
        assert reverse_spec['right'] == 'users'
        assert forward_spec['left_on'] == reverse_spec['right_on']
        assert forward_spec['right_on'] == reverse_spec['left_on']

    def test_disconnected_components(self, simple_tables):
        """Test that disconnected components are handled correctly."""
        joins = [
            # Component 1: users -> orders
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            # Component 2: products -> categories (disconnected from users/orders)
            {
                'left': 'products',
                'right': 'categories',
                'left_on': ['product_id'],
                'right_on': ['product_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Check component 1 exists
        assert 'users' in dm.join_lookup
        assert 'orders' in dm.join_lookup['users']

        # Check component 2 exists
        assert 'products' in dm.join_lookup
        assert 'categories' in dm.join_lookup['products']

        # Verify no path between disconnected components
        assert 'products' not in dm.join_lookup.get('users', {})
        assert 'users' not in dm.join_lookup.get('products', {})

    def test_empty_joins_list(self, simple_tables):
        """Test that an empty joins list creates an empty join_lookup."""
        dm = DataModel(
            tables=simple_tables,
            joins=[],
            pre_aggregations={},
            pre_agg_directory=None
        )

        assert dm.join_lookup == {}

    def test_single_table_no_joins(self):
        """Test that a single table with no joins works correctly."""
        tables = {
            'users': pl.LazyFrame({
                'user_id': [1, 2, 3],
                'name': ['Alice', 'Bob', 'Charlie']
            })
        }

        dm = DataModel(
            tables=tables,
            joins=[],
            pre_aggregations={},
            pre_agg_directory=None
        )

        assert dm.join_lookup == {}

    def test_star_schema(self):
        """Test a typical star schema: fact table joined to multiple dimension tables."""
        tables = {
            'sales': pl.LazyFrame({
                'sale_id': [1, 2, 3],
                'user_id': [1, 2, 1],
                'product_id': [10, 20, 30],
                'store_id': [100, 200, 100]
            }),
            'users': pl.LazyFrame({
                'user_id': [1, 2],
                'name': ['Alice', 'Bob']
            }),
            'products': pl.LazyFrame({
                'product_id': [10, 20, 30],
                'product_name': ['Widget', 'Gadget', 'Tool']
            }),
            'stores': pl.LazyFrame({
                'store_id': [100, 200],
                'store_name': ['Store A', 'Store B']
            })
        }

        joins = [
            {
                'left': 'sales',
                'right': 'users',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'sales',
                'right': 'products',
                'left_on': ['product_id'],
                'right_on': ['product_id'],
                'how': 'inner',
                'direction': 'right2left'
            },
            {
                'left': 'sales',
                'right': 'stores',
                'left_on': ['store_id'],
                'right_on': ['store_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # Verify all dimensions are reachable from fact table
        assert 'sales' in dm.join_lookup
        assert 'users' in dm.join_lookup['sales']
        assert 'products' in dm.join_lookup['sales']
        assert 'stores' in dm.join_lookup['sales']

        # Verify each join is direct (no transitive paths in star schema)
        assert dm.join_lookup['sales']['users']['path'] == ['sales', 'users']
        assert dm.join_lookup['sales']['products']['path'] == ['sales', 'products']
        assert dm.join_lookup['sales']['stores']['path'] == ['sales', 'stores']

    def test_join_spec_doesnt_contain_direction(self, simple_tables):
        """Test that 'direction' field is removed from join_specs in join_lookup."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        join_spec = dm.join_lookup['users']['orders']['join_specs'][0]

        # Verify required fields are present
        assert 'left' in join_spec
        assert 'right' in join_spec
        assert 'left_on' in join_spec
        assert 'right_on' in join_spec
        assert 'how' in join_spec

        # Verify direction is NOT present
        assert 'direction' not in join_spec

    def test_left_join_validation(self, simple_tables):
        """Test that left joins with direction='both' are rejected."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'left',
                'direction': 'both'
            }
        ]

        with pytest.raises(Exception, match="Left joins only make sense with direction=right2left"):
            DataModel(
                tables=simple_tables,
                joins=joins,
                pre_aggregations={},
                pre_agg_directory=None
            )

    def test_left_join_with_right2left_succeeds(self, simple_tables):
        """Test that left joins with direction='right2left' are allowed."""
        joins = [
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'left',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        assert 'users' in dm.join_lookup
        assert 'orders' in dm.join_lookup['users']
        assert dm.join_lookup['users']['orders']['join_specs'][0]['how'] == 'left'

    def test_invalid_table_reference(self, simple_tables):
        """Test that invalid table references are caught."""
        joins = [
            {
                'left': 'users',
                'right': 'nonexistent_table',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        with pytest.raises(KeyError, match="left and right join tables must exist"):
            DataModel(
                tables=simple_tables,
                joins=joins,
                pre_aggregations={},
                pre_agg_directory=None
            )

    def test_complex_multi_hop_with_bidirectional(self, simple_tables):
        """Test complex scenario with both right2left and both directions."""
        joins = [
            # users <-> orders (bidirectional)
            {
                'left': 'users',
                'right': 'orders',
                'left_on': ['user_id'],
                'right_on': ['user_id'],
                'how': 'inner',
                'direction': 'both'
            },
            # orders -> products (one-way)
            {
                'left': 'orders',
                'right': 'products',
                'left_on': ['order_id'],
                'right_on': ['order_id'],
                'how': 'inner',
                'direction': 'right2left'
            }
        ]

        dm = DataModel(
            tables=simple_tables,
            joins=joins,
            pre_aggregations={},
            pre_agg_directory=None
        )

        # users -> orders -> products should exist
        assert dm.join_lookup['users']['products']['path'] == ['users', 'orders', 'products']

        # orders -> users should exist (bidirectional)
        assert dm.join_lookup['orders']['users']['path'] == ['orders', 'users']

        # products -> orders should NOT exist (only right2left)
        assert 'products' not in dm.join_lookup or 'orders' not in dm.join_lookup.get('products', {})
