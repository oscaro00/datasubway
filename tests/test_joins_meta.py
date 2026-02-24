import pytest

from datasubway.joins_meta import Join, parse_joins

# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

SIMPLE_CHAIN = [
    {
        "left": "orders",
        "right": "customers",
        "left_on": "order_id",
        "right_on": "id",
        "how": "inner",
        "direction": "right2left",
    },
    {
        "left": "customers",
        "right": "regions",
        "left_on": "region_id",
        "right_on": "id",
        "how": "left",
        "direction": "right2left",
    },
]

BIDIRECTIONAL = [
    {
        "left": "fact_sales",
        "right": "dim_date",
        "left_on": "date_id",
        "right_on": "id",
        "how": "inner",
        "direction": "both",
    },
    {
        "left": "dim_date",
        "right": "dim_category",
        "left_on": "category_id",
        "right_on": "id",
        "how": "left",
        "direction": "right2left",
    },
]


# ---------------------------------------------------------------------------
# 1. Join dataclass — __post_init__ normalisation
# ---------------------------------------------------------------------------


def test_join_string_left_on_normalized_to_list():
    j = Join(
        left="a",
        right="b",
        left_on="x",
        right_on="y",
        how="inner",
        direction="right2left",
    )
    assert j.left_on_cols == ["x"]
    assert j.right_on_cols == ["y"]


def test_join_list_left_on_preserved():
    j = Join(
        left="a",
        right="b",
        left_on=["x", "z"],
        right_on=["y", "w"],
        how="inner",
        direction="right2left",
    )
    assert j.left_on_cols == ["x", "z"]
    assert j.right_on_cols == ["y", "w"]


# ---------------------------------------------------------------------------
# 2. parse_joins — basic structure
# ---------------------------------------------------------------------------


def test_parse_joins_empty_returns_empty_dict():
    assert parse_joins([]) == {}


def test_parse_joins_all_tables_are_keys():
    result = parse_joins(SIMPLE_CHAIN)
    assert set(result.keys()) == {"orders", "customers", "regions"}


def test_parse_joins_single_right2left_reachability():
    joins = [
        {
            "left": "a",
            "right": "b",
            "left_on": "b_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        }
    ]
    result = parse_joins(joins)
    assert "b" in result["a"]
    assert "a" not in result["b"]


def test_parse_joins_single_both_reachability():
    joins = [
        {
            "left": "a",
            "right": "b",
            "left_on": "b_id",
            "right_on": "id",
            "how": "inner",
            "direction": "both",
        }
    ]
    result = parse_joins(joins)
    assert "b" in result["a"]
    assert "a" in result["b"]


# ---------------------------------------------------------------------------
# 3. parse_joins — migrated __main__ scenarios
# ---------------------------------------------------------------------------


def test_parse_joins_chain_right2left_reachability():
    # Scenario 1: orders→customers→regions, all right2left
    result = parse_joins(SIMPLE_CHAIN)
    # orders can reach both customers and regions
    assert "customers" in result["orders"]
    assert "regions" in result["orders"]
    # customers can reach regions (direct)
    assert "regions" in result["customers"]
    # neither customers nor regions can reach orders
    assert "orders" not in result["customers"]
    assert "orders" not in result["regions"]


def test_parse_joins_chain_right2left_multihop_path():
    # Scenario 1: orders→regions path must go through two Join objects
    result = parse_joins(SIMPLE_CHAIN)
    path = result["orders"]["regions"]
    assert len(path) == 2
    assert path[0].left == "orders" and path[0].right == "customers"
    assert path[1].left == "customers" and path[1].right == "regions"


def test_parse_joins_bidirectional_with_extension_reachability():
    # Scenario 2: fact_sales↔dim_date→dim_category
    result = parse_joins(BIDIRECTIONAL)
    # fact_sales can reach dim_date (both direction)
    assert "dim_date" in result["fact_sales"]
    # fact_sales can reach dim_category via dim_date
    assert "dim_category" in result["fact_sales"]
    # dim_date can reach fact_sales (both direction) and dim_category
    assert "fact_sales" in result["dim_date"]
    assert "dim_category" in result["dim_date"]
    # dim_category cannot reach anything (it's a sink)
    assert result["dim_category"] == {}


def test_parse_joins_multiple_paths_raises():
    # Scenario 3: a→b, a→c, b→c → two paths from a to c
    joins = [
        {
            "left": "a",
            "right": "b",
            "left_on": "b_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        },
        {
            "left": "a",
            "right": "c",
            "left_on": "c_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        },
        {
            "left": "b",
            "right": "c",
            "left_on": "c_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        },
    ]
    with pytest.raises(ValueError, match="Multiple join paths"):
        parse_joins(joins)


def test_parse_joins_cycle_raises():
    # Scenario 4: a→b→c→a forms a directed cycle of 3 tables
    joins = [
        {
            "left": "a",
            "right": "b",
            "left_on": "b_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        },
        {
            "left": "b",
            "right": "c",
            "left_on": "c_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        },
        {
            "left": "c",
            "right": "a",
            "left_on": "a_id",
            "right_on": "id",
            "how": "inner",
            "direction": "right2left",
        },
    ]
    with pytest.raises(ValueError, match=r"Cycle of 3\+"):
        parse_joins(joins)


# ---------------------------------------------------------------------------
# 4. parse_joins — additional edge cases
# ---------------------------------------------------------------------------


def test_parse_joins_isolated_table_has_empty_destinations():
    # regions is a sink (nothing outgoing from it); its entry should be {}
    result = parse_joins(SIMPLE_CHAIN)
    assert result["regions"] == {}


def test_parse_joins_multi_key_join_columns_preserved():
    joins = [
        {
            "left": "orders",
            "right": "line_items",
            "left_on": ["order_id", "tenant_id"],
            "right_on": ["order_fk", "tenant_fk"],
            "how": "inner",
            "direction": "right2left",
        }
    ]
    result = parse_joins(joins)
    join_obj = result["orders"]["line_items"][0]
    assert join_obj.left_on_cols == ["order_id", "tenant_id"]
    assert join_obj.right_on_cols == ["order_fk", "tenant_fk"]


def test_parse_joins_right2left_is_strictly_one_directional():
    # In the chain A→B→C (all right2left), C must not reach A or B
    result = parse_joins(SIMPLE_CHAIN)
    regions_destinations = result["regions"]
    assert "orders" not in regions_destinations
    assert "customers" not in regions_destinations
