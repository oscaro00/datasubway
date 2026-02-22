from __future__ import annotations

from dataclasses import dataclass, field, InitVar


@dataclass
class Join:
    left: str
    right: str
    left_on: InitVar[str | list[str]]
    right_on: InitVar[str | list[str]]
    how: str
    direction: str
    left_on_cols: list[str] = field(init=False)
    right_on_cols: list[str] = field(init=False)

    def __post_init__(
        self, left_on: str | list[str], right_on: str | list[str]
    ) -> None:
        """Standardize left_on and right_on to lists and store on the instance."""
        self.left_on_cols = [left_on] if isinstance(left_on, str) else left_on
        self.right_on_cols = [right_on] if isinstance(right_on, str) else right_on


def parse_joins(joins_list: list[dict]) -> dict[str, dict[str, list[Join]]]:
    """Parse a list of join dicts into a nested lookup of join paths.

    Returns a double-nested dict: join_lookup[start_table][end_table] is an
    ordered list of Join objects that, applied in sequence, join end_table into
    a query rooted at start_table.

    Raises ValueError if:
    - Any two tables have more than one distinct join path between them.
    - A directed cycle involving 3+ distinct tables is found.
    """
    join_object_list: list[Join] = []
    for join_dict in joins_list:
        join_object_list.append(
            Join(
                left=join_dict["left"],
                right=join_dict["right"],
                left_on=join_dict["left_on"],
                right_on=join_dict["right_on"],
                how=join_dict["how"],
                direction=join_dict["direction"],
            )
        )

    # Build directed adjacency list.
    # right2left: left is the base; only left can reach right (left→right edge).
    # both:       bidirectional (left→right and right→left edges).
    adjacency: dict[str, list[tuple[str, Join]]] = {}
    all_tables: set[str] = set()

    for join in join_object_list:
        all_tables.add(join.left)
        all_tables.add(join.right)
        adjacency.setdefault(join.left, []).append((join.right, join))
        if join.direction == "both":
            adjacency.setdefault(join.right, []).append((join.left, join))

    join_lookup: dict[str, dict[str, list[Join]]] = {}

    for start in all_tables:
        join_lookup[start] = {}
        # Each stack entry: (current_node, joins_accumulated, ordered_path_of_nodes)
        stack: list[tuple[str, list[Join], list[str]]] = [(start, [], [start])]

        while stack:
            current, path_joins, path_nodes = stack.pop()
            visited = set(path_nodes)

            for neighbor, join_obj in adjacency.get(current, []):
                if neighbor in visited:
                    # A back-edge: check whether this is a 3+-table cycle.
                    # 2-node "cycles" from `both`-direction edges are fine and
                    # are simply pruned by the visited check.
                    cycle_start = path_nodes.index(neighbor)
                    cycle_nodes = path_nodes[cycle_start:]
                    if len(cycle_nodes) >= 3:
                        raise ValueError(
                            f"Cycle of 3+ tables detected: "
                            f"{' -> '.join(cycle_nodes + [neighbor])}"
                        )
                    continue

                if neighbor in join_lookup[start]:
                    raise ValueError(
                        f"Multiple join paths from '{start}' to '{neighbor}'. "
                        f"Each pair of tables must have exactly one join path."
                    )

                new_path = path_joins + [join_obj]
                join_lookup[start][neighbor] = new_path
                stack.append((neighbor, new_path, path_nodes + [neighbor]))

    return join_lookup


def _print_lookup(join_lookup: dict[str, dict[str, list[Join]]]) -> None:
    """Print each reachable path in a human-readable format."""
    for start, destinations in sorted(join_lookup.items()):
        if not destinations:
            print(f"  {start} → (nothing)")
            continue
        for end, joins in sorted(destinations.items()):
            hops = " → ".join(
                f"[{j.left} JOIN {j.right} on "
                f"{','.join(j.left_on_cols)}={','.join(j.right_on_cols)} ({j.how})]"
                for j in joins
            )
            print(f"  {start} → {end}  :  {hops}")


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Scenario 1 — Chain, all right2left
    # Three tables: orders → customers → regions
    # right2left means only left can reach right (one-directional).
    # ------------------------------------------------------------------
    print("=== Scenario 1: Chain, all right2left ===")
    s1 = parse_joins([
        {"left": "orders", "right": "customers", "left_on": "order_id", "right_on": "id", "how": "inner", "direction": "right2left"},
        {"left": "customers", "right": "regions", "left_on": "region_id", "right_on": "id", "how": "left", "direction": "right2left"},
    ])
    _print_lookup(s1)
    print()

    # ------------------------------------------------------------------
    # Scenario 2 — Bidirectional edge + extension
    # fact_sales <-> dim_date → dim_category
    # ------------------------------------------------------------------
    print("=== Scenario 2: Bidirectional edge + extension ===")
    s2 = parse_joins([
        {"left": "fact_sales", "right": "dim_date", "left_on": "date_id", "right_on": "id", "how": "inner", "direction": "both"},
        {"left": "dim_date", "right": "dim_category", "left_on": "category_id", "right_on": "id", "how": "left", "direction": "right2left"},
    ])
    _print_lookup(s2)
    print()

    # ------------------------------------------------------------------
    # Scenario 3 — Error: multiple paths from a to c
    # a→b, a→c, b→c gives two paths: a→c direct and a→b→c
    # ------------------------------------------------------------------
    print("=== Scenario 3: Error — multiple paths ===")
    try:
        parse_joins([
            {"left": "a", "right": "b", "left_on": "b_id", "right_on": "id", "how": "inner", "direction": "right2left"},
            {"left": "a", "right": "c", "left_on": "c_id", "right_on": "id", "how": "inner", "direction": "right2left"},
            {"left": "b", "right": "c", "left_on": "c_id", "right_on": "id", "how": "inner", "direction": "right2left"},
        ])
    except ValueError as e:
        print(f"  Caught expected error: {e}")
    print()

    # ------------------------------------------------------------------
    # Scenario 4 — Error: 3-table cycle
    # a→b→c→a forms a directed cycle of length 3
    # ------------------------------------------------------------------
    print("=== Scenario 4: Error — 3-table cycle ===")
    try:
        parse_joins([
            {"left": "a", "right": "b", "left_on": "b_id", "right_on": "id", "how": "inner", "direction": "right2left"},
            {"left": "b", "right": "c", "left_on": "c_id", "right_on": "id", "how": "inner", "direction": "right2left"},
            {"left": "c", "right": "a", "left_on": "a_id", "right_on": "id", "how": "inner", "direction": "right2left"},
        ])
    except ValueError as e:
        print(f"  Caught expected error: {e}")
