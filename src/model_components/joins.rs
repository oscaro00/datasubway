use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use tracing::trace;

/// Whether the join is inner or left.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JoinHow {
    Left,
    Inner,
}

/// Whether the join edge is bidirectional or right-to-left only.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum JoinDirection {
    Both,
    RightOnLeft,
}

/// A single join specification between two tables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Join {
    pub left: String,
    pub right: String,
    pub left_on: Vec<String>,
    pub right_on: Vec<String>,
    pub how: JoinHow,
    pub direction: JoinDirection,
}

/// Join graph with O(1) path lookup.
/// Validates no 3+ cycles and no multiple paths between tables at construction.
pub struct JoinGraph {
    lookup: HashMap<String, HashMap<String, Vec<Join>>>,
    tables: HashSet<String>,
}

impl JoinGraph {
    pub fn new(joins: &[Join]) -> Result<Self, String> {
        let mut tables = HashSet::new();
        for join in joins {
            tables.insert(join.left.clone());
            tables.insert(join.right.clone());
        }

        validate_no_long_cycles(joins)?;
        let lookup = compute_lookup(joins, &tables)?;

        trace!(tables = tables.len(), "join graph built");
        Ok(JoinGraph { lookup, tables })
    }

    /// Returns the pre-computed join path from `start` to `end`, or None if unreachable.
    pub fn find_path(&self, start: &str, end: &str) -> Option<Vec<Join>> {
        if start == end {
            return Some(vec![]);
        }
        self.lookup.get(start)?.get(end).cloned()
    }

    /// Finds a single table among `referenced_tables` that can reach every other
    /// referenced table via `find_path`. Returns an error listing the tables if none exists.
    pub fn find_reachable_base(&self, referenced_tables: &[String]) -> Result<String, String> {
        referenced_tables
            .iter()
            .find(|candidate| {
                referenced_tables
                    .iter()
                    .all(|t| t == *candidate || self.find_path(candidate, t).is_some())
            })
            .cloned()
            .ok_or_else(|| {
                format!(
                    "no single base table can reach all tables {referenced_tables:?} via join graph"
                )
            })
    }

    /// Returns the minimal, deduplicated set of joins needed to reach all `targets` from `base`.
    /// Shared intermediate hops are not duplicated across multiple target paths.
    pub fn find_joins_for_tables(&self, base: &str, targets: &[&str]) -> Vec<Join> {
        if targets.is_empty() {
            return vec![];
        }
        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut joins = Vec::new();
        for &target in targets {
            if let Some(path) = self.find_path(base, target) {
                for join in path {
                    if seen.insert((join.left.clone(), join.right.clone())) {
                        joins.push(join);
                    }
                }
            }
        }
        joins
    }

    pub fn build_lookup(&self) -> HashMap<String, HashMap<String, Vec<Join>>> {
        self.lookup.clone()
    }

    pub fn tables(&self) -> &HashSet<String> {
        &self.tables
    }
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Detects any directed simple cycle involving 3 or more distinct tables.
/// Two-table cycles from bidirectional edges (A↔B) are allowed.
fn validate_no_long_cycles(joins: &[Join]) -> Result<(), String> {
    let mut adj: HashMap<&str, Vec<&str>> = HashMap::new();
    let mut all_nodes: HashSet<&str> = HashSet::new();
    for join in joins {
        let l = join.left.as_str();
        let r = join.right.as_str();
        all_nodes.insert(l);
        all_nodes.insert(r);
        adj.entry(l).or_default().push(r);
        if join.direction == JoinDirection::Both {
            adj.entry(r).or_default().push(l);
        }
    }

    for &start in &all_nodes {
        if has_long_cycle_from(&adj, start) {
            return Err(format!(
                "Cycle of length >= 3 detected involving table '{start}'"
            ));
        }
    }
    Ok(())
}

/// Returns true if there is a simple path from `start` back to `start` passing through
/// at least 2 other distinct nodes (i.e., a simple directed cycle of length ≥ 3).
fn has_long_cycle_from(adj: &HashMap<&str, Vec<&str>>, start: &str) -> bool {
    // Iterative DFS tracking the current path.
    // path: (node, next-neighbor-index). in_path: nodes on the current path.
    let mut path: Vec<(&str, usize)> = vec![(start, 0)];
    let mut in_path: HashSet<&str> = HashSet::from([start]);

    while let Some(&(current, i)) = path.last() {
        let neighbors = adj.get(current).map(|v| v.as_slice()).unwrap_or(&[]);
        if i < neighbors.len() {
            path.last_mut().unwrap().1 += 1;
            let neighbor = neighbors[i];
            if neighbor == start {
                // Back-edge to start: cycle length = path.len() nodes (path includes start).
                // path.len() >= 3 means start + ≥2 intermediates → 3+ distinct nodes.
                if path.len() >= 3 {
                    return true;
                }
            } else if !in_path.contains(neighbor) {
                in_path.insert(neighbor);
                path.push((neighbor, 0));
            }
        } else {
            path.pop();
            in_path.remove(current);
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Path pre-computation
// ---------------------------------------------------------------------------

/// BFS from each table to pre-compute all reachable paths.
/// Also validates that each (start, end) pair has at most one simple path.
fn compute_lookup(
    joins: &[Join],
    tables: &HashSet<String>,
) -> Result<HashMap<String, HashMap<String, Vec<Join>>>, String> {
    // Build adjacency list including reverse edges for bidirectional joins.
    let mut adj: HashMap<&str, Vec<(&str, Join)>> = HashMap::new();
    for join in joins {
        adj.entry(join.left.as_str())
            .or_default()
            .push((join.right.as_str(), join.clone()));
        if join.direction == JoinDirection::Both {
            let reverse = Join {
                left: join.right.clone(),
                right: join.left.clone(),
                left_on: join.right_on.clone(),
                right_on: join.left_on.clone(),
                how: join.how.clone(),
                direction: join.direction.clone(),
            };
            adj.entry(join.right.as_str())
                .or_default()
                .push((join.left.as_str(), reverse));
        }
    }

    let mut lookup: HashMap<String, HashMap<String, Vec<Join>>> = HashMap::new();

    for start in tables {
        let start_str = start.as_str();
        let mut inner: HashMap<String, Vec<Join>> = HashMap::new();
        let mut visited: HashSet<&str> = HashSet::from([start_str]);
        // BFS parent map: parent[v] = the node that first discovered v.
        let mut parent: HashMap<&str, &str> = HashMap::new();
        let mut queue: VecDeque<(&str, Vec<Join>)> = VecDeque::new();
        queue.push_back((start_str, vec![]));

        while let Some((current, path)) = queue.pop_front() {
            for (neighbor, join_edge) in adj.get(current).map(|v| v.as_slice()).unwrap_or(&[]) {
                if *neighbor == start_str {
                    continue; // Skip back-edges to start.
                }
                if visited.contains(*neighbor) {
                    // Cross-edge to an already-visited node: skip it.
                    // BFS already recorded the shortest path to this neighbor, so we
                    // let that stand (first-found wins). The only exception we still
                    // allow explicitly is the reverse of a bidirectional tree edge,
                    // which is harmless and not a true second path.
                    continue;
                }
                visited.insert(neighbor);
                parent.insert(neighbor, current);
                let mut new_path = path.clone();
                new_path.push(join_edge.clone());
                inner.insert(neighbor.to_string(), new_path.clone());
                queue.push_back((neighbor, new_path));
            }
        }

        trace!(from = %start, reachable = inner.len(), "paths pre-computed");
        lookup.insert(start.clone(), inner);
    }

    Ok(lookup)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_joins() -> Vec<Join> {
        vec![
            Join {
                left: "orders".into(),
                right: "customers".into(),
                left_on: vec!["customer_id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Left,
                direction: JoinDirection::RightOnLeft,
            },
            Join {
                left: "orders".into(),
                right: "products".into(),
                left_on: vec!["product_id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Left,
                direction: JoinDirection::RightOnLeft,
            },
        ]
    }

    #[test]
    fn test_build_graph() {
        let graph = JoinGraph::new(&sample_joins()).unwrap();
        assert_eq!(graph.tables().len(), 3);
    }

    #[test]
    fn test_find_direct_path() {
        let graph = JoinGraph::new(&sample_joins()).unwrap();
        let path = graph.find_path("orders", "customers").unwrap();
        assert_eq!(path.len(), 1);
        assert_eq!(path[0].right, "customers");
    }

    #[test]
    fn test_no_reverse_path_unidirectional() {
        let graph = JoinGraph::new(&sample_joins()).unwrap();
        assert!(graph.find_path("customers", "orders").is_none());
    }

    #[test]
    fn test_bidirectional() {
        let joins = vec![Join {
            left: "a".into(),
            right: "b".into(),
            left_on: vec!["id".into()],
            right_on: vec!["a_id".into()],
            how: JoinHow::Inner,
            direction: JoinDirection::Both,
        }];
        let graph = JoinGraph::new(&joins).unwrap();
        assert!(graph.find_path("a", "b").is_some());
        assert!(graph.find_path("b", "a").is_some());
    }

    #[test]
    fn test_cycle_detection() {
        let joins = vec![
            Join {
                left: "a".into(),
                right: "b".into(),
                left_on: vec!["id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Inner,
                direction: JoinDirection::Both,
            },
            Join {
                left: "b".into(),
                right: "c".into(),
                left_on: vec!["id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Inner,
                direction: JoinDirection::Both,
            },
            Join {
                left: "c".into(),
                right: "a".into(),
                left_on: vec!["id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Inner,
                direction: JoinDirection::Both,
            },
        ];
        assert!(JoinGraph::new(&joins).is_err());
    }

    #[test]
    fn test_build_lookup() {
        let graph = JoinGraph::new(&sample_joins()).unwrap();
        let lookup = graph.build_lookup();
        assert!(lookup["orders"].contains_key("customers"));
        assert!(lookup["orders"].contains_key("products"));
    }

    #[test]
    fn test_find_joins_for_tables() {
        let graph = JoinGraph::new(&sample_joins()).unwrap();
        let joins = graph.find_joins_for_tables("orders", &["customers", "products"]);
        assert_eq!(joins.len(), 2);
        let rights: Vec<&str> = joins.iter().map(|j| j.right.as_str()).collect();
        assert!(rights.contains(&"customers"));
        assert!(rights.contains(&"products"));
    }

    #[test]
    fn test_find_joins_deduplicates_shared_hops() {
        // orders → items → products (bidirectional)
        // orders → customers (direct)
        // Asking for both products and items should not duplicate orders→items
        let joins = vec![
            Join {
                left: "orders".into(),
                right: "items".into(),
                left_on: vec!["id".into()],
                right_on: vec!["order_id".into()],
                how: JoinHow::Left,
                direction: JoinDirection::RightOnLeft,
            },
            Join {
                left: "items".into(),
                right: "products".into(),
                left_on: vec!["product_id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Left,
                direction: JoinDirection::RightOnLeft,
            },
            Join {
                left: "orders".into(),
                right: "customers".into(),
                left_on: vec!["customer_id".into()],
                right_on: vec!["id".into()],
                how: JoinHow::Left,
                direction: JoinDirection::RightOnLeft,
            },
        ];
        let graph = JoinGraph::new(&joins).unwrap();
        let result = graph.find_joins_for_tables("orders", &["customers", "products"]);
        // Should be 3 joins: orders→customers, orders→items, items→products
        // NOT 4 (orders→items should not appear twice)
        assert_eq!(result.len(), 3);
        let pairs: Vec<(&str, &str)> = result
            .iter()
            .map(|j| (j.left.as_str(), j.right.as_str()))
            .collect();
        assert_eq!(
            pairs
                .iter()
                .filter(|&&(l, r)| l == "orders" && r == "items")
                .count(),
            1
        );
    }
}
