use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

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

/// Adjacency-list based join graph.
/// Computes join paths between tables, validates no 3+ cycles.
#[derive(Debug, Clone)]
pub struct JoinGraph {
    /// adjacency[a][b] = Join specification for a→b edge
    adjacency: HashMap<String, HashMap<String, Join>>,
    /// All table names in the graph
    tables: HashSet<String>,
}

impl JoinGraph {
    pub fn new(joins: &[Join]) -> Result<Self, String> {
        let mut adjacency: HashMap<String, HashMap<String, Join>> = HashMap::new();
        let mut tables = HashSet::new();

        for join in joins {
            tables.insert(join.left.clone());
            tables.insert(join.right.clone());

            // Always add left→right edge
            adjacency
                .entry(join.left.clone())
                .or_default()
                .insert(join.right.clone(), join.clone());

            if join.direction == JoinDirection::Both {
                // Add reverse edge with swapped on-columns
                let reverse = Join {
                    left: join.right.clone(),
                    right: join.left.clone(),
                    left_on: join.right_on.clone(),
                    right_on: join.left_on.clone(),
                    how: join.how.clone(),
                    direction: join.direction.clone(),
                };
                adjacency
                    .entry(join.right.clone())
                    .or_default()
                    .insert(join.left.clone(), reverse);
            }
        }

        let graph = JoinGraph { adjacency, tables };
        graph.validate_no_long_cycles()?;
        Ok(graph)
    }

    /// Find the join path from `start` to `end` using BFS.
    /// Returns the ordered list of Joins to traverse, or None if unreachable.
    pub fn find_path(&self, start: &str, end: &str) -> Option<Vec<Join>> {
        if start == end {
            return Some(vec![]);
        }

        let mut visited = HashSet::new();
        // Queue stores (current_node, path_so_far)
        let mut queue: VecDeque<(String, Vec<Join>)> = VecDeque::new();
        visited.insert(start.to_string());
        queue.push_back((start.to_string(), vec![]));

        while let Some((current, path)) = queue.pop_front() {
            if let Some(neighbors) = self.adjacency.get(&current) {
                for (neighbor, join) in neighbors {
                    if neighbor == end {
                        let mut result = path.clone();
                        result.push(join.clone());
                        return Some(result);
                    }
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        let mut new_path = path.clone();
                        new_path.push(join.clone());
                        queue.push_back((neighbor.clone(), new_path));
                    }
                }
            }
        }
        None
    }

    /// Build full lookup: lookup[start][end] = vec of Joins to get from start to end.
    pub fn build_lookup(&self) -> HashMap<String, HashMap<String, Vec<Join>>> {
        let mut lookup = HashMap::new();
        for start in &self.tables {
            let mut inner = HashMap::new();
            for end in &self.tables {
                if start != end {
                    if let Some(path) = self.find_path(start, end) {
                        inner.insert(end.clone(), path);
                    }
                }
            }
            lookup.insert(start.clone(), inner);
        }
        lookup
    }

    /// Validate that no cycles of length >= 3 exist.
    /// 2-node cycles from bidirectional edges are allowed.
    fn validate_no_long_cycles(&self) -> Result<(), String> {
        for start in &self.tables {
            let mut visited = HashMap::new();
            visited.insert(start.clone(), 0usize);
            self.dfs_cycle_check(start, &mut visited, 1)?;
        }
        Ok(())
    }

    fn dfs_cycle_check(
        &self,
        current: &str,
        visited: &mut HashMap<String, usize>,
        depth: usize,
    ) -> Result<(), String> {
        if let Some(neighbors) = self.adjacency.get(current) {
            for neighbor in neighbors.keys() {
                if let Some(&visit_depth) = visited.get(neighbor) {
                    // Cycle detected — only allowed if length == 2 (bidirectional edge)
                    let cycle_len = depth - visit_depth;
                    if cycle_len >= 3 {
                        return Err(format!(
                            "Cycle of length {} detected involving table '{}'",
                            cycle_len, neighbor
                        ));
                    }
                } else {
                    visited.insert(neighbor.clone(), depth);
                    self.dfs_cycle_check(neighbor, visited, depth + 1)?;
                    visited.remove(neighbor);
                }
            }
        }
        Ok(())
    }

    pub fn tables(&self) -> &HashSet<String> {
        &self.tables
    }
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
        // rightonleft only: customers cannot reach orders
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
        let result = JoinGraph::new(&joins);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_lookup() {
        let graph = JoinGraph::new(&sample_joins()).unwrap();
        let lookup = graph.build_lookup();
        assert!(lookup["orders"].contains_key("customers"));
        assert!(lookup["orders"].contains_key("products"));
    }
}
