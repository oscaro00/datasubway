use grafeo::{GrafeoDB, Session, Value};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
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

/// LPG-backed join graph stored in an in-memory grafeo database.
/// Validates no 3+ cycles and no multiple paths between tables at construction.
/// Paths are pre-computed into a lookup map for O(1) retrieval.
pub struct JoinGraph {
    db: GrafeoDB,
    lookup: HashMap<String, HashMap<String, Vec<Join>>>,
    tables: HashSet<String>,
}

impl JoinGraph {
    pub fn new(joins: &[Join]) -> Result<Self, String> {
        let db = GrafeoDB::new_in_memory();
        let session = db.session();
        let mut tables = HashSet::new();

        for join in joins {
            tables.insert(join.left.clone());
            tables.insert(join.right.clone());
        }

        for table in &tables {
            let mut params = HashMap::new();
            params.insert("name".to_string(), Value::String(table.as_str().into()));
            session
                .execute_with_params("INSERT (:Table {name: $name})", params)
                .map_err(|e| e.to_string())?;
        }

        for join in joins {
            insert_edge(&session, join)?;
            if join.direction == JoinDirection::Both {
                let reverse = Join {
                    left: join.right.clone(),
                    right: join.left.clone(),
                    left_on: join.right_on.clone(),
                    right_on: join.left_on.clone(),
                    how: join.how.clone(),
                    direction: join.direction.clone(),
                };
                insert_edge(&session, &reverse)?;
            }
        }

        validate_no_long_cycles(&session)?;
        let lookup = compute_lookup(&session, &tables)?;

        trace!(tables = tables.len(), "join graph built");
        Ok(JoinGraph { db, lookup, tables })
    }

    /// Returns the pre-computed join path from `start` to `end`, or None if unreachable.
    pub fn find_path(&self, start: &str, end: &str) -> Option<Vec<Join>> {
        if start == end {
            return Some(vec![]);
        }
        self.lookup.get(start)?.get(end).cloned()
    }

    /// Returns the minimal, deduplicated ordered set of joins needed to reach all
    /// `targets` from `base`. Uses a grafeo path query so shared intermediate hops
    /// are not duplicated across multiple target paths.
    pub fn find_joins_for_tables(&self, base: &str, targets: &[&str]) -> Vec<Join> {
        if targets.is_empty() {
            return vec![];
        }
        let session = self.db.session();
        let target_list = targets
            .iter()
            .map(|t| format!("'{t}'"))
            .collect::<Vec<_>>()
            .join(", ");
        let query = format!(
            "MATCH p = (b:Table {{name: $base}})-[:JOIN*1..100]->(t:Table) \
             WHERE t.name IN [{target_list}] \
             RETURN p ORDER BY length(p)"
        );
        let mut params = HashMap::new();
        params.insert("base".to_string(), Value::String(base.into()));

        let result = match session.execute_with_params(&query, params) {
            Ok(r) => r,
            Err(e) => {
                trace!(error = %e, "find_joins_for_tables query failed");
                return vec![];
            }
        };

        let mut seen: HashSet<(String, String)> = HashSet::new();
        let mut joins = Vec::new();

        for row in result.rows() {
            if let Some(Value::Path { nodes, edges }) = row.first() {
                for (i, edge_val) in edges.iter().enumerate() {
                    let left = node_name(&nodes[i]);
                    let right = node_name(&nodes[i + 1]);
                    if seen.insert((left.clone(), right.clone())) {
                        if let Some(join) = join_from_edge(edge_val, left, right) {
                            joins.push(join);
                        }
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
// Graph construction helpers
// ---------------------------------------------------------------------------

fn insert_edge(session: &Session, join: &Join) -> Result<(), String> {
    let left_on: Vec<Value> = join
        .left_on
        .iter()
        .map(|s| Value::String(s.as_str().into()))
        .collect();
    let right_on: Vec<Value> = join
        .right_on
        .iter()
        .map(|s| Value::String(s.as_str().into()))
        .collect();
    let how_str = match join.how {
        JoinHow::Left => "left",
        JoinHow::Inner => "inner",
    };
    let dir_str = match join.direction {
        JoinDirection::Both => "both",
        JoinDirection::RightOnLeft => "rightonleft",
    };

    let mut params: HashMap<String, Value> = HashMap::new();
    params.insert("left".to_string(), Value::String(join.left.as_str().into()));
    params.insert(
        "right".to_string(),
        Value::String(join.right.as_str().into()),
    );
    params.insert("left_on".to_string(), Value::List(left_on.into()));
    params.insert("right_on".to_string(), Value::List(right_on.into()));
    params.insert("how".to_string(), Value::String(how_str.into()));
    params.insert("direction".to_string(), Value::String(dir_str.into()));

    session
        .execute_with_params(
            "MATCH (l:Table {name: $left}), (r:Table {name: $right}) \
             INSERT (l)-[:JOIN {left_on: $left_on, right_on: $right_on, \
                               how: $how, direction: $direction}]->(r)",
            params,
        )
        .map_err(|e| e.to_string())?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

/// Detects any directed cycle of 3 or more hops. Two-node cycles from
/// bidirectional edges (A→B→A) are allowed and not flagged.
fn validate_no_long_cycles(session: &Session) -> Result<(), String> {
    let result = session
        .execute(
            "MATCH (a:Table)-[:JOIN*3..100]->(b:Table) \
             WHERE a.name = b.name \
             RETURN a.name LIMIT 1",
        )
        .map_err(|e| e.to_string())?;

    if let Some(row) = result.rows().first() {
        if let Value::String(name) = &row[0] {
            return Err(format!(
                "Cycle of length >= 3 detected involving table '{name}'"
            ));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Path pre-computation
// ---------------------------------------------------------------------------

/// For every start table, validates unique paths and pre-computes the lookup.
///
/// Uses `MATCH SIMPLE` (no repeated nodes) to count paths per end node —
/// this prevents false positives from bidirectional edges re-entering the
/// start node. Then uses `ANY SHORTEST` to retrieve the canonical path.
fn compute_lookup(
    session: &Session,
    tables: &HashSet<String>,
) -> Result<HashMap<String, HashMap<String, Vec<Join>>>, String> {
    let mut lookup = HashMap::new();

    for start in tables {
        let mut params = HashMap::new();
        params.insert("start".to_string(), Value::String(start.as_str().into()));

        // Validate: at most one simple path to each reachable table.
        let count_result = session
            .execute_with_params(
                "MATCH SIMPLE (a:Table {name: $start})-[:JOIN*1..100]->(b:Table) \
                 RETURN b.name AS target, COUNT(*) AS cnt",
                params.clone(),
            )
            .map_err(|e| e.to_string())?;

        for row in count_result.rows() {
            let target = match &row[0] {
                Value::String(s) => s.to_string(),
                _ => continue,
            };
            let cnt = match &row[1] {
                Value::Int64(n) => *n,
                _ => continue,
            };
            if cnt > 1 {
                return Err(format!(
                    "Multiple paths from '{start}' to '{target}' detected"
                ));
            }
        }

        // Retrieve the canonical (shortest) path to each reachable table.
        let path_result = session
            .execute_with_params(
                "MATCH p = ANY SHORTEST (a:Table {name: $start})-[:JOIN*1..100]->(b:Table) \
                 RETURN p",
                params,
            )
            .map_err(|e| e.to_string())?;

        let mut inner: HashMap<String, Vec<Join>> = HashMap::new();
        for row in path_result.rows() {
            if let Some(Value::Path { nodes, edges }) = row.first() {
                let end = node_name(&nodes[nodes.len() - 1]);
                inner.insert(end, extract_path_joins(nodes, edges));
            }
        }

        trace!(from = %start, reachable = inner.len(), "paths pre-computed");
        lookup.insert(start.clone(), inner);
    }

    Ok(lookup)
}

fn extract_path_joins(nodes: &[Value], edges: &[Value]) -> Vec<Join> {
    edges
        .iter()
        .enumerate()
        .filter_map(|(i, edge_val)| {
            join_from_edge(edge_val, node_name(&nodes[i]), node_name(&nodes[i + 1]))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Value extraction helpers
// ---------------------------------------------------------------------------

fn string_list(val: &Value) -> Vec<String> {
    match val {
        Value::List(items) => items
            .iter()
            .filter_map(|v| match v {
                Value::String(s) => Some(s.to_string()),
                _ => None,
            })
            .collect(),
        _ => vec![],
    }
}

/// Extracts the `name` property from a node value in a path.
fn node_name(node_val: &Value) -> String {
    match node_val {
        Value::Map(map) => match map.get("name") {
            Some(Value::String(s)) => s.to_string(),
            _ => String::new(),
        },
        _ => String::new(),
    }
}

/// Reconstructs a Join from an edge value in a path.
fn join_from_edge(edge_val: &Value, left: String, right: String) -> Option<Join> {
    let map = match edge_val {
        Value::Map(m) => m,
        _ => return None,
    };
    let left_on = string_list(map.get("left_on").unwrap_or(&Value::Null));
    let right_on = string_list(map.get("right_on").unwrap_or(&Value::Null));
    let how = match map.get("how") {
        Some(Value::String(s)) if s.as_str() == "inner" => JoinHow::Inner,
        _ => JoinHow::Left,
    };
    let direction = match map.get("direction") {
        Some(Value::String(s)) if s.as_str() == "both" => JoinDirection::Both,
        _ => JoinDirection::RightOnLeft,
    };
    Some(Join {
        left,
        right,
        left_on,
        right_on,
        how,
        direction,
    })
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
