use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::Result as DFResult;
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_optimizer::OptimizerRule;
use std::collections::HashSet;

use crate::model::joins::JoinGraph;

/// Optimizer rule that eliminates unnecessary Join nodes from the plan.
///
/// When `DataModel::table()` eagerly joins all reachable tables, some of those
/// joins may be unreferenced by the query. This rule removes Join nodes whose
/// right subtree contains only tables that are neither directly referenced by
/// column expressions nor needed as intermediaries for transitive joins.
#[derive(Debug)]
pub struct EliminateUnusedJoins {
    join_graph: JoinGraph,
}

impl EliminateUnusedJoins {
    pub fn new(join_graph: JoinGraph) -> Self {
        Self { join_graph }
    }

    /// Collect all table names referenced by Column expressions in the plan,
    /// excluding join ON conditions (which reference tables by definition and
    /// should not count as "user references").
    fn collect_referenced_tables(plan: &LogicalPlan) -> HashSet<String> {
        let mut tables = HashSet::new();
        let _ = plan.apply(|node| {
            // Skip Join nodes — their ON conditions reference tables structurally
            if matches!(node, LogicalPlan::Join(_)) {
                return Ok(TreeNodeRecursion::Continue);
            }
            for expr in node.expressions() {
                let _ = expr.apply(|e| {
                    if let Expr::Column(col) = e {
                        if let Some(relation) = &col.relation {
                            tables.insert(relation.table().to_string());
                        }
                    }
                    Ok(TreeNodeRecursion::Continue)
                });
            }
            Ok(TreeNodeRecursion::Continue)
        });
        tables
    }

    /// Collect all table names from TableScan nodes in the plan.
    fn collect_scanned_tables(plan: &LogicalPlan) -> HashSet<String> {
        let mut tables = HashSet::new();
        let _ = plan.apply(|node| {
            if let LogicalPlan::TableScan(scan) = node {
                tables.insert(scan.table_name.table().to_string());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        tables
    }

    /// Find the base table — the leftmost/deepest TableScan in the join chain.
    fn find_base_table(plan: &LogicalPlan) -> Option<String> {
        match plan {
            LogicalPlan::TableScan(scan) => Some(scan.table_name.table().to_string()),
            LogicalPlan::Join(join) => {
                // Recurse into the left side to find the deepest table
                Self::find_base_table(&join.left)
            }
            _ => {
                // Recurse into the first input
                plan.inputs()
                    .into_iter()
                    .next()
                    .and_then(Self::find_base_table)
            }
        }
    }

    /// Compute the set of tables that must be kept in the plan.
    ///
    /// For each referenced table, find the join path from the base table to it.
    /// All tables along those paths (including intermediaries) must be kept.
    fn compute_keep_set(
        &self,
        base_table: &str,
        referenced_tables: &HashSet<String>,
    ) -> HashSet<String> {
        let mut keep = HashSet::new();
        keep.insert(base_table.to_string());

        for table in referenced_tables {
            if let Some(path) = self.join_graph.find_path(base_table, table) {
                // Add every table along the path
                keep.insert(base_table.to_string());
                for step in &path {
                    keep.insert(step.left.clone());
                    keep.insert(step.right.clone());
                }
            }
        }

        keep
    }

    /// Collect all tables scanned in a subtree.
    fn subtree_tables(plan: &LogicalPlan) -> HashSet<String> {
        Self::collect_scanned_tables(plan)
    }

    /// Recursively remove unnecessary joins from the plan.
    fn remove_unused_joins(
        &self,
        plan: LogicalPlan,
        keep_set: &HashSet<String>,
    ) -> DFResult<LogicalPlan> {
        if let LogicalPlan::Join(ref join) = plan {
            // First, recursively process both sides
            let left = self.remove_unused_joins((*join.left).clone(), keep_set)?;
            let right = self.remove_unused_joins((*join.right).clone(), keep_set)?;

            // Check if the right subtree contains ONLY tables NOT in the keep set
            let right_tables = Self::subtree_tables(&right);
            let right_needed = right_tables.iter().any(|t| keep_set.contains(t));

            if !right_needed {
                // Drop this join entirely — just return the left side
                return Ok(left);
            }

            // Rebuild the join with potentially pruned children
            let new_join = LogicalPlan::Join(datafusion_expr::logical_plan::Join {
                left: std::sync::Arc::new(left),
                right: std::sync::Arc::new(right),
                on: join.on.clone(),
                filter: join.filter.clone(),
                join_type: join.join_type,
                join_constraint: join.join_constraint,
                schema: join.schema.clone(),
                null_equality: join.null_equality,
            });
            return Ok(new_join);
        }

        // For non-Join nodes, recurse into children
        let inputs: Vec<LogicalPlan> = plan
            .inputs()
            .into_iter()
            .map(|input| self.remove_unused_joins(input.clone(), keep_set))
            .collect::<DFResult<Vec<_>>>()?;

        if inputs.is_empty() {
            Ok(plan)
        } else {
            plan.with_new_exprs(plan.expressions(), inputs)
        }
    }
}

impl OptimizerRule for EliminateUnusedJoins {
    fn name(&self) -> &str {
        "eliminate_unused_joins"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn datafusion_optimizer::OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        // Step 1: Collect referenced tables from Column expressions
        let referenced_tables = Self::collect_referenced_tables(&plan);

        // Step 2: Check if there are any joins to potentially eliminate
        let scanned_tables = Self::collect_scanned_tables(&plan);
        if scanned_tables.len() <= 1 {
            return Ok(Transformed::no(plan));
        }

        // Step 3: Find the base table
        let base_table = match Self::find_base_table(&plan) {
            Some(t) => t,
            None => return Ok(Transformed::no(plan)),
        };

        // Step 4: Compute the keep set
        let keep_set = self.compute_keep_set(&base_table, &referenced_tables);

        // Step 5: Check if any tables can be eliminated
        let can_eliminate = scanned_tables.iter().any(|t| !keep_set.contains(t));
        if !can_eliminate {
            return Ok(Transformed::no(plan));
        }

        // Step 6: Remove unnecessary joins
        let rewritten = self.remove_unused_joins(plan, &keep_set)?;
        Ok(Transformed::yes(rewritten))
    }
}
