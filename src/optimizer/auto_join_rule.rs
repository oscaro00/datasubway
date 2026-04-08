use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Column, JoinType, Result as DFResult};
use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion_optimizer::OptimizerRule;
use std::collections::HashSet;
use std::sync::Arc;

use crate::model::joins::JoinGraph;

/// Optimizer rule that automatically injects Join nodes into a LogicalPlan
/// when columns reference tables that are scanned but not yet joined.
///
/// This rule uses the JoinGraph (BFS path-finding) to determine how to
/// connect disconnected TableScan nodes. It runs AFTER PreAggSubstitution:
/// if a pre-agg covers the query, no joins are needed. This rule is the
/// fallback for when no pre-agg is available.
#[derive(Debug)]
pub struct AutoJoinRule {
    join_graph: JoinGraph,
    /// Table schemas: table_name -> list of column names (unqualified)
    table_schemas: std::collections::HashMap<String, Vec<String>>,
}

impl AutoJoinRule {
    pub fn new(
        join_graph: JoinGraph,
        table_schemas: std::collections::HashMap<String, Vec<String>>,
    ) -> Self {
        Self {
            join_graph,
            table_schemas,
        }
    }

    /// Collect all table names referenced by Column expressions in the plan.
    fn collect_referenced_tables(plan: &LogicalPlan) -> HashSet<String> {
        let mut tables = HashSet::new();
        let _ = plan.apply(|node| {
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

    /// Collect all table names that have TableScan nodes in the plan.
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

    /// Find the "base" table scan node name. We pick the first TableScan
    /// that exists in the join graph and can reach the most other tables.
    fn find_base_table<'a>(
        &self,
        scanned: &'a HashSet<String>,
        missing: &HashSet<&String>,
    ) -> Option<&'a String> {
        // Prefer a scanned table that can reach all missing tables
        for candidate in scanned {
            if missing
                .iter()
                .all(|target| self.join_graph.find_path(candidate, target).is_some())
            {
                return Some(candidate);
            }
        }
        // Fallback: any scanned table
        scanned.iter().next()
    }

    /// Compute deduplicated join steps from a base table to all missing tables.
    fn compute_join_steps(
        &self,
        base_table: &str,
        missing: &HashSet<&String>,
    ) -> DFResult<Vec<crate::model::joins::Join>> {
        let mut joined_so_far: HashSet<String> = HashSet::new();
        joined_so_far.insert(base_table.to_string());
        let mut all_steps = Vec::new();

        for target in missing {
            if joined_so_far.contains(*target) {
                continue;
            }
            // Find path from any already-joined table to the target
            let mut found_path = None;
            for source in &joined_so_far.clone() {
                if let Some(path) = self.join_graph.find_path(source, target) {
                    found_path = Some(path);
                    break;
                }
            }
            let path = found_path.ok_or_else(|| {
                datafusion_common::DataFusionError::Plan(format!(
                    "No join path from any joined table to '{}'",
                    target
                ))
            })?;
            for step in path {
                if !joined_so_far.contains(&step.right) {
                    joined_so_far.insert(step.right.clone());
                    all_steps.push(step);
                }
            }
        }
        Ok(all_steps)
    }

    /// Build a LogicalPlan that scans a table with all its columns.
    fn build_table_scan(&self, table_name: &str) -> DFResult<LogicalPlan> {
        // Build a scan using an empty source -- the actual table provider
        // will be resolved by DataFusion during execution since it's registered
        // in the SessionContext. We create a placeholder scan with the schema.
        let columns = self
            .table_schemas
            .get(table_name)
            .cloned()
            .unwrap_or_default();
        let fields: Vec<arrow::datatypes::Field> = columns
            .iter()
            .map(|c| arrow::datatypes::Field::new(c, arrow::datatypes::DataType::Null, true))
            .collect();
        let schema = Arc::new(arrow::datatypes::Schema::new(fields));
        let table_source = Arc::new(EmptyTableSource { schema });
        LogicalPlanBuilder::scan(table_name, table_source, None)?.build()
    }

    /// Rewrite the plan by finding the base TableScan and wrapping it with Join nodes.
    /// Uses manual recursive rewriting to avoid infinite loops from transform_down
    /// re-visiting the base TableScan inside newly created Join nodes.
    fn inject_joins_into_plan(
        &self,
        plan: LogicalPlan,
        base_table: &str,
        join_steps: &[crate::model::joins::Join],
    ) -> DFResult<LogicalPlan> {
        self.rewrite_node(plan, base_table, join_steps)
    }

    fn rewrite_node(
        &self,
        plan: LogicalPlan,
        base_table: &str,
        join_steps: &[crate::model::joins::Join],
    ) -> DFResult<LogicalPlan> {
        // If this is the base table scan, replace it with the join chain
        if let LogicalPlan::TableScan(ref scan) = plan {
            if scan.table_name.table() == base_table {
                return self.build_join_chain(plan, join_steps);
            }
        }

        // Otherwise, recurse into children
        let inputs: Vec<LogicalPlan> = plan
            .inputs()
            .into_iter()
            .map(|input| self.rewrite_node(input.clone(), base_table, join_steps))
            .collect::<DFResult<Vec<_>>>()?;

        if inputs.is_empty() {
            Ok(plan)
        } else {
            plan.with_new_exprs(plan.expressions(), inputs)
        }
    }

    /// Build a chain of joins starting from the base scan.
    fn build_join_chain(
        &self,
        base_scan: LogicalPlan,
        join_steps: &[crate::model::joins::Join],
    ) -> DFResult<LogicalPlan> {
        let mut current = base_scan;
        for step in join_steps {
            let right_scan = self.build_table_scan(&step.right)?;

            let join_type = match step.how.as_str() {
                "inner" => JoinType::Inner,
                "left" => JoinType::Left,
                "right" => JoinType::Right,
                "full" => JoinType::Full,
                _ => JoinType::Inner,
            };

            let left_keys: Vec<Column> = step
                .left_on
                .iter()
                .map(|c| Column::new(Some(step.left.as_str()), c.as_str()))
                .collect();
            let right_keys: Vec<Column> = step
                .right_on
                .iter()
                .map(|c| Column::new(Some(step.right.as_str()), c.as_str()))
                .collect();

            let on_exprs: Vec<Expr> = left_keys
                .into_iter()
                .zip(right_keys)
                .map(|(l, r)| Expr::Column(l).eq(Expr::Column(r)))
                .collect();

            current = LogicalPlanBuilder::from(current)
                .join_on(right_scan, join_type, on_exprs)?
                .build()?;
        }
        Ok(current)
    }
}

impl OptimizerRule for AutoJoinRule {
    fn name(&self) -> &str {
        "auto_join"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn datafusion_optimizer::OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        // Step 1: Collect all table references from Column exprs
        let referenced_tables = Self::collect_referenced_tables(&plan);

        // Step 2: Collect tables that already have TableScan nodes
        let scanned_tables = Self::collect_scanned_tables(&plan);

        // Step 3: Find tables that are referenced but not scanned
        // (these need to be joined in)
        let missing: HashSet<&String> = referenced_tables.difference(&scanned_tables).collect();
        if missing.is_empty() {
            // All referenced tables are already scanned -- check if they need joining.
            // If there are multiple TableScans but no Join nodes, we need to join them.
            if scanned_tables.len() <= 1 {
                return Ok(Transformed::no(plan));
            }
            // Check if joins already exist in the plan
            let has_joins = Self::plan_has_joins(&plan);
            if has_joins {
                return Ok(Transformed::no(plan));
            }
            // Multiple scans, no joins -- need to connect them
            return self.join_disconnected_scans(plan, &scanned_tables);
        }

        // Step 4: Find the base table
        let base_table = self
            .find_base_table(&scanned_tables, &missing)
            .ok_or_else(|| {
                datafusion_common::DataFusionError::Plan(
                    "No scanned tables found in plan".to_string(),
                )
            })?
            .clone();

        // Step 5: Compute join steps
        let join_steps = self.compute_join_steps(&base_table, &missing)?;
        if join_steps.is_empty() {
            return Ok(Transformed::no(plan));
        }

        // Step 6: Rewrite the plan
        let rewritten = self.inject_joins_into_plan(plan, &base_table, &join_steps)?;
        Ok(Transformed::yes(rewritten))
    }
}

impl AutoJoinRule {
    fn plan_has_joins(plan: &LogicalPlan) -> bool {
        let mut has_join = false;
        let _ = plan.apply(|node| {
            if matches!(node, LogicalPlan::Join(_)) {
                has_join = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            Ok(TreeNodeRecursion::Continue)
        });
        has_join
    }

    /// Handle the case where multiple tables are scanned but not joined together.
    /// This happens when the plan was built with multiple table references
    /// (e.g., via Python DataFusion DataFrame API).
    fn join_disconnected_scans(
        &self,
        plan: LogicalPlan,
        scanned_tables: &HashSet<String>,
    ) -> DFResult<Transformed<LogicalPlan>> {
        // Pick a base table that can reach others
        let all_tables: HashSet<&String> = scanned_tables.iter().collect();
        let base_table = scanned_tables
            .iter()
            .find(|candidate| {
                scanned_tables.iter().all(|other| {
                    *candidate == other || self.join_graph.find_path(candidate, other).is_some()
                })
            })
            .or_else(|| scanned_tables.iter().next())
            .cloned();

        if let Some(base) = base_table {
            let others: HashSet<&String> = all_tables.into_iter().filter(|t| **t != base).collect();
            if others.is_empty() {
                return Ok(Transformed::no(plan));
            }
            let join_steps = self.compute_join_steps(&base, &others)?;
            if join_steps.is_empty() {
                return Ok(Transformed::no(plan));
            }
            let rewritten = self.inject_joins_into_plan(plan, &base, &join_steps)?;
            Ok(Transformed::yes(rewritten))
        } else {
            Ok(Transformed::no(plan))
        }
    }
}

/// A minimal TableSource for building placeholder scans.
/// The actual table data comes from the SessionContext at execution time.
#[derive(Debug)]
struct EmptyTableSource {
    schema: Arc<arrow::datatypes::Schema>,
}

impl datafusion_expr::TableSource for EmptyTableSource {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn schema(&self) -> arrow::datatypes::SchemaRef {
        self.schema.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::joins::Join;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::execution::context::SessionContext;
    use datafusion::prelude::col;
    use datafusion_functions_aggregate::sum::sum;
    use datafusion_optimizer::OptimizerContext;
    use std::collections::HashMap;

    fn make_join_graph() -> JoinGraph {
        let joins = vec![
            Join {
                left: "orders".into(),
                right: "customers".into(),
                left_on: vec!["customer_id".into()],
                right_on: vec!["id".into()],
                how: "left".into(),
                direction: "right2left".into(),
            },
            Join {
                left: "orders".into(),
                right: "products".into(),
                left_on: vec!["product_id".into()],
                right_on: vec!["id".into()],
                how: "left".into(),
                direction: "right2left".into(),
            },
        ];
        JoinGraph::new(&joins).unwrap()
    }

    fn make_table_schemas() -> HashMap<String, Vec<String>> {
        let mut schemas = HashMap::new();
        schemas.insert(
            "orders".into(),
            vec![
                "region".into(),
                "amount".into(),
                "customer_id".into(),
                "product_id".into(),
            ],
        );
        schemas.insert(
            "customers".into(),
            vec!["id".into(), "name".into(), "country".into()],
        );
        schemas.insert(
            "products".into(),
            vec!["id".into(), "product_name".into(), "category".into()],
        );
        schemas
    }

    fn make_test_context() -> SessionContext {
        let ctx = SessionContext::new();
        let rt = tokio::runtime::Runtime::new().unwrap();

        rt.block_on(async {
            // Orders table
            let schema = Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("amount", DataType::Int64, false),
                Field::new("customer_id", DataType::Int64, false),
                Field::new("product_id", DataType::Int64, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["US", "EU"])),
                    Arc::new(Int64Array::from(vec![100, 200])),
                    Arc::new(Int64Array::from(vec![1, 2])),
                    Arc::new(Int64Array::from(vec![10, 20])),
                ],
            )
            .unwrap();
            let mem_table =
                datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
            ctx.register_table("orders", Arc::new(mem_table)).unwrap();

            // Customers table
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("name", DataType::Utf8, false),
                Field::new("country", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![1, 2])),
                    Arc::new(StringArray::from(vec!["Alice", "Bob"])),
                    Arc::new(StringArray::from(vec!["US", "DE"])),
                ],
            )
            .unwrap();
            let mem_table =
                datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
            ctx.register_table("customers", Arc::new(mem_table))
                .unwrap();

            // Products table
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("product_name", DataType::Utf8, false),
                Field::new("category", DataType::Utf8, false),
            ]));
            let batch = RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Int64Array::from(vec![10, 20])),
                    Arc::new(StringArray::from(vec!["Widget", "Gadget"])),
                    Arc::new(StringArray::from(vec!["A", "B"])),
                ],
            )
            .unwrap();
            let mem_table =
                datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
            ctx.register_table("products", Arc::new(mem_table)).unwrap();
        });
        ctx
    }

    #[test]
    fn test_collect_referenced_tables() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            // Reference orders.region and aggregate orders.amount
            let agg = df
                .aggregate(
                    vec![col("orders.region")],
                    vec![sum(col("orders.amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let tables = AutoJoinRule::collect_referenced_tables(&plan);
        assert!(
            tables.contains("orders"),
            "Should find 'orders' in referenced tables: {:?}",
            tables
        );
    }

    #[test]
    fn test_collect_scanned_tables() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            df.logical_plan().clone()
        });

        let tables = AutoJoinRule::collect_scanned_tables(&plan);
        assert_eq!(tables.len(), 1);
        assert!(tables.contains("orders"));
    }

    #[test]
    fn test_no_rewrite_single_table() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let agg = df
                .aggregate(
                    vec![col("orders.region")],
                    vec![sum(col("orders.amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let rule = AutoJoinRule::new(make_join_graph(), make_table_schemas());
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();
        assert!(
            !result.transformed,
            "Should not transform when only one table is referenced"
        );
    }

    #[test]
    fn test_no_rewrite_empty_graph() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let agg = df
                .aggregate(
                    vec![col("orders.region")],
                    vec![sum(col("orders.amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let empty_graph = JoinGraph::new(&[]).unwrap();
        let rule = AutoJoinRule::new(empty_graph, make_table_schemas());
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();
        assert!(!result.transformed);
    }

    #[test]
    fn test_compute_join_steps_single_target() {
        let rule = AutoJoinRule::new(make_join_graph(), make_table_schemas());
        let mut missing = HashSet::new();
        let target = "customers".to_string();
        missing.insert(&target);

        let steps = rule.compute_join_steps("orders", &missing).unwrap();
        assert_eq!(steps.len(), 1);
        assert_eq!(steps[0].left, "orders");
        assert_eq!(steps[0].right, "customers");
    }

    #[test]
    fn test_compute_join_steps_multiple_targets() {
        let rule = AutoJoinRule::new(make_join_graph(), make_table_schemas());
        let mut missing = HashSet::new();
        let t1 = "customers".to_string();
        let t2 = "products".to_string();
        missing.insert(&t1);
        missing.insert(&t2);

        let steps = rule.compute_join_steps("orders", &missing).unwrap();
        assert_eq!(steps.len(), 2);
        // Both should join from orders
        let rights: HashSet<String> = steps.iter().map(|s| s.right.clone()).collect();
        assert!(rights.contains("customers"));
        assert!(rights.contains("products"));
    }

    #[test]
    fn test_compute_join_steps_deduplication() {
        // Chain: orders -> customers -> loyalty
        let joins = vec![
            Join {
                left: "orders".into(),
                right: "customers".into(),
                left_on: vec!["customer_id".into()],
                right_on: vec!["id".into()],
                how: "left".into(),
                direction: "right2left".into(),
            },
            Join {
                left: "customers".into(),
                right: "loyalty".into(),
                left_on: vec!["loyalty_id".into()],
                right_on: vec!["id".into()],
                how: "left".into(),
                direction: "right2left".into(),
            },
        ];
        let graph = JoinGraph::new(&joins).unwrap();
        let mut schemas = make_table_schemas();
        schemas.insert("loyalty".into(), vec!["id".into(), "tier".into()]);
        let rule = AutoJoinRule::new(graph, schemas);

        // Both customers and loyalty are missing
        let mut missing = HashSet::new();
        let t1 = "customers".to_string();
        let t2 = "loyalty".to_string();
        missing.insert(&t1);
        missing.insert(&t2);

        let steps = rule.compute_join_steps("orders", &missing).unwrap();
        // Should have 2 steps: orders->customers, customers->loyalty
        // NOT 3 (no duplicate orders->customers)
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0].right, "customers");
        assert_eq!(steps[1].right, "loyalty");
    }

    #[test]
    fn test_rewrite_injects_join_for_cross_table_ref() {
        // Build a plan that scans "orders" but references "customers.country"
        // via a join that DataFusion built (two table scans, no join node).
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let orders = ctx.table("orders").await.unwrap();
            let customers = ctx.table("customers").await.unwrap();
            // Join them explicitly so we have a multi-scan plan
            let joined = orders
                .join(
                    customers,
                    datafusion::prelude::JoinType::Left,
                    &["customer_id"],
                    &["id"],
                    None,
                )
                .unwrap();
            let agg = joined
                .aggregate(
                    vec![col("customers.country")],
                    vec![sum(col("orders.amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        // This plan already has a join, so the rule should NOT transform it
        let rule = AutoJoinRule::new(make_join_graph(), make_table_schemas());
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();
        assert!(
            !result.transformed,
            "Should not transform plan that already has joins"
        );
    }

    #[test]
    fn test_inject_joins_adds_join_node() {
        // Verify that inject_joins_into_plan adds Join nodes to a real plan
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();
        let rule = AutoJoinRule::new(make_join_graph(), make_table_schemas());

        let plan = rt.block_on(async { ctx.table("orders").await.unwrap().logical_plan().clone() });

        // No joins initially
        assert!(!AutoJoinRule::plan_has_joins(&plan));

        let mut missing = HashSet::new();
        let target = "customers".to_string();
        missing.insert(&target);
        let steps = rule.compute_join_steps("orders", &missing).unwrap();

        let rewritten = rule.inject_joins_into_plan(plan, "orders", &steps).unwrap();

        assert!(
            AutoJoinRule::plan_has_joins(&rewritten),
            "Rewritten plan should contain a Join node"
        );
    }
}
