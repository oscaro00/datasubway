use datafusion_common::tree_node::Transformed;
use datafusion_common::{Result as DFResult, TableReference};
use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder};
use datafusion_optimizer::OptimizerRule;
use std::collections::{HashMap, HashSet};

use crate::model::pre_agg::{agg_needed_components, find_best_pre_agg, PreAggregation};
use crate::optimizer::agg_rewrite::rewrite_agg_expr;

/// OptimizerRule that substitutes raw table scans with pre-aggregated tables
/// when a suitable pre-aggregation exists.
#[derive(Debug)]
pub struct PreAggSubstitution {
    pre_aggs: Vec<PreAggregation>,
}

impl PreAggSubstitution {
    pub fn new(pre_aggs: Vec<PreAggregation>) -> Self {
        Self { pre_aggs }
    }

    /// Collect all column references from the plan to determine what the pre-agg must cover.
    fn collect_plan_requirements(
        plan: &LogicalPlan,
    ) -> (Vec<String>, HashMap<String, HashSet<String>>, Vec<String>) {
        let mut group_by_cols = Vec::new();
        let mut agg_components: HashMap<String, HashSet<String>> = HashMap::new();
        let mut filter_cols = Vec::new();

        Self::walk_plan(
            plan,
            &mut group_by_cols,
            &mut agg_components,
            &mut filter_cols,
        );

        (group_by_cols, agg_components, filter_cols)
    }

    fn walk_plan(
        plan: &LogicalPlan,
        group_by_cols: &mut Vec<String>,
        agg_components: &mut HashMap<String, HashSet<String>>,
        filter_cols: &mut Vec<String>,
    ) {
        match plan {
            LogicalPlan::Aggregate(agg) => {
                for expr in &agg.group_expr {
                    if let Some(name) = Self::extract_col_name(expr) {
                        group_by_cols.push(name);
                    }
                }
                for expr in &agg.aggr_expr {
                    Self::extract_agg_requirements(expr, agg_components);
                }
            }
            LogicalPlan::Filter(filter) => {
                Self::extract_filter_columns(&filter.predicate, filter_cols);
            }
            _ => {}
        }

        for child in plan.inputs() {
            Self::walk_plan(child, group_by_cols, agg_components, filter_cols);
        }
    }

    fn extract_col_name(expr: &Expr) -> Option<String> {
        match expr {
            Expr::Column(col) => Some(col.name.clone()),
            Expr::Alias(alias) => Self::extract_col_name(&alias.expr),
            _ => None,
        }
    }

    fn extract_agg_requirements(
        expr: &Expr,
        agg_components: &mut HashMap<String, HashSet<String>>,
    ) {
        match expr {
            Expr::AggregateFunction(agg) => {
                if let Some(col_name) = Self::extract_col_name(&agg.params.args[0]) {
                    let func_name = agg.func.name();
                    if let Some(components) = agg_needed_components(func_name) {
                        let entry = agg_components.entry(col_name).or_default();
                        for comp in components {
                            entry.insert(comp.to_string());
                        }
                    }
                }
            }
            Expr::Alias(alias) => {
                Self::extract_agg_requirements(&alias.expr, agg_components);
            }
            _ => {}
        }
    }

    fn extract_filter_columns(expr: &Expr, filter_cols: &mut Vec<String>) {
        match expr {
            Expr::Column(col) => {
                filter_cols.push(col.name.clone());
            }
            Expr::BinaryExpr(binary) => {
                Self::extract_filter_columns(&binary.left, filter_cols);
                Self::extract_filter_columns(&binary.right, filter_cols);
            }
            Expr::Not(inner) => {
                Self::extract_filter_columns(inner, filter_cols);
            }
            Expr::IsNull(inner) | Expr::IsNotNull(inner) => {
                Self::extract_filter_columns(inner, filter_cols);
            }
            Expr::InList(in_list) => {
                Self::extract_filter_columns(&in_list.expr, filter_cols);
            }
            _ => {}
        }
    }

    /// Rewrite the plan to use a pre-aggregated table.
    /// This replaces the Aggregate node's inputs and rewrites its expressions.
    fn rewrite_plan_with_pre_agg(
        &self,
        plan: LogicalPlan,
        pre_agg: &PreAggregation,
    ) -> DFResult<LogicalPlan> {
        match plan {
            LogicalPlan::Aggregate(agg) => {
                // Rewrite aggregate expressions to use pre-agg component columns
                let rewritten_aggs: Vec<Expr> = agg
                    .aggr_expr
                    .iter()
                    .map(|expr| rewrite_agg_expr(expr).unwrap_or_else(|| expr.clone()))
                    .collect();

                // Group-by expressions stay the same (same column names in pre-agg)
                let group_exprs = agg.group_expr.clone();

                // Build a new plan that scans the pre-agg table instead of raw tables
                // For now, we recursively rewrite children to point at the pre-agg table
                let new_input = self.replace_table_scans(&agg.input, pre_agg)?;

                LogicalPlanBuilder::from(new_input)
                    .aggregate(group_exprs, rewritten_aggs)?
                    .build()
            }
            LogicalPlan::Filter(filter) => {
                // Keep filters but rewrite the input
                let new_input = self.rewrite_plan_with_pre_agg((*filter.input).clone(), pre_agg)?;
                LogicalPlanBuilder::from(new_input)
                    .filter(filter.predicate.clone())?
                    .build()
            }
            LogicalPlan::Projection(proj) => {
                let new_input = self.rewrite_plan_with_pre_agg((*proj.input).clone(), pre_agg)?;
                LogicalPlanBuilder::from(new_input)
                    .project(proj.expr.clone())?
                    .build()
            }
            // For other nodes (TableScan, Join, etc.), replace table scans
            other => self.replace_table_scans(&other, pre_agg),
        }
    }

    /// Replace all TableScan and Join nodes with a single scan of the pre-agg table.
    fn replace_table_scans(
        &self,
        plan: &LogicalPlan,
        pre_agg: &PreAggregation,
    ) -> DFResult<LogicalPlan> {
        // The pre-agg table contains all the data we need.
        // Replace any TableScan/Join tree with a scan of the pre-agg parquet.
        let table_ref = TableReference::bare(pre_agg.name.clone());
        // Build column list: group-by cols + all component cols
        let mut columns: Vec<String> = pre_agg.group_by.clone();
        for (col, components) in &pre_agg.aggregations {
            for comp in components {
                columns.push(PreAggregation::component_column(col, comp));
            }
        }

        // Create a scan of the pre-agg table
        // Note: This assumes the pre-agg is registered as a table in the SessionContext
        // The engine must register pre-agg parquet files before optimization
        match plan {
            LogicalPlan::TableScan(scan) => {
                // Replace with pre-agg table scan
                let mut new_scan = scan.clone();
                new_scan.table_name = table_ref;
                Ok(LogicalPlan::TableScan(new_scan))
            }
            LogicalPlan::Join(_) => {
                // For joins, we need to find any table scan in the subtree
                // and replace the whole join with a single pre-agg scan
                // For now, find the first table scan and replace it
                self.find_and_replace_first_scan(plan, pre_agg)
            }
            other => {
                // Recurse into children
                let inputs: Vec<LogicalPlan> = other
                    .inputs()
                    .into_iter()
                    .map(|input| self.replace_table_scans(input, pre_agg))
                    .collect::<DFResult<Vec<_>>>()?;

                if inputs.is_empty() {
                    Ok(other.clone())
                } else {
                    other.with_new_exprs(other.expressions(), inputs)
                }
            }
        }
    }

    fn find_and_replace_first_scan(
        &self,
        plan: &LogicalPlan,
        pre_agg: &PreAggregation,
    ) -> DFResult<LogicalPlan> {
        match plan {
            LogicalPlan::TableScan(scan) => {
                let table_ref = TableReference::bare(pre_agg.name.clone());
                let mut new_scan = scan.clone();
                new_scan.table_name = table_ref;
                Ok(LogicalPlan::TableScan(new_scan))
            }
            other => {
                // Return first table scan found in depth-first order
                for input in other.inputs() {
                    if let Ok(result) = self.find_and_replace_first_scan(input, pre_agg) {
                        return Ok(result);
                    }
                }
                Ok(other.clone())
            }
        }
    }
}

/// Helper to check if a plan contains a TableScan for a given table name.
/// Used for testing plan rewriting.
pub fn plan_contains_table_scan(plan: &LogicalPlan, table_name: &str) -> bool {
    match plan {
        LogicalPlan::TableScan(scan) => scan.table_name.table() == table_name,
        _ => plan
            .inputs()
            .iter()
            .any(|child| plan_contains_table_scan(child, table_name)),
    }
}

impl OptimizerRule for PreAggSubstitution {
    fn name(&self) -> &str {
        "pre_agg_substitution"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn datafusion_optimizer::OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        if self.pre_aggs.is_empty() {
            return Ok(Transformed::no(plan));
        }

        let (group_by, agg_components, filter_cols) = Self::collect_plan_requirements(&plan);

        let best = find_best_pre_agg(&self.pre_aggs, &group_by, &agg_components, &filter_cols);

        match best {
            Some(pa) => {
                let rewritten = self.rewrite_plan_with_pre_agg(plan, pa)?;
                Ok(Transformed::yes(rewritten))
            }
            None => Ok(Transformed::no(plan)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::execution::context::SessionContext;
    use datafusion::prelude::{col, lit};
    use datafusion_functions_aggregate::sum::sum;
    use datafusion_optimizer::OptimizerContext;
    use std::sync::Arc;

    fn make_test_context() -> SessionContext {
        let ctx = SessionContext::new();
        let schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, false),
            Field::new("amount", DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["US", "EU"])),
                Arc::new(Int64Array::from(vec![100, 200])),
            ],
        )
        .unwrap();

        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async {
            let mem_table =
                datafusion::datasource::MemTable::try_new(schema.clone(), vec![vec![batch.clone()]])
                    .unwrap();
            ctx.register_table("orders", Arc::new(mem_table)).unwrap();

            // Register a pre-agg table with component columns
            let preagg_schema = Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("amount-sum", DataType::Int64, false),
            ]));
            let preagg_batch = RecordBatch::try_new(
                preagg_schema.clone(),
                vec![
                    Arc::new(StringArray::from(vec!["US", "EU"])),
                    Arc::new(Int64Array::from(vec![100, 200])),
                ],
            )
            .unwrap();
            let preagg_table = datafusion::datasource::MemTable::try_new(
                preagg_schema,
                vec![vec![preagg_batch]],
            )
            .unwrap();
            ctx.register_table("regional_revenue", Arc::new(preagg_table))
                .unwrap();
        });
        ctx
    }

    fn make_pre_agg() -> PreAggregation {
        let mut pa = PreAggregation::new(
            "regional_revenue".into(),
            vec!["region".into()],
            HashMap::from([("amount".into(), vec!["sum".into()])]),
            "_preagg/regional_revenue.parquet".into(),
        )
        .unwrap();
        pa.row_count = 2;
        pa
    }

    #[test]
    fn test_collect_plan_requirements() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let agg = df
                .aggregate(
                    vec![col("region")],
                    vec![sum(col("amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let (group_by, agg_components, filter_cols) =
            PreAggSubstitution::collect_plan_requirements(&plan);

        assert_eq!(group_by, vec!["region"]);
        assert!(
            agg_components.contains_key("amount"),
            "Should extract 'amount' as agg column, got: {:?}",
            agg_components
        );
        assert!(
            agg_components["amount"].contains("sum"),
            "Should extract 'sum' component"
        );
        assert!(filter_cols.is_empty());
    }

    #[test]
    fn test_rule_matches_covering_pre_agg() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let agg = df
                .aggregate(
                    vec![col("region")],
                    vec![sum(col("amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        assert!(plan_contains_table_scan(&plan, "orders"));

        // Verify the rule's matching logic finds the covering pre-agg
        let pre_agg = make_pre_agg();
        let (group_by, agg_components, filter_cols) =
            PreAggSubstitution::collect_plan_requirements(&plan);

        assert!(
            pre_agg.covers(&group_by, &agg_components, &filter_cols),
            "Pre-agg should cover the plan requirements: group_by={:?}, agg={:?}",
            group_by,
            agg_components
        );

        let pre_aggs = [pre_agg];
        let best = find_best_pre_agg(
            &pre_aggs,
            &group_by,
            &agg_components,
            &filter_cols,
        );
        assert!(best.is_some(), "Should find a matching pre-agg");
        assert_eq!(best.unwrap().name, "regional_revenue");
    }

    #[test]
    fn test_rule_matches_with_filter_columns() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let filtered = df.filter(col("region").eq(lit("US"))).unwrap();
            let agg = filtered
                .aggregate(
                    vec![col("region")],
                    vec![sum(col("amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let pre_agg = make_pre_agg();
        let (group_by, agg_components, filter_cols) =
            PreAggSubstitution::collect_plan_requirements(&plan);

        assert_eq!(group_by, vec!["region"]);
        assert!(filter_cols.contains(&"region".to_string()));
        assert!(
            pre_agg.covers(&group_by, &agg_components, &filter_cols),
            "Pre-agg should cover filtered plan: filter_cols={:?}",
            filter_cols
        );
    }

    #[test]
    fn test_no_rewrite_when_no_covering_pre_agg() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let agg = df
                .aggregate(
                    vec![col("region")],
                    vec![sum(col("amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        // Pre-agg that doesn't cover the request (wrong column)
        let pa = PreAggregation::new(
            "wrong_preagg".into(),
            vec!["date".into()],
            HashMap::from([("quantity".into(), vec!["sum".into()])]),
            "_preagg/wrong.parquet".into(),
        )
        .unwrap();

        let rule = PreAggSubstitution::new(vec![pa]);
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            !result.transformed,
            "Plan should NOT have been transformed when no pre-agg covers the request"
        );
    }

    #[test]
    fn test_no_rewrite_when_empty_pre_aggs() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            let agg = df
                .aggregate(
                    vec![col("region")],
                    vec![sum(col("amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let rule = PreAggSubstitution::new(vec![]);
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(!result.transformed, "Empty pre-aggs should not transform");
    }

    #[test]
    fn test_rule_rejects_uncovered_filter_column() {
        let ctx = make_test_context();
        let rt = tokio::runtime::Runtime::new().unwrap();

        let plan = rt.block_on(async {
            let df = ctx.table("orders").await.unwrap();
            // Filter on "amount" which is NOT in the pre-agg's group_by
            let filtered = df.filter(col("amount").gt(lit(100))).unwrap();
            let agg = filtered
                .aggregate(
                    vec![col("region")],
                    vec![sum(col("amount")).alias("total")],
                )
                .unwrap();
            agg.logical_plan().clone()
        });

        let pre_agg = make_pre_agg(); // group_by = ["region"], no "amount" in group_by
        let (group_by, agg_components, filter_cols) =
            PreAggSubstitution::collect_plan_requirements(&plan);

        assert!(
            filter_cols.contains(&"amount".to_string()),
            "Should extract 'amount' as a filter column"
        );
        assert!(
            !pre_agg.covers(&group_by, &agg_components, &filter_cols),
            "Pre-agg should NOT cover plan with filter on non-group-by column"
        );
    }
}
