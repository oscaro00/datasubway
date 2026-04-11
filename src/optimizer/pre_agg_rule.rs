use datafusion_common::tree_node::Transformed;
use datafusion_common::{Column, Result as DFResult};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableSource};
use datafusion_functions_aggregate::min_max::{max, min};
use datafusion_functions_aggregate::sum::sum;
use datafusion_optimizer::OptimizerRule;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::model::pre_agg::{agg_needed_components, find_best_pre_agg, PreAggregation};

// ── Aggregate expression rewriting ──────────────────────────────────────────

/// Rewrite an aggregate expression to use pre-agg component columns.
///
/// For example:
///   sum(orders.amount) → sum(orders.amount-sum)
///   avg(orders.amount) → sum(orders.amount-sum) / sum(orders.amount-count)
///   count(orders.amount) → sum(orders.amount-count)
///   min(orders.amount) → min(orders.amount-min)
///   max(orders.amount) → max(orders.amount-max)
fn rewrite_agg_expr(expr: &Expr) -> Option<Expr> {
    match expr {
        Expr::AggregateFunction(agg) => rewrite_aggregate_function(agg),
        Expr::Alias(alias) => {
            let rewritten = rewrite_agg_expr(&alias.expr)?;
            Some(rewritten.alias(&alias.name))
        }
        _ => None,
    }
}

fn agg_rewrite_col_expr(name: &str) -> Expr {
    Expr::Column(Column::from_name(name))
}

fn rewrite_aggregate_function(agg: &AggregateFunction) -> Option<Expr> {
    let col_name = extract_agg_column_name(&agg.params.args[0])?;
    let func_name = agg.func.name();

    match func_name {
        "sum" => {
            let comp_col = PreAggregation::component_column(&col_name, "sum");
            Some(sum(agg_rewrite_col_expr(&comp_col)))
        }
        "count" => {
            let comp_col = PreAggregation::component_column(&col_name, "count");
            Some(sum(agg_rewrite_col_expr(&comp_col)))
        }
        "min" => {
            let comp_col = PreAggregation::component_column(&col_name, "min");
            Some(min(agg_rewrite_col_expr(&comp_col)))
        }
        "max" => {
            let comp_col = PreAggregation::component_column(&col_name, "max");
            Some(max(agg_rewrite_col_expr(&comp_col)))
        }
        "avg" => {
            let sum_col = PreAggregation::component_column(&col_name, "sum");
            let count_col = PreAggregation::component_column(&col_name, "count");
            Some(sum(agg_rewrite_col_expr(&sum_col)) / sum(agg_rewrite_col_expr(&count_col)))
        }
        _ => None,
    }
}

fn extract_agg_column_name(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Column(col) => Some(col.name.clone()),
        _ => None,
    }
}

/// OptimizerRule that substitutes raw table scans with pre-aggregated tables
/// when a suitable pre-aggregation exists.
pub struct PreAggSubstitution {
    pre_aggs: Vec<PreAggregation>,
    /// Table sources for pre-agg tables, keyed by pre-agg name.
    /// Required so that replaced TableScan nodes have the correct source and schema.
    table_sources: HashMap<String, Arc<dyn TableSource>>,
}

impl std::fmt::Debug for PreAggSubstitution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PreAggSubstitution")
            .field("pre_aggs", &self.pre_aggs)
            .finish()
    }
}

impl PreAggSubstitution {
    pub fn new(
        pre_aggs: Vec<PreAggregation>,
        table_sources: HashMap<String, Arc<dyn TableSource>>,
    ) -> Self {
        Self {
            pre_aggs,
            table_sources,
        }
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
    ///
    /// The key insight: we alias the pre-agg scan with the original table name
    /// (via `SubqueryAlias`) so that existing column qualifiers (e.g. `orders.region`)
    /// continue to resolve correctly. Only aggregate expressions need rewriting
    /// (to reference component columns like `amount-sum`).
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

                // Group-by expressions stay the same — the SubqueryAlias on the
                // pre-agg scan preserves the original table qualifier.
                let group_exprs = agg.group_expr.clone();

                let new_input = self.rewrite_plan_with_pre_agg((*agg.input).clone(), pre_agg)?;

                LogicalPlanBuilder::from(new_input)
                    .aggregate(group_exprs, rewritten_aggs)?
                    .build()
            }
            LogicalPlan::Filter(filter) => {
                let new_input = self.rewrite_plan_with_pre_agg((*filter.input).clone(), pre_agg)?;
                LogicalPlanBuilder::from(new_input)
                    .filter(filter.predicate.clone())?
                    .build()
            }
            LogicalPlan::Projection(proj) => {
                // Skip intermediate projections — they reference original table columns
                // that don't exist in the pre-agg table. The parent Aggregate node
                // already specifies which columns it needs.
                self.rewrite_plan_with_pre_agg((*proj.input).clone(), pre_agg)
            }
            LogicalPlan::TableScan(ref scan) => {
                self.build_pre_agg_scan(pre_agg, scan.table_name.table())
            }
            LogicalPlan::Join(_) => {
                // For joins, find the first table name and use that as the alias
                let original_name =
                    Self::find_first_table_name(&plan).unwrap_or_else(|| pre_agg.name.clone());
                self.build_pre_agg_scan(pre_agg, &original_name)
            }
            // For other nodes (Sort, Limit, etc.), recurse into children.
            // Expressions keep their original qualifiers since the SubqueryAlias
            // ensures they still resolve.
            other => {
                let new_inputs: Vec<LogicalPlan> = other
                    .inputs()
                    .into_iter()
                    .map(|input| self.rewrite_plan_with_pre_agg(input.clone(), pre_agg))
                    .collect::<DFResult<Vec<_>>>()?;
                other.with_new_exprs(other.expressions(), new_inputs)
            }
        }
    }

    /// Build a pre-agg scan aliased with the original table name so that
    /// existing column qualifiers (e.g. `orders.region`) remain valid.
    fn build_pre_agg_scan(
        &self,
        pre_agg: &PreAggregation,
        original_table_name: &str,
    ) -> DFResult<LogicalPlan> {
        let source = self.table_sources.get(&pre_agg.name).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "Pre-agg table source not found for '{}'",
                pre_agg.name
            ))
        })?;
        LogicalPlanBuilder::scan(&pre_agg.name, Arc::clone(source), None)?
            .alias(original_table_name)?
            .build()
    }

    fn find_first_table_name(plan: &LogicalPlan) -> Option<String> {
        match plan {
            LogicalPlan::TableScan(scan) => Some(scan.table_name.table().to_string()),
            other => other
                .inputs()
                .into_iter()
                .find_map(Self::find_first_table_name),
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
    use datafusion_functions_aggregate::average::avg;
    use datafusion_functions_aggregate::count::count;
    use datafusion_functions_aggregate::min_max::{max as max_fn, min as min_fn};
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
        let preagg_table =
            datafusion::datasource::MemTable::try_new(preagg_schema, vec![vec![preagg_batch]])
                .unwrap();
        ctx.register_table("regional_revenue", Arc::new(preagg_table))
            .unwrap();
        ctx
    }

    fn make_pre_agg() -> PreAggregation {
        let mut pa = PreAggregation::new(
            "regional_revenue".into(),
            vec!["region".into()],
            HashMap::from([("amount".into(), vec!["sum".into()])]),
        )
        .unwrap();
        pa.row_count = 2;
        pa
    }

    #[tokio::test]
    async fn test_collect_plan_requirements() {
        let ctx = make_test_context();

        let df = ctx.table("orders").await.unwrap();
        let agg = df
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

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

    #[tokio::test]
    async fn test_rule_matches_covering_pre_agg() {
        let ctx = make_test_context();

        let df = ctx.table("orders").await.unwrap();
        let agg = df
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

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
        let best = find_best_pre_agg(&pre_aggs, &group_by, &agg_components, &filter_cols);
        assert!(best.is_some(), "Should find a matching pre-agg");
        assert_eq!(best.unwrap().name, "regional_revenue");
    }

    #[tokio::test]
    async fn test_rule_matches_with_filter_columns() {
        let ctx = make_test_context();

        let df = ctx.table("orders").await.unwrap();
        let filtered = df.filter(col("region").eq(lit("US"))).unwrap();
        let agg = filtered
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

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

    #[tokio::test]
    async fn test_no_rewrite_when_no_covering_pre_agg() {
        let ctx = make_test_context();

        let df = ctx.table("orders").await.unwrap();
        let agg = df
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

        // Pre-agg that doesn't cover the request (wrong column)
        let pa = PreAggregation::new(
            "wrong_preagg".into(),
            vec!["date".into()],
            HashMap::from([("quantity".into(), vec!["sum".into()])]),
        )
        .unwrap();

        let rule = PreAggSubstitution::new(vec![pa], HashMap::new());
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(
            !result.transformed,
            "Plan should NOT have been transformed when no pre-agg covers the request"
        );
    }

    #[tokio::test]
    async fn test_no_rewrite_when_empty_pre_aggs() {
        let ctx = make_test_context();

        let df = ctx.table("orders").await.unwrap();
        let agg = df
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

        let rule = PreAggSubstitution::new(vec![], HashMap::new());
        let config = OptimizerContext::new();
        let result = rule.rewrite(plan, &config).unwrap();

        assert!(!result.transformed, "Empty pre-aggs should not transform");
    }

    #[tokio::test]
    async fn test_rule_rejects_uncovered_filter_column() {
        let ctx = make_test_context();

        let df = ctx.table("orders").await.unwrap();
        // Filter on "amount" which is NOT in the pre-agg's group_by
        let filtered = df.filter(col("amount").gt(lit(100))).unwrap();
        let agg = filtered
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

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

    // ── agg rewrite tests ───────────────────────────────────────────────

    #[test]
    fn test_rewrite_sum() {
        let expr = sum(agg_rewrite_col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-sum"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_count() {
        let expr = count(agg_rewrite_col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_avg() {
        let expr = avg(agg_rewrite_col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-sum"), "Got: {}", s);
        assert!(s.contains("orders.amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_min() {
        let expr = min_fn(agg_rewrite_col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-min"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_max() {
        let expr = max_fn(agg_rewrite_col_expr("orders.amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("orders.amount-max"), "Got: {}", s);
    }

    #[test]
    fn test_non_agg_returns_none() {
        let expr = agg_rewrite_col_expr("orders.amount");
        assert!(rewrite_agg_expr(&expr).is_none());
    }
}
