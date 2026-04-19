use datafusion_common::tree_node::Transformed;
use datafusion_common::{Column, Result as DFResult, TableReference};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::{Expr, LogicalPlan, LogicalPlanBuilder, TableSource};
use datafusion_functions_aggregate::min_max::{max, min};
use datafusion_functions_aggregate::sum::sum;
use datafusion_optimizer::OptimizerRule;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::model::pre_agg::{agg_needed_components, find_best_pre_agg, PreAggregation};

// ── Expression rewriting ──────────────────────────────────────────────────────

/// Rewrite qualified column references so their qualifier matches the
/// SubqueryAlias on the pre-agg scan.  Columns from *any* original table
/// (e.g. `player_stats.goals`) are re-qualified to the single source table
/// (`players.goals`) because the SubqueryAlias flattens all parquet columns
/// under one qualifier.
fn rewrite_qualifier(expr: &Expr, source_table: &str) -> Expr {
    match expr {
        Expr::Column(col) if col.relation.is_some() => Expr::Column(Column::new(
            Some(TableReference::bare(source_table)),
            &col.name,
        )),
        Expr::BinaryExpr(binary) => Expr::BinaryExpr(datafusion_expr::expr::BinaryExpr::new(
            Box::new(rewrite_qualifier(&binary.left, source_table)),
            binary.op,
            Box::new(rewrite_qualifier(&binary.right, source_table)),
        )),
        Expr::Alias(alias) => rewrite_qualifier(&alias.expr, source_table).alias(&alias.name),
        Expr::Not(inner) => Expr::Not(Box::new(rewrite_qualifier(inner, source_table))),
        Expr::IsNull(inner) => Expr::IsNull(Box::new(rewrite_qualifier(inner, source_table))),
        Expr::IsNotNull(inner) => Expr::IsNotNull(Box::new(rewrite_qualifier(inner, source_table))),
        Expr::InList(in_list) => {
            let rewritten_expr = Box::new(rewrite_qualifier(&in_list.expr, source_table));
            let rewritten_list = in_list
                .list
                .iter()
                .map(|e| rewrite_qualifier(e, source_table))
                .collect();
            Expr::InList(datafusion_expr::expr::InList::new(
                rewritten_expr,
                rewritten_list,
                in_list.negated,
            ))
        }
        Expr::Cast(cast) => Expr::Cast(datafusion_expr::expr::Cast::new(
            Box::new(rewrite_qualifier(&cast.expr, source_table)),
            cast.data_type.clone(),
        )),
        Expr::TryCast(cast) => Expr::TryCast(datafusion_expr::expr::TryCast::new(
            Box::new(rewrite_qualifier(&cast.expr, source_table)),
            cast.data_type.clone(),
        )),
        Expr::AggregateFunction(agg) => {
            let new_args: Vec<Expr> = agg
                .params
                .args
                .iter()
                .map(|a| rewrite_qualifier(a, source_table))
                .collect();
            let mut new_params = agg.params.clone();
            new_params.args = new_args;
            Expr::AggregateFunction(AggregateFunction {
                func: agg.func.clone(),
                params: new_params,
            })
        }
        _ => expr.clone(),
    }
}

/// Rewrite an aggregate expression to use pre-agg component columns.
///
/// The column qualifier is set to `source_table` (matching the SubqueryAlias)
/// and the column name is changed to the bare component name.
///
/// Examples (source_table = "players"):
///   sum(player_stats.goals) → sum(players."goals-sum")
///   count(player_stats.goals) → sum(players."goals-count")
///   avg(player_stats.goals) → sum(players."goals-sum") / sum(players."goals-count")
fn rewrite_agg_expr(expr: &Expr, source_table: &str) -> Option<Expr> {
    match expr {
        Expr::AggregateFunction(agg) => rewrite_aggregate_function(agg, source_table),
        Expr::Alias(alias) => {
            let rewritten = rewrite_agg_expr(&alias.expr, source_table)?;
            Some(rewritten.alias(&alias.name))
        }
        _ => None,
    }
}

/// Create a column expression for a pre-agg component, qualified to the source table.
fn component_col_expr(original: &Column, component: &str, source_table: &str) -> Expr {
    let component_name = PreAggregation::component_column(&original.name, component);
    Expr::Column(Column::new(
        Some(TableReference::bare(source_table)),
        component_name,
    ))
}

fn rewrite_aggregate_function(agg: &AggregateFunction, source_table: &str) -> Option<Expr> {
    let col = extract_agg_column(&agg.params.args[0])?;
    let func_name = agg.func.name();

    match func_name {
        "sum" => Some(sum(component_col_expr(col, "sum", source_table))),
        "count" => Some(sum(component_col_expr(col, "count", source_table))),
        "min" => Some(min(component_col_expr(col, "min", source_table))),
        "max" => Some(max(component_col_expr(col, "max", source_table))),
        "avg" => Some(
            sum(component_col_expr(col, "sum", source_table))
                / sum(component_col_expr(col, "count", source_table)),
        ),
        _ => None,
    }
}

/// Extract the Column struct from an aggregate function argument,
/// seeing through Cast/TryCast wrappers that DataFusion's type coercion
/// optimizer pass may insert before our rule runs.
fn extract_agg_column(expr: &Expr) -> Option<&Column> {
    match expr {
        Expr::Column(col) => Some(col),
        Expr::Cast(cast) => extract_agg_column(&cast.expr),
        Expr::TryCast(cast) => extract_agg_column(&cast.expr),
        _ => None,
    }
}

/// Return the fully qualified column name (e.g. "players.player_name")
/// or just the bare name if no relation qualifier is present.
/// Used for plan requirement matching (not expression rewriting).
fn qualified_col_name(col: &Column) -> String {
    match &col.relation {
        Some(r) => format!("{}.{}", r, col.name),
        None => col.name.clone(),
    }
}

/// Collect filters that DataFusion's FilterPushdown pass pushed into TableScan nodes.
/// These must be preserved when replacing the scan/join with a pre-agg table.
fn collect_scan_filters(plan: &LogicalPlan) -> Vec<Expr> {
    let mut filters = Vec::new();
    match plan {
        LogicalPlan::TableScan(scan) => {
            filters.extend(scan.filters.clone());
        }
        _ => {
            for child in plan.inputs() {
                filters.extend(collect_scan_filters(child));
            }
        }
    }
    filters
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

    /// Try to rewrite a single plan using pre-aggregations.
    ///
    /// Returns the rewritten plan if a covering pre-agg was found, or the
    /// original plan unchanged. This is used to apply pre-agg substitution
    /// per-measure before plans are combined, avoiding the session-level
    /// optimizer seeing a multi-measure plan.
    pub fn try_rewrite(&self, plan: LogicalPlan) -> DFResult<LogicalPlan> {
        if self.pre_aggs.is_empty() {
            return Ok(plan);
        }

        let (group_by, agg_components, filter_cols) = Self::collect_plan_requirements(&plan);
        let best = find_best_pre_agg(&self.pre_aggs, &group_by, &agg_components, &filter_cols);

        match best {
            Some(pa) => {
                let source_table = pa.source_table().ok_or_else(|| {
                    datafusion_common::DataFusionError::Plan(format!(
                        "Cannot infer source table from pre-agg '{}'",
                        pa.name
                    ))
                })?;
                self.rewrite_plan_with_pre_agg(plan, pa, source_table)
            }
            None => Ok(plan),
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
            Expr::Column(col) => Some(qualified_col_name(col)),
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
                filter_cols.push(qualified_col_name(col));
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
    /// The pre-agg scan is wrapped in a SubqueryAlias matching the source table.
    /// All qualified column references are re-qualified to the source table so
    /// they resolve against the SubqueryAlias. Aggregate expressions are also
    /// changed to reference component columns.
    fn rewrite_plan_with_pre_agg(
        &self,
        plan: LogicalPlan,
        pre_agg: &PreAggregation,
        source_table: &str,
    ) -> DFResult<LogicalPlan> {
        match plan {
            LogicalPlan::Aggregate(agg) => {
                // Rewrite aggregate expressions to reference component columns,
                // and re-qualify group-by expressions to the source table.
                let rewritten_aggs: Vec<Expr> = agg
                    .aggr_expr
                    .iter()
                    .map(|expr| {
                        rewrite_agg_expr(expr, source_table)
                            .unwrap_or_else(|| rewrite_qualifier(expr, source_table))
                    })
                    .collect();

                let group_exprs: Vec<Expr> = agg
                    .group_expr
                    .iter()
                    .map(|expr| rewrite_qualifier(expr, source_table))
                    .collect();

                let new_input =
                    self.rewrite_plan_with_pre_agg((*agg.input).clone(), pre_agg, source_table)?;

                LogicalPlanBuilder::from(new_input)
                    .aggregate(group_exprs, rewritten_aggs)?
                    .build()
            }
            LogicalPlan::Filter(filter) => {
                let new_input =
                    self.rewrite_plan_with_pre_agg((*filter.input).clone(), pre_agg, source_table)?;
                let rewritten_pred = rewrite_qualifier(&filter.predicate, source_table);
                LogicalPlanBuilder::from(new_input)
                    .filter(rewritten_pred)?
                    .build()
            }
            LogicalPlan::Projection(proj) => {
                // Skip intermediate projections — they reference original table columns
                // that don't exist in the pre-agg table. The parent Aggregate node
                // already specifies which columns it needs.
                self.rewrite_plan_with_pre_agg((*proj.input).clone(), pre_agg, source_table)
            }
            LogicalPlan::TableScan(_) | LogicalPlan::Join(_) => {
                // Collect any filters that DataFusion's FilterPushdown pass
                // pushed into the TableScan nodes. These would be lost when
                // we replace the scan/join with the pre-agg table.
                let scan_filters = collect_scan_filters(&plan);
                let pre_agg_scan = self.build_pre_agg_scan(pre_agg)?;

                if scan_filters.is_empty() {
                    Ok(pre_agg_scan)
                } else {
                    let combined = scan_filters
                        .into_iter()
                        .map(|f| rewrite_qualifier(&f, source_table))
                        .reduce(|a, b| a.and(b))
                        .unwrap();
                    LogicalPlanBuilder::from(pre_agg_scan)
                        .filter(combined)?
                        .build()
                }
            }
            // For other nodes (Sort, Limit, etc.), recurse into children
            // and re-qualify column references.
            other => {
                let new_inputs: Vec<LogicalPlan> = other
                    .inputs()
                    .into_iter()
                    .map(|input| {
                        self.rewrite_plan_with_pre_agg(input.clone(), pre_agg, source_table)
                    })
                    .collect::<DFResult<Vec<_>>>()?;
                let rewritten_exprs: Vec<Expr> = other
                    .expressions()
                    .into_iter()
                    .map(|expr| rewrite_qualifier(&expr, source_table))
                    .collect();
                other.with_new_exprs(rewritten_exprs, new_inputs)
            }
        }
    }

    /// Build a scan of the pre-agg table wrapped in a SubqueryAlias.
    ///
    /// The parquet stores columns with bare names (e.g. "player_name", "goals-sum").
    /// The SubqueryAlias re-qualifies them with the source table name
    /// (e.g. players.player_name, players."goals-sum") so that rewritten column
    /// references resolve correctly.
    fn build_pre_agg_scan(&self, pre_agg: &PreAggregation) -> DFResult<LogicalPlan> {
        let source = self.table_sources.get(&pre_agg.name).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "Pre-agg table source not found for '{}'",
                pre_agg.name
            ))
        })?;

        let source_table = pre_agg.source_table().ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "Cannot infer source table from pre-agg '{}': no qualified group-by columns",
                pre_agg.name
            ))
        })?;

        let scan = LogicalPlanBuilder::scan(&pre_agg.name, Arc::clone(source), None)?.build()?;
        LogicalPlanBuilder::from(scan).alias(source_table)?.build()
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
                let source_table = pa.source_table().ok_or_else(|| {
                    datafusion_common::DataFusionError::Plan(format!(
                        "Cannot infer source table from pre-agg '{}'",
                        pa.name
                    ))
                })?;
                let rewritten = self.rewrite_plan_with_pre_agg(plan, pa, source_table)?;
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

        // Register a pre-agg table with bare column names (matching parquet convention)
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
            vec!["orders.region".into()],
            HashMap::from([("orders.amount".into(), vec!["sum".into()])]),
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

        assert_eq!(group_by, vec!["orders.region"]);
        assert!(
            agg_components.contains_key("orders.amount"),
            "Should extract 'orders.amount' as agg column, got: {:?}",
            agg_components
        );
        assert!(
            agg_components["orders.amount"].contains("sum"),
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

        assert_eq!(group_by, vec!["orders.region"]);
        assert!(filter_cols.contains(&"orders.region".to_string()));
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
            filter_cols.contains(&"orders.amount".to_string()),
            "Should extract 'orders.amount' as a filter column"
        );
        assert!(
            !pre_agg.covers(&group_by, &agg_components, &filter_cols),
            "Pre-agg should NOT cover plan with filter on non-group-by column"
        );
    }

    // ── agg rewrite tests ───────────────────────────────────────────────

    /// Helper: create a qualified column expression for test inputs.
    fn test_col(table: &str, name: &str) -> Expr {
        Expr::Column(Column::new(Some(TableReference::bare(table)), name))
    }

    #[test]
    fn test_rewrite_sum() {
        let expr = sum(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr, "orders").unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-sum"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_count() {
        let expr = count(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr, "orders").unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_avg() {
        let expr = avg(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr, "orders").unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-sum"), "Got: {}", s);
        assert!(s.contains("amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_min() {
        let expr = min_fn(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr, "orders").unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-min"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_max() {
        let expr = max_fn(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr, "orders").unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-max"), "Got: {}", s);
    }

    #[test]
    fn test_non_agg_returns_none() {
        let expr = test_col("orders", "amount");
        assert!(rewrite_agg_expr(&expr, "orders").is_none());
    }
}
