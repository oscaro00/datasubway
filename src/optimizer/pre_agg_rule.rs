use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{Column, Result as DFResult, TableReference};
use datafusion_expr::expr::AggregateFunction;
use datafusion_expr::{Expr, JoinType, LogicalPlan, LogicalPlanBuilder, TableSource};
use datafusion_functions_aggregate::min_max::{max, min};
use datafusion_functions_aggregate::sum::sum;
use datafusion_optimizer::OptimizerRule;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::model::joins::{JoinGraph, JoinHow};
use crate::model::pre_agg::{agg_needed_components, find_best_pre_agg, PreAggregation};

macro_rules! debug_log {
    ($($arg:tt)*) => {
        if std::env::var("DATASUBWAY_DEBUG").is_ok() {
            eprintln!("[datasubway optimizer] {}", format!($($arg)*));
        }
    };
}

// ── Expression rewriting ──────────────────────────────────────────────────────

/// Create a column expression for a pre-agg component.
/// Uses the original column's own qualifier so each column has its proper table relation
/// after the pre-agg projection restores qualifiers.
fn component_col_expr(original: &Column, component: &str) -> Expr {
    let component_name = format!("{}-{}", original.name, component);
    Expr::Column(Column::new(original.relation.clone(), component_name))
}

fn rewrite_aggregate_function(agg: &AggregateFunction) -> Option<Expr> {
    let col = extract_agg_column(&agg.params.args[0])?;
    let func_name = agg.func.name();

    match func_name {
        "sum" => Some(sum(component_col_expr(col, "sum"))),
        "count" => Some(sum(component_col_expr(col, "count"))),
        "min" => Some(min(component_col_expr(col, "min"))),
        "max" => Some(max(component_col_expr(col, "max"))),
        "avg" => Some(sum(component_col_expr(col, "sum")) / sum(component_col_expr(col, "count"))),
        _ => None,
    }
}

/// Rewrite an aggregate expression to use pre-agg component columns.
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
fn qualified_col_name(col: &Column) -> String {
    match &col.relation {
        Some(r) => format!("{}.{}", r, col.name),
        None => col.name.clone(),
    }
}

// ── Plan column collection ────────────────────────────────────────────────────

/// Walk all expression-bearing nodes in the plan and collect:
/// - non_agg_cols: all Column references NOT inside an aggregate function
/// - agg_cols: columns inside aggregate functions, mapped to required components
pub fn collect_plan_columns(plan: &LogicalPlan) -> (Vec<String>, HashMap<String, HashSet<String>>) {
    let mut non_agg: Vec<String> = Vec::new();
    let mut agg: HashMap<String, HashSet<String>> = HashMap::new();

    let _ = plan.apply(|node| {
        for expr in node.expressions() {
            collect_expr_columns(&expr, &mut non_agg, &mut agg);
        }
        Ok(TreeNodeRecursion::Continue)
    });

    debug_log!("collect_plan_columns: non_agg={:?} agg={:?}", non_agg, agg);
    (non_agg, agg)
}

fn collect_expr_columns(
    expr: &Expr,
    non_agg: &mut Vec<String>,
    agg: &mut HashMap<String, HashSet<String>>,
) {
    let _ = expr.apply(|e| {
        match e {
            Expr::AggregateFunction(f) => {
                if let Some(col) = extract_agg_column(&f.params.args[0]) {
                    let col_name = qualified_col_name(col);
                    let func_name = f.func.name();
                    if let Some(components) = agg_needed_components(func_name) {
                        let entry = agg.entry(col_name).or_default();
                        for c in components {
                            entry.insert(c.to_string());
                        }
                    }
                }
                // Skip children so aggregate args aren't also counted as non-agg columns
                Ok(TreeNodeRecursion::Jump)
            }
            Expr::Column(col) => {
                non_agg.push(qualified_col_name(col));
                Ok(TreeNodeRecursion::Continue)
            }
            _ => Ok(TreeNodeRecursion::Continue),
        }
    });
}

/// Collect referenced table names from the plan, EXCLUDING join ON conditions.
/// This gives us the tables that are actually needed by the query (not just join infrastructure).
fn collect_referenced_tables(plan: &LogicalPlan) -> HashSet<String> {
    let mut tables = HashSet::new();
    let _ = plan.apply(|node| {
        // Skip Join ON expressions — they reference tables structurally, not semantically
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

// ── Virtual scan helpers ──────────────────────────────────────────────────────

const VIRTUAL_PREFIX: &str = "__empty_";

/// Returns true if the plan contains any TableScan whose name starts with
/// the virtual prefix (indicating a scan of an empty virtual table from table()).
pub fn has_virtual_scan(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = plan.apply(|node| {
        if let LogicalPlan::TableScan(scan) = node {
            if scan.table_name.table().starts_with(VIRTUAL_PREFIX) {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

/// Extract the real table name from a virtual scan name (strips __empty_ prefix).
fn real_name_from_virtual(virtual_name: &str) -> &str {
    virtual_name
        .strip_prefix(VIRTUAL_PREFIX)
        .unwrap_or(virtual_name)
}

/// Returns true if the plan node is part of the virtual scan cluster:
/// - A virtual TableScan (__empty_*)
/// - A SubqueryAlias wrapping a virtual cluster
/// - A Join where both sides are virtual clusters
///
/// Explicitly returns false for query nodes (Aggregate, Filter, Projection, etc.)
/// even if their inputs are virtual clusters.
fn is_virtual_cluster(plan: &LogicalPlan) -> bool {
    match plan {
        LogicalPlan::TableScan(scan) => scan.table_name.table().starts_with(VIRTUAL_PREFIX),
        LogicalPlan::SubqueryAlias(alias) => is_virtual_cluster(&alias.input),
        LogicalPlan::Join(join) => {
            is_virtual_cluster(&join.left) && is_virtual_cluster(&join.right)
        }
        _ => false,
    }
}

/// Find the base table name from the virtual scan cluster.
/// The base table is the real name encoded in the leftmost virtual scan.
fn find_base_virtual_table(plan: &LogicalPlan) -> Option<String> {
    match plan {
        LogicalPlan::TableScan(scan) if scan.table_name.table().starts_with(VIRTUAL_PREFIX) => {
            Some(real_name_from_virtual(scan.table_name.table()).to_string())
        }
        LogicalPlan::SubqueryAlias(alias) => find_base_virtual_table(&alias.input),
        LogicalPlan::Join(join) => find_base_virtual_table(&join.left),
        other => {
            for child in other.inputs() {
                if let Some(t) = find_base_virtual_table(child) {
                    return Some(t);
                }
            }
            None
        }
    }
}

/// Recursively replace the virtual scan cluster (identified by is_virtual_cluster)
/// with the given replacement plan.
fn replace_virtual_cluster_with(
    plan: LogicalPlan,
    replacement: &LogicalPlan,
) -> DFResult<LogicalPlan> {
    if is_virtual_cluster(&plan) {
        return Ok(replacement.clone());
    }

    let new_inputs: Vec<LogicalPlan> = plan
        .inputs()
        .into_iter()
        .map(|child| replace_virtual_cluster_with(child.clone(), replacement))
        .collect::<DFResult<Vec<_>>>()?;

    if new_inputs.is_empty() {
        Ok(plan)
    } else {
        plan.with_new_exprs(plan.expressions(), new_inputs)
    }
}

/// Collect filters pushed into TableScan nodes (preserved when substituting).
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

// ── VirtualScanExpander ───────────────────────────────────────────────────────

/// Unified optimizer rule that:
/// 1. Detects virtual empty table scans produced by DataModel::table()
/// 2. Tries to substitute them with a covering pre-aggregation
/// 3. Falls back to replacing virtual scans with real join plans (only joining
///    tables actually referenced by the query)
///
/// Applied per-measure in DataModel::prepare_measure_dfs() via try_rewrite().
pub struct VirtualScanExpander {
    pre_aggs: Vec<PreAggregation>,
    /// Table sources for pre-agg parquet tables, keyed by pre-agg name.
    pre_agg_sources: HashMap<String, Arc<dyn TableSource>>,
    /// Real table sources for each registered table, keyed by table name.
    real_table_sources: HashMap<String, Arc<dyn TableSource>>,
    /// Join graph — used to build minimal real join plans.
    join_graph: Option<JoinGraph>,
}

impl std::fmt::Debug for VirtualScanExpander {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VirtualScanExpander")
            .field("pre_aggs", &self.pre_aggs)
            .finish()
    }
}

impl VirtualScanExpander {
    pub fn new(
        pre_aggs: Vec<PreAggregation>,
        pre_agg_sources: HashMap<String, Arc<dyn TableSource>>,
        real_table_sources: HashMap<String, Arc<dyn TableSource>>,
        join_graph: Option<JoinGraph>,
    ) -> Self {
        Self {
            pre_aggs,
            pre_agg_sources,
            real_table_sources,
            join_graph,
        }
    }

    /// Apply virtual-scan expansion to a combined plan that may contain multiple
    /// virtual scan clusters (one per measure sub-plan). Splits at Join nodes so
    /// each measure branch is expanded independently with its own column
    /// requirements and pre-agg selection.
    /// This cannot easily be added to the DataFusion optimizer pipeline because
    /// use_pre_agg decides whether or not the optimizer runs and the pipeline does not
    /// allow enabling/disabling rules easily.
    pub fn try_rewrite_combined(
        &self,
        plan: LogicalPlan,
        allow_pre_agg: bool,
    ) -> DFResult<LogicalPlan> {
        if !has_virtual_scan(&plan) {
            return Ok(plan);
        }
        self.rewrite_node_combined(plan, allow_pre_agg)
    }

    fn rewrite_node_combined(
        &self,
        plan: LogicalPlan,
        allow_pre_agg: bool,
    ) -> DFResult<LogicalPlan> {
        match &plan {
            LogicalPlan::Join(_) => {
                let inputs = plan.inputs();
                let left = self.rewrite_node_combined(inputs[0].clone(), allow_pre_agg)?;
                let right = self.rewrite_node_combined(inputs[1].clone(), allow_pre_agg)?;
                plan.with_new_exprs(plan.expressions(), vec![left, right])
            }
            _ if has_virtual_scan(&plan) => self.try_rewrite(plan, allow_pre_agg),
            other => Ok(other.clone()),
        }
    }

    /// Apply the virtual-scan expansion to a single measure plan.
    ///
    /// Always expands virtual (`__empty_*`) scans — either to a pre-agg scan
    /// (when `allow_pre_agg` is true and a covering pre-agg exists) or to the
    /// real join plan. Returns the plan unchanged if no virtual scans are found.
    pub fn try_rewrite(&self, plan: LogicalPlan, allow_pre_agg: bool) -> DFResult<LogicalPlan> {
        if !has_virtual_scan(&plan) {
            return Ok(plan);
        }

        debug_log!(
            "try_rewrite: virtual scan detected, allow_pre_agg={}",
            allow_pre_agg
        );

        let best = if allow_pre_agg {
            let (non_agg_cols, agg_cols) = collect_plan_columns(&plan);
            debug_log!("try_rewrite: non_agg_cols={:?}", non_agg_cols);
            debug_log!("try_rewrite: agg_cols={:?}", agg_cols);
            let chosen = find_best_pre_agg(&self.pre_aggs, &non_agg_cols, &agg_cols);
            debug_log!("try_rewrite: chose pre_agg={:?}", chosen.map(|pa| &pa.name));
            chosen
        } else {
            None
        };

        match best {
            Some(pa) => self.rewrite_with_pre_agg(plan, pa),
            None => self.expand_virtual_to_real(plan),
        }
    }

    // ── Pre-agg substitution path ─────────────────────────────────────────────

    fn rewrite_with_pre_agg(
        &self,
        plan: LogicalPlan,
        pre_agg: &PreAggregation,
    ) -> DFResult<LogicalPlan> {
        match plan {
            LogicalPlan::Aggregate(agg) => {
                let rewritten_aggs: Vec<Expr> = agg
                    .aggr_expr
                    .iter()
                    .map(|expr| rewrite_agg_expr(expr).unwrap_or_else(|| expr.clone()))
                    .collect();

                let new_input = self.rewrite_with_pre_agg((*agg.input).clone(), pre_agg)?;

                LogicalPlanBuilder::from(new_input)
                    .aggregate(agg.group_expr, rewritten_aggs)?
                    .build()
            }
            LogicalPlan::Filter(filter) => {
                let new_input = self.rewrite_with_pre_agg((*filter.input).clone(), pre_agg)?;
                LogicalPlanBuilder::from(new_input)
                    .filter(filter.predicate)?
                    .build()
            }
            LogicalPlan::Projection(proj) => {
                // Skip intermediate projections — the parent Aggregate specifies needed columns.
                self.rewrite_with_pre_agg((*proj.input).clone(), pre_agg)
            }
            // The virtual scan cluster: replace entirely with the pre-agg scan.
            node if is_virtual_cluster(&node) => {
                debug_log!(
                    "rewrite_with_pre_agg: replacing virtual cluster with pre-agg '{}'",
                    pre_agg.name
                );
                let scan_filters = collect_scan_filters(&node);
                let pre_agg_scan = self.build_pre_agg_scan(pre_agg)?;

                if scan_filters.is_empty() {
                    Ok(pre_agg_scan)
                } else {
                    let combined = scan_filters.into_iter().reduce(|a, b| a.and(b)).unwrap();
                    LogicalPlanBuilder::from(pre_agg_scan)
                        .filter(combined)?
                        .build()
                }
            }
            // For other nodes, recurse into children.
            other => {
                let new_inputs: Vec<LogicalPlan> = other
                    .inputs()
                    .into_iter()
                    .map(|input| self.rewrite_with_pre_agg(input.clone(), pre_agg))
                    .collect::<DFResult<Vec<_>>>()?;
                other.with_new_exprs(other.expressions(), new_inputs)
            }
        }
    }

    /// Build a scan of the pre-agg parquet table with a Projection that
    /// re-qualifies each encoded parquet column name back to its original
    /// table qualifier and bare name via alias_qualified.
    ///
    /// e.g. parquet field "orders__region"     → Column { relation: "orders", name: "region" }
    ///      parquet field "orders__amount-sum"  → Column { relation: "orders", name: "amount-sum" }
    ///      parquet field "players__name-count" → Column { relation: "players", name: "name-count" }
    fn build_pre_agg_scan(&self, pre_agg: &PreAggregation) -> DFResult<LogicalPlan> {
        let source = self.pre_agg_sources.get(&pre_agg.name).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "Pre-agg table source not found for '{}'",
                pre_agg.name
            ))
        })?;

        let scan = LogicalPlanBuilder::scan(&pre_agg.name, Arc::clone(source), None)?.build()?;

        let mut projections: Vec<Expr> = Vec::new();

        for col_name in &pre_agg.group_by {
            let parquet_name = PreAggregation::group_by_column_name(col_name);
            let (table, bare) = PreAggregation::decode_parquet_col_name(&parquet_name);
            projections.push(
                Expr::Column(Column::new(None::<TableReference>, &parquet_name))
                    .alias_qualified(Some(table), bare),
            );
        }

        for (col_name, components) in &pre_agg.aggregations {
            for comp in components {
                let parquet_name = PreAggregation::component_column(col_name, comp);
                let (table, bare) = PreAggregation::decode_parquet_col_name(&parquet_name);
                projections.push(
                    Expr::Column(Column::new(None::<TableReference>, &parquet_name))
                        .alias_qualified(Some(table), bare),
                );
            }
        }

        let result = LogicalPlanBuilder::from(scan)
            .project(projections)?
            .build()?;
        debug_log!(
            "build_pre_agg_scan: output schema = {:?}",
            result
                .schema()
                .columns()
                .iter()
                .map(|c| format!("{}", c))
                .collect::<Vec<_>>()
        );
        Ok(result)
    }

    // ── Real-join fallback path ───────────────────────────────────────────────

    /// Replace the virtual scan cluster with a real join plan.
    /// Only joins tables that are actually referenced by the query (excluding
    /// join ON conditions), eliminating unnecessary joins without a separate pass.
    fn expand_virtual_to_real(&self, plan: LogicalPlan) -> DFResult<LogicalPlan> {
        let base_table = find_base_virtual_table(&plan).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(
                "expand_virtual_to_real: no virtual scan found in plan".into(),
            )
        })?;

        // Collect tables referenced by the query (excluding join ON conditions)
        let referenced_tables = collect_referenced_tables(&plan);
        debug_log!(
            "expand_virtual_to_real: base_table='{}', referenced={:?}",
            base_table,
            referenced_tables
        );

        // Build the minimal real join plan
        let real_plan = self.build_real_join_plan(&base_table, &referenced_tables)?;

        // Replace the virtual cluster with the real plan
        replace_virtual_cluster_with(plan, &real_plan)
    }

    /// Build a real join plan starting from base_table, joining only the tables
    /// that appear in referenced_tables (and any intermediate tables needed for paths).
    fn build_real_join_plan(
        &self,
        base_table: &str,
        referenced_tables: &HashSet<String>,
    ) -> DFResult<LogicalPlan> {
        let base_source = self.real_table_sources.get(base_table).ok_or_else(|| {
            datafusion_common::DataFusionError::Plan(format!(
                "Real table source not found for '{}'",
                base_table
            ))
        })?;

        let mut plan =
            LogicalPlanBuilder::scan(base_table, Arc::clone(base_source), None)?.build()?;
        let mut joined: HashSet<String> = HashSet::new();
        joined.insert(base_table.to_string());

        let Some(ref jg) = self.join_graph else {
            return Ok(plan);
        };

        // For each referenced table, find the join path and add necessary joins
        for table in referenced_tables {
            if table == base_table || joined.contains(table) {
                continue;
            }
            let path = match jg.find_path(base_table, table) {
                Some(p) => p,
                None => continue,
            };
            for step in &path {
                if joined.contains(&step.right) {
                    continue;
                }
                let right_source = self.real_table_sources.get(&step.right).ok_or_else(|| {
                    datafusion_common::DataFusionError::Plan(format!(
                        "Real table source not found for '{}'",
                        step.right
                    ))
                })?;
                let right_plan =
                    LogicalPlanBuilder::scan(&step.right, Arc::clone(right_source), None)?
                        .build()?;

                let join_type = match step.how {
                    JoinHow::Inner => JoinType::Inner,
                    JoinHow::Left => JoinType::Left,
                };

                let left_keys: Vec<Column> = step
                    .left_on
                    .iter()
                    .map(|k| Column::new(Some(TableReference::bare(&*step.left)), k.as_str()))
                    .collect();
                let right_keys: Vec<Column> = step
                    .right_on
                    .iter()
                    .map(|k| Column::new(Some(TableReference::bare(&*step.right)), k.as_str()))
                    .collect();

                plan = LogicalPlanBuilder::from(plan)
                    .join(right_plan, join_type, (left_keys, right_keys), None)?
                    .build()?;

                joined.insert(step.right.clone());
            }
        }

        Ok(plan)
    }
}

// ── Utility ───────────────────────────────────────────────────────────────────

/// Helper to check if a plan contains a TableScan for a given table name.
pub fn plan_contains_table_scan(plan: &LogicalPlan, table_name: &str) -> bool {
    match plan {
        LogicalPlan::TableScan(scan) => scan.table_name.table() == table_name,
        _ => plan
            .inputs()
            .iter()
            .any(|child| plan_contains_table_scan(child, table_name)),
    }
}

// ── OptimizerRule impl ────────────────────────────────────────────────────────

impl OptimizerRule for VirtualScanExpander {
    fn name(&self) -> &str {
        "virtual_scan_expander"
    }

    fn supports_rewrite(&self) -> bool {
        true
    }

    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn datafusion_optimizer::OptimizerConfig,
    ) -> DFResult<Transformed<LogicalPlan>> {
        match self.try_rewrite(plan, true) {
            Ok(new_plan) => Ok(Transformed::yes(new_plan)),
            Err(e) => Err(e),
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::datasource::MemTable;
    use datafusion::execution::context::SessionContext;
    use datafusion::prelude::col;
    use datafusion_functions_aggregate::average::avg;
    use datafusion_functions_aggregate::count::count;
    use datafusion_functions_aggregate::min_max::{max as max_fn, min as min_fn};
    use std::sync::Arc;

    fn make_test_context() -> SessionContext {
        let ctx = SessionContext::new();

        // Real orders table
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
        let mem_table = MemTable::try_new(schema.clone(), vec![vec![batch.clone()]]).unwrap();
        ctx.register_table("orders", Arc::new(mem_table)).unwrap();

        // Virtual empty orders table
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let empty_table = MemTable::try_new(schema.clone(), vec![vec![empty_batch]]).unwrap();
        ctx.register_table("__empty_orders", Arc::new(empty_table))
            .unwrap();

        // Pre-agg table with encoded column names
        let preagg_schema = Arc::new(Schema::new(vec![
            Field::new("orders__region", DataType::Utf8, false),
            Field::new("orders__amount-sum", DataType::Int64, false),
        ]));
        let preagg_batch = RecordBatch::try_new(
            preagg_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["US", "EU"])),
                Arc::new(Int64Array::from(vec![100, 200])),
            ],
        )
        .unwrap();
        let preagg_table = MemTable::try_new(preagg_schema, vec![vec![preagg_batch]]).unwrap();
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

    fn make_expander(ctx: &SessionContext) -> VirtualScanExpander {
        use datafusion::datasource::DefaultTableSource;
        use futures::executor::block_on;

        let pre_agg = make_pre_agg();
        let provider = block_on(ctx.table_provider("regional_revenue")).unwrap();
        let pre_agg_sources = HashMap::from([(
            "regional_revenue".to_string(),
            Arc::new(DefaultTableSource::new(provider)) as Arc<dyn TableSource>,
        )]);

        let real_provider = block_on(ctx.table_provider("orders")).unwrap();
        let real_table_sources = HashMap::from([(
            "orders".to_string(),
            Arc::new(DefaultTableSource::new(real_provider)) as Arc<dyn TableSource>,
        )]);

        VirtualScanExpander::new(vec![pre_agg], pre_agg_sources, real_table_sources, None)
    }

    #[tokio::test]
    async fn test_collect_plan_columns_aggregate() {
        let ctx = make_test_context();
        let df = ctx
            .table("__empty_orders")
            .await
            .unwrap()
            .alias("orders")
            .unwrap();
        let agg = df
            .aggregate(
                vec![col("orders.region")],
                vec![sum(col("orders.amount")).alias("total")],
            )
            .unwrap();
        let plan = agg.logical_plan().clone();

        let (non_agg, agg_cols) = collect_plan_columns(&plan);
        assert!(
            non_agg.contains(&"orders.region".to_string()),
            "got: {:?}",
            non_agg
        );
        assert!(
            agg_cols.contains_key("orders.amount"),
            "got: {:?}",
            agg_cols
        );
        assert!(agg_cols["orders.amount"].contains("sum"));
    }

    #[tokio::test]
    async fn test_no_rewrite_without_virtual_scan() {
        let ctx = make_test_context();
        let expander = make_expander(&ctx);

        let df = ctx.table("orders").await.unwrap();
        let agg = df
            .aggregate(vec![col("region")], vec![sum(col("amount")).alias("total")])
            .unwrap();
        let plan = agg.logical_plan().clone();

        assert!(!has_virtual_scan(&plan));
        let result = expander.try_rewrite(plan.clone(), true).unwrap();
        assert_eq!(format!("{:?}", result), format!("{:?}", plan));
    }

    #[tokio::test]
    async fn test_expands_virtual_to_real_when_no_pre_agg_covers() {
        let ctx = make_test_context();

        let real_provider = ctx.table_provider("orders").await.unwrap();
        use datafusion::datasource::DefaultTableSource;
        let real_sources = HashMap::from([(
            "orders".to_string(),
            Arc::new(DefaultTableSource::new(real_provider)) as Arc<dyn TableSource>,
        )]);
        let expander = VirtualScanExpander::new(vec![], HashMap::new(), real_sources, None);

        let df = ctx
            .table("__empty_orders")
            .await
            .unwrap()
            .alias("orders")
            .unwrap();
        let agg = df
            .aggregate(
                vec![col("orders.region")],
                vec![sum(col("orders.amount")).alias("total")],
            )
            .unwrap();
        let plan = agg.logical_plan().clone();

        assert!(has_virtual_scan(&plan));
        let result = expander.try_rewrite(plan, true).unwrap();
        assert!(
            plan_contains_table_scan(&result, "orders"),
            "should contain real orders scan"
        );
        assert!(
            !has_virtual_scan(&result),
            "should not contain virtual scan"
        );
    }

    #[tokio::test]
    async fn test_substitutes_pre_agg_when_covering() {
        let ctx = make_test_context();
        let expander = make_expander(&ctx);

        let df = ctx
            .table("__empty_orders")
            .await
            .unwrap()
            .alias("orders")
            .unwrap();
        let agg = df
            .aggregate(
                vec![col("orders.region")],
                vec![sum(col("orders.amount")).alias("total")],
            )
            .unwrap();
        let plan = agg.logical_plan().clone();

        assert!(has_virtual_scan(&plan));
        let result = expander.try_rewrite(plan, true).unwrap();

        assert!(
            plan_contains_table_scan(&result, "regional_revenue"),
            "should contain pre-agg scan"
        );
        assert!(!has_virtual_scan(&result));
    }

    #[tokio::test]
    async fn test_no_rewrite_when_no_covering_pre_agg() {
        let ctx = make_test_context();

        let pa = PreAggregation::new(
            "wrong_preagg".into(),
            vec!["orders.date".into()],
            HashMap::from([("orders.quantity".into(), vec!["sum".into()])]),
        )
        .unwrap();
        let real_provider = ctx.table_provider("orders").await.unwrap();
        use datafusion::datasource::DefaultTableSource;
        let real_sources = HashMap::from([(
            "orders".to_string(),
            Arc::new(DefaultTableSource::new(real_provider)) as Arc<dyn TableSource>,
        )]);
        let expander = VirtualScanExpander::new(vec![pa], HashMap::new(), real_sources, None);

        let df = ctx
            .table("__empty_orders")
            .await
            .unwrap()
            .alias("orders")
            .unwrap();
        let agg = df
            .aggregate(
                vec![col("orders.region")],
                vec![sum(col("orders.amount")).alias("total")],
            )
            .unwrap();
        let plan = agg.logical_plan().clone();

        let result = expander.try_rewrite(plan, true).unwrap();
        // Falls back to real table expansion
        assert!(plan_contains_table_scan(&result, "orders"));
        assert!(!has_virtual_scan(&result));
        assert!(!plan_contains_table_scan(&result, "wrong_preagg"));
    }

    // ── agg rewrite tests ───────────────────────────────────────────────────

    fn test_col(table: &str, name: &str) -> Expr {
        Expr::Column(Column::new(Some(TableReference::bare(table)), name))
    }

    #[test]
    fn test_rewrite_sum() {
        let expr = sum(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-sum"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_count() {
        let expr = count(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_avg() {
        let expr = avg(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-sum"), "Got: {}", s);
        assert!(s.contains("amount-count"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_min() {
        let expr = min_fn(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-min"), "Got: {}", s);
    }

    #[test]
    fn test_rewrite_max() {
        let expr = max_fn(test_col("orders", "amount"));
        let rewritten = rewrite_agg_expr(&expr).unwrap();
        let s = format!("{}", rewritten);
        assert!(s.contains("amount-max"), "Got: {}", s);
    }

    #[test]
    fn test_non_agg_returns_none() {
        let expr = test_col("orders", "amount");
        assert!(rewrite_agg_expr(&expr).is_none());
    }

    #[test]
    fn test_component_col_preserves_qualifier() {
        let col = Column::new(Some(TableReference::bare("players")), "goals");
        let expr = component_col_expr(&col, "sum");
        if let Expr::Column(c) = expr {
            assert_eq!(c.name, "goals-sum");
            assert_eq!(c.relation.unwrap().table(), "players");
        } else {
            panic!("expected Column expr");
        }
    }

    #[test]
    fn test_is_virtual_cluster() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let mem = Arc::new(MemTable::try_new(schema.clone(), vec![vec![empty_batch]]).unwrap());
        use datafusion::datasource::DefaultTableSource;
        let source = Arc::new(DefaultTableSource::new(mem as _));

        let virtual_scan = LogicalPlanBuilder::scan("__empty_orders", source.clone(), None)
            .unwrap()
            .build()
            .unwrap();
        assert!(is_virtual_cluster(&virtual_scan));

        let real_scan = LogicalPlanBuilder::scan("orders", source.clone(), None)
            .unwrap()
            .build()
            .unwrap();
        assert!(!is_virtual_cluster(&real_scan));
    }

    #[test]
    fn test_find_base_virtual_table() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let empty_batch = RecordBatch::new_empty(schema.clone());
        let mem = Arc::new(MemTable::try_new(schema.clone(), vec![vec![empty_batch]]).unwrap());
        use datafusion::datasource::DefaultTableSource;
        let source = Arc::new(DefaultTableSource::new(mem as _));

        let scan = LogicalPlanBuilder::scan("__empty_player_stats", source, None)
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(
            find_base_virtual_table(&scan),
            Some("player_stats".to_string())
        );
    }
}
