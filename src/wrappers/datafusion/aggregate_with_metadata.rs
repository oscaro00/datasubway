use async_trait::async_trait;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{Column, DFSchemaRef, Result};
use datafusion::execution::context::{QueryPlanner, SessionState};
use datafusion::logical_expr::{
    Extension, LogicalPlan, LogicalPlanBuilder, SortExpr, UserDefinedLogicalNode,
    UserDefinedLogicalNodeCore,
};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::{DataFrame, Expr};
use std::collections::{BTreeMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// Replaces a standard `Aggregate` node in the logical plan and carries
/// arbitrary string metadata alongside it. Used to attach allow/exclude
/// context so it survives into the plan and can be extracted at query time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct AggregateWithMetadata {
    pub input: Arc<LogicalPlan>,
    pub group_expr: Vec<Expr>,
    pub aggr_expr: Vec<Expr>,
    pub schema: DFSchemaRef,
    pub metadata: BTreeMap<String, String>,
}

impl AggregateWithMetadata {
    /// Walk `plan`, replace every `Aggregate` node with an `AggregateWithMetadata`
    /// carrying the supplied metadata.
    pub fn inject(plan: LogicalPlan, metadata: BTreeMap<String, String>) -> Result<LogicalPlan> {
        plan.transform(|node| {
            let LogicalPlan::Aggregate(agg) = node else {
                return Ok(Transformed::no(node));
            };
            let replacement = LogicalPlan::Extension(Extension {
                node: Arc::new(AggregateWithMetadata {
                    input: agg.input,
                    group_expr: agg.group_expr,
                    aggr_expr: agg.aggr_expr,
                    schema: agg.schema,
                    metadata: metadata.clone(),
                }),
            });
            Ok(Transformed::yes(replacement))
        })
        .map(|t| t.data)
    }
}

/// Returns the `AggregateWithMetadata` node at the root of `df`'s logical plan,
/// or an error if the plan doesn't end with `.aggregate()`.
pub fn root_aggregate_node(df: &DataFrame) -> Result<&AggregateWithMetadata, String> {
    match df.logical_plan() {
        LogicalPlan::Extension(e) => e
            .node
            .as_any()
            .downcast_ref::<AggregateWithMetadata>()
            .ok_or_else(|| "expected AggregateWithMetadata at plan root".to_string()),
        _ => Err("expected AggregateWithMetadata at plan root".into()),
    }
}

/// Parses the JSON-serialized `allow_exclude` metadata entry into records,
/// defaulting to empty if absent or malformed.
pub fn allow_exclude_records(
    node: &AggregateWithMetadata,
) -> Vec<crate::column_expressions::column_context::AllowExcludeRecord> {
    node.metadata
        .get("allow_exclude")
        .and_then(|s| serde_json::from_str(s).ok())
        .unwrap_or_default()
}

impl PartialOrd for AggregateWithMetadata {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.metadata.partial_cmp(&other.metadata)
    }
}

impl UserDefinedLogicalNodeCore for AggregateWithMetadata {
    fn name(&self) -> &str {
        "AggregateWithMetadata"
    }

    fn inputs(&self) -> Vec<&LogicalPlan> {
        vec![self.input.as_ref()]
    }

    fn schema(&self) -> &DFSchemaRef {
        &self.schema
    }

    fn expressions(&self) -> Vec<Expr> {
        self.group_expr
            .iter()
            .chain(&self.aggr_expr)
            .cloned()
            .collect()
    }

    fn fmt_for_explain(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let meta_pairs: Vec<String> = self
            .metadata
            .iter()
            .map(|(k, v)| format!("{k}={}", elide(v)))
            .collect();
        write!(
            f,
            "AggregateWithMetadata: groupBy=[[{}]], aggr=[[{}]], metadata=[{}]",
            fmt_exprs(&self.group_expr),
            fmt_exprs(&self.aggr_expr),
            meta_pairs.join(", "),
        )
    }

    /// Reports which input columns the aggregate actually reads, letting
    /// `OptimizeProjections` prune the scan beneath us. Without this the
    /// optimizer cannot route requirements through an extension node and
    /// conservatively keeps every input column, which leaves the logical plan
    /// scanning all ~100 columns of a wide pre-aggregation.
    ///
    /// `output_columns` is ignored on purpose: it says which of *our* outputs
    /// the parent wants, but dropping a `group_expr` would change grouping
    /// cardinality and dropping an `aggr_expr` would desynchronise `schema`,
    /// which `with_exprs_and_inputs` clones verbatim. The columns referenced by
    /// all group and aggregate expressions are the tight, correct answer, and
    /// are what DataFusion's own `Aggregate` computes for this case.
    fn necessary_children_exprs(&self, _output_columns: &[usize]) -> Option<Vec<Vec<usize>>> {
        let input_schema = self.input.schema();

        let mut refs = HashSet::new();
        for expr in self.group_expr.iter().chain(&self.aggr_expr) {
            expr.add_column_refs(&mut refs);
        }

        let mut indices = Vec::with_capacity(refs.len());
        for col in refs {
            // An unresolvable reference (an outer/correlated column, say) means
            // we cannot answer safely. `None` falls back to "needs everything".
            indices.push(input_schema.maybe_index_of_column(col)?);
        }
        indices.sort_unstable();
        indices.dedup();
        Some(vec![indices])
    }

    fn with_exprs_and_inputs(&self, exprs: Vec<Expr>, inputs: Vec<LogicalPlan>) -> Result<Self> {
        let n_group = self.group_expr.len();
        Ok(Self {
            input: Arc::new(inputs.into_iter().next().expect("expected one input")),
            group_expr: exprs[..n_group].to_vec(),
            aggr_expr: exprs[n_group..].to_vec(),
            schema: self.schema.clone(),
            metadata: self.metadata.clone(),
        })
    }
}

// ── Physical planner ─────────────────────────────────────────────────────────

struct AggregateWithMetadataExtensionPlanner;

#[async_trait]
impl ExtensionPlanner for AggregateWithMetadataExtensionPlanner {
    async fn plan_extension(
        &self,
        planner: &dyn PhysicalPlanner,
        node: &dyn UserDefinedLogicalNode,
        _logical_inputs: &[&LogicalPlan],
        _physical_inputs: &[Arc<dyn ExecutionPlan>],
        session_state: &SessionState,
    ) -> Result<Option<Arc<dyn ExecutionPlan>>> {
        let Some(agg) = node.as_any().downcast_ref::<AggregateWithMetadata>() else {
            return Ok(None);
        };
        // Use LogicalPlanBuilder so DataFusion's normalization runs (e.g. it
        // splits BinaryExpr of aggregates into individual aggs + a projection,
        // which Aggregate::try_new does not do on its own).
        // Rebuild via LogicalPlanBuilder and run the logical optimizer before
        // physical planning. This is necessary for complex aggregate expressions
        // (e.g. sum(A) / sum(B)) which the optimizer normalizes into separate
        // aggregates + a projection. Skipping the optimizer causes the physical
        // aggregate planner to reject BinaryExpr inside aggr_expr.
        let input_plan = Arc::unwrap_or_clone(agg.input.clone());
        let logical_agg = LogicalPlanBuilder::from(input_plan)
            .aggregate(agg.group_expr.clone(), agg.aggr_expr.clone())?
            .build()?;
        let optimized = session_state.optimize(&logical_agg)?;
        let physical = planner
            .create_physical_plan(&optimized, session_state)
            .await?;
        Ok(Some(physical))
    }
}

/// A `QueryPlanner` that wraps `DefaultPhysicalPlanner` with support for
/// `AggregateWithMetadata` extension nodes. Register this on the session
/// state so `AggregateWithMetadata` nodes can be executed.
#[derive(Debug)]
pub struct AggregateWithMetadataPlanner;

#[async_trait]
impl QueryPlanner for AggregateWithMetadataPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session_state: &SessionState,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::with_extension_planners(vec![Arc::new(
            AggregateWithMetadataExtensionPlanner,
        )])
        .create_physical_plan(logical_plan, session_state)
        .await
    }
}

// ── MetadataDataFrame ─────────────────────────────────────────────────────────

/// Wraps a DataFusion `DataFrame` and intercepts `aggregate()` to inject an
/// `AggregateWithMetadata` node carrying the supplied metadata string map.
pub struct MetadataDataFrame(DataFrame);

impl MetadataDataFrame {
    pub fn new(df: DataFrame) -> Self {
        Self(df)
    }

    pub fn into_inner(self) -> DataFrame {
        self.0
    }

    pub fn filter(self, predicate: Expr) -> Result<Self> {
        Ok(Self(self.0.filter(predicate)?))
    }

    pub fn with_columns(self, exprs: Vec<Expr>) -> Result<Self> {
        let mut df = self.0;
        for expr in exprs {
            let (name, inner) = if let Expr::Alias(a) = expr {
                (a.name, *a.expr)
            } else {
                (format!("{expr}"), expr)
            };
            df = df.with_column(&name, inner)?;
        }
        Ok(Self(df))
    }

    pub fn sort(self, exprs: Vec<Expr>) -> Result<Self> {
        Ok(Self(self.0.sort_by(exprs)?))
    }

    pub fn limit(self, skip: usize, fetch: Option<usize>) -> Result<Self> {
        Ok(Self(self.0.limit(skip, fetch)?))
    }

    pub fn join(
        self,
        right: DataFrame,
        join_type: datafusion::logical_expr::JoinType,
        left_cols: &[&str],
        right_cols: &[&str],
        filter: Option<Expr>,
    ) -> Result<Self> {
        Ok(Self(
            self.0
                .join(right, join_type, left_cols, right_cols, filter)?,
        ))
    }

    pub fn distinct(self) -> Result<Self> {
        Ok(Self(self.0.distinct()?))
    }

    pub fn distinct_on(
        self,
        on_expr: Vec<Expr>,
        select_expr: Vec<Expr>,
        sort_expr: Option<Vec<SortExpr>>,
    ) -> Result<Self> {
        Ok(Self(self.0.distinct_on(on_expr, select_expr, sort_expr)?))
    }

    pub fn drop_columns(self, columns: Vec<Column>) -> Result<Self> {
        Ok(Self(self.0.drop_columns(&columns)?))
    }

    pub fn union(self, right: DataFrame) -> Result<Self> {
        Ok(Self(self.0.union(right)?))
    }

    pub fn union_distinct(self, right: DataFrame) -> Result<Self> {
        Ok(Self(self.0.union_distinct(right)?))
    }

    pub fn window(self, window_exprs: Vec<Expr>) -> Result<Self> {
        Ok(Self(self.0.window(window_exprs)?))
    }

    /// Call `aggregate()` and attach `metadata` to the resulting logical plan node.
    /// The caller is responsible for including `"allow_exclude"` in the metadata map.
    pub fn aggregate(
        self,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<Expr>,
        metadata: BTreeMap<String, String>,
    ) -> Result<DataFrame> {
        let df = self.0.aggregate(group_expr, aggr_expr)?;
        let (state, plan) = df.into_parts();
        let annotated = AggregateWithMetadata::inject(plan, metadata)?;
        Ok(DataFrame::new(state, annotated))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Metadata values longer than this are elided in explain output.
const MAX_METADATA_VALUE: usize = 40;

/// Shortens a long metadata value for explain output, keeping a digest so the
/// rendering still distinguishes values that differ.
///
/// The digest matters beyond cosmetics: `plan_merge_key` identifies a measure's
/// mergeable slot by display-formatting its input subplan, so if a rolled-up
/// pre-aggregation ever nests one of these nodes inside another's input, this
/// text becomes part of that key. Eliding values outright would collapse two
/// plans that differ only in metadata into one key and merge measures that
/// should have stayed separate.
fn elide(value: &str) -> String {
    if value.len() <= MAX_METADATA_VALUE {
        return value.to_string();
    }
    format!("<{} chars #{:08x}>", value.len(), digest(value))
}

/// FNV-1a. Hand-rolled rather than `DefaultHasher` so the same plan always
/// renders the same text — std makes no cross-run stability guarantee.
fn digest(s: &str) -> u32 {
    let mut hash: u32 = 0x811c_9dc5;
    for byte in s.as_bytes() {
        hash ^= *byte as u32;
        hash = hash.wrapping_mul(0x0100_0193);
    }
    hash
}

pub fn fmt_exprs(exprs: &[Expr]) -> String {
    exprs
        .iter()
        .map(|e| format!("{e}"))
        .collect::<Vec<_>>()
        .join(", ")
}
