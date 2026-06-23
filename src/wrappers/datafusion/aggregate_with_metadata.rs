use async_trait::async_trait;
use datafusion::common::tree_node::{Transformed, TreeNode};
use datafusion::common::{DFSchemaRef, Result};
use datafusion::execution::context::{QueryPlanner, SessionState};
use datafusion::logical_expr::{
    Aggregate, Extension, LogicalPlan, UserDefinedLogicalNode, UserDefinedLogicalNodeCore,
};
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_planner::{DefaultPhysicalPlanner, ExtensionPlanner, PhysicalPlanner};
use datafusion::prelude::{DataFrame, Expr};
use std::collections::BTreeMap;
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
            .map(|(k, v)| format!("{k}={v}"))
            .collect();
        write!(
            f,
            "AggregateWithMetadata: groupBy=[[{}]], aggr=[[{}]], metadata=[{}]",
            fmt_exprs(&self.group_expr),
            fmt_exprs(&self.aggr_expr),
            meta_pairs.join(", "),
        )
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
        let logical_agg = LogicalPlan::Aggregate(Aggregate::try_new(
            agg.input.clone(),
            agg.group_expr.clone(),
            agg.aggr_expr.clone(),
        )?);
        let physical = planner
            .create_physical_plan(&logical_agg, session_state)
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

pub fn fmt_exprs(exprs: &[Expr]) -> String {
    exprs
        .iter()
        .map(|e| format!("{e}"))
        .collect::<Vec<_>>()
        .join(", ")
}
