use std::collections::HashSet;
use std::sync::Arc;

use datafusion::arrow::datatypes::Field;
use datafusion::common::{Column, TableReference};
use datafusion::logical_expr::{
    JoinType, LogicalPlan, LogicalPlanBuilder, SortExpr, SubqueryAlias,
};
use datafusion::prelude::{DataFrame, Expr, col};

use crate::{
    column_expressions::filter_expr::json_to_expr,
    model_components::{
        agg_context::AggContext,
        measures::{MeasureMetadata, resolve_group_by_cols},
    },
};

use super::merge_optimizer::merge_measure_dfs;

use super::DataModel;

impl DataModel {
    pub(super) fn build_agg_frame(&self, qc: &AggContext) -> Result<DataFrame, String> {
        let known_measures: Vec<MeasureMetadata> =
            self.0.measure_metadata.values().cloned().collect();
        let all_columns: HashSet<String> = self
            .0
            .table_providers
            .iter()
            .flat_map(|(name, provider)| {
                let prefix = format!("{name}.");
                provider
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| {
                        if f.name().starts_with(&prefix) {
                            f.name().to_string()
                        } else {
                            format!("{prefix}{}", f.name())
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .collect();
        qc.validate(&known_measures, &all_columns)?;

        // Build all measure DataFrames before any joining.
        let mut measure_dfs: Vec<(String, DataFrame)> = Vec::with_capacity(qc.measures.len());
        for m_name in &qc.measures {
            let df = self
                .0
                .measures
                .get(m_name)
                .unwrap()
                .call(self, qc)
                .map_err(|e| format!("measure '{m_name}' failed: {e}"))?;
            measure_dfs.push((m_name.clone(), df));
        }

        // Validate group-by compatibility upfront: all measures must resolve to the
        // same group-by columns so the FULL JOIN chain has consistent join keys.
        let join_cols = resolve_group_by_cols(&measure_dfs[0].1)?;
        for (m_name, df) in &measure_dfs[1..] {
            let cols = resolve_group_by_cols(df)?;
            if cols != join_cols {
                return Err(format!(
                    "incompatible group-by columns across measures: '{m_name}' resolved {:?} but expected {:?}",
                    cols, join_cols
                ));
            }
        }

        // Merge compatible measures (identical input subplan). Group-by equivalence is
        // already guaranteed above, so the merge key only needs to cover the input plan.
        let merged = merge_measure_dfs(measure_dfs, &self.0.ctx)?;

        let mut merged_iter = merged.into_iter().enumerate();
        let (_, first) = merged_iter.next().unwrap();

        // Flatten the first measure's output: qualified group-by columns
        // (e.g. Column{relation:"orders", name:"date"}) become flat dot-named aliases
        // ("orders.date" with no qualifier). This makes them consistently referenceable
        // by Column::from_name throughout the rest of build_agg_frame.
        let mut combined = flatten_df(first)?;

        // Chain FULL JOINs for any remaining (non-merged) DataFrames.
        // enumerate() resumes at index 1 after consuming the first element, so
        // right_idx matches the alias "m1", "m2", ... that full_join_right_measure uses.
        for (right_idx, right_df) in merged_iter {
            combined =
                full_join_right_measure(&self.0.ctx, combined, right_df, &join_cols, right_idx)?;
        }

        if let Some(having_expr) = json_to_expr(&qc.havings) {
            combined = combined
                .filter(having_expr)
                .map_err(|e| format!("having filter failed: {e}"))?;
        }

        if !qc.sorts.is_empty() {
            let sort_exprs: Vec<SortExpr> = qc
                .sorts
                .iter()
                .map(|(c, d)| Expr::Column(Column::from_name(c.as_str())).sort(d != "desc", true))
                .collect();
            combined = combined
                .sort(sort_exprs)
                .map_err(|e| format!("sort failed: {e}"))?;
        }

        combined
            .limit(qc.offset, Some(qc.limit))
            .map_err(|e| format!("limit failed: {e}"))
    }
}

/// Convert a measure DataFrame's schema to flat dot-named aliases.
///
/// Qualified columns like Column{relation:"orders", name:"date"} become an alias
/// "orders.date" with no relation qualifier. Unqualified columns pass through
/// unchanged. This ensures group-by columns are consistently referenceable via
/// Column::from_name("orders.date") in havings, sorts, and subsequent joins.
pub fn flatten_df(df: DataFrame) -> Result<DataFrame, String> {
    let exprs: Vec<Expr> = df
        .schema()
        .iter()
        .map(|(qualifier, field)| match qualifier {
            Some(q) => {
                let name = format!("{}.{}", q, field.name());
                col(name.as_str()).alias(name.as_str())
            }
            None => Expr::Column(Column::from_name(field.name())),
        })
        .collect();
    df.select(exprs)
        .map_err(|e| format!("schema flatten failed: {e}"))
}

/// FULL JOIN a new measure DataFrame into the accumulating combined result.
///
/// The left side always has flat dot-named aliases at this point (from flatten_df
/// or a prior call to this function). Only the right side receives a SubqueryAlias
/// ("m{right_idx}") to give its group-by columns distinct qualifiers, preventing
/// the "duplicate qualified field name" error DataFusion raises on FULL JOIN when
/// both sides carry the same qualified column.
///
/// After joining, a projection COALESCEs each group-by column pair and strips all
/// qualifiers so the combined result stays consistently flat for subsequent joins.
fn full_join_right_measure(
    ctx: &datafusion::prelude::SessionContext,
    left: DataFrame,
    right: DataFrame,
    join_cols: &[String],
    right_idx: usize,
) -> Result<DataFrame, String> {
    let right_alias = format!("m{right_idx}");

    // Capture right's schema before aliasing to identify measure-specific columns.
    let right_schema_pre_alias: Vec<(Option<TableReference>, Arc<Field>)> = right
        .schema()
        .iter()
        .map(|(q, f)| (q.cloned(), f.clone()))
        .collect();

    // Apply SubqueryAlias to right only. After this, right's "orders.date" becomes "m1.date".
    let right_plan = Arc::new(right.into_unoptimized_plan());
    let aliased_right = LogicalPlan::SubqueryAlias(
        SubqueryAlias::try_new(right_plan, right_alias.as_str())
            .map_err(|e| format!("subquery alias failed: {e}"))?,
    );

    // Left join keys: left has flat aliases → Column::from_name("orders.date").
    // LogicalPlanBuilder::join takes Vec<impl Into<Column>>, so pass Column directly.
    let left_keys: Vec<Column> = join_cols
        .iter()
        .map(|jc| Column::from_name(jc.as_str()))
        .collect();

    // Right join keys: after SubqueryAlias("m1"), "orders.date" is accessed as "m1.date".
    // String implements Into<Column> via from_qualified_name, so "m1.date" → relation="m1", name="date".
    let right_keys: Vec<String> = join_cols
        .iter()
        .map(|jc| {
            let field = jc.split_once('.').map(|(_, n)| n).unwrap_or(jc.as_str());
            format!("{right_alias}.{field}")
        })
        .collect();

    let left_plan = left.into_unoptimized_plan();
    let joined_plan = LogicalPlanBuilder::from(left_plan)
        .join(aliased_right, JoinType::Full, (left_keys, right_keys), None)
        .map_err(|e| format!("full join failed: {e}"))?
        .build()
        .map_err(|e| format!("join plan build failed: {e}"))?;

    let joined = DataFrame::new(ctx.state(), joined_plan);
    build_post_join_projection(joined, &right_schema_pre_alias, join_cols, &right_alias)
}

/// Projection applied after each FULL JOIN.
///
/// - Join-key columns: COALESCE(left_flat_alias, right_m_alias) → flat alias.
/// - Left non-join-key columns: pass through unchanged (already flat).
/// - Right non-join-key columns: alias from "m{i}.col" → flat "col".
///   Columns already seen on the left are skipped to avoid duplicates.
fn build_post_join_projection(
    joined: DataFrame,
    right_schema_pre_alias: &[(Option<TableReference>, Arc<Field>)],
    join_cols: &[String],
    right_alias: &str,
) -> Result<DataFrame, String> {
    let join_col_set: HashSet<&str> = join_cols.iter().map(|s| s.as_str()).collect();

    let mut proj_exprs: Vec<Expr> = Vec::new();
    let mut seen_measure_cols: HashSet<String> = HashSet::new();

    // 1. COALESCE each join-key column pair (left flat alias + right aliased field).
    for jc in join_cols {
        let field = jc.split_once('.').map(|(_, n)| n).unwrap_or(jc.as_str());
        let left_expr = Expr::Column(Column::from_name(jc.as_str()));
        let right_expr = col(format!("{right_alias}.{field}").as_str());
        proj_exprs.push(case_coalesce(left_expr, right_expr).alias(jc.as_str()));
    }

    // 2. Pass through left's non-join-key columns (measure outputs, already flat aliases).
    for (qualifier, field) in joined.schema().iter() {
        // Skip join-key flat aliases (handled above).
        if qualifier.is_none() && join_col_set.contains(field.name() as &str) {
            continue;
        }
        // Skip right-side columns (those with the SubqueryAlias qualifier).
        if qualifier
            .as_ref()
            .map(|q| q.table() == right_alias)
            .unwrap_or(false)
        {
            continue;
        }
        // This is a left-side non-join-key column (flat alias, no qualifier expected).
        let col_name = field.name().to_string();
        seen_measure_cols.insert(col_name.clone());
        proj_exprs
            .push(Expr::Column(Column::from_name(col_name.as_str())).alias(col_name.as_str()));
    }

    // 3. Project right's non-join-key columns, skipping any already present on the left.
    for (q, f) in right_schema_pre_alias {
        if is_join_key_in_right_schema(q, f, join_cols) {
            continue;
        }
        let col_name = f.name().to_string();
        if seen_measure_cols.contains(&col_name) {
            continue; // left already provides this column
        }
        // Use explicit Column construction — col("m1.games.game_count") would be
        // misparse as relation="m1.games", name="game_count" if col_name contains a dot.
        proj_exprs.push(
            Expr::Column(Column::new(Some(right_alias.to_string()), col_name.clone()))
                .alias(col_name.as_str()),
        );
    }

    joined
        .select(proj_exprs)
        .map_err(|e| format!("post-join projection failed: {e}"))
}

/// True if the (qualifier, field) pair from the pre-alias right schema is one of the join keys.
fn is_join_key_in_right_schema(
    q: &Option<TableReference>,
    f: &Arc<Field>,
    join_cols: &[String],
) -> bool {
    join_cols.iter().any(|jc| {
        if let Some((table, col_name)) = jc.split_once('.') {
            q.as_ref().map(|qr| qr.table() == table).unwrap_or(false) && f.name() == col_name
        } else {
            q.is_none() && f.name() == jc.as_str()
        }
    })
}

/// CASE WHEN a IS NOT NULL THEN a ELSE b END — equivalent to COALESCE(a, b).
/// Used to pick the non-NULL value from each side of a FULL JOIN for group-by columns.
fn case_coalesce(a: Expr, b: Expr) -> Expr {
    use datafusion::logical_expr::expr::Case;
    Expr::Case(Case {
        expr: None,
        when_then_expr: vec![(Box::new(a.clone().is_not_null()), Box::new(a))],
        else_expr: Some(Box::new(b)),
    })
}
