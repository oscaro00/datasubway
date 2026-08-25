use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use datafusion::common::Column;
use datafusion::functions_aggregate::expr_fn::{max, min};
use datafusion::logical_expr::Expr;
use datafusion::prelude::{DataFrame, col};
use datafusion::sql::TableReference;
use tracing::{debug, trace};

use crate::model_components::column_values_context::{ColumnValuesContext, ColumnValuesMode};
use crate::model_components::pre_agg_store::PreAggVersion;
use crate::model_components::pre_aggregations::{pre_agg_component_col_name, to_pre_agg_col_name};

use super::DataModel;

/// Reference a physical pre-agg column on a scan of `version`.
///
/// The scan is a `TableScan` qualified by the pre-agg name, so the dunder-encoded
/// physical field must be addressed as `<pre_agg>.<physical>`. Built through
/// `Column::new` rather than a dotted string because the physical names contain
/// `__`, not `.`, and string parsing would split on the wrong separator.
fn pre_agg_col(version: &PreAggVersion, physical: &str) -> Expr {
    Expr::Column(Column::new(
        Some(TableReference::bare(version.name.clone())),
        physical,
    ))
}

impl DataModel {
    pub(super) fn build_column_values_frame(
        &self,
        ctx: &ColumnValuesContext,
    ) -> Result<DataFrame, String> {
        let (table_name, _) = ctx.column.split_once('.').unwrap();
        if !self.0.table_providers.contains_key(table_name) {
            return Err(format!("unknown table: '{table_name}'"));
        }

        if ctx.use_pre_agg && self.0.pre_agg_store.is_some() {
            let found = match ctx.mode {
                ColumnValuesMode::Distinct => self.try_distinct_from_pre_agg(ctx),
                ColumnValuesMode::Range => self.try_range_from_pre_agg(ctx),
            };
            if let Some(df) = found {
                return Ok(df);
            }
            trace!(column = %ctx.column, "no valid pre-agg, falling back");
        }

        let base = self
            .get_df_table(
                table_name,
                &HashSet::from([ctx.column.clone()]),
                &HashMap::new(),
                false,
            )
            .map_err(|e| e.to_string())?
            .inner;

        match ctx.mode {
            ColumnValuesMode::Distinct => base
                .select(vec![col(ctx.column.as_str()).alias(ctx.column.as_str())])
                .and_then(|d| d.distinct())
                .map_err(|e| e.to_string()),
            ColumnValuesMode::Range => base
                .aggregate(
                    vec![],
                    vec![
                        min(col(ctx.column.as_str())).alias("min"),
                        max(col(ctx.column.as_str())).alias("max"),
                    ],
                )
                .map_err(|e| e.to_string()),
        }
    }

    /// Try each group_by-covering pre-agg (smallest `row_count` first), returning the
    /// first that yields a distinct-values frame for `ctx.column`.
    fn try_distinct_from_pre_agg(&self, ctx: &ColumnValuesContext) -> Option<DataFrame> {
        let store = self.0.pre_agg_store.as_ref()?;
        let versions = store.versions_where(ctx.pre_agg_valid_secs, |pa| {
            pa.group_by.contains(&ctx.column)
        });

        'candidates: for version in versions {
            debug!(column = %ctx.column, pre_agg = %version.name, "using pre-agg for column values (distinct)");
            let Ok(df) = self.scan_pre_agg(&version) else {
                debug!(pre_agg = %version.name, "failed to scan pre-agg, trying next");
                continue 'candidates;
            };

            // Group-by columns are stored dunder-encoded on disk (write_pre_agg
            // aliases via to_pre_agg_col_name). Select the physical column, then
            // alias back to the logical dotted name so pre-agg- and raw-table-
            // sourced output share an identical output schema.
            let physical = to_pre_agg_col_name(&ctx.column);
            if let Ok(df) = df
                .select(vec![
                    pre_agg_col(&version, &physical).alias(ctx.column.as_str()),
                ])
                .and_then(|d| d.distinct())
            {
                return Some(df);
            }
            debug!(pre_agg = %version.name, "failed to select/distinct, trying next");
        }
        None
    }

    /// Try every pre-agg that can answer a min/max range request for `ctx.column`,
    /// smallest `row_count` first, combining two candidate shapes:
    ///   - `ctx.column` is a group_by key: aggregate the raw stored values.
    ///   - `ctx.column` is an aggregations entry with both "min" and "max" components
    ///     stored: re-aggregate those component columns directly.
    fn try_range_from_pre_agg(&self, ctx: &ColumnValuesContext) -> Option<DataFrame> {
        #[derive(Clone, Copy)]
        enum Source {
            GroupBy,
            Aggregation,
        }

        let store = self.0.pre_agg_store.as_ref()?;
        let has_min_max = |pa: &crate::model_components::pre_aggregations::PreAggregation| {
            pa.aggregations.get(&ctx.column).is_some_and(|components| {
                components.iter().any(|c| c == "min") && components.iter().any(|c| c == "max")
            })
        };

        // Two candidate shapes, each already ordered cheapest-first by the store:
        // `ctx.column` as a stored group_by key, or as an aggregation with both
        // "min" and "max" components on disk.
        let candidates: Vec<(Arc<PreAggVersion>, Source)> = store
            .versions_where(ctx.pre_agg_valid_secs, |pa| {
                pa.group_by.contains(&ctx.column)
            })
            .into_iter()
            .map(|v| (v, Source::GroupBy))
            .chain(
                store
                    .versions_where(ctx.pre_agg_valid_secs, has_min_max)
                    .into_iter()
                    .map(|v| (v, Source::Aggregation)),
            )
            .collect();

        'candidates: for (version, source) in candidates {
            debug!(column = %ctx.column, pre_agg = %version.name, "using pre-agg for column values (range)");
            let Ok(df) = self.scan_pre_agg(&version) else {
                debug!(pre_agg = %version.name, "failed to scan pre-agg, trying next");
                continue 'candidates;
            };

            let agg_result = match source {
                Source::GroupBy => {
                    let physical = to_pre_agg_col_name(&ctx.column);
                    df.aggregate(
                        vec![],
                        vec![
                            min(pre_agg_col(&version, &physical)).alias("min"),
                            max(pre_agg_col(&version, &physical)).alias("max"),
                        ],
                    )
                }
                Source::Aggregation => {
                    let min_col = pre_agg_component_col_name(&ctx.column, "min");
                    let max_col = pre_agg_component_col_name(&ctx.column, "max");
                    df.aggregate(
                        vec![],
                        vec![
                            min(pre_agg_col(&version, &min_col)).alias("min"),
                            max(pre_agg_col(&version, &max_col)).alias("max"),
                        ],
                    )
                }
            };

            if let Ok(df) = agg_result {
                return Some(df);
            }
            debug!(pre_agg = %version.name, "failed to aggregate min/max, trying next");
        }
        None
    }
}
