use std::collections::{HashMap, HashSet};

use datafusion::functions_aggregate::expr_fn::{max, min};
use datafusion::prelude::{DataFrame, col};
use tracing::{debug, trace};

use crate::model_components::column_values_context::{ColumnValuesContext, ColumnValuesMode};
use crate::model_components::pre_aggregations::{
    PreAggregation, pre_agg_component_col_name, resolve_fresh_pre_agg_path, to_pre_agg_col_name,
};

use super::DataModel;

impl DataModel {
    pub(super) fn build_column_values_frame(
        &self,
        ctx: &ColumnValuesContext,
    ) -> Result<DataFrame, String> {
        let (table_name, _) = ctx.column.split_once('.').unwrap();
        if !self.0.table_providers.contains_key(table_name) {
            return Err(format!("unknown table: '{table_name}'"));
        }

        if ctx.use_pre_agg
            && let (Some(pre_aggs_lock), Some(path)) =
                (&self.0.pre_aggs, self.0.pre_agg_path.as_deref())
        {
            let pre_aggs = pre_aggs_lock.read().unwrap();
            let found = match ctx.mode {
                ColumnValuesMode::Distinct => self.try_distinct_from_pre_agg(ctx, &pre_aggs, path),
                ColumnValuesMode::Range => self.try_range_from_pre_agg(ctx, &pre_aggs, path),
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
    fn try_distinct_from_pre_agg(
        &self,
        ctx: &ColumnValuesContext,
        pre_aggs: &[PreAggregation],
        path: &str,
    ) -> Option<DataFrame> {
        let mut candidates: Vec<&PreAggregation> = pre_aggs
            .iter()
            .filter(|pa| pa.group_by.contains(&ctx.column))
            .collect();
        candidates.sort_by_key(|pa| pa.row_count);

        'candidates: for candidate in candidates {
            let Some(pre_agg_file) =
                resolve_fresh_pre_agg_path(path, &candidate.name, ctx.pre_agg_valid_secs)
            else {
                debug!(pre_agg = %candidate.name, "no current pointer or file missing/stale, trying next");
                continue 'candidates;
            };
            debug!(column = %ctx.column, pre_agg = %candidate.name, "using pre-agg for column values (distinct)");
            let Ok(df) = self.read_parquet_sync(&pre_agg_file) else {
                debug!(pre_agg = %candidate.name, "pre-agg not found, trying next");
                continue 'candidates;
            };

            // Group-by columns are stored dunder-encoded on disk (write_pre_agg aliases
            // via to_pre_agg_col_name), and read_parquet_sync applies no SubqueryAlias
            // (unlike get_df_table), so the physical field is unqualified. Select it by
            // its physical name, then alias back to the logical dotted name so pre-agg-
            // and raw-table-sourced output share an identical output schema.
            let physical = to_pre_agg_col_name(&ctx.column);
            if let Ok(df) = df
                .select(vec![col(physical.as_str()).alias(ctx.column.as_str())])
                .and_then(|d| d.distinct())
            {
                return Some(df);
            }
            debug!(pre_agg = %candidate.name, "failed to select/distinct, trying next");
        }
        None
    }

    /// Try every pre-agg that can answer a min/max range request for `ctx.column`,
    /// smallest `row_count` first, combining two candidate shapes:
    ///   - `ctx.column` is a group_by key: aggregate the raw stored values.
    ///   - `ctx.column` is an aggregations entry with both "min" and "max" components
    ///     stored: re-aggregate those component columns directly.
    fn try_range_from_pre_agg(
        &self,
        ctx: &ColumnValuesContext,
        pre_aggs: &[PreAggregation],
        path: &str,
    ) -> Option<DataFrame> {
        enum Source {
            GroupBy,
            Aggregation,
        }

        let mut candidates: Vec<(&PreAggregation, Source)> = pre_aggs
            .iter()
            .filter(|pa| pa.group_by.contains(&ctx.column))
            .map(|pa| (pa, Source::GroupBy))
            .chain(pre_aggs.iter().filter_map(|pa| {
                let components = pa.aggregations.get(&ctx.column)?;
                let has_min = components.iter().any(|c| c == "min");
                let has_max = components.iter().any(|c| c == "max");
                (has_min && has_max).then_some((pa, Source::Aggregation))
            }))
            .collect();
        candidates.sort_by_key(|(pa, _)| pa.row_count);

        'candidates: for (candidate, source) in candidates {
            let Some(pre_agg_file) =
                resolve_fresh_pre_agg_path(path, &candidate.name, ctx.pre_agg_valid_secs)
            else {
                debug!(pre_agg = %candidate.name, "no current pointer or file missing/stale, trying next");
                continue 'candidates;
            };
            debug!(column = %ctx.column, pre_agg = %candidate.name, "using pre-agg for column values (range)");
            let Ok(df) = self.read_parquet_sync(&pre_agg_file) else {
                debug!(pre_agg = %candidate.name, "pre-agg not found, trying next");
                continue 'candidates;
            };

            let agg_result = match source {
                Source::GroupBy => {
                    let physical = to_pre_agg_col_name(&ctx.column);
                    df.aggregate(
                        vec![],
                        vec![
                            min(col(physical.as_str())).alias("min"),
                            max(col(physical.as_str())).alias("max"),
                        ],
                    )
                }
                Source::Aggregation => {
                    // No SubqueryAlias qualifier here (unlike build_pre_agg_expr's use
                    // inside get_df_table) — read_parquet_sync gives unqualified physical
                    // field names, so reference the dunder-encoded component columns bare.
                    let min_col = pre_agg_component_col_name(&ctx.column, "min");
                    let max_col = pre_agg_component_col_name(&ctx.column, "max");
                    df.aggregate(
                        vec![],
                        vec![
                            min(col(min_col.as_str())).alias("min"),
                            max(col(max_col.as_str())).alias("max"),
                        ],
                    )
                }
            };

            if let Ok(df) = agg_result {
                return Some(df);
            }
            debug!(pre_agg = %candidate.name, "failed to aggregate min/max, trying next");
        }
        None
    }
}
