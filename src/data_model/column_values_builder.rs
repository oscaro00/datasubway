use std::collections::{HashMap, HashSet};

use datafusion::common::Column;
use datafusion::prelude::{DataFrame, Expr, col};
use tracing::{debug, trace};

use crate::model_components::column_values_context::ColumnValuesContext;
use crate::model_components::pre_aggregations::resolve_pre_agg_path;

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

        if ctx.use_pre_agg {
            if let (Some(pre_aggs_lock), Some(path)) =
                (&self.0.pre_aggs, self.0.pre_agg_path.as_deref())
            {
                let pre_aggs = pre_aggs_lock.read().unwrap();
                let mut candidates: Vec<_> = pre_aggs
                    .iter()
                    .filter(|pa| pa.group_by.contains(&ctx.column))
                    .collect();
                candidates.sort_by_key(|pa| pa.row_count);

                'candidates: for candidate in candidates {
                    let Some(pre_agg_file) = resolve_pre_agg_path(path, &candidate.name) else {
                        debug!(pre_agg = %candidate.name, "no current pointer, trying next");
                        continue 'candidates;
                    };
                    if let Some(max_age) = ctx.pre_agg_valid_secs {
                        match std::fs::metadata(&pre_agg_file)
                            .ok()
                            .and_then(|m| m.modified().ok())
                        {
                            Some(modified) => {
                                let age = modified.elapsed().unwrap_or(std::time::Duration::MAX);
                                if age > std::time::Duration::from_secs(max_age) {
                                    continue 'candidates;
                                }
                            }
                            None => continue 'candidates,
                        }
                    }
                    if std::path::Path::new(&pre_agg_file).exists() {
                        debug!(column = %ctx.column, pre_agg = %candidate.name, "using pre-agg for column values");
                        if let Ok(df) = self.read_parquet_sync(&pre_agg_file) {
                            return df
                                .select(vec![Expr::Column(Column::from_name(ctx.column.as_str()))])
                                .and_then(|d| d.distinct())
                                .map_err(|e| e.to_string());
                        }
                    }
                    debug!(pre_agg = %candidate.name, "pre-agg not found, trying next");
                }
                trace!(column = %ctx.column, "no valid pre-agg, falling back");
            }
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

        base.select(vec![col(ctx.column.as_str()).alias(ctx.column.as_str())])
            .and_then(|d| d.distinct())
            .map_err(|e| e.to_string())
    }
}
