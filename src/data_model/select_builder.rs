use std::collections::{HashMap, HashSet};

use datafusion::common::{Column};
use datafusion::logical_expr::SortExpr;
use datafusion::prelude::{DataFrame, Expr, col};

use crate::{
    column_expressions::filter_expr::json_to_expr,
    model_components::select_context::SelectContext,
};

use super::DataModel;

impl DataModel {
    pub(super) fn build_select_frame(&self, vc: &SelectContext) -> Result<DataFrame, String> {
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
        vc.validate(&all_columns)?;

        let mut all_needed: HashSet<String> = vc.columns.iter().cloned().collect();
        for fc in vc.filter_columns() {
            all_needed.insert(fc);
        }

        let mut referenced_tables: Vec<String> = all_needed
            .iter()
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .collect::<HashSet<_>>()
            .into_iter()
            .collect();
        referenced_tables.sort();

        if referenced_tables.is_empty() {
            return Err("columns must be table-qualified (e.g. table.column)".into());
        }

        let base_table = referenced_tables
            .iter()
            .find(|candidate| {
                referenced_tables
                    .iter()
                    .all(|t| t == *candidate || self.0.joins.find_path(candidate, t).is_some())
            })
            .ok_or_else(|| {
                format!(
                    "no single base table can reach all tables {referenced_tables:?} via join graph"
                )
            })?
            .clone();

        let non_agg_str: HashSet<String> = all_needed.clone();
        let mut df = self
            .get_df_table(&base_table, &non_agg_str, &HashMap::new(), false)
            .map_err(|e| e.to_string())?
            .inner;

        if let Some(filter_expr) = json_to_expr(&vc.filters) {
            df = df.filter(filter_expr).map_err(|e| e.to_string())?;
        }

        let select_exprs: Vec<Expr> = vc
            .columns
            .iter()
            .map(|c| col(c.as_str()).alias(c.as_str()))
            .collect();
        df = df.select(select_exprs).map_err(|e| e.to_string())?;

        if !vc.sorts.is_empty() {
            let sort_exprs: Vec<SortExpr> = vc
                .sorts
                .iter()
                .map(|(c, d)| Expr::Column(Column::from_name(c.as_str())).sort(d != "desc", true))
                .collect();
            df = df.sort(sort_exprs).map_err(|e| e.to_string())?;
        }

        df.limit(vc.offset, Some(vc.limit))
            .map_err(|e| e.to_string())
    }
}
