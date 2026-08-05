use std::collections::{HashMap, HashSet};

use datafusion::prelude::{DataFrame, Expr, col};

use crate::{
    column_expressions::filter_expr::json_to_expr, model_components::select_context::SelectContext,
};

use super::DataModel;

impl DataModel {
    pub(super) fn build_select_frame(&self, vc: &SelectContext) -> Result<DataFrame, String> {
        let all_columns = self.known_qualified_columns();
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

        let base_table = self.0.joins.find_reachable_base(&referenced_tables)?;

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
            df = df
                .sort(super::sort_exprs(&vc.sorts))
                .map_err(|e| e.to_string())?;
        }

        df.limit(vc.offset, Some(vc.limit))
            .map_err(|e| e.to_string())
    }
}
