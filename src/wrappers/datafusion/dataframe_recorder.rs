use std::collections::{BTreeMap, HashMap, HashSet};

use datafusion::logical_expr::{JoinType, SortExpr};
use datafusion::prelude::{DataFrame, Expr};

use crate::column_expressions::column_context::{AllowExcludeRecord, IntoColsExpr, IntoFilterExpr};
use crate::data_model::DataModel;

use super::agg_expr::{
    extract_agg_exprs, qualified_name, resolve_source_col, rewrite_col_name_for_pre_agg,
    rewrite_expr_for_pre_agg, rewrite_for_pre_agg, rewrite_group_for_pre_agg,
};
use super::aggregate_with_metadata::{MetadataDataFrame, fmt_exprs};

// ── Op enum ──────────────────────────────────────────────────────────────────

pub enum DataFrameOp {
    Filter(Expr),
    Join {
        right: DataFrame,
        join_type: JoinType,
        left_cols: Vec<String>,
        right_cols: Vec<String>,
        filter: Option<Expr>,
    },
    WithColumns(Vec<Expr>),
    Aggregate {
        group_expr: Vec<Expr>,
        aggr_expr: Vec<Expr>,
    },
    Sort(Vec<Expr>),
    Limit {
        skip: usize,
        fetch: Option<usize>,
    },
    Distinct,
    DistinctOn {
        on_expr: Vec<Expr>,
        select_expr: Vec<Expr>,
        sort_expr: Option<Vec<SortExpr>>,
    },
    DropColumns(Vec<String>),
    Union(DataFrame),
    UnionDistinct(DataFrame),
    Window(Vec<Expr>),
}

// ── Recorder ─────────────────────────────────────────────────────────────────

/// Records operations on a DataFusion `DataFrame` and builds them lazily.
///
/// Key properties vs the Polars `LazyFrameRecorder`:
/// - `pre_agg_allowed`: set to `false` when `join()` or `with_columns()` is
///   called. If false, the DataModel's `get_df_table()` will skip pre-agg
///   selection for this chain.
/// - `alias_map`: tracks the transitive source column for every alias introduced
///   by `with_columns()` or `aggregate()`, enabling the pre-agg rewriter to find
///   the right component column even after multiple renaming steps.
pub struct DataFrameRecorder {
    pub table_name: String,
    pub data_model: DataModel,
    pub ops: Vec<DataFrameOp>,
    pub non_agg_cols: HashSet<String>,
    pub agg_cols: HashMap<String, Vec<String>>,
    pub allow_exclude_records: Vec<AllowExcludeRecord>,
    /// Set to `false` when `join()` or `with_columns()` is recorded.
    pub pre_agg_allowed: bool,
    /// alias_name → ultimate source column name (follows transitive chains).
    pub alias_map: HashMap<String, String>,
}

impl DataFrameRecorder {
    pub fn new(table_name: String, data_model: DataModel, use_pre_agg: bool) -> Self {
        Self {
            table_name,
            data_model,
            ops: Vec::new(),
            non_agg_cols: HashSet::new(),
            agg_cols: HashMap::new(),
            allow_exclude_records: Vec::new(),
            pre_agg_allowed: use_pre_agg,
            alias_map: HashMap::new(),
        }
    }

    // ── Methods ───────────────────────────────────────────────────────────────

    pub fn filter(mut self, predicate: impl IntoFilterExpr) -> Self {
        if let Some(pred) = predicate.into_filter(&mut self.allow_exclude_records) {
            if let Some(pruned) = prune_filter_by_tables(pred, &self.table_name, &self.data_model) {
                collect_col_names(&pruned, &mut self.non_agg_cols);
                self.ops.push(DataFrameOp::Filter(pruned));
            }
        }
        self
    }

    /// Records a DataFusion join. Sets `pre_agg_allowed = false` because the
    /// extra columns brought in by the join make pre-agg component matching
    /// unreliable for this specific chain.
    pub fn join(
        mut self,
        right: DataFrame,
        join_type: JoinType,
        left_cols: Vec<String>,
        right_cols: Vec<String>,
        filter: Option<Expr>,
    ) -> Self {
        self.pre_agg_allowed = false;
        self.ops.push(DataFrameOp::Join {
            right,
            join_type,
            left_cols,
            right_cols,
            filter,
        });
        self
    }

    /// Records `with_columns()`. Updates the alias map so downstream agg
    /// expressions can be resolved correctly. Pre-agg remains allowed because
    /// `rewrite_for_pre_agg` / `rewrite_group_for_pre_agg` resolve aliases via
    /// `alias_map`, and `build()` skips `WithColumns` ops when using a pre-agg
    /// (pre-agg schemas only have dunder-named component columns).
    pub fn with_columns(mut self, exprs: Vec<Expr>) -> Self {
        for expr in &exprs {
            update_alias_map(&mut self.alias_map, expr);
        }
        self.ops.push(DataFrameOp::WithColumns(exprs));
        self
    }

    /// Records `aggregate()`. This is the terminal recording operation for measures.
    /// Updates `agg_cols` and `alias_map` from the aggregate expressions.
    pub fn aggregate(mut self, group_expr: impl IntoColsExpr, aggr_expr: Vec<Expr>) -> Self {
        let group_exprs = group_expr.into_exprs_with_record(&mut self.allow_exclude_records);

        for expr in &group_exprs {
            let mut raw = HashSet::new();
            collect_col_names(expr, &mut raw);
            for name in raw {
                self.non_agg_cols
                    .insert(resolve_source_col(&name, &self.alias_map));
            }
        }

        for expr in &aggr_expr {
            update_alias_map(&mut self.alias_map, expr);
            let pairs = extract_agg_exprs(expr);
            let agg_col_set: HashSet<&str> = pairs.iter().map(|(c, _)| c.as_str()).collect();
            collect_col_names_filtered(expr, &agg_col_set, &mut self.non_agg_cols);
            for (col_name, agg_name) in pairs {
                let resolved = resolve_source_col(&col_name, &self.alias_map);
                self.agg_cols.entry(resolved).or_default().push(agg_name);
            }
        }

        self.ops.push(DataFrameOp::Aggregate {
            group_expr: group_exprs,
            aggr_expr,
        });
        self
    }

    pub fn sort(mut self, exprs: Vec<Expr>) -> Self {
        for expr in &exprs {
            collect_col_names(expr, &mut self.non_agg_cols);
        }
        self.ops.push(DataFrameOp::Sort(exprs));
        self
    }

    pub fn limit(mut self, skip: usize, fetch: Option<usize>) -> Self {
        self.ops.push(DataFrameOp::Limit { skip, fetch });
        self
    }

    pub fn distinct(mut self) -> Self {
        self.ops.push(DataFrameOp::Distinct);
        self
    }

    pub fn distinct_on(
        mut self,
        on_expr: Vec<Expr>,
        select_expr: Vec<Expr>,
        sort_expr: Option<Vec<SortExpr>>,
    ) -> Self {
        for e in on_expr.iter().chain(select_expr.iter()) {
            collect_col_names(e, &mut self.non_agg_cols);
        }
        self.ops.push(DataFrameOp::DistinctOn {
            on_expr,
            select_expr,
            sort_expr,
        });
        self
    }

    pub fn drop_columns(mut self, columns: Vec<String>) -> Self {
        self.ops.push(DataFrameOp::DropColumns(columns));
        self
    }

    pub fn union(mut self, right: DataFrame) -> Self {
        self.ops.push(DataFrameOp::Union(right));
        self
    }

    pub fn union_distinct(mut self, right: DataFrame) -> Self {
        self.ops.push(DataFrameOp::UnionDistinct(right));
        self
    }

    /// Window functions operate on raw row-level data and cannot be applied to
    /// pre-aggregated tables, so this sets `pre_agg_allowed = false`.
    pub fn window(mut self, window_exprs: Vec<Expr>) -> Self {
        self.pre_agg_allowed = false;
        for e in &window_exprs {
            collect_col_names(e, &mut self.non_agg_cols);
        }
        self.ops.push(DataFrameOp::Window(window_exprs));
        self
    }

    // ── Build ─────────────────────────────────────────────────────────────────

    /// Execute all recorded operations against the appropriate base table
    /// (raw or pre-aggregated) and return the resulting DataFusion DataFrame.
    ///
    /// If the base table comes from a pre-aggregation file, aggregate expressions
    /// are automatically rewritten to reference pre-computed component columns.
    pub fn build(self) -> datafusion::common::Result<DataFrame> {
        let base = self.data_model.get_df_table(
            &self.table_name,
            &self.non_agg_cols,
            &self.agg_cols,
            self.pre_agg_allowed,
        )?;

        let from_pre_agg = base.from_pre_agg;
        let pre_agg_name = base.pre_agg_name.clone().unwrap_or_default();
        let alias_map = self.alias_map.clone();
        let allow_exclude_records = self.allow_exclude_records.clone();

        let mut mdf = MetadataDataFrame::new(base.inner);

        for op in self.ops {
            mdf = match op {
                DataFrameOp::Filter(pred) => {
                    let pred = if from_pre_agg {
                        rewrite_expr_for_pre_agg(pred, &alias_map, &pre_agg_name)?
                    } else {
                        pred
                    };
                    mdf.filter(pred)?
                }

                DataFrameOp::Join {
                    right,
                    join_type,
                    left_cols,
                    right_cols,
                    filter,
                } => {
                    let l: Vec<&str> = left_cols.iter().map(|s| s.as_str()).collect();
                    let r: Vec<&str> = right_cols.iter().map(|s| s.as_str()).collect();
                    mdf.join(right, join_type, &l, &r, filter)?
                }

                DataFrameOp::WithColumns(exprs) => {
                    if from_pre_agg {
                        mdf
                    } else {
                        mdf.with_columns(exprs)?
                    }
                }

                DataFrameOp::Aggregate {
                    group_expr,
                    aggr_expr,
                } => {
                    let final_group = if from_pre_agg {
                        group_expr
                            .into_iter()
                            .map(|e| rewrite_group_for_pre_agg(e, &alias_map, &pre_agg_name))
                            .collect()
                    } else {
                        group_expr
                    };
                    let final_aggr = if from_pre_agg {
                        aggr_expr
                            .into_iter()
                            .map(|e| rewrite_for_pre_agg(e, &alias_map, &pre_agg_name))
                            .collect()
                    } else {
                        aggr_expr
                    };

                    let ae_json = serde_json::to_string(&allow_exclude_records).unwrap_or_default();
                    let metadata = BTreeMap::from([
                        ("allow_exclude".to_string(), ae_json),
                        ("group_by".to_string(), fmt_exprs(&final_group)),
                        ("aggregates".to_string(), fmt_exprs(&final_aggr)),
                    ]);

                    let df = mdf.aggregate(final_group, final_aggr, metadata)?;
                    return Ok(df);
                }

                DataFrameOp::Sort(exprs) => mdf.sort(exprs)?,

                DataFrameOp::Limit { skip, fetch } => mdf.limit(skip, fetch)?,

                DataFrameOp::Distinct => mdf.distinct()?,

                DataFrameOp::DistinctOn {
                    on_expr,
                    select_expr,
                    sort_expr,
                } => {
                    let (on_expr, select_expr) = if from_pre_agg {
                        (
                            on_expr
                                .into_iter()
                                .map(|e| rewrite_expr_for_pre_agg(e, &alias_map, &pre_agg_name))
                                .collect::<datafusion::common::Result<Vec<_>>>()?,
                            select_expr
                                .into_iter()
                                .map(|e| rewrite_expr_for_pre_agg(e, &alias_map, &pre_agg_name))
                                .collect::<datafusion::common::Result<Vec<_>>>()?,
                        )
                    } else {
                        (on_expr, select_expr)
                    };
                    mdf.distinct_on(on_expr, select_expr, sort_expr)?
                }

                DataFrameOp::DropColumns(cols) => {
                    let cols = if from_pre_agg {
                        cols.iter()
                            .map(|c| rewrite_col_name_for_pre_agg(c, &alias_map, &pre_agg_name))
                            .collect()
                    } else {
                        cols
                    };
                    mdf.drop_columns(cols)?
                }

                DataFrameOp::Union(right) => mdf.union(right)?,

                DataFrameOp::UnionDistinct(right) => mdf.union_distinct(right)?,

                DataFrameOp::Window(exprs) => mdf.window(exprs)?,
            };
        }

        Ok(mdf.into_inner())
    }
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Update `alias_map` with any alias introduced at the top level of `expr`.
/// Only tracks column aliases (e.g. `col("x").alias("y")`), not aggregate
/// aliases (e.g. `sum(col("x")).alias("y")`), to avoid poisoning the map
/// with expression strings that aren't source column names.
pub(crate) fn update_alias_map(alias_map: &mut HashMap<String, String>, expr: &Expr) {
    if let Expr::Alias(a) = expr {
        if let Some(resolved) = resolve_expr_to_source(&a.expr, alias_map) {
            alias_map.insert(a.name.to_string(), resolved);
        }
    }
}

/// Returns the ultimate source column name reachable from `expr`, or `None`
/// if `expr` is not a column or simple alias/cast chain over a column.
fn resolve_expr_to_source(expr: &Expr, alias_map: &HashMap<String, String>) -> Option<String> {
    match expr {
        Expr::Column(c) => Some(resolve_source_col(&qualified_name(c), alias_map)),
        Expr::Alias(a) => resolve_expr_to_source(&a.expr, alias_map),
        Expr::Cast(c) => resolve_expr_to_source(&c.expr, alias_map),
        Expr::TryCast(c) => resolve_expr_to_source(&c.expr, alias_map),
        _ => None,
    }
}

/// Collect all `Expr::Column` names from `expr` into `out`.
///
/// Delegates to DataFusion's own `Expr::column_refs`, which walks the full
/// expression tree via the generic `TreeNode` traversal — exhaustive over
/// every `Expr` variant (including `Case`/`when...otherwise`, `Between`,
/// `InList`, etc.) by construction, rather than a hand-maintained allowlist
/// that silently misses whichever variant nobody thought to add.
pub(crate) fn collect_col_names(expr: &Expr, out: &mut HashSet<String>) {
    for c in expr.column_refs() {
        out.insert(qualified_name(c));
    }
}

fn collect_col_names_filtered(expr: &Expr, agg_col_set: &HashSet<&str>, out: &mut HashSet<String>) {
    for c in expr.column_refs() {
        let full = qualified_name(c);
        if !agg_col_set.contains(full.as_str()) {
            out.insert(full);
        }
    }
}

/// Port of the Polars `prune_filter_by_tables`. Walks an AND/OR expression
/// tree and drops branches that reference tables unreachable from `base_table`
/// via the join graph.
fn prune_filter_by_tables(expr: Expr, base_table: &str, data_model: &DataModel) -> Option<Expr> {
    use datafusion::logical_expr::Operator;

    match expr {
        Expr::BinaryExpr(ref b) if b.op == Operator::And => {
            let Expr::BinaryExpr(b) = expr else {
                unreachable!()
            };
            let l = prune_filter_by_tables(*b.left, base_table, data_model);
            let r = prune_filter_by_tables(*b.right, base_table, data_model);
            match (l, r) {
                (Some(l), Some(r)) => Some(l.and(r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        Expr::BinaryExpr(ref b) if b.op == Operator::Or => {
            let Expr::BinaryExpr(b) = expr else {
                unreachable!()
            };
            let l = prune_filter_by_tables(*b.left, base_table, data_model);
            let r = prune_filter_by_tables(*b.right, base_table, data_model);
            match (l, r) {
                (Some(l), Some(r)) => Some(l.or(r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        other => {
            let mut cols = HashSet::new();
            collect_col_names(&other, &mut cols);
            let all_reachable = cols.iter().all(|col_name| match col_name.split_once('.') {
                Some((table, _)) => table == base_table || data_model.can_join(base_table, table),
                None => true,
            });
            if all_reachable { Some(other) } else { None }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use datafusion::prelude::{col, lit};

    fn make_dm() -> DataModel {
        use crate::model_components::joins::{Join, JoinDirection, JoinGraph, JoinHow};
        use datafusion::arrow::array::{Float64Array, StringArray};
        use datafusion::arrow::datatypes::{DataType, Field, Schema};
        use datafusion::arrow::record_batch::RecordBatch;
        use datafusion::datasource::MemTable;
        use std::sync::Arc;

        let orders_schema = Arc::new(Schema::new(vec![
            Field::new("amount", DataType::Float64, true),
            Field::new("region", DataType::Utf8, true),
        ]));
        let customers_schema = Arc::new(Schema::new(vec![
            Field::new("region", DataType::Utf8, true),
            Field::new("country", DataType::Utf8, true),
        ]));
        let products_schema = Arc::new(Schema::new(vec![Field::new(
            "price",
            DataType::Float64,
            true,
        )]));

        let orders_batch = RecordBatch::try_new(
            orders_schema.clone(),
            vec![
                Arc::new(Float64Array::from(vec![100.0, 200.0])),
                Arc::new(StringArray::from(vec!["north", "south"])),
            ],
        )
        .unwrap();
        let customers_batch = RecordBatch::try_new(
            customers_schema.clone(),
            vec![
                Arc::new(StringArray::from(vec!["north", "south"])),
                Arc::new(StringArray::from(vec!["US", "UK"])),
            ],
        )
        .unwrap();
        let products_batch = RecordBatch::try_new(
            products_schema.clone(),
            vec![Arc::new(Float64Array::from(vec![10.0, 20.0]))],
        )
        .unwrap();

        let joins = vec![Join {
            left: "orders".into(),
            right: "customers".into(),
            left_on: vec!["orders.region".into()],
            right_on: vec!["customers.region".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
        }];

        let tables = std::collections::HashMap::from([
            (
                "orders".to_string(),
                Arc::new(MemTable::try_new(orders_schema, vec![vec![orders_batch]]).unwrap())
                    as Arc<dyn datafusion::catalog::TableProvider>,
            ),
            (
                "customers".to_string(),
                Arc::new(MemTable::try_new(customers_schema, vec![vec![customers_batch]]).unwrap())
                    as Arc<dyn datafusion::catalog::TableProvider>,
            ),
            (
                "products".to_string(),
                Arc::new(MemTable::try_new(products_schema, vec![vec![products_batch]]).unwrap())
                    as Arc<dyn datafusion::catalog::TableProvider>,
            ),
        ]);

        DataModel::new(tables, JoinGraph::new(&joins).unwrap(), vec![], None)
    }

    fn filter_ops(recorder: DataFrameRecorder) -> Vec<Expr> {
        recorder
            .ops
            .into_iter()
            .filter_map(|op| match op {
                DataFrameOp::Filter(e) => Some(e),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_reachable_filter_kept() {
        let dm = make_dm();
        let rec = dm
            .table("orders", true)
            .filter(col("customers.country").eq(lit("US")));
        assert_eq!(filter_ops(rec).len(), 1);
    }

    #[test]
    fn test_unreachable_filter_dropped() {
        let dm = make_dm();
        let rec = dm
            .table("orders", true)
            .filter(col("products.price").lt(lit(50.0f64)));
        assert_eq!(filter_ops(rec).len(), 0);
    }

    #[test]
    fn test_compound_and_partial_prune() {
        let dm = make_dm();
        let rec = dm.table("orders", true).filter(
            col("orders.amount")
                .gt(lit(0.0f64))
                .and(col("products.price").lt(lit(50.0f64))),
        );
        let ops = filter_ops(rec);
        assert_eq!(ops.len(), 1);
        let mut cols = HashSet::new();
        collect_col_names(&ops[0], &mut cols);
        assert!(cols.contains("orders.amount"));
        assert!(!cols.contains("products.price"));
    }

    #[test]
    fn test_join_disables_pre_agg() {
        // join() sets pre_agg_allowed = false; with_columns alone does not
        let dm = make_dm();
        let rec_with_cols = dm
            .table("orders", true)
            .with_columns(vec![col("orders.amount").alias("amt")]);
        assert!(
            rec_with_cols.pre_agg_allowed,
            "with_columns alone should not disable pre-agg"
        );

        let customers_df = {
            let dm2 = make_dm();
            dm2.table("customers", true).build().unwrap()
        };
        let rec_join = make_dm().table("orders", true).join(
            customers_df,
            JoinType::Left,
            vec!["orders.region".into()],
            vec!["customers.region".into()],
            None,
        );
        assert!(
            !rec_join.pre_agg_allowed,
            "join should still disable pre-agg"
        );
    }

    #[test]
    fn test_agg_cols_resolved_through_alias_map() {
        use datafusion::functions_aggregate::expr_fn::sum;

        let dm = make_dm();
        let rec = dm
            .table("orders", true)
            .with_columns(vec![col("orders.amount").alias("amt")])
            .aggregate(
                vec![col("orders.region")],
                vec![sum(col("amt")).alias("total")],
            );

        // agg_cols key should be the resolved source name, not the alias
        assert!(
            rec.agg_cols.contains_key("orders.amount"),
            "expected 'orders.amount', got {:?}",
            rec.agg_cols
        );
        assert!(
            !rec.agg_cols.contains_key("amt"),
            "alias 'amt' should not appear as agg_cols key"
        );
    }

    #[test]
    fn test_with_columns_tracks_alias() {
        let dm = make_dm();
        let rec = dm
            .table("orders", true)
            .with_columns(vec![col("orders.amount").alias("amt")]);
        assert_eq!(rec.alias_map.get("amt"), Some(&"orders.amount".to_string()));
    }

    #[test]
    fn test_transitive_alias_chain() {
        let dm = make_dm();
        let rec = dm
            .table("orders", true)
            .with_columns(vec![col("orders.amount").alias("a")])
            .with_columns(vec![col("a").alias("b")])
            .with_columns(vec![col("b").alias("c")]);
        assert_eq!(rec.alias_map.get("c"), Some(&"orders.amount".to_string()));
    }
}
