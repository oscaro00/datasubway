use std::collections::{HashMap, HashSet};

use polars::prelude::*;

use super::super::super::data_model::DataModel;
use super::agg_expr_parser::extract_agg_exprs;
use super::lazyframe_wrapper::LazyFrameWrapper;
use super::lazygroupby_wrapper::LazyGroupByWrapper;
use crate::column_expressions::column_context::{
    AllowExcludeRecord, IntoFilterExpr, IntoPolarsColsExpr,
};

fn prune_filter_by_tables(expr: Expr, base_table: &str, data_model: &DataModel) -> Option<Expr> {
    match expr {
        Expr::BinaryExpr {
            left,
            op: Operator::And,
            right,
        } => {
            let l = prune_filter_by_tables((*left).clone(), base_table, data_model);
            let r = prune_filter_by_tables((*right).clone(), base_table, data_model);
            match (l, r) {
                (Some(l), Some(r)) => Some(l.and(r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        Expr::BinaryExpr {
            left,
            op: Operator::Or,
            right,
        } => {
            let l = prune_filter_by_tables((*left).clone(), base_table, data_model);
            let r = prune_filter_by_tables((*right).clone(), base_table, data_model);
            match (l, r) {
                (Some(l), Some(r)) => Some(l.or(r)),
                (Some(l), None) => Some(l),
                (None, Some(r)) => Some(r),
                (None, None) => None,
            }
        }
        other => {
            let all_reachable = other.clone().meta().root_names().iter().all(|col| {
                match col.as_str().split_once('.') {
                    Some((table, _)) => {
                        table == base_table || data_model.can_join(base_table, table)
                    }
                    None => true,
                }
            });
            if all_reachable {
                Some(other)
            } else {
                None
            }
        }
    }
}

pub enum LazyOp {
    Sort(Vec<PlSmallStr>, SortMultipleOptions),
    Filter(Expr),
    WithColumn(Expr),
    GroupBy(Vec<Expr>),
    GroupByDynamic,
    Rolling,
    Having(Expr),
    Agg(Vec<Expr>),
    Head(Option<usize>),
    Tail(Option<usize>),
    Limit(u32),
}

pub struct LazyFrameRecorder<'a> {
    pub table_name: String,
    pub data_model: &'a DataModel,
    pub lazy_ops: Vec<LazyOp>,
    pub non_agg_cols: HashSet<PlSmallStr>,
    pub agg_cols: HashMap<PlSmallStr, Vec<String>>,
    pub non_base_tables: HashSet<String>,
    pub use_pre_agg: bool,
    pub allow_exclude_records: Vec<AllowExcludeRecord>,
}

impl<'a> LazyFrameRecorder<'a> {
    pub fn sort(
        mut self,
        by: impl IntoVec<PlSmallStr>,
        sort_options: SortMultipleOptions,
    ) -> LazyFrameRecorder<'a> {
        let cols = by.into_vec();
        self.non_agg_cols.extend(cols.iter().cloned());
        self.lazy_ops.push(LazyOp::Sort(cols, sort_options));
        self
    }

    pub fn filter(mut self, predicate: impl IntoFilterExpr) -> LazyFrameRecorder<'a> {
        if let Some(pred) = predicate.into_filter(&mut self.allow_exclude_records) {
            if let Some(pruned) = prune_filter_by_tables(pred, &self.table_name, self.data_model) {
                let cols = pruned.clone().meta().root_names();
                self.non_agg_cols.extend(cols);
                self.lazy_ops.push(LazyOp::Filter(pruned));
            }
        }
        self
    }

    pub fn with_column(mut self, expr: Expr) -> LazyFrameRecorder<'a> {
        let all_root_names = expr.clone().meta().root_names();
        let agg_pairs = extract_agg_exprs(&expr);

        let agg_col_set: HashSet<&str> = agg_pairs.iter().map(|(col, _)| col.as_str()).collect();

        for name in &all_root_names {
            if !agg_col_set.contains(name.as_str()) {
                self.non_agg_cols.insert(name.clone());
            }
        }

        for (col_name, agg_name) in agg_pairs {
            self.agg_cols
                .entry(col_name.as_str().into())
                .or_default()
                .push(agg_name);
        }

        self.lazy_ops.push(LazyOp::WithColumn(expr));
        self
    }

    pub fn group_by(mut self, by: impl IntoPolarsColsExpr) -> LazyFrameRecorder<'a> {
        let exprs = by.into_exprs_with_record(&mut self.allow_exclude_records);
        for expr in &exprs {
            self.non_agg_cols.extend(expr.clone().meta().root_names());
        }
        self.lazy_ops.push(LazyOp::GroupBy(exprs));
        self
    }

    pub fn agg(mut self, exprs: Vec<Expr>) -> LazyFrameRecorder<'a> {
        for expr in &exprs {
            let all_root_names = expr.clone().meta().root_names();
            let agg_pairs = extract_agg_exprs(expr);
            let agg_col_set: HashSet<&str> =
                agg_pairs.iter().map(|(col, _)| col.as_str()).collect();

            for name in &all_root_names {
                if !agg_col_set.contains(name.as_str()) {
                    self.non_agg_cols.insert(name.clone());
                }
            }

            for (col_name, agg_name) in agg_pairs {
                self.agg_cols
                    .entry(col_name.as_str().into())
                    .or_default()
                    .push(agg_name);
            }
        }

        self.lazy_ops.push(LazyOp::Agg(exprs));
        self
    }

    pub fn limit(mut self, n: u32) -> LazyFrameRecorder<'a> {
        self.lazy_ops.push(LazyOp::Limit(n));
        self
    }

    pub fn build(self) -> LazyFrameWrapper {
        enum State {
            Frame(LazyFrameWrapper),
            GroupBy(LazyGroupByWrapper),
        }

        let mut state = State::Frame(self.data_model.get_table(
            &self.table_name,
            &self.non_agg_cols,
            &self.agg_cols,
        ));

        for op in self.lazy_ops {
            state = match (state, op) {
                (State::Frame(lfw), LazyOp::Sort(by, opts)) => {
                    State::Frame(lfw.sort(Some(by), Some(opts)))
                }
                (State::Frame(lfw), LazyOp::Filter(pred)) => State::Frame(lfw.filter(Some(pred))),
                (State::Frame(lfw), LazyOp::WithColumn(expr)) => {
                    State::Frame(lfw.with_column(expr))
                }
                (State::Frame(lfw), LazyOp::GroupBy(by)) => State::GroupBy(lfw.group_by(by)),
                (State::Frame(lfw), LazyOp::Limit(n)) => State::Frame(lfw.limit(n)),
                (State::GroupBy(lgbw), LazyOp::Having(pred)) => State::GroupBy(lgbw.having(pred)),
                (State::GroupBy(lgbw), LazyOp::Agg(aggs)) => State::Frame(lgbw.agg(aggs)),
                (State::GroupBy(lgbw), LazyOp::Head(n)) => State::Frame(lgbw.head(n)),
                (State::GroupBy(lgbw), LazyOp::Tail(n)) => State::Frame(lgbw.tail(n)),
                (_, LazyOp::GroupByDynamic) | (_, LazyOp::Rolling) => {
                    panic!("group_by_dynamic and rolling are not yet implemented")
                }
                _ => panic!("invalid LazyOp sequence"),
            };
        }

        match state {
            State::Frame(lfw) => lfw,
            State::GroupBy(_) => panic!("incomplete group-by chain: missing agg/head/tail"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_components::joins::{Join, JoinDirection, JoinGraph, JoinHow};
    use std::collections::HashMap;

    fn make_dm() -> DataModel {
        let orders = df![
            "amount"    => [100.0f64, 200.0],
            "region"    => ["north", "south"],
        ]
        .unwrap()
        .lazy();
        let customers = df![
            "region"    => ["north", "south"],
            "country"   => ["US", "UK"],
        ]
        .unwrap()
        .lazy();
        let products = df![
            "price"     => [10.0f64, 20.0],
        ]
        .unwrap()
        .lazy();

        // orders → customers (joined), products is standalone
        let joins = vec![Join {
            left: "orders".into(),
            right: "customers".into(),
            left_on: vec!["orders.region".into()],
            right_on: vec!["customers.region".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
        }];

        DataModel::new(
            HashMap::from([
                ("orders".into(), orders),
                ("customers".into(), customers),
                ("products".into(), products),
            ]),
            JoinGraph::new(&joins).unwrap(),
            vec![],
            None,
        )
    }

    fn filter_ops(recorder: LazyFrameRecorder<'_>) -> Vec<Expr> {
        recorder
            .lazy_ops
            .into_iter()
            .filter_map(|op| match op {
                LazyOp::Filter(e) => Some(e),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn test_fully_reachable_filter_is_applied() {
        let dm = make_dm();
        // customers is joined to orders — filter should be kept
        let recorder = dm
            .table("orders")
            .filter(col("customers.country").eq(lit("US")));
        assert_eq!(filter_ops(recorder).len(), 1);
    }

    #[test]
    fn test_fully_unreachable_filter_is_dropped() {
        let dm = make_dm();
        // products has no join to orders — filter should be silently ignored
        let recorder = dm
            .table("orders")
            .filter(col("products.price").lt(lit(50.0f64)));
        assert_eq!(filter_ops(recorder).len(), 0);
    }

    #[test]
    fn test_compound_and_partial_prune() {
        let dm = make_dm();
        // AND: left branch (orders) reachable, right branch (products) not
        // Expected: only the orders.amount clause survives
        let recorder = dm.table("orders").filter(
            col("orders.amount")
                .gt(lit(0.0f64))
                .and(col("products.price").lt(lit(50.0f64))),
        );
        let ops = filter_ops(recorder);
        assert_eq!(ops.len(), 1);
        // The surviving expression should reference orders.amount, not products.price
        let names = ops[0].clone().meta().root_names();
        assert!(names.iter().any(|n| n.as_str() == "orders.amount"));
        assert!(!names.iter().any(|n| n.as_str() == "products.price"));
    }

    #[test]
    fn test_compound_and_all_reachable_kept() {
        let dm = make_dm();
        let recorder = dm.table("orders").filter(
            col("orders.amount")
                .gt(lit(0.0f64))
                .and(col("customers.country").eq(lit("US"))),
        );
        let ops = filter_ops(recorder);
        assert_eq!(ops.len(), 1);
        let names = ops[0].clone().meta().root_names();
        assert!(names.iter().any(|n| n.as_str() == "orders.amount"));
        assert!(names.iter().any(|n| n.as_str() == "customers.country"));
    }

    #[test]
    fn test_compound_and_all_unreachable_dropped() {
        let dm = make_dm();
        let recorder = dm.table("orders").filter(
            col("products.price")
                .gt(lit(5.0f64))
                .and(col("products.price").lt(lit(50.0f64))),
        );
        assert_eq!(filter_ops(recorder).len(), 0);
    }
}
