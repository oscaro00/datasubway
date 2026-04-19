//! Combining measure results: joining, havings, sorting, limit/offset.
//!
//! After each measure produces a DataFrame, this module combines them
//! into a single result using DataFusion for joins, filtering, and sorting.

use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::logical_expr::SortExpr;
use datafusion::prelude::*;

use crate::model::filter_tree::filter_tree_to_expr;
use crate::model::query_context::QueryContext;

/// Combine multiple measure DataFrames into a single DataFrame.
///
/// Steps:
/// 1. Join them (cross join if no groups, full outer join on group cols)
/// 2. Apply having filters
/// 3. Apply sorts
/// 4. Apply offset + limit
pub fn combine_measure_dfs(
    measure_dfs: Vec<(&str, DataFrame)>,
    qc: &QueryContext,
) -> Result<DataFrame, DataFusionError> {
    if measure_dfs.is_empty() {
        return Err(DataFusionError::Plan("No measure results produced".into()));
    }

    let mut iter = measure_dfs.into_iter();
    let (_, mut result_df) = iter.next().unwrap();

    for (_, other_df) in iter {
        if qc.groups.is_empty() {
            // Cross join for scalar (no group-by) results
            result_df = result_df.join(other_df, JoinType::Inner, &[], &[], None)?;
        } else {
            // Full outer join on common group columns
            let left_cols: Vec<String> = result_df
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();
            let right_cols: Vec<String> = other_df
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().clone())
                .collect();

            let join_cols = find_common_group_cols(&left_cols, &right_cols, &qc.groups);

            if join_cols.is_empty() {
                // Fallback to cross join if no common columns found
                result_df = result_df.join(other_df, JoinType::Inner, &[], &[], None)?;
            } else {
                // Rename join columns on the right DF to avoid duplicate qualified
                // field names (DataFusion keeps both sides' join keys in a FULL join).
                let temp_names: Vec<String> = join_cols
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("__join_{i}"))
                    .collect();

                let rename_exprs: Vec<Expr> = other_df
                    .schema()
                    .fields()
                    .iter()
                    .map(|f| {
                        let name = f.name();
                        if let Some(idx) = join_cols.iter().position(|c| c == name) {
                            col(name.as_str()).alias(&temp_names[idx])
                        } else {
                            col(name.as_str())
                        }
                    })
                    .collect();
                let renamed_right = other_df.select(rename_exprs)?;

                let left_strs: Vec<&str> = join_cols.iter().map(|s| s.as_str()).collect();
                let right_strs: Vec<&str> = temp_names.iter().map(|s| s.as_str()).collect();
                result_df =
                    result_df.join(renamed_right, JoinType::Full, &left_strs, &right_strs, None)?;

                // Coalesce left and right join keys, then drop the temp columns.
                // Put COALESCE at the original column's position, skip the temp column.
                let select_exprs: Vec<Expr> = result_df
                    .schema()
                    .fields()
                    .iter()
                    .filter_map(|f| {
                        let name = f.name();
                        if temp_names.contains(&name.to_string()) {
                            // Skip temp column — handled by the COALESCE below
                            None
                        } else if let Some(idx) = join_cols.iter().position(|c| c == name) {
                            // Replace original with COALESCE(original, temp) AS original
                            let temp = &temp_names[idx];
                            Some(
                                coalesce(vec![col(name.as_str()), col(temp.as_str())])
                                    .alias(name.clone()),
                            )
                        } else {
                            Some(col(name.as_str()))
                        }
                    })
                    .collect();
                result_df = result_df.select(select_exprs)?;
            }
        }
    }

    // Apply having filters
    if qc.havings.is_object() && !qc.havings.as_object().unwrap().is_empty() {
        let having_expr = filter_tree_to_expr(&qc.havings)?;
        result_df = result_df.filter(having_expr)?;
    }

    // Apply sorts
    if !qc.sorts.is_empty() {
        let sort_exprs: Vec<SortExpr> = qc
            .sorts
            .iter()
            .map(|(col_name, direction)| {
                let asc = direction == "asc";
                col(col_name).sort(asc, !asc)
            })
            .collect();
        result_df = result_df.sort(sort_exprs)?;
    }

    // Apply offset
    if qc.offset > 0 {
        result_df = result_df.limit(qc.offset, Some(qc.limit))?;
    } else if qc.limit < 10000 {
        result_df = result_df.limit(0, Some(qc.limit))?;
    }

    Ok(result_df)
}

/// Combine multiple measure DataFrames and collect into RecordBatches.
pub async fn combine_measure_results(
    measure_dfs: Vec<(&str, DataFrame)>,
    qc: &QueryContext,
) -> Result<Vec<RecordBatch>, DataFusionError> {
    let df = combine_measure_dfs(measure_dfs, qc)?;
    df.collect().await
}

/// Find group columns common to both left and right column lists.
///
/// Handles both qualified (`orders.region`) and unqualified (`region`) names.
/// Returns the actual column names as they appear in both tables.
pub fn find_common_group_cols(
    left_cols: &[String],
    right_cols: &[String],
    groups: &[String],
) -> Vec<String> {
    let left_set: std::collections::HashSet<&str> = left_cols.iter().map(|s| s.as_str()).collect();
    let right_set: std::collections::HashSet<&str> =
        right_cols.iter().map(|s| s.as_str()).collect();

    let mut common = Vec::new();
    for g in groups {
        // Try qualified name first
        if left_set.contains(g.as_str()) && right_set.contains(g.as_str()) {
            common.push(g.clone());
            continue;
        }
        // Try unqualified name (strip table prefix)
        let unqualified = if let Some(pos) = g.find('.') {
            &g[pos + 1..]
        } else {
            g.as_str()
        };
        if left_set.contains(unqualified) && right_set.contains(unqualified) {
            common.push(unqualified.to_string());
        }
    }
    common
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::execution::context::SessionContext;
    use serde_json::json;
    use std::sync::Arc;

    /// Helper: register a RecordBatch as a table and return a DataFrame.
    async fn batch_to_df(ctx: &SessionContext, name: &str, batch: RecordBatch) -> DataFrame {
        let schema = batch.schema();
        let mem_table =
            datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]]).unwrap();
        ctx.register_table(name, Arc::new(mem_table)).unwrap();
        ctx.table(name).await.unwrap()
    }

    #[test]
    fn test_find_common_group_cols_qualified() {
        let left = vec!["region".into(), "revenue".into()];
        let right = vec!["region".into(), "quantity".into()];
        let groups = vec!["orders.region".into()];
        let common = find_common_group_cols(&left, &right, &groups);
        assert_eq!(common, vec!["region"]);
    }

    #[test]
    fn test_find_common_group_cols_exact() {
        let left = vec!["orders.region".into(), "revenue".into()];
        let right = vec!["orders.region".into(), "quantity".into()];
        let groups = vec!["orders.region".into()];
        let common = find_common_group_cols(&left, &right, &groups);
        assert_eq!(common, vec!["orders.region"]);
    }

    #[test]
    fn test_find_common_group_cols_no_match() {
        let left = vec!["revenue".into()];
        let right = vec!["quantity".into()];
        let groups = vec!["orders.region".into()];
        let common = find_common_group_cols(&left, &right, &groups);
        assert!(common.is_empty());
    }

    #[tokio::test]
    async fn test_combine_single_measure() {
        let ctx = SessionContext::new();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "revenue",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(arrow::array::Int64Array::from(vec![1000]))],
        )
        .unwrap();
        let df = batch_to_df(&ctx, "m0", batch).await;

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = combine_measure_results(vec![("revenue", df)], &qc)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 1);
    }

    #[tokio::test]
    async fn test_combine_with_having() {
        let ctx = SessionContext::new();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("revenue", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450])),
            ],
        )
        .unwrap();
        let df = batch_to_df(&ctx, "m0", batch).await;

        let havings = json!({"AND": [["revenue", ">", 500]]});
        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            Some(havings),
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = combine_measure_results(vec![("revenue", df)], &qc)
            .await
            .unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_combine_with_sort() {
        let ctx = SessionContext::new();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("revenue", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450])),
            ],
        )
        .unwrap();
        let df = batch_to_df(&ctx, "m0", batch).await;

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("revenue".into(), "asc".into())]),
            None,
            None,
            None,
        )
        .unwrap();

        let result = combine_measure_results(vec![("revenue", df)], &qc)
            .await
            .unwrap();
        assert_eq!(result.len(), 1);
        let col = result[0]
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 450);
        assert_eq!(col.value(1), 550);
    }

    #[tokio::test]
    async fn test_combine_with_limit_offset() {
        let ctx = SessionContext::new();
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("revenue", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU", "APAC"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450, 300])),
            ],
        )
        .unwrap();
        let df = batch_to_df(&ctx, "m0", batch).await;

        let qc = QueryContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("revenue".into(), "desc".into())]),
            Some(1),
            Some(1),
            None,
        )
        .unwrap();

        let result = combine_measure_results(vec![("revenue", df)], &qc)
            .await
            .unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
        let col = result[0]
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 450);
    }

    #[tokio::test]
    async fn test_combine_multi_measure_no_groups() {
        let ctx = SessionContext::new();

        let batch1 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "revenue",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(arrow::array::Int64Array::from(vec![1000]))],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new(
                "quantity",
                DataType::Int64,
                false,
            )])),
            vec![Arc::new(arrow::array::Int64Array::from(vec![100]))],
        )
        .unwrap();

        let df1 = batch_to_df(&ctx, "m0", batch1).await;
        let df2 = batch_to_df(&ctx, "m1", batch2).await;

        let qc = QueryContext::new(
            vec!["revenue".into(), "total_quantity".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = combine_measure_results(vec![("revenue", df1), ("total_quantity", df2)], &qc)
            .await
            .unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[tokio::test]
    async fn test_combine_multi_measure_with_groups() {
        let ctx = SessionContext::new();

        let batch1 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("revenue", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450])),
            ],
        )
        .unwrap();
        let batch2 = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("region", DataType::Utf8, false),
                Field::new("quantity", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![55, 45])),
            ],
        )
        .unwrap();

        let df1 = batch_to_df(&ctx, "m0", batch1).await;
        let df2 = batch_to_df(&ctx, "m1", batch2).await;

        let qc = QueryContext::new(
            vec!["revenue".into(), "total_quantity".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result = combine_measure_results(vec![("revenue", df1), ("total_quantity", df2)], &qc)
            .await
            .unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }

    #[tokio::test]
    async fn test_combine_multi_measure_with_qualified_groups() {
        use datafusion_functions_aggregate::sum::sum;

        let ctx = SessionContext::new();

        // Register a "players" table so columns get proper table qualifiers
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("player_name", DataType::Utf8, false),
                Field::new("goals", DataType::Int64, false),
                Field::new("assists", DataType::Int64, false),
            ])),
            vec![
                Arc::new(arrow::array::StringArray::from(vec![
                    "Alice", "Alice", "Bob", "Bob",
                ])),
                Arc::new(arrow::array::Int64Array::from(vec![3, 7, 5, 15])),
                Arc::new(arrow::array::Int64Array::from(vec![1, 4, 2, 13])),
            ],
        )
        .unwrap();
        let mem_table =
            datafusion::datasource::MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap();
        ctx.register_table("players", Arc::new(mem_table)).unwrap();

        // Simulate two measures that both group by players.player_name
        let df1 = ctx
            .table("players")
            .await
            .unwrap()
            .aggregate(
                vec![col("players.player_name")],
                vec![sum(col("goals")).alias("total_goals")],
            )
            .unwrap();
        let df2 = ctx
            .table("players")
            .await
            .unwrap()
            .aggregate(
                vec![col("players.player_name")],
                vec![sum(col("assists")).alias("total_assists")],
            )
            .unwrap();

        let qc = QueryContext::new(
            vec!["total_goals".into(), "total_assists".into()],
            None,
            Some(vec!["players.player_name".into()]),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();

        let result =
            combine_measure_results(vec![("total_goals", df1), ("total_assists", df2)], &qc)
                .await
                .unwrap();
        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);

        // Verify no duplicate columns in the output
        let schema = result[0].schema();
        let field_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        assert_eq!(
            field_names.iter().filter(|&&n| n == "player_name").count(),
            1,
            "Should have exactly one player_name column, got: {:?}",
            field_names
        );
    }
}
