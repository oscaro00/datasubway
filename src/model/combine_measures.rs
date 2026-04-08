//! Combining measure results: joining, havings, sorting, limit/offset.
//!
//! After each measure produces Arrow RecordBatches, this module combines them
//! into a single result table using DataFusion for joins, filtering, and sorting.

use std::sync::Arc;

use arrow::array::RecordBatch;
use datafusion::common::DataFusionError;
use datafusion::execution::context::SessionContext;
use datafusion::logical_expr::SortExpr;
use datafusion::prelude::*;
use tokio::runtime::Runtime;

use crate::model::filter_tree::filter_tree_to_expr;
use crate::model::query_context::QueryContext;

/// Combine multiple measure results into a single final result.
///
/// Steps:
/// 1. Register each measure's batches in a temp SessionContext
/// 2. Join them (cross join if no groups, full outer join on group cols)
/// 3. Apply having filters
/// 4. Apply sorts
/// 5. Apply offset + limit
pub fn combine_measure_results(
    rt: &Runtime,
    measure_batches: Vec<(&str, Vec<RecordBatch>)>,
    qc: &QueryContext,
) -> Result<Vec<RecordBatch>, DataFusionError> {
    rt.block_on(async { combine_measure_results_async(measure_batches, qc).await })
}

async fn combine_measure_results_async(
    measure_batches: Vec<(&str, Vec<RecordBatch>)>,
    qc: &QueryContext,
) -> Result<Vec<RecordBatch>, DataFusionError> {
    let ctx = SessionContext::new();

    // Register each measure's batches as a named table
    let mut table_names = Vec::new();
    for (i, (name, batches)) in measure_batches.iter().enumerate() {
        if batches.is_empty() {
            return Err(DataFusionError::Plan(format!(
                "Measure '{}' produced no results",
                name
            )));
        }
        let table_name = format!("_measure_{}", i);
        let schema = batches[0].schema();
        let mem_table = datafusion::datasource::MemTable::try_new(schema, vec![batches.clone()])?;
        ctx.register_table(&table_name, Arc::new(mem_table))?;
        table_names.push(table_name);
    }

    // Build the combined DataFrame
    let mut result_df = ctx.table(&table_names[0]).await?;

    for other_name in &table_names[1..] {
        let other_df = ctx.table(other_name).await?;

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
                let join_col_strs: Vec<&str> = join_cols.iter().map(|s| s.as_str()).collect();
                result_df = result_df.join(
                    other_df,
                    JoinType::Full,
                    &join_col_strs,
                    &join_col_strs,
                    None,
                )?;
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
        // DataFusion doesn't have a direct offset-only method;
        // use limit(offset + limit) then skip in Arrow, or use SQL.
        // Actually, DataFrame::limit(skip, fetch) handles both.
        result_df = result_df.limit(qc.offset, Some(qc.limit))?;
    } else if qc.limit < 10000 {
        result_df = result_df.limit(0, Some(qc.limit))?;
    }

    result_df.collect().await
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
    use serde_json::json;

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

    #[test]
    fn test_combine_single_measure() {
        let rt = Runtime::new().unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("revenue", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(arrow::array::Int64Array::from(vec![1000]))],
        )
        .unwrap();

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

        let result = combine_measure_results(&rt, vec![("revenue", vec![batch])], &qc).unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].num_rows(), 1);
    }

    #[test]
    fn test_combine_with_having() {
        let rt = Runtime::new().unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("region", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("revenue", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450])),
            ],
        )
        .unwrap();

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

        let result = combine_measure_results(&rt, vec![("revenue", vec![batch])], &qc).unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[test]
    fn test_combine_with_sort() {
        let rt = Runtime::new().unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("region", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("revenue", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450])),
            ],
        )
        .unwrap();

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

        let result = combine_measure_results(&rt, vec![("revenue", vec![batch])], &qc).unwrap();

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

    #[test]
    fn test_combine_with_limit_offset() {
        let rt = Runtime::new().unwrap();
        let schema = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("region", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("revenue", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU", "APAC"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450, 300])),
            ],
        )
        .unwrap();

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

        let result = combine_measure_results(&rt, vec![("revenue", vec![batch])], &qc).unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
        let col = result[0]
            .column_by_name("revenue")
            .unwrap()
            .as_any()
            .downcast_ref::<arrow::array::Int64Array>()
            .unwrap();
        assert_eq!(col.value(0), 450); // second row after desc sort
    }

    #[test]
    fn test_combine_multi_measure_no_groups() {
        let rt = Runtime::new().unwrap();

        let schema1 = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("revenue", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch1 = RecordBatch::try_new(
            schema1,
            vec![Arc::new(arrow::array::Int64Array::from(vec![1000]))],
        )
        .unwrap();

        let schema2 = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("quantity", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch2 = RecordBatch::try_new(
            schema2,
            vec![Arc::new(arrow::array::Int64Array::from(vec![100]))],
        )
        .unwrap();

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

        let result = combine_measure_results(
            &rt,
            vec![("revenue", vec![batch1]), ("total_quantity", vec![batch2])],
            &qc,
        )
        .unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 1);
    }

    #[test]
    fn test_combine_multi_measure_with_groups() {
        let rt = Runtime::new().unwrap();

        let schema1 = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("region", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("revenue", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch1 = RecordBatch::try_new(
            schema1,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![550, 450])),
            ],
        )
        .unwrap();

        let schema2 = Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("region", arrow::datatypes::DataType::Utf8, false),
            arrow::datatypes::Field::new("quantity", arrow::datatypes::DataType::Int64, false),
        ]));
        let batch2 = RecordBatch::try_new(
            schema2,
            vec![
                Arc::new(arrow::array::StringArray::from(vec!["US", "EU"])),
                Arc::new(arrow::array::Int64Array::from(vec![55, 45])),
            ],
        )
        .unwrap();

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

        let result = combine_measure_results(
            &rt,
            vec![("revenue", vec![batch1]), ("total_quantity", vec![batch2])],
            &qc,
        )
        .unwrap();

        let total_rows: usize = result.iter().map(|b| b.num_rows()).sum();
        assert_eq!(total_rows, 2);
    }
}
