use std::collections::HashMap;

use datafusion::logical_expr::{Aggregate, LogicalPlan};
use datafusion::prelude::{DataFrame, SessionContext};

use crate::wrappers::aggregate_with_metadata::{
    AggregateWithMetadata, fmt_exprs, root_aggregate_node,
};

/// Returns a string that uniquely identifies a measure's mergeable "slot":
/// the display-formatted input subplan + sorted display-formatted group expressions.
///
/// Two measures with the same key can have their aggr_expr merged into a single
/// aggregate node, avoiding the FULL JOIN that would otherwise combine them.
pub fn plan_merge_key(df: &DataFrame) -> Result<String, String> {
    let node = root_aggregate_node(df)?;
    Ok(format!("{}", node.input.display_graphviz()))
}

/// Merge a group of compatible DataFrames into one whose AggregateWithMetadata
/// combines all their aggr_expr. Single-element groups pass through unchanged.
fn merge_group(group: Vec<(String, DataFrame)>, ctx: &SessionContext) -> Result<DataFrame, String> {
    if group.len() == 1 {
        return Ok(group.into_iter().next().unwrap().1);
    }

    // Clone what we need from the first DataFrame; borrow is released at end of block.
    let (input_arc, group_expr, mut merged_aggr_expr, mut merged_metadata) = {
        let node = root_aggregate_node(&group[0].1)?;
        (
            node.input.clone(),
            node.group_expr.clone(),
            node.aggr_expr.clone(),
            node.metadata.clone(),
        )
    };

    // Accumulate aggr_expr from the remaining group members; first value wins for metadata.
    for (_, df) in &group[1..] {
        let node = root_aggregate_node(df)?;
        merged_aggr_expr.extend(node.aggr_expr.iter().cloned());
        for (k, v) in &node.metadata {
            merged_metadata
                .entry(k.clone())
                .or_insert_with(|| v.clone());
        }
    }

    merged_metadata.insert("aggregates".to_string(), fmt_exprs(&merged_aggr_expr));

    let agg = Aggregate::try_new(input_arc, group_expr, merged_aggr_expr)
        .map_err(|e| format!("Aggregate::try_new during merge: {e}"))?;

    let plan = AggregateWithMetadata::inject(LogicalPlan::Aggregate(agg), merged_metadata)
        .map_err(|e| format!("AggregateWithMetadata::inject during merge: {e}"))?;

    Ok(DataFrame::new(ctx.state(), plan))
}

/// Group `measure_dfs` by merge key, merge each compatible group into a single
/// DataFrame, and return the reduced list in insertion order of first occurrence.
pub fn merge_measure_dfs(
    measure_dfs: Vec<(String, DataFrame)>,
    ctx: &SessionContext,
) -> Result<Vec<DataFrame>, String> {
    let mut key_order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<(String, DataFrame)>> = HashMap::new();

    for (name, df) in measure_dfs {
        let key = plan_merge_key(&df)?;
        if !groups.contains_key(&key) {
            key_order.push(key.clone());
        }
        groups.entry(key).or_default().push((name, df));
    }

    key_order
        .into_iter()
        .map(|key| merge_group(groups.remove(&key).unwrap(), ctx))
        .collect()
}
