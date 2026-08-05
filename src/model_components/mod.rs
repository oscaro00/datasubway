pub mod agg_context;
pub mod column_values_context;
pub mod joins;
pub mod measures;
pub mod pre_aggregations;
pub mod select_context;

use std::collections::HashSet;

/// Defaults `limit` to 10000 and rejects an explicit 0.
pub(crate) fn validate_limit(limit: Option<usize>) -> Result<usize, String> {
    let limit = limit.unwrap_or(10000);
    if limit == 0 {
        return Err("limit must be > 0".into());
    }
    Ok(limit)
}

/// Checks every item is present in `valid`, returning a labeled error on the first miss.
pub(crate) fn validate_membership(
    items: &[String],
    valid: &HashSet<String>,
    label: &str,
) -> Result<(), String> {
    for item in items {
        if !valid.contains(item) {
            return Err(format!("{label}: '{item}'"));
        }
    }
    Ok(())
}

/// Checks every sort column is in `valid` and every direction is "asc"/"desc".
pub(crate) fn validate_sorts(
    sorts: &[(String, String)],
    valid: &HashSet<String>,
    unknown_col_label: &str,
) -> Result<(), String> {
    for (col, direction) in sorts {
        if !valid.contains(col) {
            return Err(format!("{unknown_col_label}: '{col}'"));
        }
        if direction != "asc" && direction != "desc" {
            return Err(format!("Invalid sort direction: '{}'", direction));
        }
    }
    Ok(())
}
