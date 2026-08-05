use serde::{Deserialize, Serialize};

/// How `ColumnValuesContext` should summarize a column's values.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum ColumnValuesMode {
    /// Return every distinct value of the column.
    Distinct,
    /// Return just the column's minimum and maximum value.
    Range,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnValuesContext {
    pub column: String,
    pub mode: ColumnValuesMode,
    pub use_pre_agg: bool,
    pub pre_agg_valid_secs: Option<u64>,
}

impl ColumnValuesContext {
    pub fn new(
        column: String,
        mode: Option<ColumnValuesMode>,
        use_pre_agg: bool,
        pre_agg_valid_secs: Option<u64>,
    ) -> Result<Self, String> {
        let parts: Vec<&str> = column.splitn(2, '.').collect();
        if parts.len() != 2 || parts[0].is_empty() || parts[1].is_empty() {
            return Err(format!(
                "column must be in 'table_name.column_name' format, got: '{column}'"
            ));
        }
        let valid = |s: &str| s.chars().all(|c| c.is_alphanumeric() || c == '_');
        if !valid(parts[0]) {
            return Err(format!("invalid table name in column: '{}'", parts[0]));
        }
        if !valid(parts[1]) {
            return Err(format!("invalid column name: '{}'", parts[1]));
        }
        Ok(ColumnValuesContext {
            column,
            mode: mode.unwrap_or(ColumnValuesMode::Distinct),
            use_pre_agg,
            pre_agg_valid_secs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_format() {
        assert!(ColumnValuesContext::new("orders.region".into(), None, false, None).is_ok());
    }

    #[test]
    fn test_invalid_format_no_dot() {
        assert!(ColumnValuesContext::new("ordersregion".into(), None, false, None).is_err());
    }

    #[test]
    fn test_invalid_format_empty_table() {
        assert!(ColumnValuesContext::new(".region".into(), None, false, None).is_err());
    }

    #[test]
    fn test_invalid_format_empty_column() {
        assert!(ColumnValuesContext::new("orders.".into(), None, false, None).is_err());
    }

    #[test]
    fn test_mode_defaults_to_distinct() {
        let ctx = ColumnValuesContext::new("orders.region".into(), None, false, None).unwrap();
        assert_eq!(ctx.mode, ColumnValuesMode::Distinct);
    }

    #[test]
    fn test_mode_range_explicit() {
        let ctx = ColumnValuesContext::new(
            "orders.amount".into(),
            Some(ColumnValuesMode::Range),
            false,
            None,
        )
        .unwrap();
        assert_eq!(ctx.mode, ColumnValuesMode::Range);
    }

    #[test]
    fn test_mode_serde_roundtrip() {
        let v = serde_json::to_value(ColumnValuesMode::Range).unwrap();
        assert_eq!(v, serde_json::json!("range"));
        let m: ColumnValuesMode = serde_json::from_value(serde_json::json!("distinct")).unwrap();
        assert_eq!(m, ColumnValuesMode::Distinct);
    }
}
