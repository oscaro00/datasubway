use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ColumnValuesContext {
    pub column: String,
    pub use_pre_agg: bool,
    pub pre_agg_valid_secs: Option<u64>,
}

impl ColumnValuesContext {
    pub fn new(
        column: String,
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
        assert!(ColumnValuesContext::new("orders.region".into(), false, None).is_ok());
    }

    #[test]
    fn test_invalid_format_no_dot() {
        assert!(ColumnValuesContext::new("ordersregion".into(), false, None).is_err());
    }

    #[test]
    fn test_invalid_format_empty_table() {
        assert!(ColumnValuesContext::new(".region".into(), false, None).is_err());
    }

    #[test]
    fn test_invalid_format_empty_column() {
        assert!(ColumnValuesContext::new("orders.".into(), false, None).is_err());
    }
}
