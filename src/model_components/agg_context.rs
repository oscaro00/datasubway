use serde::{Deserialize, Serialize};

use crate::column_expressions::filter_expr::extract_filter_cols;
use crate::model_components::measures::MeasureMetadata;
use crate::model_components::{validate_limit, validate_membership, validate_sorts};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggContext {
    pub measures: Vec<String>,
    pub filters: serde_json::Value,
    pub groups: Vec<String>,
    pub havings: serde_json::Value,
    pub sorts: Vec<(String, String)>,
    pub limit: usize,
    pub offset: usize,
    pub use_pre_agg: bool,
    pub pre_agg_valid_secs: Option<u64>,
}

impl AggContext {
    /// Each parameter is one field of the query-context payload this type
    /// mirrors, so the arity tracks the payload rather than signalling a
    /// function that does too much. Collapsing them into an options struct
    /// would be an API change across every call site; revisit if the payload
    /// grows further.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        measures: Vec<String>,
        filters: Option<serde_json::Value>,
        groups: Option<Vec<String>>,
        havings: Option<serde_json::Value>,
        sorts: Option<Vec<(String, String)>>,
        limit: Option<usize>,
        offset: Option<usize>,
        use_pre_agg: Option<bool>,
        pre_agg_valid_secs: Option<u64>,
    ) -> Result<Self, String> {
        if measures.is_empty() {
            return Err("measures must not be empty".into());
        }

        let limit = validate_limit(limit)?;

        Ok(AggContext {
            measures,
            filters: filters.unwrap_or(serde_json::Value::Object(Default::default())),
            groups: groups.unwrap_or_default(),
            havings: havings.unwrap_or(serde_json::Value::Object(Default::default())),
            sorts: sorts.unwrap_or_default(),
            limit,
            offset: offset.unwrap_or(0),
            use_pre_agg: use_pre_agg.unwrap_or(true),
            pre_agg_valid_secs,
        })
    }

    pub(crate) fn stub() -> Self {
        AggContext::new(
            vec!["_stub".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .expect("stub always valid")
    }

    /// Extract all column references from the filters JSON.
    pub fn filter_columns(&self) -> Vec<String> {
        extract_filter_cols(&self.filters)
    }

    /// Extract all column references from the havings JSON.
    pub fn having_columns(&self) -> Vec<String> {
        extract_filter_cols(&self.havings)
    }

    pub fn validate(
        &self,
        known_measures: &[MeasureMetadata],
        all_columns: &std::collections::HashSet<String>,
    ) -> Result<(), String> {
        let measure_map: std::collections::HashMap<&str, &MeasureMetadata> = known_measures
            .iter()
            .map(|m| (m.name.as_str(), m))
            .collect();

        for m in &self.measures {
            if !measure_map.contains_key(m.as_str()) {
                return Err(format!("Unknown measure: '{}'", m));
            }
        }

        validate_membership(&self.groups, all_columns, "Unknown group column")?;
        validate_membership(&self.filter_columns(), all_columns, "Unknown filter column")?;

        let mut valid_having_cols: std::collections::HashSet<String> =
            self.groups.iter().cloned().collect();
        for m in &self.measures {
            if let Some(meta) = measure_map.get(m.as_str()) {
                for col in &meta.output_columns {
                    valid_having_cols.insert(col.clone());
                }
            }
        }

        validate_membership(
            &self.having_columns(),
            &valid_having_cols,
            "Invalid having column",
        )?;
        validate_sorts(&self.sorts, &valid_having_cols, "Invalid sort column")?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_basic_creation() {
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        assert_eq!(qc.measures, vec!["revenue"]);
        assert_eq!(qc.limit, 10000);
        assert_eq!(qc.offset, 0);
        assert!(qc.use_pre_agg);
    }

    #[test]
    fn test_empty_measures_rejected() {
        let result = AggContext::new(vec![], None, None, None, None, None, None, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_limit_rejected() {
        let result = AggContext::new(
            vec!["revenue".into()],
            None,
            None,
            None,
            None,
            Some(0),
            None,
            None,
            None,
        );
        assert!(result.is_err());
    }

    fn make_measure_meta(name: &str, output_cols: &[&str]) -> MeasureMetadata {
        MeasureMetadata {
            name: name.into(),
            output_columns: output_cols.iter().map(|s| s.to_string()).collect(),
            aggregate_columns: Vec::new(),
            allow_exclude_calls: Vec::new(),
        }
    }

    fn make_all_columns() -> std::collections::HashSet<String> {
        ["orders.region", "orders.amount", "orders.date"]
            .iter()
            .map(|s| s.to_string())
            .collect()
    }

    #[test]
    fn test_validate_ok() {
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        assert!(qc.validate(&measures, &make_all_columns()).is_ok());
    }

    #[test]
    fn test_validate_unknown_measure() {
        let qc = AggContext::new(
            vec!["unknown".into()],
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown measure"));
    }

    #[test]
    fn test_validate_unknown_group_column() {
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.nonexistent".into()]),
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown group column"));
    }

    #[test]
    fn test_validate_unknown_filter_column() {
        let filters = json!({"and": [{"left": {"col": "orders.nonexistent"}, "op": "=", "right": {"lit": "US"}}]});
        let qc = AggContext::new(
            vec!["revenue".into()],
            Some(filters),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Unknown filter column"));
    }

    #[test]
    fn test_validate_invalid_having_column() {
        let havings =
            json!({"and": [{"left": {"col": "nonexistent"}, "op": ">", "right": {"lit": 500}}]});
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            Some(havings),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid having column"));
    }

    #[test]
    fn test_validate_valid_having_on_measure_output() {
        let havings =
            json!({"and": [{"left": {"col": "revenue"}, "op": ">", "right": {"lit": 500}}]});
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            Some(havings),
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        assert!(qc.validate(&measures, &make_all_columns()).is_ok());
    }

    #[test]
    fn test_validate_invalid_sort_direction() {
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("revenue".into(), "up".into())]),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid sort direction"));
    }

    #[test]
    fn test_validate_invalid_sort_column() {
        let qc = AggContext::new(
            vec!["revenue".into()],
            None,
            Some(vec!["orders.region".into()]),
            None,
            Some(vec![("nonexistent".into(), "asc".into())]),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let measures = vec![make_measure_meta("revenue", &["revenue"])];
        let err = qc.validate(&measures, &make_all_columns()).unwrap_err();
        assert!(err.contains("Invalid sort column"));
    }

    #[test]
    fn test_filter_column_extraction() {
        let filters = json!({
            "and": [
                {"left": {"col": "orders.region"}, "op": "=",  "right": {"lit": "US"}},
                {"left": {"col": "orders.date"},   "op": ">",  "right": {"lit": "2024-01-01"}}
            ]
        });
        let qc = AggContext::new(
            vec!["revenue".into()],
            Some(filters),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let cols = qc.filter_columns();
        assert!(cols.contains(&"orders.region".to_string()));
        assert!(cols.contains(&"orders.date".to_string()));
    }

    #[test]
    fn test_filter_column_extraction_with_column_ref() {
        let filters = json!({
            "and": [
                {"left": {"col": "orders.amount"}, "op": "<=", "right": {"col": "orders.limit"}}
            ]
        });
        let qc = AggContext::new(
            vec!["revenue".into()],
            Some(filters),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let cols = qc.filter_columns();
        assert!(cols.contains(&"orders.amount".to_string()));
        assert!(cols.contains(&"orders.limit".to_string()));
    }
}
