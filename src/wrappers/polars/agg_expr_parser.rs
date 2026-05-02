use polars::prelude::Expr;
use serde_json::Value;

fn col_from_value(v: &Value) -> Option<String> {
    v.as_object()?.get("Column")?.as_str().map(str::to_owned)
}

fn extract_from_agg_node(agg_val: &Value) -> Option<(String, String)> {
    let obj = agg_val.as_object()?;
    let (agg_type, inner) = obj.iter().next()?;

    let agg_name = match agg_type.as_str() {
        "Sum" => "sum",
        "Mean" => "mean",
        "Min" => "min",
        "Max" => "max",
        "Count" => "count",
        "Std" => "std",
        "Var" => "var",
        "First" => "first",
        "Last" => "last",
        "Median" => "median",
        "NUnique" => "n_unique",
        _ => return None,
    };

    let col_name = match agg_type.as_str() {
        "Sum" | "Mean" | "First" | "Last" | "Median" | "NUnique" => col_from_value(inner)?,
        "Min" | "Max" | "Count" => col_from_value(inner.get("input")?)?,
        "Std" | "Var" => col_from_value(inner.as_array()?.first()?)?,
        _ => return None,
    };

    Some((col_name, agg_name.to_string()))
}

fn walk(value: &Value, out: &mut Vec<(String, String)>) {
    match value {
        Value::Object(map) => {
            if let Some(agg_val) = map.get("Agg") {
                if let Some(pair) = extract_from_agg_node(agg_val) {
                    out.push(pair);
                    return;
                }
            }
            for v in map.values() {
                walk(v, out);
            }
        }
        Value::Array(arr) => {
            for v in arr {
                walk(v, out);
            }
        }
        _ => {}
    }
}

/// Walk a polars `Expr` and return every `(column_name, agg_name)` pair found in it.
pub fn extract_agg_exprs(expr: &Expr) -> Vec<(String, String)> {
    let s = serde_json::to_string(expr).expect("Expr serialization failed");
    let value: Value = serde_json::from_str(&s).expect("serialized Expr is not valid JSON");
    let mut out = Vec::new();
    walk(&value, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use polars::prelude::*;

    fn single(expr: &Expr) -> (String, String) {
        let mut v = extract_agg_exprs(expr);
        assert_eq!(v.len(), 1, "expected exactly one agg pair, got {v:?}");
        v.remove(0)
    }

    #[test]
    fn test_sum() {
        assert_eq!(single(&col("x").sum()), ("x".into(), "sum".into()));
    }

    #[test]
    fn test_mean() {
        assert_eq!(single(&col("x").mean()), ("x".into(), "mean".into()));
    }

    #[test]
    fn test_min() {
        assert_eq!(single(&col("x").min()), ("x".into(), "min".into()));
    }

    #[test]
    fn test_max() {
        assert_eq!(single(&col("x").max()), ("x".into(), "max".into()));
    }

    #[test]
    fn test_count() {
        assert_eq!(single(&col("x").count()), ("x".into(), "count".into()));
    }

    #[test]
    fn test_std() {
        assert_eq!(single(&col("x").std(1)), ("x".into(), "std".into()));
    }

    #[test]
    fn test_var() {
        assert_eq!(single(&col("x").var(1)), ("x".into(), "var".into()));
    }

    #[test]
    fn test_first() {
        assert_eq!(single(&col("x").first()), ("x".into(), "first".into()));
    }

    #[test]
    fn test_last() {
        assert_eq!(single(&col("x").last()), ("x".into(), "last".into()));
    }

    #[test]
    fn test_median() {
        assert_eq!(single(&col("x").median()), ("x".into(), "median".into()));
    }

    #[test]
    fn test_n_unique() {
        assert_eq!(
            single(&col("x").n_unique()),
            ("x".into(), "n_unique".into())
        );
    }

    #[test]
    fn test_multi_column_agg_list() {
        let exprs = [
            col("amount").sum(),
            col("qty").mean(),
            col("price").std(1),
            col("amount").mean().alias("amount_mean"),
        ];
        let mut all: Vec<(String, String)> =
            exprs.iter().flat_map(|e| extract_agg_exprs(e)).collect();
        all.sort();
        assert_eq!(
            all,
            vec![
                ("amount".into(), "mean".into()),
                ("amount".into(), "sum".into()),
                ("price".into(), "std".into()),
                ("qty".into(), "mean".into()),
            ]
        );
    }

    #[test]
    fn test_alias_wrapper() {
        let expr = col("x").sum().alias("total");
        assert_eq!(extract_agg_exprs(&expr), vec![("x".into(), "sum".into())]);
    }

    #[test]
    fn test_binary_expr_two_aggs() {
        let expr = col("x").sum() + col("y").sum();
        let mut got = extract_agg_exprs(&expr);
        got.sort();
        assert_eq!(
            got,
            vec![("x".into(), "sum".into()), ("y".into(), "sum".into())]
        );
    }
}
