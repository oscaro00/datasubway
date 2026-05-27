//! Polars aggregate expression serialization canary.
//!
//! Each test locks in the JSON serialization of one polars aggregate expression as of
//! polars 0.53.0. If a polars upgrade changes the `Expr` serialization format, these
//! tests will fail, signaling that any AST-walking code that depends on the format must
//! be updated.
//!
//! To regenerate expected JSON after a polars version bump:
//!   cargo test -- --ignored print_all_serialized --nocapture
//! then copy each output block into the corresponding const below.

use polars::prelude::*;

fn serialize_expr(expr: &Expr) -> serde_json::Value {
    let s = serde_json::to_string(expr).expect("Expr serialization failed");
    serde_json::from_str(&s).expect("serialized output is not valid JSON")
}

// polars 0.53.0
const SUM_EXPR_JSON: &str = r#"{"Agg":{"Sum":{"Column":"x"}}}"#;
const MEAN_EXPR_JSON: &str = r#"{"Agg":{"Mean":{"Column":"x"}}}"#;
const MIN_EXPR_JSON: &str = r#"{"Agg":{"Min":{"input":{"Column":"x"},"propagate_nans":false}}}"#;
const MAX_EXPR_JSON: &str = r#"{"Agg":{"Max":{"input":{"Column":"x"},"propagate_nans":false}}}"#;
const COUNT_EXPR_JSON: &str = r#"{"Agg":{"Count":{"input":{"Column":"x"},"include_nulls":false}}}"#;
const STD_EXPR_JSON: &str = r#"{"Agg":{"Std":[{"Column":"x"},1]}}"#;
const VAR_EXPR_JSON: &str = r#"{"Agg":{"Var":[{"Column":"x"},1]}}"#;
const FIRST_EXPR_JSON: &str = r#"{"Agg":{"First":{"Column":"x"}}}"#;
const LAST_EXPR_JSON: &str = r#"{"Agg":{"Last":{"Column":"x"}}}"#;
const MEDIAN_EXPR_JSON: &str = r#"{"Agg":{"Median":{"Column":"x"}}}"#;
const N_UNIQUE_EXPR_JSON: &str = r#"{"Agg":{"NUnique":{"Column":"x"}}}"#;

#[test]
fn test_sum_serialization() {
    let actual = serialize_expr(&col("x").sum());
    let expected: serde_json::Value =
        serde_json::from_str(SUM_EXPR_JSON).expect("SUM_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").sum() serialization changed — update SUM_EXPR_JSON"
    );
}

#[test]
fn test_mean_serialization() {
    let actual = serialize_expr(&col("x").mean());
    let expected: serde_json::Value =
        serde_json::from_str(MEAN_EXPR_JSON).expect("MEAN_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").mean() serialization changed — update MEAN_EXPR_JSON"
    );
}

#[test]
fn test_min_serialization() {
    let actual = serialize_expr(&col("x").min());
    let expected: serde_json::Value =
        serde_json::from_str(MIN_EXPR_JSON).expect("MIN_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").min() serialization changed — update MIN_EXPR_JSON"
    );
}

#[test]
fn test_max_serialization() {
    let actual = serialize_expr(&col("x").max());
    let expected: serde_json::Value =
        serde_json::from_str(MAX_EXPR_JSON).expect("MAX_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").max() serialization changed — update MAX_EXPR_JSON"
    );
}

#[test]
fn test_count_serialization() {
    let actual = serialize_expr(&col("x").count());
    let expected: serde_json::Value =
        serde_json::from_str(COUNT_EXPR_JSON).expect("COUNT_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").count() serialization changed — update COUNT_EXPR_JSON"
    );
}

#[test]
fn test_std_serialization() {
    let actual = serialize_expr(&col("x").std(1));
    let expected: serde_json::Value =
        serde_json::from_str(STD_EXPR_JSON).expect("STD_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").std(1) serialization changed — update STD_EXPR_JSON"
    );
}

#[test]
fn test_var_serialization() {
    let actual = serialize_expr(&col("x").var(1));
    let expected: serde_json::Value =
        serde_json::from_str(VAR_EXPR_JSON).expect("VAR_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").var(1) serialization changed — update VAR_EXPR_JSON"
    );
}

#[test]
fn test_first_serialization() {
    let actual = serialize_expr(&col("x").first());
    let expected: serde_json::Value =
        serde_json::from_str(FIRST_EXPR_JSON).expect("FIRST_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").first() serialization changed — update FIRST_EXPR_JSON"
    );
}

#[test]
fn test_last_serialization() {
    let actual = serialize_expr(&col("x").last());
    let expected: serde_json::Value =
        serde_json::from_str(LAST_EXPR_JSON).expect("LAST_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").last() serialization changed — update LAST_EXPR_JSON"
    );
}

#[test]
fn test_median_serialization() {
    let actual = serialize_expr(&col("x").median());
    let expected: serde_json::Value =
        serde_json::from_str(MEDIAN_EXPR_JSON).expect("MEDIAN_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").median() serialization changed — update MEDIAN_EXPR_JSON"
    );
}

#[test]
fn test_n_unique_serialization() {
    let actual = serialize_expr(&col("x").n_unique());
    let expected: serde_json::Value =
        serde_json::from_str(N_UNIQUE_EXPR_JSON).expect("N_UNIQUE_EXPR_JSON is malformed");
    assert_eq!(
        actual, expected,
        "col(\"x\").n_unique() serialization changed — update N_UNIQUE_EXPR_JSON"
    );
}

#[test]
#[ignore = "developer tool: run to regenerate expected JSON constants after a polars version bump"]
fn print_all_serialized() {
    let cases: Vec<(&str, Expr)> = vec![
        ("sum", col("x").sum()),
        ("mean", col("x").mean()),
        ("min", col("x").min()),
        ("max", col("x").max()),
        ("count", col("x").count()),
        ("std(1)", col("x").std(1)),
        ("var(1)", col("x").var(1)),
        ("first", col("x").first()),
        ("last", col("x").last()),
        ("median", col("x").median()),
        ("n_unique", col("x").n_unique()),
    ];
    for (name, expr) in &cases {
        println!(
            "// {name}\n{}\n",
            serde_json::to_string_pretty(expr)
                .unwrap_or_else(|e| panic!("failed to serialize {name}: {e}"))
        );
    }
}
