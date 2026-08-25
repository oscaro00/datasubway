use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use datafusion::prelude::col;

use crate::data_model::DataModel;
use crate::model_components::pre_agg_store::{PreAggVersion, ReclaimReport};

/// Canonical aggregation kinds and the stored component column suffixes each needs.
/// e.g. a mean/avg needs both "sum" and "count" components stored.
enum AggKind {
    Sum,
    Count,
    Min,
    Max,
    Mean,
    StdVar,
}

impl AggKind {
    fn components(&self) -> &'static [&'static str] {
        match self {
            AggKind::Sum => &["sum"],
            AggKind::Count => &["count"],
            AggKind::Min => &["min"],
            AggKind::Max => &["max"],
            AggKind::Mean => &["sum", "count"],
            AggKind::StdVar => &["sum", "sumsq", "count"],
        }
    }
}

/// Maps user-facing agg names to stored component column suffixes.
/// e.g. "mean" needs both "sum" and "count" components stored.
pub fn agg_expansion(agg_name: &str) -> Result<Vec<&'static str>, String> {
    let kind = match agg_name {
        "sum" => AggKind::Sum,
        "count" => AggKind::Count,
        "min" => AggKind::Min,
        "max" => AggKind::Max,
        "mean" => AggKind::Mean,
        "std" | "var" => AggKind::StdVar,
        _ => return Err(format!("Unknown aggregation type: {}", agg_name)),
    };
    Ok(kind.components().to_vec())
}

/// Returns the column name for a pre-agg component stored in parquet.
/// e.g. ("orders.amount", "sum") → "orders.amount-sum"
pub fn component_col_name(col: &str, component: &str) -> String {
    format!("{col}-{component}")
}

/// Convert a qualified column name to a parquet-safe dunder name.
/// Replaces `.` with `__` to avoid clashing with DataFusion's `table.column` separator.
/// e.g. "players.player_name" → "players__player_name"
pub fn to_pre_agg_col_name(qualified: &str) -> String {
    qualified.replace('.', "__")
}

/// Generate the parquet column name for a pre-agg component column.
/// e.g. ("player_stats.goals", "sum") → "player_stats__goals__sum"
pub fn pre_agg_component_col_name(qualified_col: &str, component: &str) -> String {
    format!("{}__{component}", to_pre_agg_col_name(qualified_col))
}

/// Maps a DataFusion aggregate function name to the pre-agg components it requires.
pub fn agg_needed_components(agg_fn_name: &str) -> Option<Vec<&'static str>> {
    let kind = match agg_fn_name {
        "sum" | "SUM" => AggKind::Sum,
        "count" | "COUNT" => AggKind::Count,
        "min" | "MIN" => AggKind::Min,
        "max" | "MAX" => AggKind::Max,
        "avg" | "AVG" | "mean" => AggKind::Mean,
        "stddev" | "STDDEV" | "stddev_pop" | "STDDEV_POP" => AggKind::StdVar,
        "variance" | "VARIANCE" | "var_pop" | "VAR_POP" => AggKind::StdVar,
        _ => return None,
    };
    Some(kind.components().to_vec())
}

/// A pre-aggregation *definition*: which columns to group by and which agg
/// components to store.
///
/// Deliberately carries no row count or write timestamp — those describe a
/// particular file on disk, not the definition, and live on
/// [`PreAggVersion`](crate::model_components::pre_agg_store::PreAggVersion).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreAggregation {
    pub name: String,
    pub group_by: Vec<String>,
    /// col_name -> list of component suffixes stored (e.g. "orders.amount" -> ["sum", "count"])
    pub aggregations: HashMap<String, Vec<String>>,
}

impl PreAggregation {
    /// Create a new PreAggregation, expanding raw agg specs to component lists.
    pub fn new(
        name: String,
        group_by: Vec<String>,
        raw_aggregations: HashMap<String, Vec<String>>,
    ) -> Result<Self, String> {
        if group_by.is_empty() {
            return Err("group_by must not be empty".into());
        }
        if raw_aggregations.is_empty() {
            return Err("aggregations must not be empty".into());
        }

        let mut aggregations: HashMap<String, Vec<String>> = HashMap::new();
        for (col, agg_names) in &raw_aggregations {
            let mut components = HashSet::new();
            for agg_name in agg_names {
                for comp in agg_expansion(agg_name)? {
                    components.insert(comp.to_string());
                }
            }
            let mut sorted: Vec<String> = components.into_iter().collect();
            sorted.sort();
            aggregations.insert(col.clone(), sorted);
        }

        Ok(PreAggregation {
            name,
            group_by,
            aggregations,
        })
    }

    /// Check if this pre-agg covers the requested column requirements.
    ///
    /// A pre-agg covers a request if:
    /// 1. All non-aggregate column references (group-by, filter, projection, join keys, etc.)
    ///    are in self.group_by
    /// 2. For each (col, needed_components) aggregate column reference, all components are stored
    pub fn covers(
        &self,
        non_agg_cols: &[String],
        agg_cols: &HashMap<String, HashSet<String>>,
    ) -> bool {
        let my_group_set: HashSet<&str> = self.group_by.iter().map(|s| s.as_str()).collect();

        // Every non-aggregate column reference must be in this pre-agg's group_by
        for col in non_agg_cols {
            if !my_group_set.contains(col.as_str()) {
                return false;
            }
        }

        // Check agg component coverage
        for (col, needed) in agg_cols {
            match self.aggregations.get(col) {
                None => {
                    return false;
                }
                Some(stored) => {
                    let stored_set: HashSet<&str> = stored.iter().map(|s| s.as_str()).collect();
                    for comp in needed {
                        if !stored_set.contains(comp.as_str()) {
                            return false;
                        }
                    }
                }
            }
        }

        true
    }
}

// ── DataModel write methods ───────────────────────────────────────────────────

impl DataModel {
    /// Compute and write parquet files for the named pre-aggregations, then
    /// reclaim whatever that supersedes.
    pub fn write_pre_aggs(&self, names: &[&str]) -> Result<(), String> {
        let store = self
            .0
            .pre_agg_store
            .as_ref()
            .ok_or("no pre-aggregations registered on this DataModel")?;

        let pre_agg_defs: Vec<PreAggregation> = names
            .iter()
            .map(|&name| {
                store
                    .def(name)
                    .cloned()
                    .ok_or_else(|| format!("pre-aggregation '{name}' not found"))
            })
            .collect::<Result<Vec<_>, _>>()?;

        for pa_def in &pre_agg_defs {
            self.write_pre_agg(pa_def)?;
        }

        // Safe to run unconditionally: reclaim only deletes versions it can prove
        // nobody holds. This is what lets a rebuild clean up after itself.
        store.reclaim();
        Ok(())
    }

    /// Delete superseded pre-aggregation versions that are provably unreferenced,
    /// and sweep untracked leftovers past their grace window.
    ///
    /// A version is removed only once no in-flight query holds a lease on it (see
    /// [`LeaseScope`](crate::model_components::pre_agg_store::LeaseScope)) *and*
    /// it is past the retired grace floor. Cheap enough to call on a schedule.
    pub fn reclaim_pre_agg_versions(&self) -> Result<ReclaimReport, String> {
        let store = self
            .0
            .pre_agg_store
            .as_ref()
            .ok_or("no pre-aggregations registered on this DataModel")?;
        Ok(store.reclaim())
    }

    /// How long a superseded version is kept even once it looks unreferenced.
    /// Zero is correct and safe for a single-process deployment — the reference
    /// count is the real guard. Raise it if another process may share the
    /// directory, since the count cannot see readers outside this process.
    pub fn set_pre_agg_retired_grace(&self, grace: std::time::Duration) {
        if let Some(store) = &self.0.pre_agg_store {
            store.set_retired_grace(grace);
        }
    }

    /// How long an untracked versioned file (left by a crash or another process)
    /// is kept before being swept.
    pub fn set_pre_agg_orphan_grace(&self, grace: std::time::Duration) {
        if let Some(store) = &self.0.pre_agg_store {
            store.set_orphan_grace(grace);
        }
    }

    fn write_pre_agg(&self, pa: &PreAggregation) -> Result<(), String> {
        use datafusion::dataframe::DataFrameWriteOptions;
        use datafusion::functions_aggregate::expr_fn::{count, max, min, sum};
        use datafusion::prelude::Expr;

        let store = self
            .0
            .pre_agg_store
            .as_ref()
            .ok_or("pre_agg_path not set on DataModel")?;
        let path = store.base_path().to_path_buf();

        let all_col_names = pa.group_by.iter().chain(pa.aggregations.keys());
        let mut referenced_tables: Vec<String> = all_col_names
            .filter_map(|c| c.split_once('.').map(|(t, _)| t.to_string()))
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        referenced_tables.sort();

        if referenced_tables.is_empty() {
            return Err("all columns must be table-qualified (e.g. orders.amount)".into());
        }

        let base_table = self.0.joins.find_reachable_base(&referenced_tables)?;

        let non_agg_str: std::collections::HashSet<String> = pa.group_by.iter().cloned().collect();
        let agg_str: HashMap<String, Vec<String>> = pa
            .aggregations
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        let mut df = self
            .get_df_table(&base_table, &non_agg_str, &agg_str, false)
            .map_err(|e| e.to_string())?
            .inner;

        let group_by_exprs: Vec<Expr> = pa
            .group_by
            .iter()
            .map(|c| col(c.as_str()).alias(to_pre_agg_col_name(c).as_str()))
            .collect();

        let mut agg_exprs: Vec<Expr> = Vec::new();
        for (qcol, components) in &pa.aggregations {
            for component in components {
                let alias = pre_agg_component_col_name(qcol, component);
                let qcol_expr = || col(qcol.as_str());
                let expr = match component.as_str() {
                    "sum" => sum(qcol_expr()).alias(&alias),
                    "count" => count(qcol_expr()).alias(&alias),
                    "min" => min(qcol_expr()).alias(&alias),
                    "max" => max(qcol_expr()).alias(&alias),
                    "sumsq" => sum(qcol_expr() * qcol_expr()).alias(&alias),
                    other => return Err(format!("unknown pre-agg component '{other}'")),
                };
                agg_exprs.push(expr);
            }
        }

        df = df
            .aggregate(group_by_exprs, agg_exprs)
            .map_err(|e| format!("failed to aggregate for pre-agg: {e}"))?;

        // Two rebuilds inside one millisecond would otherwise collide on the same
        // filename, and publishing would retire the file it just wrote.
        let mut unix_millis = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis();
        let mut versioned_path = path.join(format!("{}.{unix_millis}.parquet", pa.name));
        while versioned_path.exists() {
            unix_millis += 1;
            versioned_path = path.join(format!("{}.{unix_millis}.parquet", pa.name));
        }
        let versioned_filename = format!("{}.{unix_millis}.parquet", pa.name);
        let pointer_path = path.join(format!("{}.current", pa.name));
        let tmp_pointer_path = path.join(format!("{}.current.tmp", pa.name));
        let versioned_str = versioned_path.to_string_lossy().to_string();

        match tokio::runtime::Handle::try_current() {
            Ok(handle) => tokio::task::block_in_place(|| {
                handle.block_on(df.write_parquet(
                    &versioned_str,
                    DataFrameWriteOptions::new(),
                    None,
                ))
            }),
            Err(_) => tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("tokio runtime")
                .block_on(df.write_parquet(&versioned_str, DataFrameWriteOptions::new(), None)),
        }
        .map_err(|e| format!("failed to write parquet: {e}"))?;

        // One footer read gives both the row count and the schema, so the version
        // opens without any async schema inference.
        let version = PreAggVersion::open(&pa.name, versioned_path)?;

        // Atomic pointer swap: write to tmp then rename (POSIX atomic).
        std::fs::write(&tmp_pointer_path, &versioned_filename)
            .map_err(|e| format!("failed to write tmp pointer: {e}"))?;
        std::fs::rename(&tmp_pointer_path, &pointer_path)
            .map_err(|e| format!("failed to rename pointer: {e}"))?;

        store.publish(Arc::new(version));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_pre_agg() -> PreAggregation {
        PreAggregation::new(
            "daily_revenue".into(),
            vec!["orders.date".into(), "orders.region".into()],
            HashMap::from([
                ("orders.amount".into(), vec!["sum".into(), "mean".into()]),
                ("orders.quantity".into(), vec!["sum".into()]),
            ]),
        )
        .unwrap()
    }

    #[test]
    fn test_agg_expansion() {
        assert_eq!(agg_expansion("sum").unwrap(), vec!["sum"]);
        assert!(agg_expansion("mean").unwrap().contains(&"sum"));
        assert!(agg_expansion("mean").unwrap().contains(&"count"));
        assert!(agg_expansion("unknown").is_err());
    }

    #[test]
    fn test_pre_agg_creation() {
        let pa = sample_pre_agg();
        // "mean" expands to sum+count, "sum" adds sum → merged = {sum, count}
        assert!(pa.aggregations["orders.amount"].contains(&"sum".to_string()));
        assert!(pa.aggregations["orders.amount"].contains(&"count".to_string()));
        assert_eq!(pa.aggregations["orders.quantity"], vec!["sum".to_string()]);
    }

    #[test]
    fn test_covers_exact() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.date".into(), "orders.region".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_subset_group_by() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.region".into()]; // subset
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_missing_group_col() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.store".into()]; // not in pre-agg
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(!pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_missing_agg_component() {
        let pa = sample_pre_agg();
        let non_agg_cols = vec!["orders.date".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sumsq".to_string()]), // not stored
        )]);
        assert!(!pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_filter_column_in_group_by() {
        let pa = sample_pre_agg();
        // Filter on region (non-agg, in group_by) → ok
        let non_agg_cols = vec!["orders.date".into(), "orders.region".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(pa.covers(&non_agg_cols, &agg_cols));
    }

    #[test]
    fn test_covers_filter_column_not_in_group_by() {
        let pa = sample_pre_agg();
        // Filter on store, which is NOT in group_by → rejected
        let non_agg_cols = vec!["orders.date".into(), "orders.store".into()];
        let agg_cols = HashMap::from([(
            "orders.amount".to_string(),
            HashSet::from(["sum".to_string()]),
        )]);
        assert!(!pa.covers(&non_agg_cols, &agg_cols));
    }
}
