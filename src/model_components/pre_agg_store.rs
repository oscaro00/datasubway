//! Version registry for pre-aggregation parquet files.
//!
//! Each rebuild of a pre-aggregation writes a new immutable file
//! `<name>.<unix_millis>.parquet` and swaps the `<name>.current` pointer. The old
//! file cannot simply be unlinked: a query resolves its version synchronously at
//! plan-build time but does not read the bytes until `collect().await`, and
//! DataFusion opens the file lazily at execution time (so POSIX unlink-on-open
//! does not protect us — `object_store`'s local backend re-opens by path per range
//! request).
//!
//! So each physical version is reference counted. [`PreAggVersion`] is handed out
//! as an `Arc` and pinned for the life of a query by [`LeaseScope`]; [`PreAggStore::reclaim`]
//! deletes a superseded version only once the count proves nobody holds it. A
//! configurable grace floor sits on top as headroom for readers the count cannot
//! see — a raw `DataFrame` used outside `DataModel::execute`, or another process
//! sharing the directory.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

use async_trait::async_trait;
use datafusion::arrow::datatypes::{Fields, Schema, SchemaRef};
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::Constraints;
use datafusion::datasource::file_format::parquet::ParquetFormat;
use datafusion::datasource::listing::{
    ListingOptions, ListingTable, ListingTableConfig, ListingTableUrl,
};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use tracing::debug;

use crate::model_components::pre_aggregations::PreAggregation;

/// Default time a superseded version is kept even once its refcount says it is
/// unreferenced. Safe to set to zero for a single-process deployment — the
/// refcount is the real guard — but non-zero gives headroom for readers this
/// process cannot see.
pub const DEFAULT_RETIRED_GRACE: Duration = Duration::from_secs(60);

/// Default time an untracked versioned file (left by a crash, or written by
/// another process) is kept before being swept.
pub const DEFAULT_ORPHAN_GRACE: Duration = Duration::from_secs(3600);

// ── PreAggVersion ─────────────────────────────────────────────────────────────

/// One immutable, physical version of a pre-aggregation on disk.
///
/// It is also its own [`TableProvider`], so a logical plan that scans this version
/// holds a reference to it — a free extra layer of protection on top of the
/// explicit lease taken by [`LeaseScope`].
pub struct PreAggVersion {
    pub name: String,
    /// Unix millis parsed out of the filename; `0` if the name is not versioned.
    pub version: u128,
    pub path: PathBuf,
    pub row_count: u64,
    pub written_at: SystemTime,
    inner: ListingTable,
}

impl fmt::Debug for PreAggVersion {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreAggVersion")
            .field("name", &self.name)
            .field("version", &self.version)
            .field("path", &self.path)
            .field("row_count", &self.row_count)
            .finish()
    }
}

impl PreAggVersion {
    /// Open a written version. A single parquet footer read yields both the row
    /// count and the Arrow schema, so the `ListingTable` is built without schema
    /// inference — no async, no blocking listing on the query path.
    pub(crate) fn open(name: &str, path: PathBuf) -> Result<Self, String> {
        let (row_count, schema) = read_footer(&path)?;
        let written_at = std::fs::metadata(&path)
            .and_then(|m| m.modified())
            .unwrap_or_else(|_| SystemTime::now());

        let url = ListingTableUrl::parse(path.to_string_lossy())
            .map_err(|e| format!("bad pre-agg path '{}': {e}", path.display()))?;
        let options =
            ListingOptions::new(Arc::new(ParquetFormat::default())).with_file_extension(".parquet");
        let config = ListingTableConfig::new(url)
            .with_listing_options(options)
            .with_schema(schema);
        let inner = ListingTable::try_new(config)
            .map_err(|e| format!("failed to open pre-agg '{}': {e}", path.display()))?;

        Ok(PreAggVersion {
            name: name.to_string(),
            version: version_of(name, &path).unwrap_or(0),
            path,
            row_count,
            written_at,
            inner,
        })
    }

    /// How long ago this version was written, per the file's mtime.
    pub fn age(&self) -> Duration {
        self.written_at.elapsed().unwrap_or_default()
    }
}

#[async_trait]
impl TableProvider for PreAggVersion {
    fn schema(&self) -> SchemaRef {
        self.inner.schema()
    }

    fn constraints(&self) -> Option<&Constraints> {
        self.inner.constraints()
    }

    fn table_type(&self) -> TableType {
        self.inner.table_type()
    }

    fn get_table_definition(&self) -> Option<&str> {
        self.inner.get_table_definition()
    }

    fn get_column_default(&self, column: &str) -> Option<&Expr> {
        self.inner.get_column_default(column)
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        self.inner.scan(state, projection, filters, limit).await
    }

    // `ListingTable` implements `scan_with_args` in its own right (it is where
    // limit and pushdown handling lives), so delegate it rather than letting the
    // trait's default route back through `scan`.
    async fn scan_with_args<'a>(
        &self,
        state: &dyn Session,
        args: ScanArgs<'a>,
    ) -> datafusion::common::Result<ScanResult> {
        self.inner.scan_with_args(state, args).await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        self.inner.supports_filters_pushdown(filters)
    }
}

// ── File-name and footer helpers ──────────────────────────────────────────────

/// Parse the version millis out of `<name>.<digits>.parquet`, anchored on the
/// full filename. Returns `None` for anything else — so a pre-agg named `goals`
/// never matches a file belonging to one named `goals.foo`.
pub(crate) fn version_of(name: &str, path: &Path) -> Option<u128> {
    version_of_filename(name, path.file_name()?.to_str()?)
}

pub(crate) fn version_of_filename(name: &str, filename: &str) -> Option<u128> {
    filename
        .strip_prefix(name)?
        .strip_prefix('.')?
        .strip_suffix(".parquet")?
        .parse()
        .ok()
}

/// Reads a parquet file's (or directory of parquet parts') footer to return the
/// total row count and the Arrow schema.
///
/// Field metadata is cleared to match `ParquetFormat`'s own inference, which
/// defaults to `skip_metadata = true`; leaving it on would make the schema we
/// hand `ListingTable` disagree with the file schema it sees at scan time.
fn read_footer(path: &Path) -> Result<(u64, SchemaRef), String> {
    // Via DataFusion's re-export rather than a direct `parquet` dependency:
    // a separately-versioned parquet crate pulls in its own arrow stack, and
    // its `RecordBatch` is then a different type from DataFusion's.
    use datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

    let read_one = |p: &Path| -> Result<(i64, SchemaRef), String> {
        let f =
            std::fs::File::open(p).map_err(|e| format!("cannot open '{}': {e}", p.display()))?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(f)
            .map_err(|e| format!("cannot read parquet footer '{}': {e}", p.display()))?;
        let rows = builder.metadata().file_metadata().num_rows();
        Ok((rows, Arc::clone(builder.schema())))
    };

    let (rows, schema) = if path.is_dir() {
        let mut total = 0i64;
        let mut schema: Option<SchemaRef> = None;
        let entries = std::fs::read_dir(path)
            .map_err(|e| format!("cannot read '{}': {e}", path.display()))?;
        for entry in entries.flatten() {
            let p = entry.path();
            if p.extension().is_some_and(|ext| ext == "parquet") {
                let (rows, s) = read_one(&p)?;
                total += rows;
                schema.get_or_insert(s);
            }
        }
        let schema = schema.ok_or_else(|| format!("no parquet parts in '{}'", path.display()))?;
        (total, schema)
    } else {
        read_one(path)?
    };

    let cleared = Schema::new(
        schema
            .fields()
            .iter()
            .map(|f| f.as_ref().clone().with_metadata(Default::default()))
            .collect::<Fields>(),
    );
    Ok((rows.max(0) as u64, Arc::new(cleared)))
}

/// Reads the pointer file for a pre-agg and returns the full path to the current
/// versioned parquet. The pointer holds a bare filename, not a path.
pub(crate) fn resolve_pointer(base_path: &Path, name: &str) -> Option<PathBuf> {
    let pointer = base_path.join(format!("{name}.current"));
    let versioned = std::fs::read_to_string(&pointer).ok()?;
    Some(base_path.join(versioned.trim()))
}

fn remove_path(path: &Path) -> std::io::Result<()> {
    if path.is_dir() {
        std::fs::remove_dir_all(path)
    } else {
        std::fs::remove_file(path)
    }
}

fn older_than(path: &Path, grace: Duration) -> bool {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .map(|t| t.elapsed().unwrap_or(Duration::MAX) >= grace)
        .unwrap_or(false)
}

// ── PreAggStore ───────────────────────────────────────────────────────────────

/// What a call to [`PreAggStore::reclaim`] did.
#[derive(Debug, Default)]
pub struct ReclaimReport {
    pub deleted: Vec<PathBuf>,
    /// Superseded, but a query still holds a lease on it.
    pub retained_in_use: Vec<PathBuf>,
    /// Superseded or orphaned, but still inside its grace window.
    pub retained_in_grace: Vec<PathBuf>,
    pub errors: Vec<(PathBuf, String)>,
}

struct Retired {
    version: Arc<PreAggVersion>,
    retired_at: Instant,
}

#[derive(Default)]
struct State {
    current: HashMap<String, Arc<PreAggVersion>>,
    retired: Vec<Retired>,
}

pub(crate) struct PreAggStore {
    base_path: PathBuf,
    /// Definitions, immutable after construction. Row counts and timestamps live
    /// on [`PreAggVersion`] instead — they describe a file, not a definition.
    defs: Vec<PreAggregation>,
    state: RwLock<State>,
    retired_grace_ms: AtomicU64,
    orphan_grace_ms: AtomicU64,
}

impl PreAggStore {
    /// Build a store and rehydrate the current version of each definition from
    /// the `<name>.current` pointers already on disk.
    pub(crate) fn load(base_path: PathBuf, defs: Vec<PreAggregation>) -> Self {
        let mut current = HashMap::new();
        for def in &defs {
            let Some(path) = resolve_pointer(&base_path, &def.name) else {
                continue;
            };
            if !path.exists() {
                debug!(pre_agg = %def.name, path = %path.display(), "pointer target missing");
                continue;
            }
            match PreAggVersion::open(&def.name, path) {
                Ok(v) => {
                    current.insert(def.name.clone(), Arc::new(v));
                }
                Err(e) => debug!(pre_agg = %def.name, err = %e, "failed to open current version"),
            }
        }

        PreAggStore {
            base_path,
            defs,
            state: RwLock::new(State {
                current,
                retired: Vec::new(),
            }),
            retired_grace_ms: AtomicU64::new(DEFAULT_RETIRED_GRACE.as_millis() as u64),
            orphan_grace_ms: AtomicU64::new(DEFAULT_ORPHAN_GRACE.as_millis() as u64),
        }
    }

    pub(crate) fn base_path(&self) -> &Path {
        &self.base_path
    }

    pub(crate) fn def(&self, name: &str) -> Option<&PreAggregation> {
        self.defs.iter().find(|d| d.name == name)
    }

    pub(crate) fn set_retired_grace(&self, grace: Duration) {
        self.retired_grace_ms
            .store(grace.as_millis() as u64, Ordering::Relaxed);
    }

    pub(crate) fn set_orphan_grace(&self, grace: Duration) {
        self.orphan_grace_ms
            .store(grace.as_millis() as u64, Ordering::Relaxed);
    }

    /// The live version of `name`, if one has been written.
    #[cfg(test)]
    pub(crate) fn acquire(&self, name: &str) -> Option<Arc<PreAggVersion>> {
        self.state.read().unwrap().current.get(name).cloned()
    }

    /// Every live version whose definition covers the request, cheapest first.
    ///
    /// Replaces the old "filter definitions, then re-read a pointer file per
    /// candidate" two-step with a single in-memory pass. `max_age` drops versions
    /// written longer ago than the caller will accept.
    pub(crate) fn covering_versions(
        &self,
        non_agg_cols: &[String],
        agg_cols: &HashMap<String, HashSet<String>>,
        max_age: Option<u64>,
    ) -> Vec<Arc<PreAggVersion>> {
        let state = self.state.read().unwrap();
        let mut out: Vec<Arc<PreAggVersion>> = self
            .defs
            .iter()
            .filter(|def| def.covers(non_agg_cols, agg_cols))
            .filter_map(|def| state.current.get(&def.name))
            .filter(|v| max_age.is_none_or(|secs| v.age() <= Duration::from_secs(secs)))
            .map(Arc::clone)
            .collect();
        out.sort_by_key(|v| v.row_count);
        out
    }

    /// Every live version that satisfies `predicate` on its definition, cheapest
    /// first. Used by the column-values path, which selects on shapes that
    /// `covers()` does not express.
    pub(crate) fn versions_where(
        &self,
        max_age: Option<u64>,
        mut predicate: impl FnMut(&PreAggregation) -> bool,
    ) -> Vec<Arc<PreAggVersion>> {
        let state = self.state.read().unwrap();
        let mut out: Vec<Arc<PreAggVersion>> = self
            .defs
            .iter()
            .filter(|def| predicate(def))
            .filter_map(|def| state.current.get(&def.name))
            .filter(|v| max_age.is_none_or(|secs| v.age() <= Duration::from_secs(secs)))
            .map(Arc::clone)
            .collect();
        out.sort_by_key(|v| v.row_count);
        out
    }

    /// Make `version` current, retiring whatever it replaces.
    pub(crate) fn publish(&self, version: Arc<PreAggVersion>) {
        let new_path = version.path.clone();
        let mut state = self.state.write().unwrap();
        let Some(old) = state.current.insert(version.name.clone(), version) else {
            return;
        };
        // A rebuild that lands on the same filename (two writes inside one
        // millisecond) would otherwise retire the file it just published.
        if old.path == new_path {
            return;
        }
        state.retired.push(Retired {
            version: old,
            retired_at: Instant::now(),
        });
    }

    /// Delete superseded versions that are provably unreferenced, then sweep
    /// untracked leftovers.
    pub(crate) fn reclaim(&self) -> ReclaimReport {
        let retired_grace = Duration::from_millis(self.retired_grace_ms.load(Ordering::Relaxed));
        let orphan_grace = Duration::from_millis(self.orphan_grace_ms.load(Ordering::Relaxed));
        let mut report = ReclaimReport::default();

        let mut state = self.state.write().unwrap();

        let mut kept = Vec::new();
        for entry in std::mem::take(&mut state.retired) {
            let path = entry.version.path.clone();
            if entry.retired_at.elapsed() < retired_grace {
                report.retained_in_grace.push(path);
                kept.push(entry);
            } else if Arc::strong_count(&entry.version) > 1 {
                // Someone still holds a lease on this version.
                report.retained_in_use.push(path);
                kept.push(entry);
            } else if let Err(e) = remove_path(&path) {
                // Dropped from `retired` regardless; the orphan sweep will retry.
                report.errors.push((path, e.to_string()));
            } else {
                debug!(pre_agg = %entry.version.name, path = %path.display(), "reclaimed pre-agg version");
                report.deleted.push(path);
            }
        }
        state.retired = kept;

        let tracked: HashSet<PathBuf> = state
            .current
            .values()
            .map(|v| v.path.clone())
            .chain(state.retired.iter().map(|r| r.version.path.clone()))
            .collect();
        drop(state);

        self.sweep_orphans(&tracked, orphan_grace, &mut report);
        report
    }

    /// Remove versioned files belonging to a known definition that the store is
    /// not tracking — left by a crash, or by another process — plus stale pointer
    /// temp files. Only names matching a registered definition are ever touched.
    fn sweep_orphans(
        &self,
        tracked: &HashSet<PathBuf>,
        grace: Duration,
        report: &mut ReclaimReport,
    ) {
        let Ok(entries) = std::fs::read_dir(&self.base_path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name();
            let Some(file_name) = file_name.to_str() else {
                continue;
            };

            // A crash between writing and renaming the pointer temp file leaves
            // this behind, and nothing else cleans it up.
            if file_name.ends_with(".current.tmp") {
                if older_than(&path, grace) {
                    match std::fs::remove_file(&path) {
                        Ok(()) => report.deleted.push(path),
                        Err(e) => report.errors.push((path, e.to_string())),
                    }
                }
                continue;
            }

            let belongs = self
                .defs
                .iter()
                .any(|def| version_of_filename(&def.name, file_name).is_some());
            if !belongs || tracked.contains(&path) {
                continue;
            }
            if !older_than(&path, grace) {
                report.retained_in_grace.push(path);
                continue;
            }
            match remove_path(&path) {
                Ok(()) => {
                    debug!(path = %path.display(), "swept orphaned pre-agg version");
                    report.deleted.push(path);
                }
                Err(e) => report.errors.push((path, e.to_string())),
            }
        }
    }
}

// ── Query lease scope ─────────────────────────────────────────────────────────
//
// Measures are `fn(&DataModel, &AggContext) -> Result<DataFrame>` and call
// `DataFrameRecorder::build()` internally, which returns a bare `DataFrame` — so
// there is no return channel for a lease without breaking every user-written
// measure signature. And a lease cannot ride inside the plan either:
// `DataFrame::collect` hands the physical plan to `execute_stream` by value and
// drops it once the stream exists, so nothing from the plan survives into
// execution.
//
// Hence an ambient scope. It is sound as a thread-local because plan building is
// entirely synchronous — `build_agg_frame` → measure closure → `build()` →
// `get_df_table` all run on one thread, and the only await is `collect()`, which
// happens after `take()` has moved the leases into a binding held across it.

#[derive(Default)]
struct ActiveQuery {
    leases: Vec<Arc<PreAggVersion>>,
    /// `pre_agg_valid_secs` for the query being planned. Carried here because
    /// `DataFrameRecorder::build()` reaches `get_df_table` without the context.
    max_age: Option<u64>,
}

thread_local! {
    static ACTIVE: RefCell<Option<ActiveQuery>> = const { RefCell::new(None) };
}

/// Collects the pre-agg versions bound while building one query's plan.
pub(crate) struct LeaseScope {
    prev: Option<ActiveQuery>,
    taken: bool,
}

impl LeaseScope {
    pub(crate) fn enter(max_age: Option<u64>) -> Self {
        let prev = ACTIVE.with(|cell| {
            cell.borrow_mut().replace(ActiveQuery {
                leases: Vec::new(),
                max_age,
            })
        });
        LeaseScope { prev, taken: false }
    }

    /// End the scope and return the leases acquired inside it. Hold the result
    /// for as long as the query may still read those files.
    pub(crate) fn take(mut self) -> Vec<Arc<PreAggVersion>> {
        self.taken = true;
        let prev = self.prev.take();
        ACTIVE
            .with(|cell| std::mem::replace(&mut *cell.borrow_mut(), prev))
            .map(|active| active.leases)
            .unwrap_or_default()
    }
}

impl Drop for LeaseScope {
    fn drop(&mut self) {
        if self.taken {
            return;
        }
        // Plan building bailed out early; restore the enclosing scope.
        let prev = self.prev.take();
        ACTIVE.with(|cell| *cell.borrow_mut() = prev);
    }
}

/// Pin `version` for the active query, if one is building a plan on this thread.
pub(crate) fn record_lease(version: &Arc<PreAggVersion>) {
    ACTIVE.with(|cell| {
        if let Some(active) = cell.borrow_mut().as_mut() {
            active.leases.push(Arc::clone(version));
        }
    });
}

/// The staleness bound the active query asked for, if any.
pub(crate) fn active_max_age() -> Option<u64> {
    ACTIVE.with(|cell| cell.borrow().as_ref().and_then(|a| a.max_age))
}
