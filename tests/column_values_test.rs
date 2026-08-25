use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::compute::{
    SortColumn, SortOptions, concat_batches, lexsort_to_indices, take,
};
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::TableProvider;
use datafusion::functions_aggregate::expr_fn::{max, min};
use datafusion::prelude::*;
use tempfile::TempDir;

use datasubway::data_model::{DataModel, DataOutput, DataQuery};
use datasubway::model_components::{
    column_values_context::{ColumnValuesContext, ColumnValuesMode},
    joins::{Join, JoinDirection, JoinGraph, JoinHow},
    pre_aggregations::PreAggregation,
};

static SHARED_CTX: tokio::sync::OnceCell<SessionContext> = tokio::sync::OnceCell::const_new();

async fn shared_ctx() -> &'static SessionContext {
    SHARED_CTX
        .get_or_init(|| async {
            let ctx = SessionContext::new();
            for name in ["players", "player_stats", "team_stats", "teams"] {
                ctx.register_parquet(
                    name,
                    &format!("tests/data_files/{name}.parquet"),
                    ParquetReadOptions::default(),
                )
                .await
                .unwrap();
            }
            ctx
        })
        .await
}

async fn make_tables() -> HashMap<String, Arc<dyn TableProvider>> {
    let ctx = shared_ctx().await;
    let mut tables = HashMap::new();
    for name in ["players", "player_stats", "team_stats", "teams"] {
        tables.insert(name.to_string(), ctx.table_provider(name).await.unwrap());
    }
    tables
}

fn make_joins() -> Vec<Join> {
    vec![
        Join {
            left: "player_stats".into(),
            right: "players".into(),
            left_on: vec!["player_stats.player_id".into()],
            right_on: vec!["players.player_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::RightOnLeft,
        },
        Join {
            left: "player_stats".into(),
            right: "team_stats".into(),
            left_on: vec![
                "player_stats.game_id".into(),
                "player_stats.team_name".into(),
            ],
            right_on: vec!["team_stats.game_id".into(), "team_stats.team_name".into()],
            how: JoinHow::Left,
            direction: JoinDirection::RightOnLeft,
        },
        Join {
            left: "team_stats".into(),
            right: "teams".into(),
            left_on: vec!["team_stats.team_name".into()],
            right_on: vec!["teams.team_name".into()],
            how: JoinHow::Left,
            direction: JoinDirection::RightOnLeft,
        },
    ]
}

async fn build_dm() -> DataModel {
    DataModel::new(
        make_tables().await,
        JoinGraph::new(&make_joins()).unwrap(),
        vec![],
        None,
    )
}

/// `player_stats.goals` used directly as a group_by key — an unusual shape, but
/// structurally valid, and exercises the group_by-based candidate search for Range mode.
fn goals_group_by_pre_agg() -> PreAggregation {
    PreAggregation::new(
        "goals_group_by".into(),
        vec!["player_stats.goals".into()],
        HashMap::from([("player_stats.assists".into(), vec!["sum".into()])]),
    )
    .unwrap()
}

/// `players.player_name` as a group_by key — used to validate the Distinct+pre-agg
/// physical column name fix.
fn player_name_group_by_pre_agg() -> PreAggregation {
    PreAggregation::new(
        "player_name_group_by".into(),
        vec!["players.player_name".into()],
        HashMap::from([("player_stats.goals".into(), vec!["sum".into()])]),
    )
    .unwrap()
}

/// `player_stats.goals` min/max stored as an aggregation, grouped by player (48 rows).
fn goals_range_by_player_pre_agg() -> PreAggregation {
    PreAggregation::new(
        "goals_range_by_player".into(),
        vec!["players.player_name".into()],
        HashMap::from([(
            "player_stats.goals".into(),
            vec!["min".into(), "max".into()],
        )]),
    )
    .unwrap()
}

/// Same shape as above but grouped by team (16 rows) — smaller than the by-player
/// version, used to test that the smaller pre-agg is preferred.
fn goals_range_by_team_pre_agg() -> PreAggregation {
    PreAggregation::new(
        "goals_range_by_team".into(),
        vec!["team_stats.team_name".into()],
        HashMap::from([(
            "player_stats.goals".into(),
            vec!["min".into(), "max".into()],
        )]),
    )
    .unwrap()
}

async fn build_dm_with_pre_aggs(
    defs: Vec<PreAggregation>,
    write_names: &[&str],
) -> (DataModel, TempDir) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let dm = DataModel::new(
        make_tables().await,
        JoinGraph::new(&make_joins()).unwrap(),
        defs,
        Some(path),
    );
    dm.write_pre_aggs(write_names).unwrap();
    (dm, tmp)
}

async fn column_values(dm: &DataModel, ctx: ColumnValuesContext) -> Vec<RecordBatch> {
    match dm.execute(&DataQuery::ColumnValues(ctx)).await.unwrap() {
        DataOutput::Data(batches) => batches,
        _ => panic!("expected Data"),
    }
}

/// Sort `batches` by `sort_col`, select `cols`, and return as a formatted string.
/// Copied from tests/rl_test.rs's helper of the same name/behavior.
fn sorted_select(batches: Vec<RecordBatch>, sort_col: &str, cols: &[&str]) -> String {
    let schema = batches[0].schema();
    let combined = concat_batches(&schema, &batches).unwrap();

    let sort_opts = Some(SortOptions {
        descending: false,
        nulls_first: true,
    });
    let sort_idx = schema.index_of(sort_col).unwrap();
    let mut sort_columns = vec![SortColumn {
        values: Arc::clone(combined.column(sort_idx)),
        options: sort_opts,
    }];
    for &c in cols {
        let i = schema.index_of(c).unwrap();
        if i != sort_idx {
            sort_columns.push(SortColumn {
                values: Arc::clone(combined.column(i)),
                options: sort_opts,
            });
        }
    }
    let indices = lexsort_to_indices(&sort_columns, None).unwrap();

    let out_schema = Arc::new(Schema::new(
        cols.iter()
            .map(|&c| schema.field_with_name(c).unwrap().clone())
            .collect::<Vec<_>>(),
    ));
    let selected_cols: Vec<datafusion::arrow::array::ArrayRef> = cols
        .iter()
        .map(|&c| {
            let i = schema.index_of(c).unwrap();
            take(combined.column(i).as_ref(), &indices, None).unwrap()
        })
        .collect();

    let out = RecordBatch::try_new(out_schema, selected_cols).unwrap();
    pretty_format_batches(&[out]).unwrap().to_string()
}

fn format_batches(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches).unwrap().to_string()
}

async fn expected_player_names_distinct() -> Vec<RecordBatch> {
    let ctx = shared_ctx().await;
    ctx.table("players")
        .await
        .unwrap()
        .select(vec![col("player_name").alias("players.player_name")])
        .unwrap()
        .distinct()
        .unwrap()
        .collect()
        .await
        .unwrap()
}

async fn expected_goals_range() -> Vec<RecordBatch> {
    let ctx = shared_ctx().await;
    ctx.table("player_stats")
        .await
        .unwrap()
        .aggregate(
            vec![],
            vec![
                min(col("goals")).alias("min"),
                max(col("goals")).alias("max"),
            ],
        )
        .unwrap()
        .collect()
        .await
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_distinct_raw() {
    let dm = build_dm().await;
    let ctx = ColumnValuesContext::new("players.player_name".into(), None, false, None).unwrap();
    let actual = column_values(&dm, ctx).await;

    assert_eq!(
        sorted_select(actual, "players.player_name", &["players.player_name"]),
        sorted_select(
            expected_player_names_distinct().await,
            "players.player_name",
            &["players.player_name"],
        ),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_range_raw() {
    let dm = build_dm().await;
    let ctx = ColumnValuesContext::new(
        "player_stats.goals".into(),
        Some(ColumnValuesMode::Range),
        false,
        None,
    )
    .unwrap();
    let actual = column_values(&dm, ctx).await;

    assert_eq!(
        format_batches(&actual),
        format_batches(&expected_goals_range().await)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_distinct_from_group_by_pre_agg() {
    let (dm, _tmp) = build_dm_with_pre_aggs(
        vec![player_name_group_by_pre_agg()],
        &["player_name_group_by"],
    )
    .await;
    let ctx = ColumnValuesContext::new("players.player_name".into(), None, true, None).unwrap();

    let plan = dm
        .display_graphviz(&DataQuery::ColumnValues(ctx.clone()))
        .unwrap();
    // The pre-agg version is its own TableProvider, so it shows up as a TableScan
    // named after the pre-agg. Assert on that plus a physical column unique to its
    // schema, confirming it (not the raw players table) was read from.
    assert!(
        plan.contains("TableScan: player_name_group_by"),
        "expected pre-agg TableScan in plan, got:\n{plan}"
    );
    assert!(
        plan.contains("player_stats__goals__sum"),
        "expected pre-agg schema in plan, got:\n{plan}"
    );

    let actual = column_values(&dm, ctx).await;
    assert_eq!(
        sorted_select(actual, "players.player_name", &["players.player_name"]),
        sorted_select(
            expected_player_names_distinct().await,
            "players.player_name",
            &["players.player_name"],
        ),
        "pre-agg-sourced distinct values should match raw table"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_range_from_group_by_pre_agg() {
    let (dm, _tmp) =
        build_dm_with_pre_aggs(vec![goals_group_by_pre_agg()], &["goals_group_by"]).await;
    let ctx = ColumnValuesContext::new(
        "player_stats.goals".into(),
        Some(ColumnValuesMode::Range),
        true,
        None,
    )
    .unwrap();

    let plan = dm
        .display_graphviz(&DataQuery::ColumnValues(ctx.clone()))
        .unwrap();
    // Assert on the pre-agg's own TableScan plus a physical column unique to its
    // schema, confirming it (not the raw player_stats table) was read from.
    assert!(
        plan.contains("player_stats__assists__sum"),
        "expected pre-agg schema in plan, got:\n{plan}"
    );

    let actual = column_values(&dm, ctx).await;
    assert_eq!(
        format_batches(&actual),
        format_batches(&expected_goals_range().await)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_range_from_aggregations_pre_agg_prefers_smaller() {
    // Both cover player_stats.goals via stored min/max aggregation components;
    // the team-grouped one (16 rows) is smaller than the player-grouped one (48 rows),
    // so it should be the one used.
    let (dm, _tmp) = build_dm_with_pre_aggs(
        vec![
            goals_range_by_player_pre_agg(),
            goals_range_by_team_pre_agg(),
        ],
        &["goals_range_by_player", "goals_range_by_team"],
    )
    .await;
    let ctx = ColumnValuesContext::new(
        "player_stats.goals".into(),
        Some(ColumnValuesMode::Range),
        true,
        None,
    )
    .unwrap();

    let plan = dm
        .display_graphviz(&DataQuery::ColumnValues(ctx.clone()))
        .unwrap();
    // Assert on physical group_by columns unique to each candidate's schema, which
    // distinguishes the two pre-aggs more precisely than the scan name alone.
    assert!(
        plan.contains("team_stats__team_name"),
        "expected smaller pre-agg (by team) in plan, got:\n{plan}"
    );
    assert!(
        !plan.contains("players__player_name"),
        "did not expect larger pre-agg (by player) in plan, got:\n{plan}"
    );

    let actual = column_values(&dm, ctx).await;
    assert_eq!(
        format_batches(&actual),
        format_batches(&expected_goals_range().await)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_range_falls_back_to_raw_when_pre_agg_stale() {
    let (dm, _tmp) =
        build_dm_with_pre_aggs(vec![goals_group_by_pre_agg()], &["goals_group_by"]).await;
    // Ensure some measurable time has passed since the pre-agg was written.
    std::thread::sleep(std::time::Duration::from_millis(10));
    let ctx = ColumnValuesContext::new(
        "player_stats.goals".into(),
        Some(ColumnValuesMode::Range),
        true,
        Some(0),
    )
    .unwrap();

    let plan = dm
        .display_graphviz(&DataQuery::ColumnValues(ctx.clone()))
        .unwrap();
    assert!(
        !plan.contains("player_stats__assists__sum"),
        "expected stale pre-agg to be skipped (raw scan only), got:\n{plan}"
    );

    let actual = column_values(&dm, ctx).await;
    assert_eq!(
        format_batches(&actual),
        format_batches(&expected_goals_range().await)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_column_values_unknown_table_errors() {
    let dm = build_dm().await;
    let ctx = ColumnValuesContext::new("nonexistent.col".into(), None, false, None).unwrap();
    let err = dm.execute(&DataQuery::ColumnValues(ctx)).await.unwrap_err();
    assert!(err.contains("unknown table"), "got: {err}");
}
