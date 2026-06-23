use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::compute::{
    SortColumn, SortOptions, concat_batches, lexsort_to_indices, take,
};
use datafusion::arrow::datatypes::Schema;
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::TableProvider;
use datafusion::common::Column;
use datafusion::functions_aggregate::expr_fn::{count, sum};
use datafusion::prelude::*;
use tempfile::TempDir;

use datasubway::column_expressions::column_context::{
    ColumnContext, ColumnInclude, ColumnPattern, allow,
};
use datasubway::data_model::{DataModel, DataOutput, DataQuery, normalize_schema};
use datasubway::model_components::{
    agg_context::AggContext,
    joins::{Join, JoinDirection, JoinGraph, JoinHow},
    measures::DfMeasure,
    pre_aggregations::PreAggregation,
};

static SHARED_CTX: tokio::sync::OnceCell<SessionContext> = tokio::sync::OnceCell::const_new();

async fn shared_ctx() -> &'static SessionContext {
    SHARED_CTX
        .get_or_init(|| async {
            let ctx = SessionContext::new();
            for name in [
                "players",
                "games",
                "player_stats",
                "team_stats",
                "teams",
                "groups",
                "groups_bridge",
            ] {
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
    let names = [
        "players",
        "games",
        "player_stats",
        "team_stats",
        "teams",
        "groups",
        "groups_bridge",
    ];
    let mut tables = HashMap::new();
    for name in names {
        tables.insert(name.to_string(), ctx.table_provider(name).await.unwrap());
    }
    tables
}

fn make_joins() -> Vec<Join> {
    vec![
        // player_stats is the central fact table
        Join {
            left: "player_stats".into(),
            right: "games".into(),
            left_on: vec!["player_stats.game_id".into()],
            right_on: vec!["games.game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
        },
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
        Join {
            left: "games".into(),
            right: "groups_bridge".into(),
            left_on: vec!["games.game_id".into()],
            right_on: vec!["groups_bridge.game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
        },
        Join {
            left: "groups_bridge".into(),
            right: "groups".into(),
            left_on: vec!["groups_bridge.group_id".into()],
            right_on: vec!["groups.group_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
        },
    ]
}

async fn build_dm() -> DataModel {
    let mut dm = DataModel::new(
        make_tables().await,
        JoinGraph::new(&make_joins()).unwrap(),
        vec![],
        None,
    );
    dm.add_measure(DfMeasure::new("player_goals", player_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("team_goals", team_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("game_count", game_count))
        .unwrap();
    dm
}

/// Pre-aggregation for player_goals grouped by player_name.
fn player_goals_pre_agg() -> PreAggregation {
    PreAggregation::new(
        "player_goals_by_player".into(),
        vec!["players.player_name".into()],
        HashMap::from([("player_stats.goals".into(), vec!["sum".into()])]),
    )
    .unwrap()
}

async fn build_dm_with_pre_agg() -> (DataModel, TempDir) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let mut dm = DataModel::new(
        make_tables().await,
        JoinGraph::new(&make_joins()).unwrap(),
        vec![player_goals_pre_agg()],
        Some(path),
    );
    dm.add_measure(DfMeasure::new("player_goals", player_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("team_goals", team_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("game_count", game_count))
        .unwrap();
    dm.write_pre_aggs(&["player_goals_by_player"]).unwrap();
    (dm, tmp)
}

fn player_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    let mut recorder = dm.table("player_stats");
    recorder.pre_agg_allowed = qc.use_pre_agg;
    recorder
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![sum(col("player_stats.goals")).alias("player_stats.goals")],
        )
        .build()
}

fn team_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    let mut recorder = dm.table("team_stats");
    recorder.pre_agg_allowed = qc.use_pre_agg;
    recorder
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![sum(col("team_stats.goals")).alias("team_stats.goals")],
        )
        .build()
}

fn game_count(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    let mut recorder = dm.table("games");
    recorder.pre_agg_allowed = qc.use_pre_agg;
    recorder
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![count(col("games.game_id")).alias("games.game_count")],
        )
        .build()
}

static SHARED_DM: tokio::sync::OnceCell<DataModel> = tokio::sync::OnceCell::const_new();

async fn shared_dm() -> &'static DataModel {
    SHARED_DM.get_or_init(|| async { build_dm().await }).await
}

static SHARED_PRE_AGG_DM: tokio::sync::OnceCell<(DataModel, TempDir)> =
    tokio::sync::OnceCell::const_new();

async fn shared_pre_agg_dm() -> &'static DataModel {
    &SHARED_PRE_AGG_DM
        .get_or_init(|| async { build_dm_with_pre_agg().await })
        .await
        .0
}

async fn dm_query(dm: &DataModel, qc: AggContext) -> Vec<RecordBatch> {
    match dm.execute(&DataQuery::Agg(qc)).await.unwrap() {
        DataOutput::Data(batches) => batches,
        _ => panic!("expected Data"),
    }
}

/// Sort `batches` by `sort_col`, select `cols`, and return as a formatted string.
/// Uses Arrow compute directly — no SessionContext overhead.
fn sorted_select(batches: Vec<RecordBatch>, sort_col: &str, cols: &[&str]) -> String {
    let schema = batches[0].schema();
    let combined = concat_batches(&schema, &batches).unwrap();

    let sort_opts = Some(SortOptions {
        descending: false,
        nulls_first: true,
    });
    let sort_idx = schema.index_of(sort_col).unwrap();
    // Primary: sort_col. Secondary: each selected column in order — stabilizes ties
    // (e.g. two rows with NULL primary key produced by different sides of a FULL JOIN).
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_player_goals_by_player_name() {
    let dm = shared_pre_agg_dm().await;
    let qc = AggContext::new(
        vec!["player_goals".to_string()],
        None,
        Some(vec!["players.player_name".to_string()]),
        None,
        None,
        None,
        None,
        Some(true),
        None,
    )
    .unwrap();
    let qc_clone = qc.clone();
    let plan = dm.display_graphviz(&DataQuery::Agg(qc_clone)).unwrap();
    println!("{plan}");
    assert!(
        plan.contains("player_goals_by_player"),
        "use_pre_agg=true: expected pre-agg table in plan, got:\n{plan}"
    );
    let actual = dm_query(&dm, qc).await;

    let ctx = shared_ctx().await;
    // Join raw scans (DataFusion qualifies columns automatically), aggregate, then
    // normalize_schema once at the end — matching the DataModel's own pipeline.
    let expected = normalize_schema(
        ctx.table("player_stats")
            .await
            .unwrap()
            .join(
                ctx.table("players").await.unwrap(),
                JoinType::Left,
                &["player_stats.player_id"],
                &["players.player_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("players.player_name")],
                vec![sum(col("player_stats.goals")).alias("player_stats.goals")],
            )
            .unwrap(),
    )
    .unwrap()
    .collect()
    .await
    .unwrap();

    let cols = &["players.player_name", "player_stats.goals"];
    assert_eq!(
        sorted_select(actual, "players.player_name", cols),
        sorted_select(expected, "players.player_name", cols),
        "player_goals by player_name mismatch"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_team_goals_by_team_name() {
    let dm = shared_dm().await;
    let qc = AggContext::new(
        vec!["team_goals".to_string()],
        None,
        Some(vec!["team_stats.team_name".to_string()]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let actual = dm_query(&dm, qc).await;

    let ctx = shared_ctx().await;
    let expected = ctx
        .table("team_stats")
        .await
        .unwrap()
        .select(vec![
            col("team_name").alias("team_stats.team_name"),
            col("goals").alias("team_stats.goals"),
        ])
        .unwrap()
        .aggregate(
            vec![Expr::Column(Column::from_name("team_stats.team_name"))],
            vec![
                sum(Expr::Column(Column::from_name("team_stats.goals"))).alias("team_stats.goals"),
            ],
        )
        .unwrap()
        .collect()
        .await
        .unwrap();

    let cols = &["team_stats.team_name", "team_stats.goals"];
    assert_eq!(
        sorted_select(actual, "team_stats.team_name", cols),
        sorted_select(expected, "team_stats.team_name", cols),
        "team_goals by team_name mismatch"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_game_count_by_group() {
    let dm = shared_dm().await;
    let qc = AggContext::new(
        vec!["game_count".to_string()],
        None,
        Some(vec!["groups.group_name".to_string()]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let actual = dm_query(&dm, qc).await;

    let ctx = shared_ctx().await;
    let expected = normalize_schema(
        ctx.table("games")
            .await
            .unwrap()
            .join(
                ctx.table("groups_bridge").await.unwrap(),
                JoinType::Left,
                &["games.game_id"],
                &["groups_bridge.game_id"],
                None,
            )
            .unwrap()
            .join(
                ctx.table("groups").await.unwrap(),
                JoinType::Left,
                &["groups_bridge.group_id"],
                &["groups.group_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("groups.group_name")],
                vec![count(col("games.game_id")).alias("games.game_count")],
            )
            .unwrap(),
    )
    .unwrap()
    .collect()
    .await
    .unwrap();

    let cols = &["groups.group_name", "games.game_count"];
    assert_eq!(
        sorted_select(actual, "groups.group_name", cols),
        sorted_select(expected, "groups.group_name", cols),
        "game_count by group_name mismatch"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_player_goals_by_platform() {
    let dm = shared_dm().await;
    let qc = AggContext::new(
        vec!["player_goals".to_string()],
        None,
        Some(vec!["players.platform".to_string()]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let actual = dm_query(&dm, qc).await;

    let ctx = shared_ctx().await;
    let expected = normalize_schema(
        ctx.table("player_stats")
            .await
            .unwrap()
            .join(
                ctx.table("players").await.unwrap(),
                JoinType::Left,
                &["player_stats.player_id"],
                &["players.player_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("players.platform")],
                vec![sum(col("player_stats.goals")).alias("player_stats.goals")],
            )
            .unwrap(),
    )
    .unwrap()
    .collect()
    .await
    .unwrap();

    let cols = &["players.platform", "player_stats.goals"];
    assert_eq!(
        sorted_select(actual, "players.platform", cols),
        sorted_select(expected, "players.platform", cols),
        "player_goals by platform mismatch"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_multi_measure_join() {
    let dm = shared_dm().await;
    let qc = AggContext::new(
        vec!["player_goals".to_string(), "game_count".to_string()],
        None,
        Some(vec!["players.player_name".to_string()]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let qc_clone = qc.clone();
    println!(
        "{}",
        dm.display_graphviz(&DataQuery::Agg(qc_clone)).unwrap()
    );
    let actual = dm_query(&dm, qc).await;

    let ctx = shared_ctx().await;

    // player_goals by player_name: player_stats → players
    let player_goals_df = normalize_schema(
        ctx.table("player_stats")
            .await
            .unwrap()
            .join(
                ctx.table("players").await.unwrap(),
                JoinType::Left,
                &["player_stats.player_id"],
                &["players.player_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("players.player_name")],
                vec![sum(col("player_stats.goals")).alias("player_stats.goals")],
            )
            .unwrap(),
    )
    .unwrap();

    // game_count by player_name: games → player_stats → players
    let game_count_df = normalize_schema(
        ctx.table("games")
            .await
            .unwrap()
            .join(
                ctx.table("player_stats").await.unwrap(),
                JoinType::Left,
                &["games.game_id"],
                &["player_stats.game_id"],
                None,
            )
            .unwrap()
            .join(
                ctx.table("players").await.unwrap(),
                JoinType::Left,
                &["player_stats.player_id"],
                &["players.player_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("players.player_name")],
                vec![count(col("games.game_id")).alias("games.game_count")],
            )
            .unwrap(),
    )
    .unwrap();

    // Merge the two normalized measure DFs via SQL FULL OUTER JOIN with COALESCE.
    // Collecting first and registering as MemTables avoids the duplicate-qualified-
    // column-name error that DataFusion raises when both DFs share the same
    // qualified group column in a logical-plan Full join.
    let pg_batches = player_goals_df.collect().await.unwrap();
    let gc_batches = game_count_df.collect().await.unwrap();
    let merge_ctx = SessionContext::new();
    merge_ctx
        .register_table(
            "pg",
            Arc::new(
                datafusion::datasource::MemTable::try_new(pg_batches[0].schema(), vec![pg_batches])
                    .unwrap(),
            ),
        )
        .unwrap();
    merge_ctx
        .register_table(
            "gc",
            Arc::new(
                datafusion::datasource::MemTable::try_new(gc_batches[0].schema(), vec![gc_batches])
                    .unwrap(),
            ),
        )
        .unwrap();
    let expected = merge_ctx
        .sql(r#"SELECT
                COALESCE(pg."players.player_name", gc."players.player_name") AS "players.player_name",
                pg."player_stats.goals",
                gc."games.game_count"
               FROM pg FULL OUTER JOIN gc
               ON pg."players.player_name" = gc."players.player_name""#)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let sort_col = "players.player_name";
    let measure_cols = &["player_stats.goals", "games.game_count"];
    assert_eq!(
        sorted_select(actual, sort_col, measure_cols),
        sorted_select(expected, sort_col, measure_cols),
        "multi-measure (player_goals + game_count) by player_name mismatch"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_player_goals_and_game_count_by_group() {
    let dm = shared_dm().await;
    let qc = AggContext::new(
        vec!["player_goals".to_string(), "game_count".to_string()],
        None,
        Some(vec!["groups.group_name".to_string()]),
        None,
        None,
        None,
        None,
        None,
        None,
    )
    .unwrap();
    let actual = dm_query(&dm, qc).await;

    let ctx = shared_ctx().await;

    // player_goals by group: player_stats → games → groups_bridge → groups
    let player_goals_df = normalize_schema(
        ctx.table("player_stats")
            .await
            .unwrap()
            .join(
                ctx.table("games").await.unwrap(),
                JoinType::Left,
                &["player_stats.game_id"],
                &["games.game_id"],
                None,
            )
            .unwrap()
            .join(
                ctx.table("groups_bridge").await.unwrap(),
                JoinType::Left,
                &["games.game_id"],
                &["groups_bridge.game_id"],
                None,
            )
            .unwrap()
            .join(
                ctx.table("groups").await.unwrap(),
                JoinType::Left,
                &["groups_bridge.group_id"],
                &["groups.group_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("groups.group_name")],
                vec![sum(col("player_stats.goals")).alias("player_stats.goals")],
            )
            .unwrap(),
    )
    .unwrap();

    // game_count by group: games → groups_bridge → groups
    let game_count_df = normalize_schema(
        ctx.table("games")
            .await
            .unwrap()
            .join(
                ctx.table("groups_bridge").await.unwrap(),
                JoinType::Left,
                &["games.game_id"],
                &["groups_bridge.game_id"],
                None,
            )
            .unwrap()
            .join(
                ctx.table("groups").await.unwrap(),
                JoinType::Left,
                &["groups_bridge.group_id"],
                &["groups.group_id"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("groups.group_name")],
                vec![count(col("games.game_id")).alias("games.game_count")],
            )
            .unwrap(),
    )
    .unwrap();

    // Rename game_count's group key before joining to avoid the duplicate-column
    // error DataFusion raises on FULL JOIN when both sides share a flat-aliased name.
    let game_count_aliased = game_count_df
        .select(vec![
            Expr::Column(Column::from_name("groups.group_name")).alias("__gc_group__"),
            Expr::Column(Column::from_name("games.game_count")),
        ])
        .unwrap();

    // FULL JOIN with standard equality — NULL != NULL is intentional: a NULL
    // group_name from one measure is not assumed to match a NULL from another.
    let joined = player_goals_df
        .join_on(
            game_count_aliased,
            JoinType::Full,
            [Expr::Column(Column::from_name("groups.group_name"))
                .eq(Expr::Column(Column::from_name("__gc_group__")))],
        )
        .unwrap();

    let expected = joined
        .select(vec![
            coalesce(vec![
                Expr::Column(Column::from_name("groups.group_name")),
                Expr::Column(Column::from_name("__gc_group__")),
            ])
            .alias("groups.group_name"),
            Expr::Column(Column::from_name("player_stats.goals")),
            Expr::Column(Column::from_name("games.game_count")),
        ])
        .unwrap()
        .collect()
        .await
        .unwrap();

    let sort_col = "groups.group_name";
    let measure_cols = &["player_stats.goals", "games.game_count"];
    assert_eq!(
        sorted_select(actual, sort_col, measure_cols),
        sorted_select(expected, sort_col, measure_cols),
        "multi-measure (player_goals + game_count) by group_name mismatch"
    );
}

/// Verify that `use_pre_agg` in AggContext controls whether the pre-aggregation
/// parquet file appears in the measure's logical plan.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pre_agg_explain_toggle() {
    let (dm, _tmp) = build_dm_with_pre_agg().await;

    let make_qc = |use_pre_agg: bool| {
        AggContext::new(
            vec!["player_goals".to_string()],
            None,
            Some(vec!["players.player_name".to_string()]),
            None,
            None,
            None,
            None,
            Some(use_pre_agg),
            None,
        )
        .unwrap()
    };

    let df_with = player_goals(&dm, &make_qc(true)).unwrap();
    let df_without = player_goals(&dm, &make_qc(false)).unwrap();

    let plan_with = pretty_format_batches(
        &df_with
            .explain(false, false)
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();
    let plan_without = pretty_format_batches(
        &df_without
            .explain(false, false)
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();

    assert!(
        plan_with.contains("player_goals_by_player"),
        "use_pre_agg=true: expected pre-agg file in plan, got:\n{plan_with}"
    );
    assert!(
        !plan_without.contains("player_goals_by_player"),
        "use_pre_agg=false: expected no pre-agg file in plan, got:\n{plan_without}"
    );
}
