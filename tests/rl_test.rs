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
use datafusion::functions_aggregate::expr_fn::{count_distinct, sum};
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
        // player_stats is the central fact table; games, players, and team_stats
        // are all dimensions that join onto it (RightOnLeft = one-way from left to right).
        Join {
            left: "player_stats".into(),
            right: "games".into(),
            left_on: vec!["player_stats.game_id".into()],
            right_on: vec!["games.game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::RightOnLeft,
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
            left: "games".into(),
            right: "team_stats".into(),
            left_on: vec!["games.game_id".into()],
            right_on: vec!["team_stats.game_id".into()],
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
    dm.add_measure(DfMeasure::new(
        "player_percent_team_boost_collected",
        player_percent_of_team_boost_collected,
    ))
    .unwrap();
    dm
}

/// Pre-aggregations
fn player_goals_pre_agg() -> Vec<PreAggregation> {
    vec![
        PreAggregation::new(
            "player_goals_by_player".into(),
            vec!["players.player_name".into()],
            HashMap::from([("player_stats.goals".into(), vec!["sum".into()])]),
        )
        .unwrap(),
        PreAggregation::new(
            "player_team_boost_collected".into(),
            vec!["players.player_name".into()],
            HashMap::from([
                ("player_stats.amount_collected".into(), vec!["sum".into()]),
                ("team_stats.amount_collected".into(), vec!["sum".into()]),
            ]),
        )
        .unwrap(),
    ]
}

async fn build_dm_with_pre_agg() -> (DataModel, TempDir) {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let mut dm = DataModel::new(
        make_tables().await,
        JoinGraph::new(&make_joins()).unwrap(),
        player_goals_pre_agg(),
        Some(path),
    );
    dm.add_measure(DfMeasure::new("player_goals", player_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("player_assists", player_assists))
        .unwrap();
    dm.add_measure(DfMeasure::new("team_goals", team_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("game_count", game_count))
        .unwrap();
    dm.add_measure(DfMeasure::new(
        "player_percent_team_boost_collected",
        player_percent_of_team_boost_collected,
    ))
    .unwrap();
    dm.write_pre_aggs(&["player_goals_by_player"]).unwrap();
    dm.write_pre_aggs(&["player_team_boost_collected"]).unwrap();
    (dm, tmp)
}

fn player_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    dm.table("player_stats", qc.use_pre_agg)
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

fn player_assists(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    dm.table("player_stats", qc.use_pre_agg)
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![sum(col("player_stats.assists")).alias("player_stats.assists")],
        )
        .build()
}

fn team_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    dm.table("team_stats", qc.use_pre_agg)
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
    dm.table("player_stats", qc.use_pre_agg)
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![count_distinct(col("player_stats.game_id")).alias("games.game_count")],
        )
        .build()
}

fn player_percent_of_team_boost_collected(
    dm: &DataModel,
    qc: &AggContext,
) -> datafusion::common::Result<DataFrame> {
    dm.table("player_stats", qc.use_pre_agg)
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![
                (sum(col("player_stats.amount_collected"))
                    / sum(col("team_stats.amount_collected")))
                .alias("player_percent_of_team_boost_collected"),
            ],
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
    let actual = dm_query(dm, qc).await;

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
    let actual = dm_query(dm, qc).await;

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
    let actual = dm_query(dm, qc).await;

    let ctx = shared_ctx().await;
    let expected = normalize_schema(
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
                vec![count_distinct(col("player_stats.game_id")).alias("games.game_count")],
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
    let actual = dm_query(dm, qc).await;

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
    let actual = dm_query(dm, qc).await;

    // Both measures start from player_stats and join to players — same subplan,
    // so the merge optimizer combines them into a single aggregate (no Full join).
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
                vec![col("players.player_name")],
                vec![
                    sum(col("player_stats.goals")).alias("player_stats.goals"),
                    count_distinct(col("player_stats.game_id")).alias("games.game_count"),
                ],
            )
            .unwrap(),
    )
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
    let actual = dm_query(dm, qc).await;

    // Both measures start from player_stats with the same path to groups — same
    // subplan, so the merge optimizer combines them into a single aggregate.
    let ctx = shared_ctx().await;
    let expected = normalize_schema(
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
                vec![
                    sum(col("player_stats.goals")).alias("player_stats.goals"),
                    count_distinct(col("player_stats.game_id")).alias("games.game_count"),
                ],
            )
            .unwrap(),
    )
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

/// Verify that player_goals and player_assists (both sourced from player_stats) are
/// merged into a single aggregate node when use_pre_agg is disabled, so no Full join
/// appears between them in the logical plan.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_player_goals_and_assists_merged() {
    // shared_pre_agg_dm already has both player_goals and player_assists registered.
    // use_pre_agg = Some(false) ensures both measures scan the same raw player_stats
    // table, giving them identical input subplans and making them candidates for merging.
    let dm = shared_pre_agg_dm().await;

    let qc = AggContext::new(
        vec!["player_goals".to_string(), "player_assists".to_string()],
        None,
        Some(vec!["players.player_name".to_string()]),
        None,
        None,
        None,
        None,
        Some(false),
        None,
    )
    .unwrap();

    let plan = dm.display_graphviz(&DataQuery::Agg(qc.clone())).unwrap();
    println!("{plan}");
    assert!(
        !plan.contains("Full"),
        "merged measures should not produce a Full join:\n{plan}"
    );

    let actual = dm_query(dm, qc).await;

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
                vec![col("players.player_name")],
                vec![
                    sum(col("player_stats.goals")).alias("player_stats.goals"),
                    sum(col("player_stats.assists")).alias("player_stats.assists"),
                ],
            )
            .unwrap(),
    )
    .unwrap()
    .collect()
    .await
    .unwrap();

    let cols = &[
        "players.player_name",
        "player_stats.goals",
        "player_stats.assists",
    ];
    assert_eq!(
        sorted_select(actual, "players.player_name", cols),
        sorted_select(expected, "players.player_name", cols),
        "merged player_goals + player_assists by player_name mismatch"
    );
}

/// Verify that player_percent_of_team_boost_collected uses the player_team_boost_collected
/// pre-agg when use_pre_agg=true and produces correct results.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_player_percent_team_boost_pre_agg() {
    let (dm, _tmp) = build_dm_with_pre_agg().await;

    let qc = AggContext::new(
        vec!["player_percent_team_boost_collected".to_string()],
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

    let plan = dm.display_graphviz(&DataQuery::Agg(qc.clone())).unwrap();
    println!("{plan}");
    assert!(
        plan.contains("player_team_boost_collected"),
        "use_pre_agg=true: expected pre-agg table in plan, got:\n{plan}"
    );

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
            .join(
                ctx.table("team_stats").await.unwrap(),
                JoinType::Left,
                &["player_stats.game_id", "player_stats.team_name"],
                &["team_stats.game_id", "team_stats.team_name"],
                None,
            )
            .unwrap()
            .aggregate(
                vec![col("players.player_name")],
                vec![
                    (sum(col("player_stats.amount_collected"))
                        / sum(col("team_stats.amount_collected")))
                    .alias("player_percent_of_team_boost_collected"),
                ],
            )
            .unwrap(),
    )
    .unwrap()
    .collect()
    .await
    .unwrap();

    let cols = &[
        "players.player_name",
        "player_percent_of_team_boost_collected",
    ];
    assert_eq!(
        sorted_select(actual, "players.player_name", cols),
        sorted_select(expected, "players.player_name", cols),
        "player_percent_of_team_boost_collected by player_name mismatch"
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

// ── Derived statistics rolled up from a pre-aggregation ───────────────────────

fn player_avg_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    use datafusion::functions_aggregate::expr_fn::avg;
    dm.table("player_stats", qc.use_pre_agg)
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![avg(col("player_stats.goals")).alias("player_stats.avg_goals")],
        )
        .build()
}

fn player_stddev_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    use datafusion::functions_aggregate::expr_fn::stddev_pop;
    dm.table("player_stats", qc.use_pre_agg)
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![stddev_pop(col("player_stats.goals")).alias("player_stats.stddev_goals")],
        )
        .build()
}

fn player_variance_goals(dm: &DataModel, qc: &AggContext) -> datafusion::common::Result<DataFrame> {
    use datafusion::functions_aggregate::expr_fn::var_pop;
    dm.table("player_stats", qc.use_pre_agg)
        .aggregate(
            allow(
                ColumnPattern::OnePattern("*".into()),
                ColumnContext::MultipleStrings(qc.groups.clone()),
                ColumnInclude::None,
            ),
            vec![var_pop(col("player_stats.goals")).alias("player_stats.variance_goals")],
        )
        .build()
}

/// A model whose pre-agg stores the components (`sum`, `count`, `sumsq`) that
/// mean/stddev/variance are reconstructed from.
async fn build_dm_with_derived_stats_pre_agg() -> (DataModel, TempDir) {
    let tmp = TempDir::new().unwrap();
    let mut dm = DataModel::new(
        make_tables().await,
        JoinGraph::new(&make_joins()).unwrap(),
        vec![
            PreAggregation::new(
                "player_goal_stats_by_player".into(),
                vec!["players.player_name".into()],
                HashMap::from([(
                    "player_stats.goals".into(),
                    vec!["mean".into(), "std".into(), "var".into()],
                )]),
            )
            .unwrap(),
        ],
        Some(tmp.path().to_str().unwrap().to_string()),
    );
    dm.add_measure(DfMeasure::new("player_avg_goals", player_avg_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new("player_stddev_goals", player_stddev_goals))
        .unwrap();
    dm.add_measure(DfMeasure::new(
        "player_variance_goals",
        player_variance_goals,
    ))
    .unwrap();
    dm.write_pre_aggs(&["player_goal_stats_by_player"]).unwrap();
    (dm, tmp)
}

/// Pull `(group, value)` pairs out of a result, sorted by group, with the value
/// as `f64` regardless of its physical numeric type.
fn numeric_by_group(
    batches: Vec<RecordBatch>,
    group_col: &str,
    value_col: &str,
) -> Vec<(String, f64)> {
    use datafusion::arrow::array::{Array, Float64Array, StringArray};
    use datafusion::arrow::compute::cast;
    use datafusion::arrow::datatypes::DataType;

    let schema = batches[0].schema();
    let combined = concat_batches(&schema, &batches).unwrap();

    let groups = cast(
        combined.column(schema.index_of(group_col).unwrap()),
        &DataType::Utf8,
    )
    .unwrap();
    let groups = groups.as_any().downcast_ref::<StringArray>().unwrap();

    let values = cast(
        combined.column(schema.index_of(value_col).unwrap()),
        &DataType::Float64,
    )
    .unwrap();
    let values = values.as_any().downcast_ref::<Float64Array>().unwrap();

    let mut out: Vec<(String, f64)> = (0..combined.num_rows())
        .filter(|&i| !groups.is_null(i) && !values.is_null(i))
        .map(|i| (groups.value(i).to_string(), values.value(i)))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

/// Rolling a mean, stddev or variance out of a pre-aggregation must produce the
/// same numbers as computing it straight from the base table.
///
/// Regression: the rollup divided the stored components directly, and `sum`,
/// `count` and `sumsq` over an integer source column are all `Int64`, so
/// DataFusion performed *integer* division. Every ratio below 1 collapsed to
/// zero — a player averaging 0.4375 goals came back as 0, as did any win rate.
///
/// Compared with a tolerance rather than exactly: the rollup evaluates variance
/// as `sumsq/n - mean^2` while the direct path uses DataFusion's streaming
/// accumulator, so the two agree to within floating-point noise rather than
/// bit-for-bit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_pre_agg_derived_stats_match_direct_computation() {
    let (dm, _tmp) = build_dm_with_derived_stats_pre_agg().await;

    let measures = [
        ("player_avg_goals", "player_stats.avg_goals"),
        ("player_stddev_goals", "player_stats.stddev_goals"),
        ("player_variance_goals", "player_stats.variance_goals"),
    ];

    for (measure, output_col) in measures {
        let query = |use_pre_agg: bool| {
            AggContext::new(
                vec![measure.into()],
                None,
                Some(vec!["players.player_name".into()]),
                None,
                None,
                None,
                None,
                Some(use_pre_agg),
                None,
            )
            .unwrap()
        };

        let from_pre_agg = numeric_by_group(
            dm_query(&dm, query(true)).await,
            "players.player_name",
            output_col,
        );
        let direct = numeric_by_group(
            dm_query(&dm, query(false)).await,
            "players.player_name",
            output_col,
        );

        assert_eq!(
            from_pre_agg.len(),
            direct.len(),
            "{measure}: row counts differ between the two paths"
        );
        assert!(!from_pre_agg.is_empty(), "{measure}: no rows to compare");

        for ((group, pre), (direct_group, want)) in from_pre_agg.iter().zip(&direct) {
            assert_eq!(group, direct_group, "{measure}: group ordering differs");
            assert!(
                (pre - want).abs() <= 1e-9 * want.abs().max(1.0),
                "{measure} for {group}: pre-agg gave {pre}, direct gave {want}"
            );
        }

        // Guards against the assertions above passing on trivially equal data:
        // `goals` is an Int32 column whose per-player statistics are genuinely
        // fractional, which is exactly what integer division destroyed.
        assert!(
            from_pre_agg.iter().any(|(_, v)| v.fract() > 0.0),
            "{measure}: expected fractional values, got only whole numbers"
        );
    }
}

// ── Projection pushdown through AggregateWithMetadata ─────────────────────────

/// Extracts the `projection=[...]` list for the named table from a plan string.
fn scan_projection(plan: &str, table: &str) -> Vec<String> {
    let needle = format!("TableScan: {table} projection=[");
    let start = plan
        .find(&needle)
        .unwrap_or_else(|| panic!("no TableScan for '{table}' in plan:\n{plan}"))
        + needle.len();
    let end = plan[start..]
        .find(']')
        .expect("unterminated projection list");
    plan[start..start + end]
        .split(", ")
        .map(|s| s.trim().to_string())
        .collect()
}

/// `AggregateWithMetadata` reports its required input columns via
/// `necessary_children_exprs`, so `OptimizeProjections` must prune the scan
/// beneath it instead of conservatively keeping all 99 `player_stats` columns.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn test_projection_pushdown_through_aggregate_with_metadata() {
    let (dm, _tmp) = build_dm_with_pre_agg().await;

    let qc = AggContext::new(
        vec!["player_goals".to_string()],
        None,
        Some(vec!["players.player_name".to_string()]),
        None,
        None,
        None,
        None,
        Some(false), // scan the base table, which is 99 columns wide
        None,
    )
    .unwrap();

    let plan = pretty_format_batches(
        &dm.explain(&DataQuery::Agg(qc), Default::default())
            .unwrap()
            .collect()
            .await
            .unwrap(),
    )
    .unwrap()
    .to_string();

    let cols = scan_projection(&plan, "player_stats");

    // Only the aggregated column and the join key survive. Without
    // `necessary_children_exprs` this scan keeps all 99 columns.
    assert!(
        cols.contains(&"goals".to_string()),
        "aggregated column missing from scan: {cols:?}"
    );
    assert!(
        cols.contains(&"player_id".to_string()),
        "join key missing from scan: {cols:?}"
    );
    assert!(
        !cols.contains(&"time_powerslide".to_string()),
        "unrelated column was not pruned: {cols:?}"
    );
    assert!(
        cols.len() < 10,
        "expected a pruned scan, got {} columns:\n{plan}",
        cols.len()
    );

    // The metadata blob is elided in explain output, but keeps a digest so two
    // plans differing only in metadata still render differently.
    assert!(
        plan.contains("allow_exclude=<"),
        "expected elided metadata in plan:\n{plan}"
    );
    assert!(
        !plan.contains("OnePattern"),
        "raw allow_exclude JSON leaked into plan:\n{plan}"
    );
}
