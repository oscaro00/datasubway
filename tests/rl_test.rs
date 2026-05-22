use std::collections::HashMap;

use datasubway::column_expressions::column_context::{
    allow, ColumnContext, ColumnInclude, ColumnPattern,
};
use datasubway::data_model::{DataModel, QueryOutput};
use datasubway::model_components::{
    joins::{Join, JoinDirection, JoinGraph, JoinHow},
    measures::Measure,
    query_context::QueryContext,
};
use datasubway::wrappers::polars::lazyframe_recorder::LazyFrameRecorder;
use polars::prelude::*;

fn build_dm() -> DataModel {
    let tables = HashMap::from([
        (
            "players".to_string(),
            LazyFrame::scan_parquet(
                "tests/data_files/players.parquet".into(),
                Default::default(),
            )
            .unwrap(),
        ),
        (
            "games".to_string(),
            LazyFrame::scan_parquet("tests/data_files/games.parquet".into(), Default::default())
                .unwrap(),
        ),
        (
            "player_stats".to_string(),
            LazyFrame::scan_parquet(
                "tests/data_files/player_stats.parquet".into(),
                Default::default(),
            )
            .unwrap(),
        ),
        (
            "team_stats".to_string(),
            LazyFrame::scan_parquet(
                "tests/data_files/team_stats.parquet".into(),
                Default::default(),
            )
            .unwrap(),
        ),
        (
            "teams".to_string(),
            LazyFrame::scan_parquet("tests/data_files/teams.parquet".into(), Default::default())
                .unwrap(),
        ),
        (
            "groups".to_string(),
            LazyFrame::scan_parquet("tests/data_files/groups.parquet".into(), Default::default())
                .unwrap(),
        ),
        (
            "groups_bridge".to_string(),
            LazyFrame::scan_parquet(
                "tests/data_files/groups_bridge.parquet".into(),
                Default::default(),
            )
            .unwrap(),
        ),
    ]);

    let joins = vec![
        Join {
            left: "games".into(),
            right: "player_stats".into(),
            left_on: vec!["games.game_id".into()],
            right_on: vec!["player_stats.game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
        },
        Join {
            left: "games".into(),
            right: "team_stats".into(),
            left_on: vec!["games.game_id".into()],
            right_on: vec!["team_stats.game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Both,
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
        Join {
            left: "player_stats".into(),
            right: "players".into(),
            left_on: vec!["player_stats.player_id".into()],
            right_on: vec!["players.player_id".into()],
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
    ];

    let mut dm = DataModel::new(tables, JoinGraph::new(&joins).unwrap(), vec![], None);

    dm.add_measure(Measure::new("player_goals", player_goals))
        .unwrap();
    dm.add_measure(Measure::new("team_goals", team_goals))
        .unwrap();
    dm.add_measure(Measure::new("game_count", game_count))
        .unwrap();

    dm
}

fn player_goals<'a>(dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
    dm.table("player_stats")
        .group_by(allow(
            ColumnPattern::OnePattern("*".into()),
            ColumnContext::MultipleStrings(qc.groups.clone()),
            ColumnInclude::None,
        ))
        .agg(vec![col("player_stats.goals")
            .sum()
            .alias("player_stats.goals")])
}

fn team_goals<'a>(dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
    dm.table("team_stats")
        .group_by(allow(
            ColumnPattern::OnePattern("*".into()),
            ColumnContext::MultipleStrings(qc.groups.clone()),
            ColumnInclude::None,
        ))
        .agg(vec![col("team_stats.goals")
            .sum()
            .alias("team_stats.goals")])
}

fn game_count<'a>(dm: &'a DataModel, qc: &QueryContext) -> LazyFrameRecorder<'a> {
    dm.table("games")
        .group_by(allow(
            ColumnPattern::OnePattern("*".into()),
            ColumnContext::MultipleStrings(qc.groups.clone()),
            ColumnInclude::None,
        ))
        .agg(vec![col("games.game_id").count().alias("games.game_count")])
}

fn dm_query(dm: &DataModel, qc: &QueryContext) -> DataFrame {
    match dm.query(qc, false).unwrap() {
        QueryOutput::Data(df) => df,
        _ => panic!("expected Data"),
    }
}

fn scan(path: &str) -> LazyFrame {
    LazyFrame::scan_parquet(path.into(), Default::default()).unwrap()
}

fn sorted_select(df: DataFrame, sort_by: &str, cols: &[&str]) -> DataFrame {
    df.select(cols)
        .unwrap()
        .sort(
            [sort_by],
            SortMultipleOptions::default().with_nulls_last(true),
        )
        .unwrap()
}

#[test]
fn test_player_goals_by_player_name() {
    let dm = build_dm();
    let qc = QueryContext::new(
        vec!["player_goals".to_string()],
        None,
        Some(vec!["players.player_name".to_string()]),
        None, None, None, None, None,
    )
    .unwrap();
    let actual = dm_query(&dm, &qc);

    let ps = scan("tests/data_files/player_stats.parquet").select([
        col("player_id").alias("player_stats.player_id"),
        col("goals").alias("player_stats.goals"),
    ]);
    let pl = scan("tests/data_files/players.parquet").select([
        col("player_id").alias("players.player_id"),
        col("player_name").alias("players.player_name"),
    ]);
    let expected = ps
        .join(
            pl,
            [col("player_stats.player_id")],
            [col("players.player_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("players.player_name")])
        .agg([col("player_stats.goals").sum().alias("player_stats.goals")])
        .collect()
        .unwrap();

    let cols = &["players.player_name", "player_stats.goals"];
    assert!(
        sorted_select(actual, "players.player_name", cols)
            .equals_missing(&sorted_select(expected, "players.player_name", cols)),
        "player_goals by player_name mismatch"
    );
}

#[test]
fn test_team_goals_by_team_name() {
    let dm = build_dm();
    let qc = QueryContext::new(
        vec!["team_goals".to_string()],
        None,
        Some(vec!["team_stats.team_name".to_string()]),
        None, None, None, None, None,
    )
    .unwrap();
    let actual = dm_query(&dm, &qc);

    let expected = scan("tests/data_files/team_stats.parquet")
        .select([
            col("team_name").alias("team_stats.team_name"),
            col("goals").alias("team_stats.goals"),
        ])
        .group_by([col("team_stats.team_name")])
        .agg([col("team_stats.goals").sum().alias("team_stats.goals")])
        .collect()
        .unwrap();

    let cols = &["team_stats.team_name", "team_stats.goals"];
    assert!(
        sorted_select(actual, "team_stats.team_name", cols)
            .equals_missing(&sorted_select(expected, "team_stats.team_name", cols)),
        "team_goals by team_name mismatch"
    );
}

#[test]
fn test_game_count_by_group() {
    let dm = build_dm();
    let qc = QueryContext::new(
        vec!["game_count".to_string()],
        None,
        Some(vec!["groups.group_name".to_string()]),
        None, None, None, None, None,
    )
    .unwrap();
    let actual = dm_query(&dm, &qc);

    let games = scan("tests/data_files/games.parquet")
        .select([col("game_id").alias("games.game_id")]);
    let gb = scan("tests/data_files/groups_bridge.parquet").select([
        col("game_id").alias("groups_bridge.game_id"),
        col("group_id").alias("groups_bridge.group_id"),
    ]);
    let groups = scan("tests/data_files/groups.parquet").select([
        col("group_id").alias("groups.group_id"),
        col("group_name").alias("groups.group_name"),
    ]);
    let expected = games
        .join(
            gb,
            [col("games.game_id")],
            [col("groups_bridge.game_id")],
            JoinArgs::new(JoinType::Left),
        )
        .join(
            groups,
            [col("groups_bridge.group_id")],
            [col("groups.group_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("groups.group_name")])
        .agg([col("games.game_id").count().alias("games.game_count")])
        .collect()
        .unwrap();

    let cols = &["groups.group_name", "games.game_count"];
    assert!(
        sorted_select(actual, "groups.group_name", cols)
            .equals_missing(&sorted_select(expected, "groups.group_name", cols)),
        "game_count by group_name mismatch"
    );
}

#[test]
fn test_player_goals_by_platform() {
    let dm = build_dm();
    let qc = QueryContext::new(
        vec!["player_goals".to_string()],
        None,
        Some(vec!["players.platform".to_string()]),
        None, None, None, None, None,
    )
    .unwrap();
    let actual = dm_query(&dm, &qc);

    let ps = scan("tests/data_files/player_stats.parquet").select([
        col("player_id").alias("player_stats.player_id"),
        col("goals").alias("player_stats.goals"),
    ]);
    let pl = scan("tests/data_files/players.parquet").select([
        col("player_id").alias("players.player_id"),
        col("platform").alias("players.platform"),
    ]);
    let expected = ps
        .join(
            pl,
            [col("player_stats.player_id")],
            [col("players.player_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("players.platform")])
        .agg([col("player_stats.goals").sum().alias("player_stats.goals")])
        .collect()
        .unwrap();

    assert!(
        expected.height() <= 10,
        "expected few distinct platforms, got {}",
        expected.height()
    );
    let cols = &["players.platform", "player_stats.goals"];
    assert!(
        sorted_select(actual, "players.platform", cols)
            .equals_missing(&sorted_select(expected, "players.platform", cols)),
        "player_goals by platform mismatch"
    );
}

#[test]
fn test_multi_measure_join() {
    let dm = build_dm();
    let qc = QueryContext::new(
        vec!["player_goals".to_string(), "game_count".to_string()],
        None,
        Some(vec!["players.player_name".to_string()]),
        None, None, None, None, None,
    )
    .unwrap();
    let actual = dm_query(&dm, &qc);

    let ps = scan("tests/data_files/player_stats.parquet").select([
        col("player_id").alias("player_stats.player_id"),
        col("goals").alias("player_stats.goals"),
    ]);
    let pl = scan("tests/data_files/players.parquet").select([
        col("player_id").alias("players.player_id"),
        col("player_name").alias("players.player_name"),
    ]);
    let player_goals_df = ps
        .join(
            pl.clone(),
            [col("player_stats.player_id")],
            [col("players.player_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("players.player_name")])
        .agg([col("player_stats.goals").sum().alias("player_stats.goals")])
        .collect()
        .unwrap();

    let games = scan("tests/data_files/games.parquet")
        .select([col("game_id").alias("games.game_id")]);
    let ps2 = scan("tests/data_files/player_stats.parquet").select([
        col("game_id").alias("player_stats.game_id"),
        col("player_id").alias("player_stats.player_id"),
    ]);
    let game_count_df = games
        .join(
            ps2,
            [col("games.game_id")],
            [col("player_stats.game_id")],
            JoinArgs::new(JoinType::Left),
        )
        .join(
            pl,
            [col("player_stats.player_id")],
            [col("players.player_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("players.player_name")])
        .agg([col("games.game_id").count().alias("games.game_count")])
        .collect()
        .unwrap();

    let expected = player_goals_df
        .lazy()
        .join(
            game_count_df.lazy(),
            [col("players.player_name")],
            [col("players.player_name")],
            JoinArgs::new(JoinType::Full),
        )
        .collect()
        .unwrap();

    let sort_col = "players.player_name";
    let measure_cols = &["player_stats.goals", "games.game_count"];
    let actual_sorted = actual
        .sort([sort_col], SortMultipleOptions::default().with_nulls_last(true))
        .unwrap()
        .select(measure_cols)
        .unwrap();
    let expected_sorted = expected
        .sort([sort_col], SortMultipleOptions::default().with_nulls_last(true))
        .unwrap()
        .select(measure_cols)
        .unwrap();
    assert!(
        actual_sorted.equals_missing(&expected_sorted),
        "multi-measure (player_goals + game_count) by player_name mismatch"
    );
}

#[test]
fn test_player_goals_and_game_count_by_group() {
    let dm = build_dm();
    let qc = QueryContext::new(
        vec!["player_goals".to_string(), "game_count".to_string()],
        None,
        Some(vec!["groups.group_name".to_string()]),
        None, None, None, None, None,
    )
    .unwrap();
    let actual = dm_query(&dm, &qc);

    // player_goals by group: player_stats -> groups_bridge -> groups
    let ps = scan("tests/data_files/player_stats.parquet").select([
        col("game_id").alias("player_stats.game_id"),
        col("goals").alias("player_stats.goals"),
    ]);
    let gb = scan("tests/data_files/groups_bridge.parquet").select([
        col("game_id").alias("groups_bridge.game_id"),
        col("group_id").alias("groups_bridge.group_id"),
    ]);
    let groups = scan("tests/data_files/groups.parquet").select([
        col("group_id").alias("groups.group_id"),
        col("group_name").alias("groups.group_name"),
    ]);
    let player_goals_df = ps
        .join(
            gb.clone(),
            [col("player_stats.game_id")],
            [col("groups_bridge.game_id")],
            JoinArgs::new(JoinType::Left),
        )
        .join(
            groups.clone(),
            [col("groups_bridge.group_id")],
            [col("groups.group_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("groups.group_name")])
        .agg([col("player_stats.goals").sum().alias("player_stats.goals")])
        .collect()
        .unwrap();

    // game_count by group: games -> groups_bridge -> groups
    let games = scan("tests/data_files/games.parquet")
        .select([col("game_id").alias("games.game_id")]);
    let game_count_df = games
        .join(
            gb,
            [col("games.game_id")],
            [col("groups_bridge.game_id")],
            JoinArgs::new(JoinType::Left),
        )
        .join(
            groups,
            [col("groups_bridge.group_id")],
            [col("groups.group_id")],
            JoinArgs::new(JoinType::Left),
        )
        .group_by([col("groups.group_name")])
        .agg([col("games.game_id").count().alias("games.game_count")])
        .collect()
        .unwrap();

    let expected = player_goals_df
        .lazy()
        .join(
            game_count_df.lazy(),
            [col("groups.group_name")],
            [col("groups.group_name")],
            JoinArgs::new(JoinType::Full),
        )
        .collect()
        .unwrap();

    let sort_col = "groups.group_name";
    let measure_cols = &["player_stats.goals", "games.game_count"];
    let actual_sorted = actual
        .sort([sort_col], SortMultipleOptions::default().with_nulls_last(true))
        .unwrap()
        .select(measure_cols)
        .unwrap();
    let expected_sorted = expected
        .sort([sort_col], SortMultipleOptions::default().with_nulls_last(true))
        .unwrap()
        .select(measure_cols)
        .unwrap();
    assert!(
        actual_sorted.equals_missing(&expected_sorted),
        "multi-measure (player_goals + game_count) by group_name mismatch"
    );
}
