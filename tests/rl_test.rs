use std::collections::HashMap;
use std::sync::Arc;

use datafusion::prelude::*;
use datafusion_functions_aggregate::average::avg;
use datafusion_functions_aggregate::sum::sum;
use serde_json::json;

use datasubway::data_model::DataModel;
use datasubway::model::column_context::ColumnInput::*;
use datasubway::model::joins::{Join, JoinDirection, JoinHow};
use datasubway::model::pre_agg::PreAggregation;
use datasubway::model::query_context::QueryContext;

const DATA_DIR: &str = "tests/data_files";
const PRE_AGG_DIR: &str = "tests/data_files/pre_aggs";

pub async fn create_data_model() -> Result<DataModel, Box<dyn std::error::Error>> {
    let mut dm = DataModel::new();

    dm.set_pre_agg_path(PRE_AGG_DIR);

    let tables = [
        "games",
        "groups",
        "groups_bridge",
        "player_stats",
        "players",
        "team_stats",
        "teams",
    ];

    for table in &tables {
        dm.register_parquet(table, &format!("{DATA_DIR}/{table}.parquet"))
            .await?;
    }

    dm.set_joins(&[
        Join {
            left: "games".into(),
            right: "groups_bridge".into(),
            left_on: vec!["game_id".into()],
            right_on: vec!["game_id".into()],
            how: JoinHow::Inner,
            direction: JoinDirection::Both,
        },
        Join {
            left: "groups".into(),
            right: "groups_bridge".into(),
            left_on: vec!["group_id".into()],
            right_on: vec!["group_id".into()],
            how: JoinHow::Inner,
            direction: JoinDirection::Both,
        },
        Join {
            left: "player_stats".into(),
            right: "games".into(),
            left_on: vec!["game_id".into()],
            right_on: vec!["game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Right2Left,
        },
        Join {
            left: "player_stats".into(),
            right: "players".into(),
            left_on: vec!["player_id".into()],
            right_on: vec!["player_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Right2Left,
        },
        Join {
            left: "team_stats".into(),
            right: "games".into(),
            left_on: vec!["game_id".into()],
            right_on: vec!["game_id".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Right2Left,
        },
        Join {
            left: "team_stats".into(),
            right: "teams".into(),
            left_on: vec!["team_name".into()],
            right_on: vec!["team_name".into()],
            how: JoinHow::Left,
            direction: JoinDirection::Right2Left,
        },
    ])?;

    let pa = PreAggregation::new(
        "player_goals".into(),
        vec!["players.player_name".into(), "players.platform".into()],
        HashMap::from([("player_stats.goals".into(), vec!["sum".into()])]),
    )
    .map_err(|e| Box::<dyn std::error::Error>::from(e))?;
    dm.write_pre_agg(vec![pa]).await?;

    // dm.add_custom_optimizers().await?;

    dm.register_measure(
        "total_player_goals",
        Arc::new(|qc, dm| {
            Box::pin(async move {
                let filter_expr = dm
                    .allow(&["*".into()], FilterTree(&qc.filters), None)?
                    .into_filter_expr();
                let group_exprs = dm
                    .allow(&["*".into()], Columns(&qc.groups), None)?
                    .into_exprs();
                dm.table("player_stats")
                    .await?
                    .filter(filter_expr)?
                    .aggregate(
                        group_exprs,
                        vec![sum(col("goals")).alias("total_player_goals")],
                    )
            })
        }),
    )
    .await?;

    dm.register_measure(
        "total_player_assists",
        Arc::new(|qc, dm| {
            Box::pin(async move {
                let filter_expr = dm
                    .allow(&["*".into()], FilterTree(&qc.filters), None)?
                    .into_filter_expr();
                let group_exprs = dm
                    .allow(&["*".into()], Columns(&qc.groups), None)?
                    .into_exprs();
                dm.table("player_stats")
                    .await?
                    .filter(filter_expr)?
                    .aggregate(
                        group_exprs,
                        vec![sum(col("assists")).alias("total_player_assists")],
                    )
            })
        }),
    )
    .await?;

    dm.register_measure(
        "winning_team_avg_possession_time",
        Arc::new(|qc, dm| {
            Box::pin(async move {
                let filter_expr = dm
                    .allow(&["*".into()], FilterTree(&qc.filters), None)?
                    .into_filter_expr();
                let group_exprs = dm
                    .allow(&["*".into()], Columns(&qc.groups), None)?
                    .into_exprs();
                dm.table("team_stats")
                    .await?
                    .filter(col("team_stats.goals").gt(col("team_stats.goals_against")))?
                    .filter(filter_expr)?
                    .aggregate(
                        group_exprs,
                        vec![avg(col("possession_time")).alias("winning_team_avg_possession_time")],
                    )
            })
        }),
    )
    .await?;

    Ok(dm)
}

// run with cargo test --test rl_test -- --nocapture
// DATASUBWAY_DEBUG=1 cargo test --test rl_test -- --nocapture
#[tokio::test]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dm = create_data_model().await?;

    // Query with cross-table grouping and a filter (eager joins from table())
    let qc = QueryContext::new(
        vec!["total_player_goals".into(), "total_player_assists".into()],
        Some(json!({"AND": [["players.platform", "=", "steam"]]})),
        Some(vec!["players.player_name".into()]),
        None,
        Some(vec![("total_player_goals".into(), "desc".into())]),
        None,
        None,
        Some(true),
    )?;

    let explain_df = dm.explain(&qc, false, false).await?;
    explain_df.show().await?;

    let results = dm.collect(&qc).await?;
    for batch in &results {
        println!("{:?}", batch);
    }

    Ok(())
}
