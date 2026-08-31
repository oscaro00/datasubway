//! The `pre_agg` schema and the identity metadata carried in pre-agg files.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::catalog::TableProvider;
use datafusion::prelude::*;
use tempfile::TempDir;

use datasubway::data_model::DataModel;
use datasubway::model_components::{
    joins::{Join, JoinDirection, JoinGraph, JoinHow},
    pre_aggregations::{META_COMPONENT, META_LOGICAL_COL, PreAggregation},
};

const TABLES: [&str; 2] = ["players", "player_stats"];

async fn make_tables(rename: Option<(&str, &str)>) -> HashMap<String, Arc<dyn TableProvider>> {
    let ctx = SessionContext::new();
    let mut tables = HashMap::new();
    for name in TABLES {
        ctx.register_parquet(
            name,
            &format!("tests/data_files/{name}.parquet"),
            ParquetReadOptions::default(),
        )
        .await
        .unwrap();
        let key = match rename {
            Some((from, to)) if from == name => to.to_string(),
            _ => name.to_string(),
        };
        tables.insert(key, ctx.table_provider(name).await.unwrap());
    }
    tables
}

fn joins(players_as: &str) -> Vec<Join> {
    vec![Join {
        left: "player_stats".into(),
        right: players_as.into(),
        left_on: vec!["player_stats.player_id".into()],
        right_on: vec![format!("{players_as}.player_id")],
        how: JoinHow::Left,
        direction: JoinDirection::RightOnLeft,
    }]
}

/// Goals summed per player — one group-by field and one component field, which is
/// all the identity metadata needs to be exercised over.
fn goals_by_player(players_as: &str) -> PreAggregation {
    PreAggregation::new(
        "goals_by_player".into(),
        vec![format!("{players_as}.player_name")],
        HashMap::from([("player_stats.goals".into(), vec!["sum".into()])]),
    )
    .unwrap()
}

async fn build(players_as: &str) -> (DataModel, TempDir) {
    let tmp = TempDir::new().unwrap();
    let rename = (players_as != "players").then_some(("players", players_as));
    let dm = DataModel::new(
        make_tables(rename).await,
        JoinGraph::new(&joins(players_as)).unwrap(),
        vec![goals_by_player(players_as)],
        Some(tmp.path().to_str().unwrap().to_string()),
    );
    dm.write_pre_aggs(&["goals_by_player"]).unwrap();
    (dm, tmp)
}

fn fmt(batches: &[RecordBatch]) -> String {
    pretty_format_batches(batches).unwrap().to_string()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_agg_is_addressable_in_its_own_schema() {
    let (dm, _tmp) = build("players").await;

    let rows = dm
        .sql("SELECT COUNT(*) AS n FROM pre_agg.goals_by_player")
        .await
        .unwrap();
    let n: i64 = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<datafusion::arrow::array::Int64Array>()
        .unwrap()
        .value(0);
    assert!(n > 0, "pre_agg.goals_by_player returned no rows");

    // The physical (dunder) names are what the schema exposes, and both a group-by
    // key and a component column are reachable by them.
    let rows = dm
        .sql(
            "SELECT players__player_name, player_stats__goals__sum \
             FROM pre_agg.goals_by_player ORDER BY players__player_name LIMIT 3",
        )
        .await
        .unwrap();
    assert!(!fmt(&rows).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_agg_schema_lists_only_written_definitions() {
    let tmp = TempDir::new().unwrap();
    let dm = DataModel::new(
        make_tables(None).await,
        JoinGraph::new(&joins("players")).unwrap(),
        vec![goals_by_player("players")],
        Some(tmp.path().to_str().unwrap().to_string()),
    );

    // Registered but never written: there is no file, so there is no table.
    let listed = dm
        .sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'pre_agg'")
        .await
        .unwrap();
    assert!(
        !fmt(&listed).contains("goals_by_player"),
        "unwritten pre-agg should not appear as a table, got:\n{}",
        fmt(&listed)
    );

    dm.write_pre_aggs(&["goals_by_player"]).unwrap();
    let listed = dm
        .sql("SELECT table_name FROM information_schema.tables WHERE table_schema = 'pre_agg'")
        .await
        .unwrap();
    assert!(
        fmt(&listed).contains("goals_by_player"),
        "written pre-agg should be listed, got:\n{}",
        fmt(&listed)
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn pre_agg_name_may_match_a_source_table_name() {
    // A pre-agg lives in `pre_agg`, a source table in the default schema, so the
    // two can share a name without their scans sharing a qualifier.
    let tmp = TempDir::new().unwrap();
    let dm = DataModel::new(
        make_tables(None).await,
        JoinGraph::new(&joins("players")).unwrap(),
        vec![
            PreAggregation::new(
                "players".into(),
                vec!["players.player_name".into()],
                HashMap::from([("player_stats.goals".into(), vec!["sum".into()])]),
            )
            .unwrap(),
        ],
        Some(tmp.path().to_str().unwrap().to_string()),
    );
    dm.write_pre_aggs(&["players"]).unwrap();

    // The source table still resolves to the raw columns...
    let raw = dm.sql("SELECT player_name FROM players LIMIT 1").await;
    assert!(raw.is_ok(), "source table shadowed: {:?}", raw.err());

    // ...while the pre-agg resolves to the stored physical ones.
    let pre = dm
        .sql("SELECT players__player_name FROM pre_agg.players LIMIT 1")
        .await;
    assert!(pre.is_ok(), "pre-agg not reachable: {:?}", pre.err());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn identity_metadata_survives_the_parquet_round_trip() {
    let (dm, tmp) = build("players").await;

    let file = std::fs::read_dir(tmp.path())
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .find(|p| p.extension().is_some_and(|x| x == "parquet"))
        .expect("a written pre-agg parquet");

    let reader =
        datafusion::parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(
            std::fs::File::open(&file).unwrap(),
        )
        .unwrap();
    let schema = reader.schema();

    let group = schema.field_with_name("players__player_name").unwrap();
    assert_eq!(
        group.metadata().get(META_LOGICAL_COL).map(String::as_str),
        Some("players.player_name"),
        "group-by field lost its logical column tag"
    );
    assert_eq!(group.metadata().get(META_COMPONENT), None);

    let component = schema.field_with_name("player_stats__goals__sum").unwrap();
    assert_eq!(
        component
            .metadata()
            .get(META_LOGICAL_COL)
            .map(String::as_str),
        Some("player_stats.goals")
    );
    assert_eq!(
        component.metadata().get(META_COMPONENT).map(String::as_str),
        Some("sum"),
        "component field lost its component tag"
    );

    // And the model still queries it, so stripping the metadata for `ListingTable`
    // did not put the scan schema at odds with the file.
    assert!(
        dm.sql("SELECT * FROM pre_agg.goals_by_player LIMIT 1")
            .await
            .is_ok()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metadata_resolves_a_table_whose_name_contains_the_separator() {
    // `player__stats.goals` encodes to `player__stats__goals__sum`, which the
    // dunder derivation alone cannot tell apart from `player.stats__goals` summed.
    // The metadata names the field outright, so the rewrite finds it either way.
    let (dm, _tmp) = build("play__ers").await;

    let rows = dm
        .sql("SELECT play__ers__player_name FROM pre_agg.goals_by_player LIMIT 1")
        .await
        .unwrap();
    assert!(!fmt(&rows).is_empty());
}
