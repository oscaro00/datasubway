import asyncio
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
from polars.testing import assert_frame_equal

from datasubway.column_context import allow, exclude
from datasubway.data_model import DataModel
from datasubway.measure_decorator import measure
from datasubway.query_context import QueryContext
from tests.examples.cpg.setup import (
    create_cpg_joins,
    create_cpg_pre_aggs,
    create_cpg_tables,
)

_tmpdir = TemporaryDirectory()


def cpg_data_model() -> DataModel:
    random.seed(20260227)

    tables = create_cpg_tables()
    joins = create_cpg_joins()
    pre_aggs = create_cpg_pre_aggs()

    dm = DataModel(
        tables=tables,
        joins=joins,
        pre_aggregations=pre_aggs,
        pre_agg_directory=Path(_tmpdir.name),
    )
    all_names = [p.name for p in dm.pre_agg_objects]
    dm.write_pre_aggs(all_names)
    return dm


dm = cpg_data_model()


def test_data_model_created():
    assert dm is not None


def test_pre_agg_parquets_written():
    for pre_agg in dm.pre_agg_objects:
        assert pre_agg.file_path.exists()
    assert (dm.pre_agg_directory / "_metadata.json").exists()


def test_pre_agg_row_counts_populated():
    assert all(p.row_count > 0 for p in dm.pre_agg_objects)


@measure(dm)
def sales_revenue(qc: QueryContext):
    return (
        dm.table("fact_sales")
        .filter(allow(pattern="*", context=qc.filters))
        .group_by(allow(pattern="*", context=qc.groups))
        .agg(pl.col("fact_sales.revenue").sum().alias("revenue"))
    )


@measure(dm)
def sales_units(qc: QueryContext):
    return (
        dm.table("fact_sales")
        .filter(allow(pattern="*", context=qc.filters))
        .group_by(allow(pattern="*", context=qc.groups))
        .agg(pl.col("fact_sales.units").sum().alias("units"))
    )


def test_complex_query():
    query = {
        "measures": ["sales_revenue"],
        "filters": {
            "AND": [
                ("dim_product.category", "in", ["categoryA", "categoryC"]),
                (
                    "dim_synd_product.category",
                    "!=",
                    "categoryB",
                ),  # this shouldn't affect the results
            ]
        },
        "groups": ["dim_product.category", "dim_store.store_id"],
        "havings": {"OR": [("revenue", ">=", 1000), ("dim_store.store_id", "<", 30)]},
        "sorts": [("revenue", "desc")],
        "limit": 3,
        "offset": 2,
    }

    datasubway_explain = asyncio.run(dm.query(query, explain=True))
    print(datasubway_explain)

    datasubway_result = asyncio.run(dm.query(query))
    print(datasubway_result)

    polars_result = (
        dm.tables["fact_sales"]
        .join(
            dm.tables["dim_store"],
            left_on="fact_sales.store_id",
            right_on="dim_store.store_id",
            how="left",
        )
        .join(
            dm.tables["dim_product"],
            left_on="fact_sales.product_id",
            right_on="dim_product.product_id",
            how="left",
        )
        .filter(pl.col("dim_product.category").is_in(["categoryA", "categoryC"]))
        .group_by("dim_product.category", "fact_sales.store_id")
        .agg(pl.col("fact_sales.revenue").sum().alias("revenue"))
        .filter((pl.col("revenue") >= 1000) | (pl.col("fact_sales.store_id") < 30))
        .sort("revenue", descending=True)
        .slice(offset=2, length=3)
        .rename({"fact_sales.store_id": "dim_store.store_id"})
        .collect()
    )
    print(polars_result)

    assert_frame_equal(datasubway_result, polars_result, check_column_order=False)


def test_complex_query_with_pre_agg():
    query = {
        "measures": ["sales_revenue", "sales_units"],
        "filters": {
            "AND": [
                ("dim_product.category", "in", ["categoryA", "categoryC"]),
                (
                    "dim_synd_product.category",
                    "!=",
                    "categoryB",
                ),  # this shouldn't affect the results
            ]
        },
        "groups": ["dim_product.category", "dim_date.month"],
        "havings": {"OR": [("revenue", ">=", 1000), ("units", "<", 1000)]},
        "sorts": [("dim_date.month", "asc"), ("revenue", "desc")],
        "limit": 5,
        "offset": 1,
    }

    datasubway_explain = asyncio.run(dm.query(query, explain=True))
    print(datasubway_explain)

    datasubway_result = asyncio.run(dm.query(query))
    print(datasubway_result)

    polars_result = (
        dm.tables["fact_sales"]
        .join(
            dm.tables["dim_date"],
            left_on="fact_sales.date",
            right_on="dim_date.date",
            how="inner",
        )
        .join(
            dm.tables["dim_product"],
            left_on="fact_sales.product_id",
            right_on="dim_product.product_id",
            how="left",
        )
        .filter(pl.col("dim_product.category").is_in(["categoryA", "categoryC"]))
        .group_by("dim_product.category", "dim_date.month")
        .agg(
            pl.col("fact_sales.revenue").sum().alias("revenue"),
            pl.col("fact_sales.units").sum().alias("units"),
        )
        .filter((pl.col("revenue") >= 1000) | (pl.col("units") < 1000))
        .sort("dim_date.month", "revenue", descending=[False, True])
        .slice(offset=1, length=5)
        # .rename({"fact_sales.store_id": "dim_store.store_id"})
        .collect()
    )
    print(polars_result)

    assert_frame_equal(datasubway_result, polars_result, check_column_order=False)


@measure(dm)
def store_share_of_total_revenue(qc: QueryContext):
    numerator = (
        dm.table("fact_sales")
        .filter(allow(pattern="*", context=qc.filters))
        .group_by(allow(pattern="*", include=["dim_store.store_id"], context=qc.groups))
        .agg(pl.col("fact_sales.revenue").sum().alias("numerator_revenue"))
    )

    denominator = (
        dm.table("fact_sales")
        .filter(exclude(pattern="dim_store.*", context=qc.filters))
        .group_by(exclude(pattern="dim_store.*", context=qc.groups))
        .agg(pl.col("fact_sales.revenue").sum().alias("total_revenue"))
    )

    join_on = qc.groups if len(qc.groups) >= 1 else None
    join_how = "inner" if len(qc.groups) >= 1 else "cross"

    return (
        numerator.join(denominator, on=join_on, how=join_how)
        .group_by(allow(pattern="*", include=["dim_store.store_id"], context=qc.groups))
        .agg(
            (pl.col("numerator_revenue") / pl.col("total_revenue") * 100)
            .round(1)
            .first()
            .alias("revenue_percentage")
        )
    )


def test_share_of_total_measure():
    query = {
        "measures": ["store_share_of_total_revenue"],
        "filters": {
            "AND": [
                ("dim_product.category", "in", ["categoryA", "categoryC"]),
                ("dim_store.region", "!=", "North"),
                (
                    "dim_synd_product.category",
                    "!=",
                    "categoryB",
                ),  # this shouldn't affect the results
            ]
        },
        "groups": ["dim_product.category"],
        # "havings": {"OR": [("revenue", ">=", 1000), ("units", "<", 1000)]},
        "sorts": [("dim_store.store_id", "asc"), ("dim_product.category", "desc")],
        "limit": 5,
        "offset": 0,
    }

    datasubway_explain = asyncio.run(dm.query(query, explain=True))
    print(datasubway_explain)

    datasubway_result = asyncio.run(dm.query(query))
    print(datasubway_result)

    numerator = (
        dm.tables["fact_sales"]
        .join(
            dm.tables["dim_store"],
            left_on="fact_sales.store_id",
            right_on="dim_store.store_id",
            how="left",
        )
        .join(
            dm.tables["dim_product"],
            left_on="fact_sales.product_id",
            right_on="dim_product.product_id",
            how="left",
        )
        .filter(
            (pl.col("dim_product.category").is_in(["categoryA", "categoryC"]))
            & (pl.col("dim_store.region") != "North")
        )
        .group_by("fact_sales.store_id", "dim_product.category")
        .agg(pl.col("fact_sales.revenue").sum().alias("numerator_revenue"))
    )

    denominator = (
        dm.tables["fact_sales"]
        .join(
            dm.tables["dim_store"],
            left_on="fact_sales.store_id",
            right_on="dim_store.store_id",
            how="left",
        )
        .join(
            dm.tables["dim_product"],
            left_on="fact_sales.product_id",
            right_on="dim_product.product_id",
            how="left",
        )
        .filter(pl.col("dim_product.category").is_in(["categoryA", "categoryC"]))
        .group_by("dim_product.category")
        .agg(pl.col("fact_sales.revenue").sum().alias("total_revenue"))
    )

    polars_result = (
        numerator.join(denominator, on="dim_product.category", how="inner")
        .group_by("fact_sales.store_id", "dim_product.category")
        .agg(
            (pl.col("numerator_revenue") / pl.col("total_revenue") * 100)
            .round(1)
            .first()
            .alias("revenue_percentage")
        )
        .sort("fact_sales.store_id", "dim_product.category", descending=[False, True])
        .slice(offset=0, length=5)
        .rename({"fact_sales.store_id": "dim_store.store_id"})
        .collect()
    )

    print(polars_result)

    assert_frame_equal(datasubway_result, polars_result, check_column_order=False)


@measure(dm)
def rolling_3_day_average_revenue(qc: QueryContext):
    return (
        dm.table("fact_sales")
        .filter(allow(pattern="*", context=qc.filters))
        .sort("fact_sales.date")
        .group_by_dynamic(
            "fact_sales.date",
            every="1d",
            period="3d",
            group_by=allow(pattern="*", context=qc.groups),
        )
        .agg(pl.col("fact_sales.revenue").mean().alias("average_3_day_rolling_revenue"))
    )


def test_rolling_3_day_average_measure():
    query = {
        "measures": ["rolling_3_day_average_revenue"],
        "filters": {
            "AND": [
                ("dim_product.category", "in", ["categoryA", "categoryC"]),
                (
                    "dim_synd_product.category",
                    "!=",
                    "categoryB",
                ),  # this shouldn't affect the results
            ]
        },
        "groups": ["fact_sales.date", "dim_product.category"],
        # "havings": {"OR": [("revenue", ">=", 1000), ("units", "<", 1000)]},
        "sorts": [("fact_sales.date", "asc"), ("dim_product.category", "desc")],
        "limit": 10,
        "offset": 0,
    }

    datasubway_explain = asyncio.run(dm.query(query, explain=True))
    print(datasubway_explain)

    datasubway_result = asyncio.run(dm.query(query))
    print(datasubway_result)

    polars_result = (
        dm.tables["fact_sales"]
        .join(
            dm.tables["dim_product"],
            left_on="fact_sales.product_id",
            right_on="dim_product.product_id",
            how="left",
        )
        .filter(pl.col("dim_product.category").is_in(["categoryA", "categoryC"]))
        .sort("fact_sales.date")
        .group_by_dynamic(
            "fact_sales.date", every="1d", period="3d", group_by="dim_product.category"
        )
        .agg(pl.col("fact_sales.revenue").mean().alias("average_3_day_rolling_revenue"))
        .sort("fact_sales.date", "dim_product.category", descending=[False, True])
        .slice(offset=0, length=10)
        .collect()
    )

    print(polars_result)

    assert_frame_equal(datasubway_result, polars_result, check_column_order=False)
