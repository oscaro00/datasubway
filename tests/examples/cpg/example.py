import asyncio
import random
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import pytest
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

with TemporaryDirectory() as temp_dir:

    def cpg_data_model() -> DataModel:
        random.seed(20260227)

        tables = create_cpg_tables()
        joins = create_cpg_joins()
        pre_aggs = create_cpg_pre_aggs()

        dm = DataModel(
            tables=tables,
            joins=joins,
            pre_aggregations=pre_aggs,
            pre_agg_directory=Path(temp_dir),
        )
        all_names = [p.name for p in dm.pre_agg_objects]
        dm.write_pre_aggs(all_names)
        return dm

    dm = cpg_data_model()

    def test_data_model_created():
        assert dm is not None

    def test_pre_agg_parquets_written(tmp_path):
        for pre_agg in dm.pre_agg_objects:
            assert (tmp_path / f"{pre_agg.name}.parquet").exists()
        assert (tmp_path / "_metadata.json").exists()

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
            "havings": {
                "OR": [("revenue", ">=", 1000), ("dim_store.store_id", "<", 30)]
            },
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
