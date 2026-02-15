from __future__ import annotations

import polars as pl
from datasubway.polars_wrappers.lazyframe_wrapper import LazyFrameWrapper

if __name__ == "__main__":
    lf = pl.LazyFrame(
        {
            "store_id": [1, 2, 3, 4, 5],
            "product_id": [9, 8, 7, 6, 5],
            "revenue": [23, 67, 34, 78, 34],
        }
    )

    lfw = LazyFrameWrapper(lf)

    print(lfw.group_by())
    print(lfw.filter().collect())

    print(
        lfw.filter()
        .group_by()
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )

    print(
        lfw.filter(pl.col("store_id") <= 3)
        .group_by(pl.col("product_id"))
        .agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )

    lf_agg = pl.LazyFrame(
        {
            "store_id": [1, 2, 3, 4, 5],
            "revenue-sum": [100, 200, 300, 400, 500],
            "revenue-count": [7, 6, 5, 4, 3],
        }
    )

    lfw_agg = LazyFrameWrapper(lf_agg)

    print(
        lfw_agg.filter(pl.col("store_id") <= 3)
        .group_by()
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("revenue").mean().round(2).alias("average_revenue"),
        )
        .collect()
    )

    print(
        lfw_agg.filter()
        .group_by("store_id")
        .agg(
            pl.col("revenue").sum().alias("total_revenue"),
            pl.col("revenue").mean().round(2).alias("average_revenue"),
        )
        .collect()
    )
