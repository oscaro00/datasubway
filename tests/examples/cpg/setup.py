import random
from datetime import date, datetime, timedelta

import polars as pl

start_date_str = "1/1/2024"
end_date_str = "12/31/2025"
time_format = "%m/%d/%Y"

start_date_seconds = datetime.strptime(start_date_str, time_format)
end_date_seconds = datetime.strptime(end_date_str, time_format)


def random_date(start, end):
    delta = end - start
    total_seconds = int(delta.total_seconds())

    random_second = random.randrange(total_seconds)

    return start + timedelta(seconds=random_second)


def create_period_lf(name, start, end):
    return pl.LazyFrame(
        {
            "period": name,
            "date": pl.date_range(start=start, end=end, interval="1d", eager=True),
        }
    )


def create_df_date():
    df_dim_date = pl.LazyFrame(
        {
            "date": pl.date_range(
                start=date(2024, 1, 1),
                end=date(2025, 12, 31),
                interval="1d",
                eager=True,
            )
        }
    ).with_columns(
        pl.col("date").dt.month().alias("month"),
        pl.col("date").dt.year().alias("year"),
        pl.col("date").dt.day().alias("day"),
        pl.col("date").dt.is_leap_year().alias("is_leap_year"),
        pl.col("date").dt.quarter().alias("quarter"),
        pl.col("date").dt.week().alias("week"),
        pl.col("date").dt.weekday().alias("weekday"),
    )

    return df_dim_date


def create_dim_period():
    df_dim_period = pl.concat(
        [
            create_period_lf("current_year", date(2025, 1, 1), date(2025, 12, 31)),
            create_period_lf("prior_year", date(2024, 1, 1), date(2024, 12, 31)),
            create_period_lf("current_month", date(2025, 12, 1), date(2025, 12, 31)),
            create_period_lf("prior_month", date(2025, 11, 1), date(2025, 11, 30)),
            create_period_lf("current_quarter", date(2025, 10, 1), date(2025, 12, 31)),
            create_period_lf("prior_quarter", date(2025, 7, 1), date(2025, 10, 31)),
            create_period_lf("last_1_month", date(2025, 12, 1), date(2025, 12, 31)),
            create_period_lf("last_3_months", date(2025, 10, 1), date(2025, 12, 31)),
            create_period_lf("last_6_months", date(2025, 7, 1), date(2025, 12, 31)),
            create_period_lf("last_12_months", date(2025, 1, 1), date(2025, 12, 31)),
        ]
    )

    return df_dim_period


def create_dim_store():
    df_dim_store = (
        pl.select(pl.int_range(0, 10).alias("store_id"))
        .lazy()
        .with_columns(
            pl.format("store_{}", pl.col("store_id")).alias("store_name"),
            pl.when(pl.col("store_id") % 4 == 0)
            .then(pl.lit("North"))
            .when(pl.col("store_id") % 4 == 1)
            .then(pl.lit("East"))
            .when(pl.col("store_id") % 4 == 2)
            .then(pl.lit("South"))
            .when(pl.col("store_id") % 4 == 3)
            .then(pl.lit("West"))
            .otherwise(pl.lit("NA"))
            .alias("region"),
        )
    )

    return df_dim_store


def create_dim_product():
    df_dim_product = (
        pl.select(pl.int_range(0, 50).alias("product_id"))
        .lazy()
        .with_columns(
            pl.format("product_{}", pl.col("product_id")).alias("product_name"),
            pl.when(pl.col("product_id") % 4 == 0)
            .then(pl.lit("categoryA"))
            .when(pl.col("product_id") % 4 == 1)
            .then(pl.lit("categoryB"))
            .when(pl.col("product_id") % 4 == 2)
            .then(pl.lit("categoryC"))
            .when(pl.col("product_id") % 4 == 3)
            .then(pl.lit("categoryD"))
            .otherwise(pl.lit("NA"))
            .alias("category"),
        )
    )

    return df_dim_product


def create_fact_sales():
    df_fact_sales = pl.LazyFrame(
        {
            "date": [
                random_date(start_date_seconds, end_date_seconds) for _ in range(10000)
            ],
            "store_id": [random.binomialvariate(10, 0.5) for _ in range(10000)],
            "product_id": [random.binomialvariate(50, 0.2) for _ in range(10000)],
            "revenue": [round(random.gammavariate(2, 2) * 10, 2) for _ in range(10000)],
            "units": [int(random.triangular(3, 10, 15) * 10) for _ in range(10000)],
        }
    ).with_columns(pl.col("date").cast(pl.Date))

    return df_fact_sales


def create_fact_syndicated():
    df_fact_syndicated = (
        pl.LazyFrame(
            {
                "date": [
                    random_date(start_date_seconds, end_date_seconds)
                    for _ in range(1000)
                ],
                "product_id": [random.binomialvariate(50, 0.2) for _ in range(1000)],
                "revenue": [
                    round(random.gammavariate(2, 2) * 100, 2) for _ in range(1000)
                ],
                "units": [int(random.triangular(3, 10, 15) * 100) for _ in range(1000)],
            }
        )
        .with_columns(pl.col("date").cast(pl.Date))
        .with_columns(pl.int_range(0, 1000).alias("region"))
        .with_columns(
            pl.when(pl.col("region") % 4 == 0)
            .then(pl.lit("North"))
            .when(pl.col("region") % 4 == 1)
            .then(pl.lit("East"))
            .when(pl.col("region") % 4 == 2)
            .then(pl.lit("South"))
            .when(pl.col("region") % 4 == 3)
            .then(pl.lit("West"))
            .otherwise(pl.lit("NA"))
            .alias("region")
        )
    )

    return df_fact_syndicated


def create_dim_synd_product():
    df_dim_synd_product = (
        pl.select(pl.int_range(0, 50).alias("product_id"))
        .lazy()
        .with_columns(
            pl.format("product_{}", pl.col("product_id")).alias("product_name"),
            pl.when(pl.col("product_id") % 4 == 0)
            .then(pl.lit("categoryA"))
            .when(pl.col("product_id") % 4 == 1)
            .then(pl.lit("categoryB"))
            .when(pl.col("product_id") % 4 == 2)
            .then(pl.lit("categoryC"))
            .when(pl.col("product_id") % 4 == 3)
            .then(pl.lit("categoryD"))
            .otherwise(pl.lit("NA"))
            .alias("category"),
        )
    )

    return df_dim_synd_product


def create_cpg_tables():
    return {
        "dim_date": create_df_date(),
        "dim_period": create_dim_period(),
        "dim_store": create_dim_store(),
        "dim_product": create_dim_product(),
        "fact_sales": create_fact_sales(),
        "fact_syndicated": create_fact_syndicated(),
        "dim_synd_product": create_dim_synd_product(),
    }


def create_cpg_joins():
    return [
        {
            "left": "dim_date",
            "right": "dim_period",
            "left_on": ["date"],
            "right_on": ["date"],
            "how": "inner",
            "direction": "both",
        },
        {
            "left": "fact_sales",
            "right": "dim_date",
            "left_on": ["date"],
            "right_on": ["date"],
            "how": "inner",
            "direction": "right2left",
        },
        {
            "left": "fact_sales",
            "right": "dim_store",
            "left_on": ["store_id"],
            "right_on": ["store_id"],
            "how": "left",
            "direction": "right2left",
        },
        {
            "left": "fact_sales",
            "right": "dim_product",
            "left_on": ["product_id"],
            "right_on": ["product_id"],
            "how": "left",
            "direction": "right2left",
        },
        {
            "left": "fact_syndicated",
            "right": "dim_synd_product",
            "left_on": ["product_id"],
            "right_on": ["product_id"],
            "how": "left",
            "direction": "right2left",
        },
        {
            "left": "fact_syndicated",
            "right": "dim_date",
            "left_on": ["date"],
            "right_on": ["date"],
            "how": "inner",
            "direction": "right2left",
        },
    ]


def create_cpg_pre_aggs():
    return {
        "sales_syndicated_by_period": {
            "group_by": ["dim_period.period"],
            "aggregations": {
                "fact_sales.revenue": ["sum", "mean", "std"],
                "fact_sales.units": ["sum", "mean", "std"],
                "fact_syndicated.revenue": ["sum", "mean", "std"],
                "fact_syndicated.units": ["sum", "mean", "std"],
            },
        },
        "sales_by_month_week": {
            "group_by": ["dim_date.month", "dim_date.week"],
            "aggregations": {
                "fact_sales.revenue": ["sum", "mean", "var"],
                "fact_sales.units": ["sum", "mean", "var"],
            },
        },
        "sales_by_region_month_week": {
            "group_by": ["dim_date.month", "dim_date.week", "dim_store.region"],
            "aggregations": {
                "fact_sales.revenue": ["sum", "mean", "var"],
                "fact_sales.units": ["sum", "mean", "var"],
            },
        },
        "sales_by_category_month_week": {
            "group_by": ["dim_date.month", "dim_date.week", "dim_product.category"],
            "aggregations": {
                "fact_sales.revenue": ["sum", "mean", "var"],
                "fact_sales.units": ["sum", "mean", "var"],
            },
        },
        "sales_by_region_category_month_week": {
            "group_by": [
                "dim_date.month",
                "dim_date.week",
                "dim_store.region",
                "dim_product.category",
            ],
            "aggregations": {
                "fact_sales.revenue": ["sum", "mean", "var"],
                "fact_sales.units": ["sum", "mean", "var"],
            },
        },
        "syndicated_by_category_month_week": {
            "group_by": [
                "dim_date.month",
                "dim_date.week",
                "dim_synd_product.category",
            ],
            "aggregations": {
                "fact_syndicated.revenue": ["sum", "mean", "var"],
                "fact_syndicated.units": ["sum", "mean", "var"],
            },
        },
    }
