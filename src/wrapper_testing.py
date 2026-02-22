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

    lfw_agg = LazyFrameWrapper(lf_agg, from_pre_agg=True)

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

    # ── Scenario 1: PreAggregation.covers() ──────────────────────────────────
    print("\n=== Scenario 1: PreAggregation.covers() ===")
    from pathlib import Path

    from datasubway.pre_agg_meta import PreAggregation

    pa = PreAggregation(
        name="orders_daily",
        group_by=["orders.date", "orders.region"],
        raw_aggregations={"orders.revenue": "mean"},  # expands to ['count', 'sum']
        # file_path=Path("_pre_aggregations/orders_daily.parquet"),
        # row_count=100,
    )
    assert pa.aggregations == {"orders.revenue": ["count", "sum"]}, (
        f"'mean' should expand to ['count', 'sum'], got {pa.aggregations}"
    )

    # Should cover: exact group-by, mean needs sum+count (both stored after expansion)
    assert pa.covers(["orders.date", "orders.region"], {"orders.revenue": {"Mean"}}), (
        "Should cover mean (sum+count stored)"
    )

    # Should cover: subset of group-by
    assert pa.covers(["orders.date"], {"orders.revenue": {"Sum"}}), (
        "Should cover sum with partial group-by"
    )

    # Should NOT cover: group-by column not in pre-agg
    assert not pa.covers(["orders.store_id"], {"orders.revenue": {"Sum"}}), (
        "Should not cover unknown group-by col"
    )

    # Should NOT cover: std without sumsq (only mean stored → sum+count, no sumsq)
    pa_mean_only = PreAggregation(
        name="orders_mean_only",
        group_by=["orders.date"],
        raw_aggregations={"orders.revenue": "mean"},  # expands to sum+count, no sumsq
        file_path=Path("_pre_aggregations/orders_mean_only.parquet"),
    )
    assert not pa_mean_only.covers(["orders.date"], {"orders.revenue": {"Std"}}), (
        "Should not cover std without sumsq stored"
    )

    print("Scenario 1 PASSED")

    # ── Scenario 2: extract_agg_requirements() ───────────────────────────────
    print("\n=== Scenario 2: extract_agg_requirements() ===")
    from datasubway.polars_wrappers.pre_agg_expr import extract_agg_requirements

    reqs_sum = extract_agg_requirements(pl.col("revenue").sum())
    assert reqs_sum == {"revenue": {"Sum"}}, (
        f"Expected {{'revenue': {{'Sum'}}}}, got {reqs_sum}"
    )

    reqs_mean = extract_agg_requirements(pl.col("revenue").mean())
    assert reqs_mean == {"revenue": {"Mean"}}, f"Expected mean, got {reqs_mean}"

    reqs_std = extract_agg_requirements(pl.col("a").std())
    assert reqs_std == {"a": {"Std"}}, f"Expected std, got {reqs_std}"

    reqs_aliased = extract_agg_requirements(pl.col("revenue").sum().alias("total"))
    assert reqs_aliased == {"revenue": {"Sum"}}, (
        f"Alias should not affect, got {reqs_aliased}"
    )

    print("Scenario 2 PASSED")

    # ── Scenario 3: proxy.resolve() with pre-agg hit ─────────────────────────
    print("\n=== Scenario 3: proxy.resolve() with pre-agg hit ===")
    import tempfile

    from datasubway.data_model import DataModel

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        source_lf = pl.LazyFrame(
            {
                "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
                "revenue": [100, 200, 300],
            }
        )

        dm = DataModel(
            tables={"orders": source_lf},
            pre_aggregations={
                "orders_daily": {
                    "group_by": ["orders.date"],
                    "aggregations": {"orders.revenue": "mean"},  # expanded to sum+count
                }
            },
            pre_agg_directory=tmp,
        )

        # Write the pre-agg via DataModel — metadata (row_count, written_at) is recorded automatically
        pre_agg_lf = pl.LazyFrame(
            {
                "date": ["2024-01-01", "2024-01-02"],
                "revenue-sum": [1000, 2000],
                "revenue-count": [10, 20],
            }
        )
        written = dm.write_pre_agg("orders_daily", pre_agg_lf)
        assert written.row_count == 2, f"Expected row_count=2, got {written.row_count}"
        assert written.written_at is not None, "written_at should be set"

        proxy = dm.table("orders")
        result_proxy = (
            proxy.filter(pl.col("date") >= "2024-01-01")
            .group_by(["orders.date"])
            .agg(pl.col("revenue").mean().alias("avg_revenue"))
        )

        resolved = result_proxy.resolve()
        assert resolved.from_pre_agg is True, "Should have used pre-agg source"

        df = resolved.lf.collect()
        assert "avg_revenue" in df.columns, (
            f"Expected avg_revenue column, got {df.columns}"
        )
        print(f"  Result: {df}")

    print("Scenario 3 PASSED")

    # ── Scenario 4: proxy.resolve() with pre-agg miss ────────────────────────
    print("\n=== Scenario 4: proxy.resolve() with pre-agg miss ===")
    source_lf2 = pl.LazyFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "revenue": [100, 200, 300],
        }
    )

    dm_no_preagg = DataModel(tables={"orders": source_lf2})

    proxy2 = dm_no_preagg.table("orders")
    result_proxy2 = (
        proxy2.filter(pl.col("date") >= "2024-01-01")
        .group_by(["date"])
        .agg(pl.col("revenue").sum().alias("total_revenue"))
    )

    resolved2 = result_proxy2.resolve()
    assert resolved2.from_pre_agg is False, "Should have fallen back to source table"

    df2 = resolved2.lf.collect()
    assert "total_revenue" in df2.columns, (
        f"Expected total_revenue column, got {df2.columns}"
    )
    print(f"  Result: {df2}")

    print("Scenario 4 PASSED")

    # ── Scenario 5: Join recording ────────────────────────────────────────────
    print("\n=== Scenario 5: Join recording ===")
    orders_lf = pl.LazyFrame(
        {
            "order_id": [1, 2, 3],
            "product_id": [10, 20, 10],
            "revenue": [100, 200, 150],
        }
    )
    products_lf = pl.LazyFrame(
        {
            "product_id": [10, 20],
            "active": [True, False],
        }
    )

    dm_join = DataModel(tables={"orders": orders_lf, "products": products_lf})

    proxy_join = (
        dm_join.table("orders")
        .join(
            dm_join.table("products").filter(pl.col("active")),
            left_on="product_id",
            right_on="product_id",
        )
        .group_by(["order_id"])
        .agg(pl.col("revenue").sum().alias("total"))
    )

    assert proxy_join.has_join is True, "Should have recorded join"
    resolved_join = proxy_join.resolve()
    assert resolved_join.from_pre_agg is False, "Join path should use source table"

    df_join = resolved_join.lf.collect()
    assert "total" in df_join.columns, f"Expected total column, got {df_join.columns}"
    print(f"  Result: {df_join}")

    print("Scenario 5 PASSED")

    # ── Scenario 6: Multi-measure lazy join ──────────────────────────────────
    print("\n=== Scenario 6: Multi-measure lazy join ===")
    sales_lf = pl.LazyFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "revenue": [100, 200, 300],
            "units": [1, 2, 3],
        }
    )

    dm_multi = DataModel(tables={"sales": sales_lf})

    def measure_revenue(dm: DataModel) -> "LazyFrameProxy":
        return (
            dm.table("sales")
            .group_by(["date"])
            .agg(pl.col("revenue").sum().alias("total_revenue"))
        )

    def measure_units(dm: DataModel) -> "LazyFrameProxy":
        return (
            dm.table("sales")
            .group_by(["date"])
            .agg(pl.col("units").sum().alias("total_units"))
        )

    lf1 = measure_revenue(dm_multi).resolve()
    lf2 = measure_units(dm_multi).resolve()

    # Join lazy frames BEFORE collect — single combined plan
    combined = lf1.lf.join(lf2.lf, on="date", how="left").collect()
    assert "total_revenue" in combined.columns, (
        f"Missing total_revenue: {combined.columns}"
    )
    assert "total_units" in combined.columns, f"Missing total_units: {combined.columns}"
    print(f"  Combined result: {combined}")

    print("Scenario 6 PASSED")

    # ── Scenario 7: lazygroupby_wrapper bug fix ───────────────────────────────
    print("\n=== Scenario 7: lazygroupby_wrapper bug fix (from_pre_agg) ===")
    from datasubway.polars_wrappers.lazygroupby_wrapper import LazyGroupByWrapper

    lf_source = pl.LazyFrame(
        {"store": [1, 2], "revenue-sum": [100, 200], "revenue-count": [5, 10]}
    )
    lfw_source = LazyFrameWrapper(lf_source, from_pre_agg=True)

    lgbw = lfw_source.group_by("store")
    assert isinstance(lgbw, LazyGroupByWrapper), "Should return LazyGroupByWrapper"
    assert lgbw.from_pre_agg is True, (
        "from_pre_agg should be True (was broken before fix)"
    )

    result7 = lgbw.agg(pl.col("revenue").mean().alias("avg")).collect()
    assert "avg" in result7.columns, f"Expected avg column, got {result7.columns}"
    print(f"  Result: {result7}")

    print("Scenario 7 PASSED")

    print("\n=== All scenarios PASSED ===")
