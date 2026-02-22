"""Explore the JSON format produced by pl.Expr.meta.serialize(format='json').

Run: python src2/demo_serialize.py
"""

# TODO: use this file for a reference to create tests that verify the serialization hasn't changed when polars versions change

from __future__ import annotations

import json

import polars as pl


def show(label: str, expr: pl.Expr) -> None:
    tree = json.loads(expr.meta.serialize(format="json"))
    print(f"--- {label} ---")
    print(f"  expr:  {expr}")
    print(f"  json:  {json.dumps(tree, indent=4)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════
# 1. LEAF NODES — the building blocks
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("1. LEAF NODES")
print("=" * 70, "\n")

show("column reference", pl.col("revenue"))
show("literal int", pl.lit(42))
show("literal float", pl.lit(3.14))
show("literal string", pl.lit("hello"))
show("literal null", pl.lit(None))

# ═══════════════════════════════════════════════════════════════════════════
# 2. AGGREGATIONS — wrapped in {"Agg": {<type>: ...}}
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("2. AGGREGATIONS")
print("=" * 70, "\n")

show("sum", pl.col("revenue").sum())
show("mean", pl.col("revenue").mean())
show("min", pl.col("revenue").min())
show("max", pl.col("revenue").max())
show("count", pl.col("revenue").count())
show("std (ddof=1)", pl.col("revenue").std())
show("var (ddof=1)", pl.col("revenue").var())
show("first", pl.col("revenue").first())
show("last", pl.col("revenue").last())
show("median", pl.col("revenue").median())
show("any", pl.col("revenue").any())
show("all", pl.col("revenue").all())
show("n unique", pl.col("revenue").n_unique())
show("approx n unique", pl.col("revenue").approx_n_unique())
show("arg max", pl.col("revenue").arg_max())
show("arg min", pl.col("revenue").arg_min())
show("len", pl.col("revenue").len())
show("null count", pl.col("revenue").null_count())
show("product", pl.col("revenue").product())

# ═══════════════════════════════════════════════════════════════════════════
# 3. FUNCTIONS — wrapped in {"Function": {"input": [...], "function": ...}}
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("3. FUNCTIONS (post-processing)")
print("=" * 70, "\n")

show("round", pl.col("revenue").round(2))
show("cast to float", pl.col("revenue").cast(pl.Float64))
show("abs", pl.col("revenue").abs())
show("sqrt", pl.col("revenue").sqrt())
show("pow", pl.col("revenue").pow(2))
show("log", pl.col("revenue").log(10))
show("fill_null", pl.col("revenue").fill_null(0))

# ═══════════════════════════════════════════════════════════════════════════
# 4. ALIAS — wraps any expression in {"Alias": [<inner>, "name"]}
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("4. ALIAS")
print("=" * 70, "\n")

show("simple alias", pl.col("revenue").alias("rev"))
show("agg + alias", pl.col("revenue").sum().alias("total"))

# ═══════════════════════════════════════════════════════════════════════════
# 5. BINARY EXPRESSIONS — {"BinaryExpr": {"left": ..., "op": ..., "right": ...}}
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("5. BINARY EXPRESSIONS")
print("=" * 70, "\n")

show("add two cols", pl.col("a") + pl.col("b"))
show("multiply by literal", pl.col("a") * 100)
show("divide aggs", pl.col("revenue").sum() / pl.col("orders").count())

# ═══════════════════════════════════════════════════════════════════════════
# 6. CHAINED EXPRESSIONS — nesting shows how chains compose
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("6. CHAINED EXPRESSIONS")
print("=" * 70, "\n")

show(
    "sum -> round -> alias",
    pl.col("revenue").sum().round(2).alias("total"),
)
show(
    "mean -> abs -> cast -> alias",
    pl.col("revenue").mean().abs().cast(pl.Int64).alias("result"),
)
show(
    "(sum + mean) -> round -> alias",
    (pl.col("revenue").sum() + pl.col("cost").mean()).round(1).alias("metric"),
)

# ═══════════════════════════════════════════════════════════════════════════
# 7. SORTING / RANKING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("7. SORT / RANK")
print("=" * 70, "\n")

show("sort_by", pl.col("revenue").sort())
show("rank", pl.col("revenue").rank("dense"))

# ═══════════════════════════════════════════════════════════════════════════
# 7.5 FILTERING
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("7.5 FILTERING")
print("=" * 70, "\n")

show("single filter", pl.col("revenue") > 30)
show(
    "complex filter",
    (
        (pl.col("country") == "US")
        | (pl.col("country") == "CA")
        | (pl.col("country") == "IR")
    )
    & ~(pl.col("revenue") >= 1000),
)


# ═══════════════════════════════════════════════════════════════════════════
# 8. ROUND-TRIP: serialize -> modify -> deserialize
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("8. ROUND-TRIP DEMO")
print("=" * 70, "\n")

original = pl.col("revenue").sum().round(2).alias("total")
tree = json.loads(original.meta.serialize(format="json"))

print("Original tree:")
print(json.dumps(tree, indent=4))

# Walk into the tree and change the column name
tree["Alias"][0]["Function"]["input"][0]["Agg"]["Sum"]["Column"] = "revenue-sum"

print("\nModified tree:")
print(json.dumps(tree, indent=4))

rebuilt = pl.Expr.deserialize(json.dumps(tree).encode(), format="json")
print(f"\nOriginal expr:  {original}")
print(f"Rebuilt expr:   {rebuilt}")

# Verify it actually works
df = pl.DataFrame({"revenue-sum": [10, 20, 30]})
print(f"Result:         {df.select(rebuilt)}")
