from typing import Any, Iterable

import numpy as np
import polars as pl
from polars._typing import IntoExpr, IntoExprColumn
from polars.lazyframe.group_by import LazyGroupBy


def main() -> None:
    @pl.api.register_lazyframe_namespace("ds")
    class DataSubwayOperations:
        def __init__(self, ldf: pl.LazyFrame) -> None:
            self._ldf = ldf
            self.schema = ldf.collect_schema()
            self.column_names = ldf.collect_schema().names()
            self.column_types = ldf.collect_schema().dtypes()

        def group_by(
            self,
            *by: IntoExpr | Iterable[IntoExpr],
            maintain_order: bool = False,
            **named_by: IntoExpr,
        ) -> pl.LazyFrame | LazyGroupBy:

            if len(by) == 0 and len(named_by) == 0:
                return self._ldf
            else:
                return self._ldf.group_by(
                    *by, maintain_order=maintain_order, **named_by
                )

        def filter(
            self,
            *predicates: IntoExprColumn
            | Iterable[IntoExprColumn]
            | bool
            | list[bool]
            | np.ndarray[Any, Any],
            **constraints: Any,
        ) -> pl.LazyFrame:

            if len(predicates) == 0 and len(constraints) == 0:
                return self._ldf
            else:
                return self._ldf.filter(*predicates, **constraints)

        def agg(
            self,
            *aggs: IntoExpr | Iterable[IntoExpr],
            **named_aggs: IntoExpr,
        ) -> pl.LazyFrame:

            if len(aggs) == 0 and len(named_aggs) == 0:
                return self._ldf
            elif isinstance(self._ldf, LazyGroupBy):
                return self._ldf.agg(*aggs, **named_aggs)
            else:
                return self._ldf.select(*aggs, **named_aggs)

    ldf = pl.LazyFrame(
        {
            "store_id": [1, 2, 3, 4, 5],
            "product_id": [9, 8, 7, 6, 5],
            "revenue": [23, 67, 34, 78, 34],
        }
    )

    print(ldf.ds.group_by().collect())
    print(ldf.ds.filter().collect())

    print(
        ldf.ds.filter()
        .ds.group_by()
        .ds.agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )

    print(
        ldf.ds.filter(pl.col("store_id") <= 3)
        .ds.group_by(pl.col("product_id"))
        .ds.agg(pl.col("revenue").sum().alias("total_revenue"))
        .collect()
    )


if __name__ == "__main__":
    main()
