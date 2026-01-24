"""Pure functions for building libcst expressions for LazyFrame access.

These functions generate CST nodes that represent:
- Pre-aggregation parquet scans
- Direct table access
- Join chains between tables
"""
from typing import List, Dict
import libcst as cst


def build_pre_agg_cst(pre_agg_name: str) -> cst.BaseExpression:
    """
    Build CST node for: pl.scan_parquet(self.pre_agg_directory / 'pre_agg_name.parquet')

    This format allows later transformers to detect pre-agg usage by checking
    if the code contains self.pre_agg_directory.

    Args:
        pre_agg_name: Name of the pre-aggregation (without .parquet extension)

    Returns:
        CST expression node for scanning the pre-aggregation parquet file
    """
    return cst.Call(
        func=cst.Attribute(
            value=cst.Name("pl"),
            attr=cst.Name("scan_parquet")
        ),
        args=[
            cst.Arg(
                value=cst.BinaryOperation(
                    left=cst.Attribute(
                        value=cst.Name("self"),
                        attr=cst.Name("pre_agg_directory")
                    ),
                    operator=cst.Divide(),  # / operator
                    right=cst.SimpleString(f"'{pre_agg_name}.parquet'")
                )
            )
        ]
    )


def build_table_access_cst(table_name: str) -> cst.BaseExpression:
    """
    Build CST node for: self.tables['table_name']

    Args:
        table_name: Name of the table to access

    Returns:
        CST expression node for accessing the table from self.tables
    """
    return cst.Subscript(
        value=cst.Attribute(
            value=cst.Name("self"),
            attr=cst.Name("tables")
        ),
        slice=[
            cst.SubscriptElement(
                slice=cst.Index(
                    value=cst.SimpleString(f"'{table_name}'")
                )
            )
        ]
    )


def build_join_chain_cst(
    base_table: str,
    join_specs: List[Dict]
) -> cst.BaseExpression:
    """
    Build CST node for inline join chain.

    Example output:
        self.tables['base'].join(self.tables['t2'], left_on=['col1'], right_on=['col2'], how='inner')

    Args:
        base_table: Starting table name
        join_specs: List of join specifications from join_lookup.
                   Each spec should have: 'right', 'left_on', 'right_on', 'how'

    Returns:
        CST node representing the chained join expression
    """
    # Start with base table access
    result = build_table_access_cst(base_table)

    # Chain each join
    for join_spec in join_specs:
        right_table = join_spec['right']
        left_on = join_spec['left_on']
        right_on = join_spec['right_on']
        how = join_spec['how']

        # Build join call
        result = cst.Call(
            func=cst.Attribute(
                value=result,
                attr=cst.Name("join")
            ),
            args=[
                # First arg: self.tables['right_table']
                cst.Arg(value=build_table_access_cst(right_table)),
                # left_on keyword arg
                cst.Arg(
                    keyword=cst.Name("left_on"),
                    value=cst.List([
                        cst.Element(value=cst.SimpleString(f"'{col}'"))
                        for col in left_on
                    ])
                ),
                # right_on keyword arg
                cst.Arg(
                    keyword=cst.Name("right_on"),
                    value=cst.List([
                        cst.Element(value=cst.SimpleString(f"'{col}'"))
                        for col in right_on
                    ])
                ),
                # how keyword arg
                cst.Arg(
                    keyword=cst.Name("how"),
                    value=cst.SimpleString(f"'{how}'")
                )
            ]
        )

    return result
