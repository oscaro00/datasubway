"""PreAggManager class for managing pre-aggregations."""

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal
import re

import polars as pl

from datasubway.column_utils import columns_match


# Type aliases for aggregation functions
AggFuncLiteral = Literal[
    'sum', 'mean', 'min', 'max', 'count', 'len',
    'null_count', 'first', 'last', 'n_unique', 'std', 'var'
]
AggFuncType = Union[AggFuncLiteral, List[AggFuncLiteral]]


class PreAggManager:
    """Manages pre-aggregation writing, matching, and metadata."""

    def __init__(
        self,
        tables: Dict[str, pl.LazyFrame],
        pre_aggregations: Dict[str, Any],
        pre_agg_directory: Path,
        join_lookup: Dict[str, Dict[str, Any]]
    ):
        """Initialize PreAggManager.

        Args:
            tables: Dictionary mapping table names to LazyFrames
            pre_aggregations: Pre-aggregation configuration
            pre_agg_directory: Directory to store parquet files
            join_lookup: Join lookup dictionary from JoinGraph
        """
        self.tables = tables
        self.pre_aggregations = pre_aggregations
        self.pre_agg_directory = pre_agg_directory
        self.join_lookup = join_lookup
        self.metadata: List[Dict] = []

    def write(self, write: Union[str, List[str]]) -> None:
        """Write pre-aggregations to parquet files.

        Args:
            write: 'all' to write all, or name(s) of specific pre-aggregations

        Raises:
            Exception: If pre-aggregation name not found
        """
        pre_aggs_to_write = []
        if write == 'all' or write == ['all']:
            pre_aggs_to_write = list(self.pre_aggregations.keys())
        elif isinstance(write, str):
            if write in self.pre_aggregations.keys():
                pre_aggs_to_write.append(write)
            else:
                raise Exception(f'{write} not in list of defined pre aggregations')
        else:
            for pre_agg_name in write:
                if pre_agg_name in self.pre_aggregations.keys():
                    pre_aggs_to_write.append(pre_agg_name)
                else:
                    raise Exception(f'{pre_agg_name} not in list of defined pre aggregations')

        # Ensure pre_agg_directory exists
        self.pre_agg_directory.mkdir(parents=True, exist_ok=True)

        # Write each pre-aggregation
        for pre_agg_name in pre_aggs_to_write:
            pre_agg_config = self.pre_aggregations[pre_agg_name]

            # Extract configuration
            group_by_cols = pre_agg_config['group_by']
            aggregations = pre_agg_config['aggregations']
            normalized_aggs = self._normalize_aggregations(aggregations)

            # Build list of all expressions using _get_calculation
            all_exprs = []
            for col_name, agg_funcs in normalized_aggs.items():
                if not agg_funcs:
                    raise ValueError(
                        f"Pre-aggregation '{pre_agg_name}' has empty aggregation list for column '{col_name}'. "
                        f"Provide at least one aggregation function."
                    )
                for agg_func in agg_funcs:
                    exprs = self._get_calculation(col_name, agg_func)
                    all_exprs.extend(exprs)

            # Determine which tables are needed and build base LazyFrame
            agg_columns = list(normalized_aggs.keys())
            all_columns = agg_columns + group_by_cols
            base_lf = self._resolve_tables(all_columns, base_table_hint=agg_columns[0])

            # Execute aggregation (strip table prefixes from group_by columns)
            group_by_col_names = [col.split('.')[-1] if '.' in col else col for col in group_by_cols]
            result = base_lf.group_by(group_by_col_names).agg(all_exprs)

            # Write to parquet
            output_path = self.pre_agg_directory / f'{pre_agg_name}.parquet'
            result.collect().write_parquet(output_path)

            # Get row count for metadata
            row_count = result.select(pl.len()).collect().item()

            # Update metadata
            self.metadata.append({
                'name': pre_agg_name,
                'path': str(output_path),
                'last_modified': datetime.now(),
                'group_by': group_by_cols,
                'aggregations': normalized_aggs,
                'row_count': row_count
            })

        # Sort metadata by row count
        self.metadata.sort(key=lambda x: x['row_count'])

    def _resolve_tables(self, columns: List[str], base_table_hint: str) -> pl.LazyFrame:
        """Build LazyFrame with all tables needed for the given columns.

        Args:
            columns: List of column references (may include table prefixes)
            base_table_hint: Column name to extract base table from

        Returns:
            LazyFrame with all necessary tables joined together

        Raises:
            ValueError: If columns don't have table prefixes or tables can't be joined
        """
        # Extract table names from column prefixes
        tables_needed = set()
        for col in columns:
            if '.' in col:
                table_name = col.split('.')[0]
                tables_needed.add(table_name)
            else:
                raise ValueError(f"Column '{col}' must have table prefix (e.g., 'table.column')")

        if not tables_needed:
            raise ValueError("Cannot determine tables needed - columns must have table prefix")

        if len(tables_needed) == 1:
            # Single table - no joins needed
            return self.tables[tables_needed.pop()]

        # Multiple tables - start with the table from base_table_hint
        base_table = base_table_hint.split('.')[0] if '.' in base_table_hint else base_table_hint
        remaining_tables = tables_needed - {base_table}

        # Start with the base table
        result = self.tables[base_table]

        # Join each remaining table to the base
        for target_table in remaining_tables:
            if base_table not in self.join_lookup:
                raise ValueError(f"No join path exists from {base_table} to {target_table}")

            if target_table not in self.join_lookup[base_table]:
                raise ValueError(f"No join path found from {base_table} to {target_table}")

            # Get join information
            join_info = self.join_lookup[base_table][target_table]
            join_specs = join_info['join_specs']

            # Apply each join in the chain
            for join_spec in join_specs:
                right_table = join_spec['right']
                result = result.join(
                    self.tables[right_table],
                    left_on=join_spec['left_on'],
                    right_on=join_spec['right_on'],
                    how=join_spec['how']
                )

        return result

    def _get_calculation(
        self,
        col_name: str,
        agg_func: AggFuncLiteral
    ) -> List[pl.Expr]:
        """Generate expressions to store pre-aggregated components.

        Args:
            col_name: Original column name (may include table prefix)
            agg_func: Aggregation function type

        Returns:
            List of pl.Expr with proper aliases for storage
        """
        # Extract just column name (strip table prefix if present)
        output_col_name = col_name.split('.')[-1] if '.' in col_name else col_name
        col_ref = output_col_name

        match agg_func:
            # Simple aggregations (no decomposition needed)
            case 'sum':
                return [pl.col(col_ref).sum().alias(f'{output_col_name}-sum')]

            case 'min':
                return [pl.col(col_ref).min().alias(f'{output_col_name}-min')]

            case 'max':
                return [pl.col(col_ref).max().alias(f'{output_col_name}-max')]

            case 'count' | 'len':
                return [pl.len().alias(f'{output_col_name}-count')]

            case 'null_count':
                return [pl.col(col_ref).null_count().alias(f'{output_col_name}-null_count')]

            case 'first':
                return [pl.col(col_ref).first().alias(f'{output_col_name}-first')]

            case 'last':
                return [pl.col(col_ref).last().alias(f'{output_col_name}-last')]

            # Complex aggregations (require decomposition)
            case 'mean':
                return [
                    pl.col(col_ref).sum().alias(f'{output_col_name}-mean-sum'),
                    pl.len().alias(f'{output_col_name}-mean-count')
                ]

            case 'std':
                return [
                    pl.col(col_ref).sum().alias(f'{output_col_name}-std-sum'),
                    pl.col(col_ref).pow(2).sum().alias(f'{output_col_name}-std-sumsq'),
                    pl.len().alias(f'{output_col_name}-std-count')
                ]

            case 'var':
                return [
                    pl.col(col_ref).sum().alias(f'{output_col_name}-var-sum'),
                    pl.col(col_ref).pow(2).sum().alias(f'{output_col_name}-var-sumsq'),
                    pl.len().alias(f'{output_col_name}-var-count')
                ]

            # Special cases (expensive - store arrays)
            case 'n_unique':
                return [pl.col(col_ref).unique().alias(f'{output_col_name}-unique-set')]

            case _:
                raise ValueError(f"Unsupported aggregation function: {agg_func}")

    def _normalize_aggregations(
        self,
        aggregations: Dict[str, AggFuncType]
    ) -> Dict[str, List[AggFuncLiteral]]:
        """Normalize aggregations dict to always use lists.

        Args:
            aggregations: Dict mapping column names to agg functions

        Returns:
            Dict mapping column names to lists of agg functions
        """
        normalized = {}
        for col_name, agg_func in aggregations.items():
            if isinstance(agg_func, list):
                normalized[col_name] = agg_func
            else:
                normalized[col_name] = [agg_func]
        return normalized

    def find_matching(
        self,
        group_by_cols: List[str],
        agg_cols: Dict[str, str],
        original_table: str
    ) -> Optional[Dict]:
        """Find the smallest pre-agg that satisfies measure requirements.

        Matching criteria (ALL must be true):
        1. Pre-agg group_by must equal or be superset of measure group_by
        2. Pre-agg must contain all required aggregation columns
        3. Pre-agg aggregation functions must exactly match

        Args:
            group_by_cols: Columns being grouped by in the query
            agg_cols: Dict mapping column to aggregation function
            original_table: The base table being queried

        Returns:
            Pre-agg metadata dict or None
        """
        for pre_agg in self.metadata:  # Already sorted by row_count
            # Check 1: Pre-agg group_by must be superset
            all_group_by_match = True
            for measure_col in group_by_cols:
                found_match = False
                for pre_col in pre_agg['group_by']:
                    if columns_match(measure_col, pre_col):
                        found_match = True
                        break
                if not found_match:
                    all_group_by_match = False
                    break

            if not all_group_by_match:
                continue

            # Check 2 & 3: Aggregation columns and functions must match
            pre_agg_aggs = pre_agg['aggregations']
            all_match = True

            for col, agg_func in agg_cols.items():
                found_match = False
                for pre_col, pre_funcs in pre_agg_aggs.items():
                    if columns_match(col, pre_col):
                        if agg_func in pre_funcs:
                            found_match = True
                            break
                        # Special case: 'rank' can use 'sum' pre-aggregations
                        elif agg_func == 'rank' and 'sum' in pre_funcs:
                            found_match = True
                            break

                if not found_match:
                    all_match = False
                    break

            if not all_match:
                continue

            # All checks passed!
            return pre_agg

        return None

    def extract_metadata_from_code(self, code: str) -> Optional[Dict[str, Any]]:
        """Extract pre-agg metadata from code by parsing the parquet filename.

        Args:
            code: Source code to parse

        Returns:
            Pre-agg metadata dict or None if not found
        """
        # Find parquet filename in code
        pattern = r"self\.pre_agg_directory\s*/\s*['\"]([^'\"]+\.parquet)['\"]"
        match = re.search(pattern, code)

        if not match:
            return None

        parquet_filename = match.group(1)
        pre_agg_name = parquet_filename.replace('.parquet', '')

        # Find matching pre-agg metadata
        for pre_agg in self.metadata:
            if pre_agg['name'] == pre_agg_name:
                return pre_agg

        return None
