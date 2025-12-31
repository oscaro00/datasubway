from typing import Dict, Self, List
import polars as pl
from query_context.filter_context import Filter

class QueryContext:
    """
    QueryContext is a json like object that is used to produce a specific query by a DataModel.
    The main functionality of the QueryContext is to validate the user input dictionary and
    convert components of the dictionary to be easier to use.

    An example:
    {
        'measure': ['total_revenue', 'average_revenue'],
        'filter': {
            'OR': [
                ('geography.country', '=', 'US'),
                {
                    'AND': [
                        ('geography.country', '=', 'CA'),
                        ('sales.revenue', '>', 1000)
                    ]
                }
            ]
        },
        'group': ['time.month'],
        # 'having': None # similar to filter,
        'sort': [('time.month', 'desc')]
    }
    """

    def __init__(self, context: Dict) -> Self:
        if context == {}:
            raise Exception('Query context cannot be empty')
        
        self.context = context

        self.validate_context()
        self.set_default_limit_offset()
    

    def validate_context(self) -> None:
        for key, val in self.context.items():
            if key not in ['measure', 'filter', 'group', 'having', 'sort', 'limit', 'offset', 'allow_pre_aggs']:
                raise KeyError(f"key: {key} not in ['measure', 'filter', 'group', 'having', 'sort', 'limit', 'offset', 'allow_pre_aggs']")
            
            match key:
                case 'measure':
                    for measure in val:
                        if not isinstance(measure, str):
                            raise TypeError('Measures must be strings')

                case 'filter':
                    # Validate filter structure (skip if None)
                    if val is not None:
                        try:
                            filter_obj = Filter(val)
                        except (TypeError, ValueError) as e:
                            raise ValueError(f"Invalid filter: {e}") from e

                case 'having':
                    # Validate having structure (skip if None)
                    if val is not None:
                        try:
                            # Allow unprefixed columns for having (aggregated column names)
                            having_obj = Filter(val, allow_unprefixed_columns=True)
                        except (TypeError, ValueError) as e:
                            raise ValueError(f"Invalid having clause: {e}") from e

                case 'sort':
                    # Validate sort structure (skip if None)
                    if val is not None:
                        if not isinstance(val, list):
                            raise TypeError(f"sort must be a list of tuples, got: {type(val).__name__}")

                        for i, sort_item in enumerate(val):
                            if not isinstance(sort_item, tuple):
                                raise TypeError(f"sort[{i}] must be a tuple, got: {type(sort_item).__name__}")

                            if len(sort_item) != 2:
                                raise ValueError(
                                    f"sort[{i}] must be a tuple of (column, direction), got length {len(sort_item)}"
                                )

                            column, direction = sort_item

                            if not isinstance(column, str):
                                raise TypeError(f"sort[{i}] column must be a string, got: {type(column).__name__}")

                            if direction not in ('asc', 'desc'):
                                raise ValueError(
                                    f"sort[{i}] direction must be 'asc' or 'desc', got: '{direction}'"
                                )

                case 'allow_pre_aggs':
                    if not isinstance(val, bool):
                        raise TypeError('allow_pre_aggs must be a boolean')

                # TODO: add other validations here (maybe make this its own method)

    def set_default_limit_offset(self) -> None:
        if 'limit' not in self.context.keys():
            self.context['limit'] = 10000

        if 'offset' not in self.context.keys():
            self.context['offset'] = 0

        if 'filter' not in self.context.keys():
            self.context['filter'] = None

        if 'having' not in self.context.keys():
            self.context['having'] = None

        if 'sort' not in self.context.keys():
            self.context['sort'] = None

        if 'group' not in self.context.keys():
            self.context['group'] = []

    def get_allow_pre_aggs(self) -> bool:
        """
        Get allow_pre_aggs flag, defaulting to True if not specified.

        Returns:
            Boolean indicating whether pre-aggregations should be used
        """
        return self.context.get('allow_pre_aggs', True)

    def get_filter_columns(self) -> List[str]:
        """
        Extract all column references from the filter.

        Returns:
            List of column names in 'table_name.column_name' format
        """
        if 'filter' not in self.context or self.context['filter'] is None:
            return []

        filter_obj = Filter(self.context['filter'])
        return filter_obj.get_columns()

    def get_having_columns(self) -> List[str]:
        """
        Extract all column references from the having clause.

        Returns:
            List of column names in 'table_name.column_name' format
        """
        if 'having' not in self.context or self.context['having'] is None:
            return []

        having_obj = Filter(self.context['having'])
        return having_obj.get_columns()