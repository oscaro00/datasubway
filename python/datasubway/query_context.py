"""QueryContext: validated container for query specifications."""

from __future__ import annotations

from typing import Any


class QueryContext:
    """Structured container for query specifications with validation."""

    def __init__(self, qc_dict: dict[str, Any]) -> None:
        # measures — required
        measures = qc_dict.get("measures")
        if not measures or not isinstance(measures, list):
            raise ValueError("measures must be a non-empty list of strings")
        if not all(isinstance(m, str) for m in measures):
            raise ValueError("measures must be a list of strings")
        self.measures: list[str] = measures

        # filters — default {}
        self.filters: dict = qc_dict.get("filters", {})
        if not isinstance(self.filters, dict):
            raise ValueError("filters must be a dict")

        # groups — default []
        self.groups: list[str] = qc_dict.get("groups", [])
        if not isinstance(self.groups, list):
            raise ValueError("groups must be a list")
        if not all(isinstance(g, str) for g in self.groups):
            raise ValueError("groups must be a list of strings")

        # havings — default {}
        self.havings: dict = qc_dict.get("havings", {})
        if not isinstance(self.havings, dict):
            raise ValueError("havings must be a dict")

        # sorts — default []
        self.sorts: list[tuple[str, str]] = qc_dict.get("sorts", [])
        if not isinstance(self.sorts, list):
            raise ValueError("sorts must be a list")
        for s in self.sorts:
            if not (isinstance(s, (list, tuple)) and len(s) == 2):
                raise ValueError("sorts must be a list of (column, direction) pairs")

        # limit — default 10000
        self.limit: int = qc_dict.get("limit", 10000)
        if not isinstance(self.limit, int) or self.limit <= 0:
            raise ValueError("limit must be a positive integer")

        # offset — default 0
        self.offset: int = qc_dict.get("offset", 0)
        if not isinstance(self.offset, int) or self.offset < 0:
            raise ValueError("offset must be a non-negative integer")

        # use_pre_agg — default True
        self.use_pre_agg: bool = qc_dict.get("use_pre_agg", True)
        if not isinstance(self.use_pre_agg, bool):
            raise ValueError("use_pre_agg must be a bool")
