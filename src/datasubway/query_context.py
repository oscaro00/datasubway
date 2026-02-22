from __future__ import annotations

from dataclasses import dataclass


@dataclass
class QueryContext:
    measures: list[str]
    filters: dict = {}
    groups: list[str] = []
    havings: dict = {}
    sorts: list[tuple[str, str]] = []
    limit: int = 1000
    offset: int = 0
