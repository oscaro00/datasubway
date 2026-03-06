from typing import TypedDict


class GroupingContext(TypedDict):
    type: str                        # "allow" or "exclude"
    pattern: str | list[str]         # actual pattern arg (before normalization)
    include: str | list[str] | None  # actual include arg, or None if not set
