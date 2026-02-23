from __future__ import annotations


class QueryContext:
    def __init__(self, qc_dict: dict) -> None:
        # TODO: probably could do more work validating inputs here...

        if "measures" not in qc_dict.keys():
            raise KeyError("Query context must include measures key")
        if not isinstance(qc_dict["measures"], list):
            raise ValueError(
                "Query context measures component must be a list of strings"
            )
        self.measures: list[str] = qc_dict["measures"]

        if not isinstance(qc_dict.get("filters", {}), dict):
            raise ValueError("Query context filters component must be a dictionary")
        self.filters: dict = qc_dict.get("filters", {})

        if not isinstance(qc_dict.get("groups", []), list):
            raise ValueError("Query context groups component must be a list of strings")
        self.groups: list[str] = qc_dict.get("groups", [])

        if not isinstance(qc_dict.get("havings", {}), dict):
            raise ValueError("Query context havings component must be a dictionary")
        self.havings: dict = qc_dict.get("havings", {})

        if not isinstance(qc_dict.get("sorts", []), list):
            raise ValueError("Query context sorts component must be a list of tuples")
        self.sorts: list[tuple[str, str]] = qc_dict.get("sorts", [])

        if (
            not isinstance(qc_dict.get("limit", 10000), int)
            or qc_dict.get("limit", 10000) < 1
        ):
            raise ValueError("Query context limit component must be a positive integer")
        self.limit: int = qc_dict.get("limit", 10000)

        if (
            not isinstance(qc_dict.get("offset", 0), int)
            or qc_dict.get("offset", 0) < 0
        ):
            raise ValueError(
                "Query context offset component must be a non-negative integer"
            )
        self.offset: int = qc_dict.get("offset", 0)

        if not isinstance(qc_dict.get("use_pre_agg", True), bool):
            raise ValueError("Query context use_pre_agg component must be a boolean")
        self.use_pre_agg: bool = qc_dict.get("use_pre_agg", True)
