from typing import Set
import inspect
import textwrap
import libcst as cst
import libcst.matchers as m
from libcst.display import dump
import polars as pl

class GetColumnContext(m.MatcherDecoratableVisitor):
    """
    Given a function name, extract all column context instances.
    Essentially, pull occurrences of Allow() and Exclude() out of polars methods.
    """

    def __init__(self, functions: Set[str]) -> None:
        super().__init__()