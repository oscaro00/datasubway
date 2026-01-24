"""
datasubway - A semantic layer library for data modeling with Polars.
"""

from datasubway.data_model import DataModel
from datasubway.decorators import measure
from datasubway.column_context import Allow, Exclude

__version__ = "0.1.0"
__all__ = ["DataModel", "measure", "Allow", "Exclude", "__version__"]
