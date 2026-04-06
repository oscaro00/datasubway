"""Column context resolution: allow() and exclude() functions.

Delegates to Rust implementation (src/model/column_context.rs).
Polymorphic: returns list[str] for group-by context, DataFusion Expr for filter dicts.
"""

from datasubway._engine import allow, exclude

__all__ = ["allow", "exclude"]
