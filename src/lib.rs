pub mod column_expressions;
pub mod data_model;
pub mod model_components;
pub mod wrappers;

/// Returns a [`LazyFrame`] for the named table, ready for polars method chaining.
/// Equivalent to `dm.table(name).build()`.
#[macro_export]
macro_rules! table {
    ($dm:expr, $name:expr) => {
        $dm.table($name).build()
    };
}
