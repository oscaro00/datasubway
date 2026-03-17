pub mod engine;
pub mod model;
pub mod optimizer;

use pyo3::prelude::*;

use crate::engine::PyEngine;
use crate::model::joins::PyJoinGraph;
use crate::model::pre_agg::PyPreAggregation;

#[pymodule]
fn _engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyJoinGraph>()?;
    m.add_class::<PyPreAggregation>()?;
    Ok(())
}
