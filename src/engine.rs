use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::pyarrow::ToPyArrow;
use datafusion::execution::context::SessionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use tokio::runtime::Runtime;

use crate::model::joins::PyJoinGraph;
use crate::model::pre_agg::{PreAggregation, PyPreAggregation};
use crate::optimizer::auto_join_rule::AutoJoinRule;
use crate::optimizer::pre_agg_rule::PreAggSubstitution;

/// Core engine wrapping a DataFusion SessionContext.
/// Handles table registration, optimization, and execution.
#[pyclass(name = "Engine")]
pub struct PyEngine {
    ctx: SessionContext,
    rt: Arc<Runtime>,
    pre_aggs: Vec<PreAggregation>,
}

#[pymethods]
impl PyEngine {
    #[new]
    fn new() -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create tokio runtime: {}", e))
        })?;
        let ctx = SessionContext::new();
        Ok(PyEngine {
            ctx,
            rt: Arc::new(rt),
            pre_aggs: Vec::new(),
        })
    }

    /// Register a parquet file as a named table.
    fn register_parquet(&self, name: &str, path: &str) -> PyResult<()> {
        self.rt
            .block_on(async {
                self.ctx
                    .register_parquet(name, path, Default::default())
                    .await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to register parquet: {}", e)))
    }

    /// Register a CSV file as a named table.
    fn register_csv(&self, name: &str, path: &str) -> PyResult<()> {
        self.rt
            .block_on(async { self.ctx.register_csv(name, path, Default::default()).await })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to register CSV: {}", e)))
    }

    /// Register an Arrow RecordBatch as a named table.
    fn register_record_batch(&self, name: &str, batch: &Bound<'_, PyAny>) -> PyResult<()> {
        let batch: RecordBatch = arrow::pyarrow::FromPyArrow::from_pyarrow_bound(batch)?;
        let schema = batch.schema();
        let mem_table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create MemTable: {}", e)))?;
        self.rt.block_on(async {
            self.ctx
                .register_table(name, Arc::new(mem_table))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to register table: {}", e)))
        })?;
        Ok(())
    }

    /// List all registered table names.
    fn table_names(&self) -> Vec<String> {
        self.rt.block_on(async {
            self.ctx
                .catalog_names()
                .into_iter()
                .flat_map(|catalog| {
                    let catalog_provider = self.ctx.catalog(&catalog).unwrap();
                    catalog_provider
                        .schema_names()
                        .into_iter()
                        .flat_map(move |schema| {
                            let schema_provider = catalog_provider.schema(&schema).unwrap();
                            schema_provider.table_names()
                        })
                        .collect::<Vec<_>>()
                })
                .collect()
        })
    }

    /// Execute a SQL query and return results as a list of PyArrow RecordBatches.
    fn sql<'py>(&self, query: &str, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let batches = self
            .rt
            .block_on(async {
                let df = self.ctx.sql(query).await?;
                df.collect().await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("SQL execution failed: {}", e)))?;

        let py_batches: Vec<Bound<'py, PyAny>> = batches
            .into_iter()
            .map(|batch| batch.to_pyarrow(py))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(pyo3::types::PyList::new(py, &py_batches)?.into_any())
    }

    /// Register pre-aggregations with the engine.
    fn set_pre_aggs(&mut self, pre_aggs: Vec<PyPreAggregation>) {
        self.pre_aggs = pre_aggs.into_iter().map(|pa| pa.inner).collect();
    }

    /// Register the PreAggSubstitution optimizer rule with the SessionContext.
    fn add_pre_agg_optimizer_rule(&mut self) {
        let rule = PreAggSubstitution::new(self.pre_aggs.clone());
        self.ctx.add_optimizer_rule(Arc::new(rule));
    }

    /// Register the AutoJoin optimizer rule with the SessionContext.
    /// This should be called AFTER add_pre_agg_optimizer_rule so that
    /// pre-agg substitution runs first (and may eliminate the need for joins).
    fn add_auto_join_optimizer_rule(
        &mut self,
        join_graph: &PyJoinGraph,
        table_schemas: std::collections::HashMap<String, Vec<String>>,
    ) {
        let rule = AutoJoinRule::new(join_graph.inner.clone(), table_schemas);
        self.ctx.add_optimizer_rule(Arc::new(rule));
    }

    /// Accept Substrait plan bytes, deserialize into a LogicalPlan in our
    /// SessionContext (which has PreAggSubstitution registered), optimize,
    /// execute, and return Arrow RecordBatches.
    fn optimize_and_collect_substrait<'py>(
        &self,
        substrait_bytes: &[u8],
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let batches = self.rt.block_on(async {
            // Deserialize Substrait bytes → substrait Plan proto
            let plan_proto: datafusion_substrait::substrait::proto::Plan =
                prost::Message::decode(substrait_bytes).map_err(|e| {
                    PyRuntimeError::new_err(format!("Substrait decode failed: {e}"))
                })?;

            // Convert substrait Plan → DataFusion LogicalPlan
            let state = self.ctx.state();
            let logical_plan = datafusion_substrait::logical_plan::consumer::from_substrait_plan(
                &state,
                &plan_proto,
            )
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("Substrait consume failed: {e}")))?;

            // Execute through our SessionContext (which has PreAggSubstitution optimizer registered)
            let df = self
                .ctx
                .execute_logical_plan(logical_plan)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Execute failed: {e}")))?;
            df.collect()
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("Collect failed: {e}")))
        })?;

        // Convert to PyArrow
        let py_batches: Vec<_> = batches
            .into_iter()
            .map(|b| b.to_pyarrow(py))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyList::new(py, &py_batches)?.into_any())
    }
}
