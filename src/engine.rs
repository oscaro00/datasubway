use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::pyarrow::ToPyArrow;
use datafusion::execution::context::SessionContext;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyList;
use tokio::runtime::Runtime;

use crate::model::joins::PyJoinGraph;
use crate::model::pre_agg::{self, PreAggregation, PyPreAggregation};
use crate::model::query_context::PyQueryContext;
use crate::optimizer::auto_join_rule::AutoJoinRule;
use crate::optimizer::pre_agg_rule::PreAggSubstitution;
use crate::post_process;

/// Core engine wrapping a DataFusion SessionContext.
/// Handles table registration, optimization, and execution.
#[pyclass(name = "Engine")]
pub struct PyEngine {
    ctx: SessionContext,
    rt: Arc<Runtime>,
    pre_aggs: Vec<PreAggregation>,
    table_schemas: HashMap<String, Vec<String>>,
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
            table_schemas: HashMap::new(),
        })
    }

    /// Register a parquet file as a named table.
    fn register_parquet(&mut self, name: &str, path: &str) -> PyResult<()> {
        self.rt
            .block_on(async {
                self.ctx
                    .register_parquet(name, path, Default::default())
                    .await
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to register parquet: {}", e)))?;

        // Store schema
        let schema_names = self
            .rt
            .block_on(async {
                let df = self.ctx.table(name).await?;
                Ok::<Vec<String>, datafusion::common::DataFusionError>(
                    df.schema()
                        .fields()
                        .iter()
                        .map(|f| format!("{}.{}", name, f.name()))
                        .collect(),
                )
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read schema: {}", e)))?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register a CSV file as a named table.
    fn register_csv(&mut self, name: &str, path: &str) -> PyResult<()> {
        self.rt
            .block_on(async { self.ctx.register_csv(name, path, Default::default()).await })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to register CSV: {}", e)))?;

        // Store schema
        let schema_names = self
            .rt
            .block_on(async {
                let df = self.ctx.table(name).await?;
                Ok::<Vec<String>, datafusion::common::DataFusionError>(
                    df.schema()
                        .fields()
                        .iter()
                        .map(|f| format!("{}.{}", name, f.name()))
                        .collect(),
                )
            })
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to read schema: {}", e)))?;
        self.table_schemas.insert(name.to_string(), schema_names);
        Ok(())
    }

    /// Register an Arrow RecordBatch as a named table.
    fn register_record_batch(&mut self, name: &str, batch: &Bound<'_, PyAny>) -> PyResult<()> {
        let batch: RecordBatch = arrow::pyarrow::FromPyArrow::from_pyarrow_bound(batch)?;
        let schema_names: Vec<String> = batch
            .schema()
            .fields()
            .iter()
            .map(|f| format!("{}.{}", name, f.name()))
            .collect();
        let schema = batch.schema();
        let mem_table = datafusion::datasource::MemTable::try_new(schema, vec![vec![batch]])
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to create MemTable: {}", e)))?;
        self.rt.block_on(async {
            self.ctx
                .register_table(name, Arc::new(mem_table))
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to register table: {}", e)))
        })?;
        self.table_schemas.insert(name.to_string(), schema_names);
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

    /// Return all qualified column names across all registered tables.
    fn all_columns(&self) -> Vec<String> {
        self.table_schemas
            .values()
            .flat_map(|cols| cols.iter().cloned())
            .collect()
    }

    /// Return the qualified column names for a specific table.
    fn table_schema(&self, name: &str) -> Vec<String> {
        self.table_schemas
            .get(name)
            .cloned()
            .unwrap_or_default()
    }

    /// Find the best pre-aggregation covering the given requirements.
    fn find_best_pre_agg(
        &self,
        group_by: Vec<String>,
        agg_components: std::collections::HashMap<String, std::collections::HashSet<String>>,
        filter_columns: Vec<String>,
    ) -> Option<PyPreAggregation> {
        pre_agg::find_best_pre_agg(&self.pre_aggs, &group_by, &agg_components, &filter_columns)
            .map(|pa| PyPreAggregation {
                inner: pa.clone(),
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

    /// Check whether the root node of a Substrait plan is an Aggregate.
    /// Used by the @measure decorator to validate measures end with .aggregate().
    fn is_aggregate_plan(&self, substrait_bytes: &[u8]) -> PyResult<bool> {
        self.rt.block_on(async {
            let plan_proto: datafusion_substrait::substrait::proto::Plan =
                prost::Message::decode(substrait_bytes).map_err(|e| {
                    PyRuntimeError::new_err(format!("Substrait decode failed: {e}"))
                })?;

            let state = self.ctx.state();
            let logical_plan =
                datafusion_substrait::logical_plan::consumer::from_substrait_plan(
                    &state,
                    &plan_proto,
                )
                .await
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Substrait consume failed: {e}"))
                })?;

            Ok(matches!(
                logical_plan,
                datafusion::logical_expr::LogicalPlan::Aggregate(_)
            ))
        })
    }

    /// Post-process measure results: join, apply havings, sorts, limit/offset.
    ///
    /// Takes a list of (measure_name, list_of_record_batches) pairs and a QueryContext.
    /// Returns the final result as a list of RecordBatches.
    fn post_process_measures<'py>(
        &self,
        measure_batches: Vec<(String, Vec<Bound<'py, PyAny>>)>,
        qc: &PyQueryContext,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Convert PyArrow RecordBatches to Rust RecordBatches
        let rust_batches: Vec<(&str, Vec<RecordBatch>)> = measure_batches
            .iter()
            .map(|(name, py_batches)| {
                let batches: Vec<RecordBatch> = py_batches
                    .iter()
                    .map(|b| arrow::pyarrow::FromPyArrow::from_pyarrow_bound(b))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok((name.as_str(), batches))
            })
            .collect::<PyResult<Vec<_>>>()?;

        let result_batches = post_process::post_process_measure_results(
            &self.rt,
            rust_batches,
            &qc.inner,
        )
        .map_err(|e| PyRuntimeError::new_err(format!("Post-process failed: {e}")))?;

        // Convert back to PyArrow
        let py_batches: Vec<Bound<'py, PyAny>> = result_batches
            .into_iter()
            .map(|b| b.to_pyarrow(py))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(PyList::new(py, &py_batches)?.into_any())
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
