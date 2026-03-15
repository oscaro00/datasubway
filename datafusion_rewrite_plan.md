# DataFusion Engine Rewrite Plan

## Context

The current Polars-based implementation works but fights against Polars' design — query plan inspection, expression rewriting via JSON serialization, and the record/resolve/replay proxy pattern are all workarounds for Polars not supporting plan manipulation. DataFusion is purpose-built for custom query optimization, making it a natural fit.

**Decisions made:**
- Rust crate + maturin/PyO3 (single repo)
- Clean rewrite on `datafusion_engine` branch (existing Polars code stays on `main`)
- Python API: same concepts, open to improvements
- Measures written in Python using `datafusion-python` DataFrame API
- `allow()`/`exclude()` resolved in Python before plan hits Rust optimizer
- Rust handles: plan optimization (pre-agg substitution, agg rewriting) + execution

---

## Architecture Overview

```
Python (user-facing)                    Rust (engine)
─────────────────────                   ──────────────
DataModel
  - data sources: files, Arrow, DB connections
    (registered as DataFusion TableProviders)
  - joins (passed to Rust as metadata)
  - pre-agg registry (passed to Rust)
  - measures (Python functions)

@measure decorator
  - validates measure returns DataFrame
  - validates plan ends with Aggregate node
  - extracts output columns

allow() / exclude()
  - resolves column context from QueryContext
  - runs before plan creation

query(QueryContext)
  1. call measure functions → DataFrame
  2. extract LogicalPlan ──────────────→ 3. optimize(plan, pre_aggs, joins)
                                            - substitute pre-agg tables
                                            - rewrite aggregations
                                            - eliminate unnecessary joins
                                         4. execute plan
                                       ← 5. return RecordBatch/Arrow
  6. convert to Python result
```

---

## Project Structure

```
datasubway/
├── Cargo.toml                  # Rust crate config (lib type: cdylib)
├── pyproject.toml              # maturin build backend
├── src/
│   ├── lib.rs                  # PyO3 module entry point
│   ├── optimizer/
│   │   ├── mod.rs
│   │   ├── pre_agg_rule.rs     # OptimizerRule: pre-agg substitution
│   │   └── agg_rewrite.rs      # sum→sum, mean→sum/count, std→sum/sumsq/count
│   ├── model/
│   │   ├── mod.rs
│   │   ├── pre_agg.rs          # PreAggregation struct, covers() logic
│   │   ├── joins.rs            # Join graph, path computation, validation
│   │   └── query_context.rs    # QueryContext struct (validated in Rust)
│   └── engine.rs               # Core engine: optimize + execute entry point
├── python/
│   └── datasubway/
│       ├── __init__.py         # DataModel, measure, allow, exclude exports
│       ├── data_model.py       # DataModel class (thin, delegates to Rust)
│       ├── measure.py          # @measure decorator
│       ├── column_context.py   # allow() / exclude()
│       └── query_context.py    # QueryContext construction
└── tests/
    ├── rust/                   # Rust unit tests (in-file #[cfg(test)])
    └── python/                 # pytest integration tests
```

---

## Implementation Phases

### Phase 1: Project Scaffolding
- Initialize maturin project with PyO3
- Set up `datafusion` and `datafusion-python` dependencies
- Create basic Rust module exposable to Python
- Verify: `import datasubway` works from Python with a trivial Rust function

### Phase 2: Data Model Foundation (Python side)
- `DataModel.__init__()`: accept data sources — file paths, Arrow tables, or database connections (DataFusion supports multiple `TableProvider` types)
- Register data sources with a DataFusion `SessionContext` as named tables
- No manual column renaming — DataFusion's logical plan tracks `table.column` references natively via qualified identifiers
- `allow()` / `exclude()` functions (pure Python, similar to current)
- `QueryContext` construction and validation
- Verify: can create a DataModel, register various data source types, call `allow()`/`exclude()`

### Phase 3: Measure System (Python side)
- `@measure` decorator:
  - Validate measure returns a DataFusion `DataFrame`
  - Validate the logical plan ends with an `Aggregate` node (measures must terminate with `.aggregate()`)
  - Extract output column names from the DataFrame's schema
- Measure functions use `datafusion-python` DataFrame API
- Extract `LogicalPlan` from the DataFrame for Rust-side optimization
- Verify: measures produce valid LogicalPlans; decorator rejects measures missing final `.aggregate()`

### Phase 4: Join Graph (Rust side)
- `JoinGraph` struct with adjacency list
- Path computation (DFS, same approach as current `joins_meta.py`)
- Cycle validation (no 3+ cycles, single path between tables)
- Expose to Python via PyO3
- Verify: Rust join graph matches current Python behavior

### Phase 5: Pre-Aggregation Registry (Rust side)
- `PreAggregation` struct: name, group_by, aggregations (expanded components), filter_columns, file_path, row_count
- `covers()` must check **all** columns referenced in the measure's logical plan, not just group-by and agg columns:
  - `group_by` columns ⊆ pre-agg group-by
  - Agg component columns available
  - **Filter columns** must exist in the pre-agg (filters need the raw column values to evaluate correctly)
  - Any other columns referenced in the plan (e.g., in expressions, case statements)
- `find_best(candidates, plan_requirements) -> Option<PreAggregation>` (minimize row_count among covering candidates)
- Pre-agg compute and write (build from raw tables, write parquet, store metadata)
- Expose to Python via PyO3
- Verify: covers() correctly rejects pre-aggs missing filter columns; selection logic picks smallest covering pre-agg

### Phase 6: Optimizer Rule (Rust side) — core of the rewrite
- Implement `OptimizerRule` trait for pre-agg substitution:
  1. Walk the `LogicalPlan` tree
  2. **Collect all column references** from the entire plan — group-by, aggregations, filters, projections, expressions, sort columns, etc.
  3. Check if a pre-agg covers all referenced columns (via `covers()` from Phase 5)
  4. If yes: replace `TableScan` with pre-agg table, remove unnecessary joins, rewrite agg expressions
  5. If no: keep original plan (raw tables + joins)
- Important: filter columns need the pre-aggregated values at the correct granularity — a filter on `region` only works if `region` is a group-by column in the pre-agg (not aggregated away)
- Agg rewriting logic (Rust `Expr` pattern matching):
  - `sum(col)` → `sum(col-sum)`
  - `mean(col)` → `sum(col-sum) / sum(col-count)`
  - `std(col)` → reconstructed from sum, sumsq, count
  - `count(col)` → `sum(col-count)`
  - `min(col)` → `min(col-min)`, `max(col)` → `max(col-max)`
- Verify: optimizer produces correct rewritten plans for single and multi-measure queries

### Phase 7: Engine Integration (Rust + Python)
- `Engine` struct in Rust: holds `SessionContext`, pre-agg registry, join graph
- `optimize_and_execute(logical_plan) -> RecordBatch` exposed to Python
- `DataModel.query(query_context)` in Python:
  1. Validate QueryContext
  2. For each measure: call function → get DataFrame → extract LogicalPlan
  3. Pass plan to Rust engine for optimization + execution
  4. Join multi-measure results on group-by columns
  5. Apply havings, sorts, limit/offset
- Verify: end-to-end query produces correct results

### Phase 8: Pre-agg Writing
- `DataModel.write_pre_aggs(names)` triggers Rust-side compute + parquet write
- Metadata stored alongside parquet (row_count, columns, etc.)
- Verify: written pre-aggs are selected and produce correct query results

---

## Key Design Details

### Pre-agg Optimizer Rule Strategy
The optimizer rule needs to handle the case where a measure involves multiple tables (fact + dimensions joined). When a pre-agg exists:
- The pre-agg already contains the joined/aggregated data
- So the rule must remove join nodes from the plan, not just swap the table scan
- This is the trickiest part — start with single-table measures, then add multi-table support

### Python ↔ Rust Boundary
- `LogicalPlan` objects from `datafusion-python` can be passed to Rust since they share the same underlying DataFusion types
- Arrow RecordBatches cross the boundary zero-copy via PyArrow
- Pre-agg metadata (group_by cols, agg components) passed as Python dicts → Rust structs via PyO3

### What Stays in Python vs. Moves to Rust
| Python | Rust |
|--------|------|
| DataModel orchestration | Join graph + validation |
| @measure decorator | Pre-agg registry + covers() |
| allow() / exclude() | OptimizerRule (pre-agg substitution) |
| QueryContext construction | Agg expression rewriting |
| Multi-measure join + post-processing | Plan execution |

---

## Verification Plan
1. **Unit tests (Rust)**: join graph, pre-agg covers(), agg rewriting, optimizer rule on synthetic plans
2. **Integration tests (Python)**: port the CPG example — same tables, joins, pre-aggs, measures, queries
3. **Correctness check**: compare DataFusion query results against current Polars implementation for the same inputs
4. **Key test cases**:
   - Single measure, no pre-agg (raw table scan)
   - Single measure, pre-agg selected (agg rewriting)
   - Multi-measure query (join on group-by columns)
   - Measure with multi-table join, pre-agg eliminates joins
   - allow/exclude with various filter/group contexts
   - mean/std measures using pre-agg components
   - Pre-agg write + subsequent query using it
   - Measure with filter on column not in pre-agg group-by → falls back to raw table
   - Measure with filter on column that IS a pre-agg group-by → pre-agg selected
   - Database connection as data source (not just parquet files)
