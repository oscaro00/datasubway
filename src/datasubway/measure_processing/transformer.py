"""CST transformation pipeline for measure processing."""

from typing import Dict, Any, Optional, List, TYPE_CHECKING

if TYPE_CHECKING:
    from datasubway.data_model import DataModel


def apply_transformation_pipeline(
    source_code: str,
    measure_name: str,
    qc_context: Dict[str, Any],
    table_schemas: Dict[str, List[str]],
    data_model: 'DataModel',
    decorator_variable_name: Optional[str]
) -> str:
    """Apply all CST transformation steps to measure source code.

    This function applies the transformation pipeline in order:
    1. Resolve Allow/Exclude to column lists
    2. Inject parameters into table() calls
    3. Replace dm.table() calls with actual LazyFrame code
    4. Strip table prefixes from pl.col() calls
    5. Remove empty polars methods
    6. Transform pre-agg expressions (if using pre-agg)

    Args:
        source_code: Measure function source code (decorator already stripped)
        measure_name: Name of the measure function
        qc_context: Query context dictionary
        table_schemas: Dict mapping table names to column lists
        data_model: DataModel instance for table resolution
        decorator_variable_name: Custom variable name from @measure decorator

    Returns:
        Fully transformed source code ready for execution
    """
    from datasubway.cst.transformers.replace_context_with_table_columns import resolve_table_columns
    from datasubway.cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods
    from datasubway.cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions
    from datasubway.cst.transformers.replace_table_calls import replace_table_calls
    from datasubway.cst.transformers.inject_table_parameters import inject_table_parameters
    from datasubway.cst.transformers.strip_table_prefixes import strip_table_prefixes

    current_code = source_code

    # 1. Resolve Allow/Exclude to column lists (PRESERVING table prefixes)
    current_code = resolve_table_columns(
        source_code=current_code,
        function_name=measure_name,
        runtime_context={'qc': qc_context},
        output_type='polar_col'
    )

    # 2. Inject parameters into table() calls based on method chain analysis
    valid_var_names = ['dm', 'self', 'data_model']
    if decorator_variable_name is not None:
        valid_var_names.append(decorator_variable_name)

    current_code = inject_table_parameters(
        source_code=current_code,
        function_name=measure_name,
        runtime_context={
            'qc': qc_context,
            'valid_var_names': valid_var_names,
            'table_schemas': table_schemas
        }
    )

    # 3. Replace dm.table() calls with actual LazyFrame code (joins)
    replace_context = {
        'dm': data_model,
        'self': data_model,
        'data_model': data_model,
        'qc': qc_context
    }
    if decorator_variable_name is not None:
        replace_context[decorator_variable_name] = data_model

    current_code = replace_table_calls(
        source_code=current_code,
        function_name=measure_name,
        runtime_context=replace_context
    )

    # 4. Strip table prefixes from pl.col() calls for Polars execution
    current_code = strip_table_prefixes(
        source_code=current_code,
        function_name=measure_name
    )

    # 5. Remove empty polars methods
    current_code = remove_empty_polars_methods(
        source_code=current_code,
        function_name=measure_name
    )

    # 6. Transform pre-agg expressions (only if using pre-agg)
    if 'self.pre_agg_directory' in current_code:
        pre_agg_metadata = data_model._pre_agg_manager.extract_metadata_from_code(current_code)
        current_code = transform_pre_agg_expressions(
            source_code=current_code,
            function_name=measure_name,
            pre_agg_metadata=pre_agg_metadata
        )

    return current_code


def apply_transformation_pipeline_with_tracking(
    source_code: str,
    measure_name: str,
    qc_context: Dict[str, Any],
    table_schemas: Dict[str, List[str]],
    data_model: 'DataModel',
    decorator_variable_name: Optional[str]
) -> Dict[str, Optional[str]]:
    """Apply all CST transformation steps, tracking code state after each step.

    Same as apply_transformation_pipeline but returns a dictionary of all
    intermediate states for debugging purposes.

    Args:
        source_code: Measure function source code (decorator already stripped)
        measure_name: Name of the measure function
        qc_context: Query context dictionary
        table_schemas: Dict mapping table names to column lists
        data_model: DataModel instance for table resolution
        decorator_variable_name: Custom variable name from @measure decorator

    Returns:
        Dictionary mapping step names to code state after each transformation.
        Keys are numbered (e.g., '0_original', '1_resolve_table_columns').
        Values are either code strings or None for skipped steps.
    """
    from datasubway.cst.transformers.replace_context_with_table_columns import resolve_table_columns
    from datasubway.cst.transformers.remove_empty_polars_methods import remove_empty_polars_methods
    from datasubway.cst.transformers.transform_pre_agg_expressions import transform_pre_agg_expressions
    from datasubway.cst.transformers.replace_table_calls import replace_table_calls
    from datasubway.cst.transformers.inject_table_parameters import inject_table_parameters
    from datasubway.cst.transformers.strip_table_prefixes import strip_table_prefixes

    steps = {}

    # STEP 0: Original source code (after decorator strip)
    current_code = source_code
    steps['0_original'] = current_code

    # STEP 1: Resolve Allow/Exclude to column lists
    current_code = resolve_table_columns(
        source_code=current_code,
        function_name=measure_name,
        runtime_context={'qc': qc_context},
        output_type='polar_col'
    )
    steps['1_resolve_table_columns'] = current_code

    # STEP 2: Inject parameters into table() calls
    valid_var_names = ['dm', 'self', 'data_model']
    if decorator_variable_name is not None:
        valid_var_names.append(decorator_variable_name)

    current_code = inject_table_parameters(
        source_code=current_code,
        function_name=measure_name,
        runtime_context={
            'qc': qc_context,
            'valid_var_names': valid_var_names,
            'table_schemas': table_schemas
        }
    )
    steps['2_inject_table_parameters'] = current_code

    # STEP 3: Replace dm.table() calls with actual LazyFrame code
    replace_context = {
        'dm': data_model,
        'self': data_model,
        'data_model': data_model,
        'qc': qc_context
    }
    if decorator_variable_name is not None:
        replace_context[decorator_variable_name] = data_model

    current_code = replace_table_calls(
        source_code=current_code,
        function_name=measure_name,
        runtime_context=replace_context
    )
    steps['3_replace_table_calls'] = current_code

    # STEP 4: Strip table prefixes from pl.col() calls
    current_code = strip_table_prefixes(
        source_code=current_code,
        function_name=measure_name
    )
    steps['4_strip_table_prefixes'] = current_code

    # STEP 5: Remove empty polars methods
    current_code = remove_empty_polars_methods(
        source_code=current_code,
        function_name=measure_name
    )
    steps['5_remove_empty_polars_methods'] = current_code

    # STEP 6: Transform pre-agg expressions (conditional)
    if 'self.pre_agg_directory' in current_code:
        pre_agg_metadata = data_model._pre_agg_manager.extract_metadata_from_code(current_code)
        current_code = transform_pre_agg_expressions(
            source_code=current_code,
            function_name=measure_name,
            pre_agg_metadata=pre_agg_metadata
        )
        steps['6_transform_pre_agg_expressions'] = current_code
    else:
        steps['6_transform_pre_agg_expressions'] = None  # Mark as skipped

    return steps


def print_transformation_steps(
    measure_name: str,
    steps: Dict[str, Optional[str]]
) -> None:
    """Pretty-print transformation steps to console.

    Args:
        measure_name: Name of the measure being transformed
        steps: Dictionary of transformation steps
    """
    print("=" * 79)
    print(f"Transformation Pipeline for Measure: '{measure_name}'")
    print("=" * 79)
    print()

    for i, (step_name, code) in enumerate(steps.items()):
        # Extract readable step name (remove numbered prefix)
        readable_name = step_name.split('_', 1)[1] if '_' in step_name else step_name

        if code is None:
            # Skipped step
            print(f"[STEP {i}] After: {readable_name} [SKIPPED]")
            print("-" * 79)
            print()
        else:
            # Show transformed code
            print(f"[STEP {i}] After: {readable_name}")
            print("-" * 79)
            print(code)
            print()

    print("=" * 79)
    print("Transformation Complete")
    print("=" * 79)
