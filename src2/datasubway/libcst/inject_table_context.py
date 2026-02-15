"""
When a measure is registered with the @measure decorator,
add parameters for non_agg_context and agg_context in table() calls.
Additionally, within the allow() and exclude() calls that are added as parameters to table() calls,
need to add the parameter include_tables=True, so the joins can know which tables are necessary.
"""
