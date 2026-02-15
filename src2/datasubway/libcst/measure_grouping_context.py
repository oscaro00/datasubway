"""
When a measure is registered with the @measure decorator,
make sure the measure ends with .group_by()/.group_by_dynamic()/.rolling()
followed by .agg().

Additionally, get the allow() or exclude() call from the final grouping method,
and save it to the data model object upon @measure decorator validation.
"""
