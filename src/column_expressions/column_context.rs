use regex::Regex;

use polars::lazy::dsl::Expr;

use super::column::{validate_table_columns, ReturnKind, TableColumn, TableColumnsReturn};

pub enum ColumnPattern {
    OnePattern(&'static str),
    MultiplePatterns(Vec<&'static str>),
}

pub enum ColumnContext {
    OneString(&'static str),
    MultipleStrings(Vec<&'static str>),
    // TODO: need to implement HashMap for parsing filter expressions
    // HashMap(HashMap<&'static str, &'static str>),
}

pub enum ColumnInclude {
    OneString(&'static str),
    MultipleStrings(Vec<&'static str>),
}

pub enum ColumnReturn {
    Strings(Vec<String>),
    PolarsExpr(Expr),
}

fn normalize_column_pattern(pattern: ColumnPattern) -> Vec<TableColumn> {
    let pattern_vec = match pattern {
        ColumnPattern::OnePattern(s) => vec![s],
        ColumnPattern::MultiplePatterns(v) => v,
    };

    let re = Regex::new(r"^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$").unwrap();

    pattern_vec
        .into_iter()
        .map(|p| {
            if p == "*" {
                TableColumn::new("*", "*").unwrap()
            } else {
                let caps = re.captures(p).unwrap();
                TableColumn::new(caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str())
                    .unwrap()
            }
        })
        .collect()
}

pub fn parse_column_pattern(pattern: &str) -> Option<TableColumn> {
    if pattern == "*" {
        return TableColumn::new("*", "*").ok();
    }
    let (table, col) = pattern.split_once('.')?;
    TableColumn::new(table, col).ok()
}

pub fn match_context_pattern(context_column: &TableColumn, patterns: &[TableColumn]) -> bool {
    for pat in patterns.iter() {
        if pat.table() == "*" {
            return true;
        } else if pat.table() == context_column.table()
            && (pat.column() == "*" || pat.column() == context_column.column())
        {
            return true;
        }
    }

    false
}

// Might need to make allow()/exclude() macros, so that a fourth argument for the current schema gets passed
// to the functions to make sure filter() can drop columns that cannot possibly apply to the base table
pub fn allow(
    pattern: ColumnPattern,
    context: ColumnContext,
    include: ColumnInclude,
) -> ColumnReturn {
    let include_return = match include {
        ColumnInclude::OneString(s) => validate_table_columns(vec![s], ReturnKind::Strings),
        ColumnInclude::MultipleStrings(v) => validate_table_columns(v, ReturnKind::Strings),
    };
    let mut allowed_columns = match include_return {
        TableColumnsReturn::Strings(v) => v,
        _ => unreachable!(),
    };

    let normalized_patterns = normalize_column_pattern(pattern);

    let context_return = match context {
        ColumnContext::OneString(s) => validate_table_columns(vec![s], ReturnKind::TableColumns),
        ColumnContext::MultipleStrings(v) => validate_table_columns(v, ReturnKind::TableColumns),
    };
    let context_vec = match context_return {
        TableColumnsReturn::TableColumns(v) => v,
        _ => unreachable!(),
    };

    let mut allowed_context: Vec<String> = context_vec
        .into_iter()
        .filter(|c| match_context_pattern(c, &normalized_patterns))
        .map(|c| c.table_column())
        .collect();

    allowed_columns.append(&mut allowed_context);
    allowed_columns.sort();
    allowed_columns.dedup();

    ColumnReturn::Strings(allowed_columns)
}

pub fn exclude(
    pattern: ColumnPattern,
    context: ColumnContext,
    include: ColumnInclude,
) -> ColumnReturn {
    todo!()
}
