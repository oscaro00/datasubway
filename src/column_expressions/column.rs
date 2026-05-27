use regex::Regex;

pub struct TableColumn {
    table: String,
    column: String,
}

impl TableColumn {
    pub fn new(table: &str, column: &str) -> Result<Self, String> {
        Self::validate_part(table, "table")?;
        Self::validate_part(column, "column")?;
        Ok(Self {
            table: table.to_string(),
            column: column.to_string(),
        })
    }

    fn validate_part(s: &str, field: &str) -> Result<(), String> {
        if s.is_empty() {
            return Err(format!("{field} cannot be empty"));
        }
        if s != "*" && !s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
            return Err(format!(
                "{field} must be '*' or alphanumeric with underscores, got '{s}'"
            ));
        }
        Ok(())
    }

    pub fn table(&self) -> &str {
        &self.table
    }

    pub fn column(&self) -> &str {
        &self.column
    }

    pub fn table_column(&self) -> String {
        format!("{}.{}", self.table, self.column)
    }
}

pub enum TableColumnsReturn {
    Strings(Vec<String>),
    TableColumns(Vec<TableColumn>),
}

pub enum ReturnKind {
    Strings,
    TableColumns,
}

pub fn validate_table_columns(
    table_columns: Vec<&str>,
    return_type: ReturnKind,
) -> TableColumnsReturn {
    let re = Regex::new(r"^([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)$").unwrap();

    match return_type {
        ReturnKind::Strings => TableColumnsReturn::Strings(
            table_columns
                .into_iter()
                .map(|tc| {
                    let caps = re.captures(tc).unwrap();
                    TableColumn::new(caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str())
                        .unwrap()
                        .table_column()
                })
                .collect(),
        ),
        ReturnKind::TableColumns => TableColumnsReturn::TableColumns(
            table_columns
                .into_iter()
                .map(|tc| {
                    let caps = re.captures(tc).unwrap();
                    TableColumn::new(caps.get(1).unwrap().as_str(), caps.get(2).unwrap().as_str())
                        .unwrap()
                })
                .collect(),
        ),
    }
}
