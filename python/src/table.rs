use arrow::{
    ffi_stream::ArrowArrayStreamReader,
    pyarrow::{FromPyArrow, ToPyArrow},
};
use lancedb::table::{AddDataMode, Table as LanceDbTable};
use pyo3::{
    exceptions::{PyRuntimeError, PyValueError},
    pyclass, pymethods, PyAny, PyRef, PyResult, Python,
};
use pyo3_asyncio::tokio::future_into_py;

use crate::{error::PythonErrorExt, index::IndexBuilder};

#[pyclass]
pub struct Table {
    // We keep a copy of the name to use if the inner table is dropped
    name: String,
    inner: Option<LanceDbTable>,
}

impl Table {
    pub(crate) fn new(inner: LanceDbTable) -> Self {
        Self {
            name: inner.name().to_string(),
            inner: Some(inner),
        }
    }
}

impl Table {
    fn inner_ref(&self) -> PyResult<&LanceDbTable> {
        self.inner
            .as_ref()
            .ok_or_else(|| PyRuntimeError::new_err(format!("Table {} is closed", self.name)))
    }
}

#[pymethods]
impl Table {
    pub fn name(&self) -> String {
        self.name.clone()
    }

    pub fn is_open(&self) -> bool {
        self.inner.is_some()
    }

    pub fn close(&mut self) -> () {
        self.inner.take();
    }

    pub fn schema(self_: PyRef<'_, Self>) -> PyResult<&PyAny> {
        let inner = self_.inner_ref()?.clone();
        future_into_py(self_.py(), async move {
            let schema = inner.schema().await.infer_error()?;
            Python::with_gil(|py| schema.to_pyarrow(py))
        })
    }

    pub fn add<'a>(self_: PyRef<'a, Self>, data: &PyAny, mode: String) -> PyResult<&'a PyAny> {
        let batches = Box::new(ArrowArrayStreamReader::from_pyarrow(data)?);
        let mut op = self_.inner_ref()?.add(batches);
        if mode == "append" {
            op = op.mode(AddDataMode::Append);
        } else if mode == "overwrite" {
            op = op.mode(AddDataMode::Overwrite);
        } else {
            return Err(PyValueError::new_err(format!("Invalid mode: {}", mode)));
        }

        future_into_py(self_.py(), async move {
            op.execute().await.infer_error()?;
            Ok(())
        })
    }

    pub fn count_rows(self_: PyRef<'_, Self>, filter: Option<String>) -> PyResult<&PyAny> {
        let inner = self_.inner_ref()?.clone();
        future_into_py(self_.py(), async move {
            inner.count_rows(filter).await.infer_error()
        })
    }

    pub fn create_index(
        &self,
        column: Option<&str>,
        replace: Option<bool>,
    ) -> PyResult<IndexBuilder> {
        let mut builder = self.inner_ref()?.create_index();
        if let Some(column) = column {
            builder = builder.column(column);
        }
        if let Some(replace) = replace {
            builder = builder.replace(replace);
        }
        Ok(IndexBuilder::new(builder))
    }

    pub fn __repr__(&self) -> String {
        match &self.inner {
            None => format!("ClosedTable({})", self.name),
            Some(inner) => inner.to_string(),
        }
    }
}
