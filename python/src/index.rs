// Copyright 2024 Lance Developers.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use lancedb::index::{
    DistanceType, IndexBuilder as LanceDbIndexBuilder,
    ScalarIndexBuilder as LanceDbScalarIndexBuilder,
    VectorIndexBuilder as LanceDbVectorIndexBuilder,
};
use pyo3::{exceptions::PyValueError, pyclass, pymethods, PyAny, PyRef, PyResult};
use pyo3_asyncio::tokio::future_into_py;

use crate::{error::PythonErrorExt, util::BuilderWrapper};

#[pyclass]
pub struct VectorIndexBuilder {
    inner: BuilderWrapper<LanceDbVectorIndexBuilder>,
}

impl VectorIndexBuilder {
    fn new(inner: LanceDbVectorIndexBuilder) -> Self {
        Self {
            inner: BuilderWrapper::new("VectorIndexBuilder", inner),
        }
    }
}

#[pymethods]
impl VectorIndexBuilder {
    pub fn ivf_pq(
        self_: PyRef<'_, Self>,
        distance_type: Option<String>,
        num_partitions: Option<u32>,
        num_sub_vectors: Option<u32>,
        max_iterations: Option<u32>,
        sample_rate: Option<u32>,
    ) -> PyResult<&PyAny> {
        let mut ivf_pq_builder = self_.inner.consume(|b| b.ivf_pq())?;
        if let Some(distance_type) = distance_type {
            let distance_type = match distance_type.as_str() {
                "l2" => Ok(DistanceType::L2),
                "cosine" => Ok(DistanceType::Cosine),
                "dot" => Ok(DistanceType::Dot),
                _ => Err(PyValueError::new_err(format!(
                    "Invalid distance type '{}'.  Must be one of l2, cosine, or dot",
                    distance_type
                ))),
            }?;
            ivf_pq_builder = ivf_pq_builder.distance_type(distance_type.into());
        }
        if let Some(num_partitions) = num_partitions {
            ivf_pq_builder = ivf_pq_builder.num_partitions(num_partitions);
        }
        if let Some(num_sub_vectors) = num_sub_vectors {
            ivf_pq_builder = ivf_pq_builder.num_sub_vectors(num_sub_vectors);
        }
        if let Some(max_iterations) = max_iterations {
            ivf_pq_builder = ivf_pq_builder.max_iterations(max_iterations);
        }
        if let Some(sample_rate) = sample_rate {
            ivf_pq_builder = ivf_pq_builder.sample_rate(sample_rate);
        }

        future_into_py(self_.py(), async move {
            Ok(ivf_pq_builder.execute().await.infer_error()?)
        })
    }
}

#[pyclass]
pub struct ScalarIndexBuilder {
    inner: BuilderWrapper<LanceDbScalarIndexBuilder>,
}

impl ScalarIndexBuilder {
    fn new(inner: LanceDbScalarIndexBuilder) -> Self {
        Self {
            inner: BuilderWrapper::new("ScalarIndexBuilder", inner),
        }
    }
}

#[pymethods]
impl ScalarIndexBuilder {
    pub fn btree(self_: PyRef<'_, Self>) -> PyResult<&PyAny> {
        let btree_builder = self_.inner.consume(|b| b.btree())?;
        future_into_py(self_.py(), async move {
            Ok(btree_builder.execute().await.infer_error()?)
        })
    }
}

#[pyclass]
pub struct IndexBuilder {
    inner: BuilderWrapper<LanceDbIndexBuilder>,
}

impl IndexBuilder {
    pub(crate) fn new(inner: LanceDbIndexBuilder) -> Self {
        Self {
            inner: BuilderWrapper::new("IndexBuilder", inner),
        }
    }
}

#[pymethods]
impl IndexBuilder {
    pub fn scalar(&self) -> PyResult<ScalarIndexBuilder> {
        Ok(ScalarIndexBuilder::new(self.inner.consume(|b| b.scalar())?))
    }

    pub fn vector(&self) -> PyResult<VectorIndexBuilder> {
        Ok(VectorIndexBuilder::new(self.inner.consume(|b| b.vector())?))
    }
}
