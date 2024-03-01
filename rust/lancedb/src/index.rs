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

use std::{cmp::max, sync::Arc};

pub use lance_linalg::distance::DistanceType;

pub mod vector;

use crate::{table::TableInternal, Result};

/// Index Parameters.
pub enum IndexParams {
    Scalar {
        replace: bool,
    },
    IvfPq {
        replace: bool,
        distance_type: DistanceType,
        num_partitions: u64,
        num_sub_vectors: u32,
        num_bits: u32,
        sample_rate: u32,
        max_iterations: u32,
    },
}

/// Builder for creating an IVF PQ index.
///
/// See [VectorIndexBuilder::ivf_pq] for more details.
pub struct IvfPqIndexBuilder {
    parent: Arc<dyn TableInternal>,
    pub(crate) common: IndexBuilder,

    pub(crate) distance_type: DistanceType,
    pub(crate) num_partitions: Option<u32>,
    pub(crate) num_sub_vectors: Option<u32>,
    pub(crate) sample_rate: u32,
    pub(crate) max_iterations: u32,
}

/// Builder for creating some kind of index.
///
/// The methods on this builder are used to specify the type of index to create and return
/// builders specific to that index type.
pub struct IndexBuilder {
    parent: Arc<dyn TableInternal>,
    pub(crate) columns: Option<Vec<String>>,
    pub(crate) replace: bool,
}

/// Builder for creating some kind of vector index.
///
/// See [IndexBuilder::vector] for more details.
pub struct VectorIndexBuilder {
    parent: Arc<dyn TableInternal>,
    pub(crate) common: IndexBuilder,
}

/// Builder for creating some kind of scalar index.
///
/// See [IndexBuilder::scalar] for more details.
pub struct ScalarIndexBuilder {
    parent: Arc<dyn TableInternal>,
    pub(crate) common: IndexBuilder,
}

/// Builder for creating a btree index.
///
/// See [ScalarIndexBuilder::btree] for more details.
pub struct BTreeIndexBuilder {
    parent: Arc<dyn TableInternal>,
    pub(crate) common: IndexBuilder,
}

impl VectorIndexBuilder {
    pub(crate) fn new(parent: Arc<dyn TableInternal>, common: IndexBuilder) -> Self {
        Self { parent, common }
    }

    /// Create an IVF PQ index.
    ///
    /// This index stores a compressed (quantized) copy of every vector.  These vectors
    /// are grouped into partitions of similar vectors.  Each partition keeps track of
    /// a centroid which is the average value of all vectors in the group.
    ///
    /// During a query the centroids are compared with the query vector to find the closest
    /// partitions.  The compressed vectors in these partitions are then searched to find
    /// the closest vectors.
    ///
    /// The compression scheme is called product quantization.  Each vector is divided into
    /// subvectors and then each subvector is quantized into a small number of bits.  the
    /// parameters `num_bits` and `num_subvectors` control this process, providing a tradeoff
    /// between index size (and thus search speed) and index accuracy.
    ///
    /// The partitioning process is called IVF and the `num_partitions` parameter controls how
    /// many groups to create.
    pub fn ivf_pq(self) -> IvfPqIndexBuilder {
        IvfPqIndexBuilder::new(self.parent, self.common)
    }
}

impl ScalarIndexBuilder {
    pub(crate) fn new(parent: Arc<dyn TableInternal>, common: IndexBuilder) -> Self {
        Self { parent, common }
    }

    /// Create a btree index
    ///
    /// A btree index is an index on scalar columns.  The index stores a copy of the column
    /// in sorted order.  A header entry is created for each block of rows (currently the
    /// block size is fixed at 4096).  These header entries are stored in a separate
    /// cacheable structure (a btree).  To search for data the header is used to determine
    /// which blocks need to be read from disk.
    ///
    /// For example, a btree index in a table with 1Bi rows requires sizeof(Scalar) * 256Ki
    /// bytes of memory and will generally need to read sizeof(Scalar) * 4096 bytes to find
    /// the correct row ids.
    ///
    /// This index is good for scalar columns with mostly distinct values and does best when
    /// the query is highly selective.
    ///
    /// The btree index does not currently have any parameters though parameters such as the
    /// block size may be added in the future.
    pub fn btree(self) -> BTreeIndexBuilder {
        BTreeIndexBuilder::new(self.parent, self.common)
    }
}

impl IvfPqIndexBuilder {
    pub(crate) fn new(parent: Arc<dyn TableInternal>, common: IndexBuilder) -> Self {
        Self {
            parent,
            common,
            distance_type: DistanceType::L2,
            num_partitions: None,
            num_sub_vectors: None,
            sample_rate: 256,
            max_iterations: 50,
        }
    }

    /// [DistanceType] to use to build the index.
    ///
    /// Default value is [DistanceType::L2].
    ///
    /// This is used when training the index to calculate the IVF partitions (vectors are
    /// grouped in partitions with similar vectors according to this distance type) and to
    /// calculate a subvector's code during quantization.
    ///
    /// The metric type used to train an index MUST match the metric type used to search the
    /// index.  Failure to do so will yield inaccurate results.
    pub fn distance_type(mut self, distance_type: DistanceType) -> Self {
        self.distance_type = distance_type;
        self
    }

    /// The number of IVF partitions to create.
    ///
    /// This value should generally scale with the number of rows in the dataset.  By default
    /// the number of partitions is the square root of the number of rows.
    ///
    /// If this value is too large then the first part of the search (picking the right partition)
    /// will be slow.  If this value is too small then the second part of the search (searching
    /// within a partition) will be slow.
    pub fn num_partitions(mut self, num_partitions: u32) -> Self {
        self.num_partitions = Some(num_partitions);
        self
    }

    /// Number of sub-vectors of PQ.
    ///
    /// This value controls how much the vector is compressed during the quantization step.
    /// The more sub vectors there are the less the vector is compressed.  The default is
    /// the dimension of the vector divided by 16.  If the dimension is not evenly divisible
    /// by 16 we use the dimension divded by 8.
    ///
    /// The above two cases are highly preferred.  Having 8 or 16 values per subvector allows
    /// us to use efficient SIMD instructions.
    ///
    /// If the dimension is not visible by 8 then we use 1 subvector.  This is not ideal and
    /// will likely result in poor performance.
    pub fn num_sub_vectors(mut self, num_sub_vectors: u32) -> Self {
        self.num_sub_vectors = Some(num_sub_vectors);
        self
    }

    /// The rate used to calculate the number of training vectors for kmeans.
    ///
    /// When an IVF PQ index is trained, we need to calculate partitions.  These are groups
    /// of vectors that are similar to each other.  To do this we use an algorithm called kmeans.
    ///
    /// Running kmeans on a large dataset can be slow.  To speed this up we run kmeans on a
    /// random sample of the data.  This parameter controls the size of the sample.  The total
    /// number of vectors used to train the index is `sample_rate * num_partitions`.
    ///
    /// Increasing this value might improve the quality of the index but in most cases the
    /// default should be sufficient.
    ///
    /// The default value is 256.
    pub fn sample_rate(mut self, sample_rate: u32) -> Self {
        self.sample_rate = sample_rate;
        self
    }

    /// Max iteration to train kmeans.
    ///
    /// When training an IVF PQ index we use kmeans to calculate the partitions.  This parameter
    /// controls how many iterations of kmeans to run.
    ///
    /// Increasing this might improve the quality of the index but in most cases these extra
    /// iterations have diminishing returns.
    ///
    /// The default value is 50.
    pub fn max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }

    /// Train the IVF PQ index
    ///
    /// Note that training an IVF PQ index on a large dataset is a slow operation and
    /// currently is also a memory intensive operation.
    pub async fn execute(self) -> Result<()> {
        self.parent.clone().create_ivf_pq_index(self).await
    }
}

impl BTreeIndexBuilder {
    pub(crate) fn new(parent: Arc<dyn TableInternal>, common: IndexBuilder) -> Self {
        Self { parent, common }
    }

    /// Train the btree index
    ///
    /// Note that building a btree index on a large dataset may require a large amount
    /// of RAM.
    pub async fn execute(self) -> Result<()> {
        self.parent.clone().create_btree_index(self).await
    }
}

impl IndexBuilder {
    pub(crate) fn new(parent: Arc<dyn TableInternal>) -> Self {
        Self {
            parent,
            columns: None,
            replace: true,
        }
    }

    /// The column to index.
    ///
    /// When building a scalar index this must be set.
    ///
    /// When building a vector index, this is optional.  The default will look
    /// for any columns of type fixed-size-list with floating point values.  If
    /// there is only one column of this type then it will be used.  Otherwise
    /// an error will be returned.
    pub fn column(mut self, column: impl Into<String>) -> Self {
        self.columns = Some(vec![column.into()]);
        self
    }

    /// Whether to replace the existing index, the default is `true`.
    ///
    /// If this is false, and another index already exists on the same columns
    /// and the same name, then an error will be returned.  This is true even if
    /// that index is out of date.
    pub fn replace(mut self, v: bool) -> Self {
        self.replace = v;
        self
    }

    /// Create a scalar index.
    ///
    /// Scalar indices are exact indices that are used to quickly satisfy a variety of filters
    /// against a column of scalar values.
    ///
    /// Scalar indices are currently supported on numeric, string, boolean, and temporal columns.
    ///
    /// A scalar index will help with queries with filters like `x > 10`, `x < 10`, `x = 10`,
    /// etc.  Scalar indices can also speed up prefiltering for vector searches.  A single
    /// vector search with prefiltering can use both a scalar index and a vector index.
    pub fn scalar(self) -> ScalarIndexBuilder {
        ScalarIndexBuilder::new(self.parent.clone(), self)
    }

    /// Create a vector index.
    ///
    /// Vector indices are approximate indices that are used to find rows similar to
    /// a query vector.  Vector indices speed up vector searches.
    ///
    /// Vector indices are only supported on fixed-size-list (tensor) columns of floating point
    /// values
    pub fn vector(self) -> VectorIndexBuilder {
        VectorIndexBuilder::new(self.parent.clone(), self)
    }
}

pub(crate) fn suggested_num_partitions(rows: usize) -> u32 {
    let num_partitions = (rows as f64).sqrt() as u32;
    max(1, num_partitions)
}

pub(crate) fn suggested_num_sub_vectors(dim: u32) -> u32 {
    if dim % 16 == 0 {
        // Should be more aggressive than this default.
        dim / 16
    } else if dim % 8 == 0 {
        dim / 8
    } else {
        log::warn!(
            "The dimension of the vector is not divisible by 8 or 16, \
                which may cause performance degradation in PQ"
        );
        1
    }
}
