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

import {
  IndexBuilder as LanceDbIndexBuilder,
  VectorIndexBuilder as LanceDbVectorIndexBuilder,
  ScalarIndexBuilder as LanceDbScalarIndexBuilder,
} from "./native";

/** Options to create `IVF_PQ` index */
export interface IvfPQOptions {
  /** The number of IVF partitions to create.
   *
   * This value should generally scale with the number of rows in the dataset.
   * By default the number of partitions is the square root of the number of
   * rows.
   *
   * If this value is too large then the first part of the search (picking the
   * right partition) will be slow.  If this value is too small then the second
   * part of the search (searching within a partition) will be slow.
   */
  num_partitions?: number;

  /** Number of sub-vectors of PQ.
   *
   * This value controls how much the vector is compressed during the quantization step.
   * The more sub vectors there are the less the vector is compressed.  The default is
   * the dimension of the vector divided by 16.  If the dimension is not evenly divisible
   * by 16 we use the dimension divded by 8.
   *
   * The above two cases are highly preferred.  Having 8 or 16 values per subvector allows
   * us to use efficient SIMD instructions.
   *
   * If the dimension is not visible by 8 then we use 1 subvector.  This is not ideal and
   * will likely result in poor performance.
   */
  num_sub_vectors?: number;

  /** [DistanceType] to use to build the index.
   *
   * Default value is [DistanceType::L2].
   *
   * This is used when training the index to calculate the IVF partitions
   * (vectors are grouped in partitions with similar vectors according to this
   * distance type) and to calculate a subvector's code during quantization.
   *
   * The distance type used to train an index MUST match the distance type used
   * to search the index.  Failure to do so will yield inaccurate results.
   *
   * The following distance types are available:
   *
   * "l2" - Euclidean distance. This is a very common distance metric that
   * accounts for both magnitude and direction when determining the distance
   * between vectors. L2 distance has a range of [0, ∞).
   *
   * "cosine" - Cosine distance.  Cosine distance is a distance metric
   * calculated from the cosine similarity between two vectors. Cosine
   * similarity is a measure of similarity between two non-zero vectors of an
   * inner product space. It is defined to equal the cosine of the angle
   * between them.  Unlike L2, the cosine distance is not affected by the
   * magnitude of the vectors.  Cosine distance has a range of [0, 2].
   *
   * Note: the cosine distance is undefined when one (or both) of the vectors
   * are all zeros (there is no direction).  These vectors are invalid and may
   * never be returned from a vector search.
   *
   * "dot" - Dot product. Dot distance is the dot product of two vectors. Dot
   * distance has a range of (-∞, ∞). If the vectors are normalized (i.e. their
   * L2 norm is 1), then dot distance is equivalent to the cosine distance.
   */
  distance_type?: "l2" | "cosine" | "dot";

  /** Max iteration to train kmeans.
   *
   * When training an IVF PQ index we use kmeans to calculate the partitions.  This parameter
   * controls how many iterations of kmeans to run.
   *
   * Increasing this might improve the quality of the index but in most cases these extra
   * iterations have diminishing returns.
   *
   * The default value is 50.
   */
  max_iterations?: number;

  /** The rate used to calculate the number of training vectors for kmeans.
   *
   * When an IVF PQ index is trained, we need to calculate partitions.  These are groups
   * of vectors that are similar to each other.  To do this we use an algorithm called kmeans.
   *
   * Running kmeans on a large dataset can be slow.  To speed this up we run kmeans on a
   * random sample of the data.  This parameter controls the size of the sample.  The total
   * number of vectors used to train the index is `sample_rate * num_partitions`.
   *
   * Increasing this value might improve the quality of the index but in most cases the
   * default should be sufficient.
   *
   * The default value is 256.
   */
  sample_rate?: number;
}

/** Builder for creating a vector index
 *
 * @see {@link IndexBuilder.vector} for more details.
 */
export class VectorIndexBuilder {
  private inner: LanceDbVectorIndexBuilder;

  constructor(inner: LanceDbVectorIndexBuilder) {
    this.inner = inner;
  }

  /** Create an IVF PQ index.
   *
   * This index stores a compressed (quantized) copy of every vector.  These vectors
   * are grouped into partitions of similar vectors.  Each partition keeps track of
   * a centroid which is the average value of all vectors in the group.
   *
   * During a query the centroids are compared with the query vector to find the closest
   * partitions.  The compressed vectors in these partitions are then searched to find
   * the closest vectors.
   *
   * The compression scheme is called product quantization.  Each vector is divided into
   * subvectors and then each subvector is quantized into a small number of bits.  the
   * parameters `num_bits` and `num_subvectors` control this process, providing a tradeoff
   * between index size (and thus search speed) and index accuracy.
   *
   * The partitioning process is called IVF and the `num_partitions` parameter controls how
   * many groups to create.
   *
   * Note that training an IVF PQ index on a large dataset is a slow operation and
   * currently is also a memory intensive operation.
   */
  async ivf_pq(options?: IvfPQOptions): Promise<void> {
    await this.inner.ivfPq(
      options?.distance_type,
      options?.num_partitions,
      options?.num_sub_vectors,
      options?.max_iterations,
      options?.sample_rate
    );
  }
}

export class ScalarIndexBuilder {
  private inner: LanceDbScalarIndexBuilder;

  constructor(inner: LanceDbScalarIndexBuilder) {
    this.inner = inner;
  }

  /** Train a btree index
   *
   * A btree index is an index on scalar columns.  The index stores a copy of the column
   * in sorted order.  A header entry is created for each block of rows (currently the
   * block size is fixed at 4096).  These header entries are stored in a separate
   * cacheable structure (a btree).  To search for data the header is used to determine
   * which blocks need to be read from disk.
   *
   * For example, a btree index in a table with 1Bi rows requires sizeof(Scalar) * 256Ki
   * bytes of memory and will generally need to read sizeof(Scalar) * 4096 bytes to find
   * the correct row ids.
   *
   * This index is good for scalar columns with mostly distinct values and does best when
   * the query is highly selective.
   *
   * The btree index does not currently have any parameters though parameters such as the
   * block size may be added in the future.
   *
   * Note that building a btree index on a large dataset may require a large amount
   * of RAM.
   */
  async btree(): Promise<void> {
    await this.inner.btree();
  }
}

export interface IndexOptions {
  /** The column to index.
   *
   * When building a scalar index this must be set.
   *
   * When building a vector index, this is optional.  The default will look
   * for any columns of type fixed-size-list with floating point values.  If
   * there is only one column of this type then it will be used.  Otherwise
   * an error will be returned.
   */
  column?: string;
  /** Whether to replace the existing index
   *
   * If this is false, and another index already exists on the same columns
   * and the same name, then an error will be returned.  This is true even if
   * that index is out of date.
   *
   * The default is true
   */
  replace?: boolean;
}

/**
 * Builder to create an index on LanceDB {@link Table}
 *
 * @see {@link Table.createIndex} for detailed usage.
 */
export class IndexBuilder {
  private inner: LanceDbIndexBuilder;

  constructor(inner: LanceDbIndexBuilder) {
    this.inner = inner;
  }

  /** Create a vector index.
   *
   * Vector indices are approximate indices that are used to find rows similar to
   * a query vector.  Vector indices speed up vector searches.
   *
   * Vector indices are only supported on fixed-size-list (tensor) columns of floating
   * point values.
   */
  vector(): VectorIndexBuilder {
    return new VectorIndexBuilder(this.inner.vector());
  }

  /** Create a scalar index.
   *
   * Scalar indices are exact indices that are used to quickly satisfy a variety of filters
   * against a column of scalar values.
   *
   * Scalar indices are currently supported on numeric, string, boolean, and temporal columns.
   *
   * A scalar index will help with queries with filters like `x > 10`, `x < 10`, `x = 10`,
   * etc.  Scalar indices can also speed up prefiltering for vector searches.  A single
   * vector search with prefiltering can use both a scalar index and a vector index.
   */
  scalar(): ScalarIndexBuilder {
    return new ScalarIndexBuilder(this.inner.scalar());
  }
}
