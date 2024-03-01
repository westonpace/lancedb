from datetime import timedelta

import pyarrow as pa
import pytest
import pytest_asyncio
from lancedb import AsyncConnection, AsyncTable, connect_async


@pytest_asyncio.fixture
async def db_async(tmp_path) -> AsyncConnection:
    return await connect_async(tmp_path, read_consistency_interval=timedelta(seconds=0))


def sample_fixed_size_list_array(nrows, dim):
    vector_data = pa.array([float(i) for i in range(dim * nrows)], pa.float32())
    return pa.FixedSizeListArray.from_arrays(vector_data, dim)


DIM = 8
NROWS = 256


@pytest_asyncio.fixture
async def some_table(db_async):
    data = pa.Table.from_pydict(
        {
            "id": list(range(256)),
            "vector": sample_fixed_size_list_array(NROWS, DIM),
        }
    )
    return await db_async.create_table(
        "some_table",
        data,
    )


@pytest_asyncio.fixture
async def two_vec_columns_table(db_async):
    data = pa.Table.from_pydict(
        {
            "id": list(range(256)),
            "vector1": sample_fixed_size_list_array(NROWS, DIM),
            "vector2": sample_fixed_size_list_array(NROWS, DIM),
        }
    )
    return await db_async.create_table(
        "some_table",
        data,
    )


@pytest.mark.asyncio
async def test_create_scalar_index(some_table: AsyncTable):
    # Can create
    await some_table.create_index(column="id").scalar().btree()
    # Can recreate if replace=True
    await some_table.create_index(column="id", replace=True).scalar().btree()
    # Can't recreate if replace=False
    with pytest.raises(RuntimeError, match="already exists"):
        await some_table.create_index(column="id", replace=False).scalar().btree()
    # Can't create without column
    with pytest.raises(ValueError, match="column must be specified"):
        await some_table.create_index().scalar().btree()


@pytest.mark.asyncio
async def test_create_vector_index(some_table: AsyncTable):
    # Can create
    await some_table.create_index().vector().ivf_pq()
    # Can recreate if replace=True
    await some_table.create_index(replace=True).vector().ivf_pq()
    # Can't recreate if replace=False
    with pytest.raises(RuntimeError, match="already exists"):
        await some_table.create_index(replace=False).vector().ivf_pq()


@pytest.mark.asyncio
async def test_create_vector_index_two_vector_cols(
    two_vec_columns_table: AsyncTable,
):
    # Cannot create if column not specified
    with pytest.raises(ValueError, match="specify the column to index"):
        await two_vec_columns_table.create_index().vector().ivf_pq()
    # Can create if column is specified
    await two_vec_columns_table.create_index(column="vector1").vector().ivf_pq()
