from typing import Optional

import pyarrow as pa

class VectorIndexBuilder(object):
    async def ivf_pq(
        self,
        distance_type: Optional[str],
        num_partitions: Optional[int],
        num_sub_vectors: Optional[int],
        max_iterations: Optional[int],
        sample_rate: Optional[int],
    ) -> None: ...

class ScalarIndexBuilder(object):
    async def btree(self) -> None: ...

class IndexBuilder(object):
    def scalar(self) -> ScalarIndexBuilder: ...
    def vector(self) -> VectorIndexBuilder: ...

class Connection(object):
    async def table_names(self) -> list[str]: ...
    async def create_table(
        self, name: str, mode: str, data: pa.RecordBatchReader
    ) -> Table: ...
    async def create_empty_table(
        self, name: str, mode: str, schema: pa.Schema
    ) -> Table: ...

class Table(object):
    def name(self) -> str: ...
    def __repr__(self) -> str: ...
    async def schema(self) -> pa.Schema: ...
    async def add(self, data: pa.RecordBatchReader, mode: str) -> None: ...
    async def count_rows(self, filter: Optional[str]) -> int: ...
    def create_index(
        self, column: Optional[str], replace: Optional[bool]
    ) -> IndexBuilder: ...

async def connect(
    uri: str,
    api_key: Optional[str],
    region: Optional[str],
    host_override: Optional[str],
    read_consistency_interval: Optional[float],
) -> Connection: ...
