<div align="center">
<p align="center">

<img width="275" alt="LanceDB Logo" src="https://user-images.githubusercontent.com/917119/226205734-6063d87a-1ecc-45fe-85be-1dea6383a3d8.png">

**Developer-friendly, serverless vector database for AI applications**

<a href='https://github.com/lancedb/vectordb-recipes/tree/main' target="_blank"><img alt='LanceDB' src='https://img.shields.io/badge/VectorDB_Recipes-100000?style=for-the-badge&logo=LanceDB&logoColor=white&labelColor=645cfb&color=645cfb'/></a>
<a href='https://lancedb.github.io/lancedb/' target="_blank"><img alt='lancdb' src='https://img.shields.io/badge/DOCS-100000?style=for-the-badge&logo=lancdb&logoColor=white&labelColor=645cfb&color=645cfb'/></a>
[![Medium](https://img.shields.io/badge/Medium-12100E?style=for-the-badge&logo=medium&logoColor=white)](https://blog.lancedb.com/)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/zMM32dvNtd)
[![Twitter](https://img.shields.io/badge/Twitter-%231DA1F2.svg?style=for-the-badge&logo=Twitter&logoColor=white)](https://twitter.com/lancedb)

</p>

<img max-width="750px" alt="LanceDB Multimodal Search" src="https://github.com/lancedb/lancedb/assets/917119/09c5afc5-7816-4687-bae4-f2ca194426ec">

</p>
</div>

<hr />

LanceDB is an open-source database for vector-search built with persistent storage, which greatly simplifies retrevial, filtering and management of embeddings.

The key features of LanceDB include:

* Production-scale vector search with no servers to manage.

* Store, query and filter vectors, metadata and multi-modal data (text, images, videos, point clouds, and more).

* Support for vector similarity search, full-text search and SQL.

* Native Python and Javascript/Typescript support.

* Zero-copy, automatic versioning, manage versions of your data without needing extra infrastructure.

* GPU support in building vector index(*).

* Ecosystem integrations with [LangChain 🦜️🔗](https://python.langchain.com/en/latest/modules/indexes/vectorstores/examples/lanecdb.html), [LlamaIndex 🦙](https://gpt-index.readthedocs.io/en/latest/examples/vector_stores/LanceDBIndexDemo.html), Apache-Arrow, Pandas, Polars, DuckDB and more on the way.

LanceDB's core is written in Rust 🦀 and is built using <a href="https://github.com/lancedb/lance">Lance</a>, an open-source columnar format designed for performant ML workloads.

## Quick Start

**Javascript**
```shell
npm install @lancedb/lancedb
```

```javascript
import * as lancedb from "@lancedb/lancedb";
const db = await lancedb.connect('data/sample-lancedb');

const table = await db.createTable({
  name: 'vectors',
  data:  [
    { id: 1, vector: [0.1, 0.2], item: "foo", price: 10 },
    { id: 2, vector: [1.1, 1.2], item: "bar", price: 50 }
  ]
})

const query = table.vectorSearch([0.1, 0.3]).limit(2);
const results = await query.toArray();

// You can also search for rows by specific criteria without involving a vector search.
const rowsByCriteria = await table.query.where("price >= 10").toArray();
```

**Python**
```shell
pip install lancedb
```

```python
import lancedb

uri = "data/sample-lancedb"
db = lancedb.connect(uri)
table = db.create_table("my_table",
                         data=[{"vector": [3.1, 4.1], "item": "foo", "price": 10.0},
                               {"vector": [5.9, 26.5], "item": "bar", "price": 20.0}])
result = table.search([100, 100]).limit(2).to_pandas()
```

**Rust**
```shell
cargo add lancedb
```

```rust
use lancedb::{connect}

#[tokio::main]
async fn main() -> Result<()> {
    let uri = "data/sample-lancedb";
    let db = connect(uri).execute().await.unwrap();
    let data: RecordBatchIterator = ...
    let table = db
        .create_table("my_table", initial_data)
        .execute()
        .await
        .unwrap();
    table.create_index(&["vector"], Index::Auto).execute().await;
    let batches = table
        .query()
        .limit(2)
        .nearest_to(&[3.1, 4.1])
        .unwrap()
        .execute()
        .await
        .unwrap()
        .try_collect::<Vec<_>>()
        .await;
    println!("{:?}", batches);
```

## Blogs, Tutorials & Videos
* 📈 <a href="https://blog.eto.ai/benchmarking-random-access-in-lance-ed690757a826">2000x better performance with Lance over Parquet</a>
* 🤖 <a href="https://github.com/lancedb/lancedb/blob/main/docs/src/notebooks/youtube_transcript_search.ipynb">Build a question and answer bot with LanceDB</a>
