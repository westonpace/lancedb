import * as vectordb from "@lancedb/lancedb";

(async () => {
  const db = await vectordb.connect("data/sample-lancedb");

  let data = [];
  for (let i = 0; i < 10_000; i++) {
    data.push({
      vector: Array(1536).fill(i),
      id: i,
      item: `item ${i}`,
      strId: `${i}`,
    });
  }
  const tbl = await db.createTable("myVectors", data);

  // --8<-- [start:search]
  let result = await tbl
    .vectorSearch(Array(1536).fill(0.5))
    .limit(1)
    .where("id = 10");
  // --8<-- [end:search]

  // --8<-- [start:vec_search]
  await tbl
    .vectorSearch(Array(1536).fill(0))
    .where("(item IN ('item 0', 'item 2')) AND (id > 10)");
  // --8<-- [end:vec_search]

  // --8<-- [start:sql_search]
  await tbl.query().where("id = 10").limit(10);
  // --8<-- [end:sql_search]

  console.log("SQL search: done");
})();
