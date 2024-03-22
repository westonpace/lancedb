// --8<-- [start:import]
import * as lancedb from "@lancedb/lancedb";
// --8<-- [end:import]

(async () => {
  // --8<-- [start:ingest]
  const db = await lancedb.connect("data/sample-lancedb");

  let data = [];
  for (let i = 0; i < 10_000; i++) {
    data.push({
      vector: Array(1536).fill(i),
      id: `${i}`,
      content: "",
      longId: `${i}`,
    });
  }
  const table = await db.createTable("my_vectors", data);
  await table.createIndex("vector", {
    config: lancedb.Index.ivfPq({
      numPartitions: 16,
      numSubVectors: 48,
    }),
  });
  // --8<-- [end:ingest]

  // --8<-- [start:search1]
  const results_1 = await table
    .vectorSearch(Array(1536).fill(1.2))
    .limit(2)
    .nprobes(20)
    .refineFactor(10);
  // --8<-- [end:search1]

  // --8<-- [start:search2]
  const results_2 = await table
    .vectorSearch(Array(1536).fill(1.2))
    .where("id != '1141'")
    .limit(2);
  // --8<-- [end:search2]

  // --8<-- [start:search3]
  const results_3 = await table
    .vectorSearch(Array(1536).fill(1.2))
    .select(["id"])
    .limit(2);
  // --8<-- [end:search3]

  console.log("Ann indexes: done");
})();
