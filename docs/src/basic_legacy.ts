// --8<-- [start:import]
import * as lancedb from "@lancedb/lancedb";
// --8<-- [end:import]
import {
  Schema,
  Field,
  FixedSizeList,
  Int32,
  Float16,
  Utf8,
} from "apache-arrow";
import * as fs from "fs";

const example = async () => {
  fs.rmSync("data/sample-lancedb", { recursive: true, force: true });
  // --8<-- [start:open_db]
  const uri = "data/sample-lancedb";
  const db = await lancedb.connect(uri);
  // --8<-- [end:open_db]

  // --8<-- [start:list_names]
  const names = await db.tableNames();
  // --8<-- [end:list_names]

  // Separate scope to avoid shadowing
  {
    // --8<-- [start:create_table]
    const tbl = await db.createTable("myTable", [
      { vector: [3.1, 4.1], item: "foo", price: 10.0 },
      { vector: [5.9, 26.5], item: "bar", price: 20.0 },
    ]);
    // --8<-- [end:create_table]
  }

  // --8<-- [start:open_table]
  const tbl = await db.openTable("myTable");
  // --8<-- [end:open_table]

  // --8<-- [start:add]
  const newData = Array.from({ length: 500 }, (_, i) => ({
    vector: [i, i + 1],
    item: "fizz",
    price: i * 0.1,
  }));
  await tbl.add(newData);
  // --8<-- [end:add]

  // --8<-- [start:create_index]
  await tbl.createIndex("vector");
  // --8<-- [end:create_index]

  // --8<-- [start:create_empty_table]
  const schema = new Schema([
    new Field("id", new Int32()),
    new Field("name", new Utf8()),
  ]);
  const empty_tbl = await db.createEmptyTable("empty_table", schema);
  // --8<-- [end:create_empty_table]

  // --8<-- [start:create_f16_table]
  // This is a dummy function to simulate getting an Arrow table from
  // some external source.  If your data is not already an Arrow Table then
  // you should just pass the data  directly to add / createTable / CreateEmptyTable
  // and LanceDB will convert it to an Arrow Table for you.
  function getArrowTable() {
    const dim = 16;
    const total = 10;
    const f16_schema = new Schema([
      new Field("id", new Int32()),
      new Field(
        "vector",
        new FixedSizeList(dim, new Field("item", new Float16(), true)),
        false,
      ),
    ]);
    return lancedb.makeArrowTable(
      Array.from(Array(total), (_, i) => ({
        id: i,
        vector: Array.from(Array(dim), Math.random),
      })),
      { schema: f16_schema },
    );
  }
  const data = getArrowTable();
  const table = await db.createTable("f16_tbl", data);
  // --8<-- [end:create_f16_table]

  // --8<-- [start:search]
  const query = await tbl.vectorSearch([100, 100]).limit(2);
  for await (const batch of query) {
    console.log(batch);
  }
  // --8<-- [end:search]

  // --8<-- [start:delete]
  await tbl.delete('item = "fizz"');
  // --8<-- [end:delete]

  // --8<-- [start:drop_table]
  await db.dropTable("myTable");
  // --8<-- [end:drop_table]
};

async function main() {
  await example();
  console.log("Basic example: done");
}

main();
