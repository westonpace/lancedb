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

/* global console */

"use strict";

import * as lancedb from "@lancedb/lancedb";

async function example() {
  const db = await lancedb.connect("data/sample-lancedb");

  const data = [
    { id: 1, vector: [0.1, 0.2], price: 10 },
    { id: 2, vector: [1.1, 1.2], price: 50 },
  ];

  const table = await db.createTable("vectors", data, { existOk: true });
  console.log(await db.tableNames());

  const results = await table.vectorSearch([0.1, 0.3]).limit(20);
  for await (const batch of results) {
    console.log(batch.toArray());
  }
}

example()
  .then(() => console.log("All done!"))
  .catch(console.error);
