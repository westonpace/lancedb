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

use std::sync::Arc;

use arrow_array::RecordBatchReader;

use crate::Result;

use super::TableInternal;

/// A builder to specify how exactly to update matched rows
#[derive(Default)]
pub struct WhenMatchedBuilder {
    pub(super) condition: Option<String>,
    // To come someday...
    // - update all columns (requires same schema, same
    //   behavior as today)
    // - update partial columns (allows subset schema)
    // - update expressions (similar to projection)
    // pub(super) update_type: Option<UpdateType>,
}

impl WhenMatchedBuilder {
    /// Only update rows matching the given condition
    ///
    /// Any rows that do not satisfy the condition will be left as
    /// they are.  Failing to satisfy the condition does not cause
    /// a "matched row" to become a "not matched" row.
    ///
    /// The condition should be an SQL string.  Use the prefix
    /// "target." to refer to rows in the target table (old data)
    /// and the prefix "source." to refer to rows in the source
    /// table (new data).
    ///
    /// For example, "target.last_update < source.last_update"
    pub fn only_if(&mut self, condition: impl Into<String>) -> &mut Self {
        self.condition = Some(condition.into());
        self
    }
}

/// A builder to specify how exactly to insert rows that exist
/// only in the source table
///
/// Right now there are no methods on this builder.  We expect
/// the incoming rows to have the same schema as the target table
/// and we insert them fully.
///
/// This object exists to future-proof the API for a time when
/// things like default values or dynamic values are supported.
#[derive(Default)]
pub struct WhenNotMatchedBuilder {
    // To come someday...
    // - default values (allows for input to have partial schema)
    // pub(super) default_values: Option<DefaultValues>,
}

/// An enum to specify what to do with rows that exist only in
/// the target table
///
/// Currently this is limited to deleting such rows (matching
/// an additional filter).
///
/// In the future we may support updating these rows using some
/// kind of dynamic update statement
pub enum WhenNotMatchedBySourceBuilder {
    Delete(String),
    // To come someday...
    // Update(WhenNotMatchedBySourceUpdateBuilder),
}

/// A builder used to create and run a merge insert operation
///
/// See [`super::Table::merge_insert`] for more context
///
/// Note that this class is not a traditional fluent builder.  Each of the
/// methods returns a different builder object that can used to customize
/// the behavior of the specific operation.
///
/// # Example
///
/// ```
/// # use arrow_array::RecordBatchReader;
/// # use lancedb::table::Table;
/// # use lancedb::table::merge::MergeInsertBuilder;
/// # async fn merge_insert_example(table: Table, data: Box<dyn RecordBatchReader + Send>) {
/// let mut builder = table.merge_insert(&["id"]);
/// // when_matched_update returns a WhenMatchedBuilder
/// builder.when_matched_update().only_if("target.last_update < source.last_update".to_string());
/// builder.when_not_matched_insert();
/// builder.execute(data).await.unwrap();
/// // As a result, you cannot chain methods like this:
/// // table.merge_insert(vec!["id"]).when_matched_update().when_not_matched_insert();
/// # }
/// ```

pub struct MergeInsertBuilder {
    table: Arc<dyn TableInternal>,
    pub(super) on: Vec<String>,
    pub(super) when_matched: Option<WhenMatchedBuilder>,
    pub(super) when_not_matched: Option<WhenNotMatchedBuilder>,
    pub(super) when_not_matched_by_source: Option<WhenNotMatchedBySourceBuilder>,
}

impl MergeInsertBuilder {
    pub(super) fn new(table: Arc<dyn TableInternal>, on: Vec<String>) -> Self {
        Self {
            table,
            on,
            when_matched: None,
            when_not_matched: None,
            when_not_matched_by_source: None,
        }
    }

    /// Rows that exist in both the source table (new data) and
    /// the target table (old data) will be updated, replacing
    /// the old row with the corresponding matching row.
    ///
    /// If there are multiple matches then the behavior is undefined.
    /// Currently this causes multiple copies of the row to be created
    /// but that behavior is subject to change.
    ///
    /// By default this will update all rows that match.  To customize
    /// that behavior see the methods on the returned builder.
    pub fn when_matched_update(&mut self) -> &mut WhenMatchedBuilder {
        self.when_matched = Some(WhenMatchedBuilder::default());
        self.when_matched.as_mut().unwrap()
    }

    /// Rows that exist only in the source table (new data) should
    /// be inserted into the target table.
    pub fn when_not_matched_insert(&mut self) -> &mut WhenNotMatchedBuilder {
        self.when_not_matched = Some(WhenNotMatchedBuilder::default());
        self.when_not_matched.as_mut().unwrap()
    }

    /// Rows that exist only in the target table (old data) will be
    /// deleted.  A condition must be provided to limit what data is
    /// deleted.  If you want to delete all such rows then you can
    /// use the string "true" as the condition.
    ///
    /// # Arguments
    ///
    /// * `condition` - All rows which satisfy this condition, and
    ///   do not match any row in the source table, will be deleted.
    pub fn when_not_matched_by_source_delete(&mut self, filter: impl Into<String>) {
        self.when_not_matched_by_source =
            Some(WhenNotMatchedBySourceBuilder::Delete(filter.into()));
    }

    /// Executes the merge insert operation
    ///
    /// Nothing is returned but the [`super::Table`] is updated
    pub async fn execute(self, new_data: Box<dyn RecordBatchReader + Send>) -> Result<()> {
        self.table.clone().merge_insert(self, new_data).await
    }
}
