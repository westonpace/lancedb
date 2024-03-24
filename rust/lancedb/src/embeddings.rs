use std::{collections::HashMap, sync::Arc};

use arrow_array::Array;
use arrow_schema::DataType;
use async_trait::async_trait;

use crate::error::Result;

/// Trait for embedding functions
///
/// An embedding function is a function that is applied to a column of input data
/// to produce an "embedding" of that input.  This embedding is then stored in the
/// database alongside (or instead of) the original input.
///
/// An "embedding" is often a lower-dimensional representation of the input data.
/// For example, sentence-transformers can be used to embed sentences into a 768-dimensional
/// vector space.  This is useful for tasks like similarity search, where we want to find
/// similar sentences to a query sentence.
///
/// To use an embedding function you must first register it with the `EmbeddingsRegistry`.
/// Then you can define it on a column in the table schema.  That embedding will then be used
/// to embed the data in that column.
#[async_trait]
pub trait EmbeddingFunction: std::fmt::Debug + Send + Sync {
    fn source_type(&self) -> &DataType;
    fn dest_type(&self) -> &DataType;
    async fn embed(&self, source: Arc<dyn Array>) -> Result<Arc<dyn Array>>;
}

#[derive(Debug)]
pub struct EmbeddingsRegistry {
    functions: HashMap<String, Box<dyn EmbeddingFunction>>,
}

impl EmbeddingsRegistry {
    /// Create a new `EmbeddingsRegistry`
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register a new embedding function
    pub fn register<F>(&mut self, name: &str, function: F)
    where
        F: EmbeddingFunction + 'static,
    {
        self.functions.insert(name.to_string(), Box::new(function));
    }

    /// Get an embedding function by name
    pub fn get(&self, name: &str) -> Option<&Box<dyn EmbeddingFunction>> {
        self.functions.get(name)
    }
}
