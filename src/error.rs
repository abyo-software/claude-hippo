//! claude-hippo の統一エラー型。

use thiserror::Error;

#[derive(Debug, Error)]
pub enum HippoError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    #[error("serde_json: {0}")]
    Json(#[from] serde_json::Error),

    #[error("schema: {0}")]
    Schema(String),

    #[error("embedding: {0}")]
    Embedding(String),

    #[error("config: {0}")]
    Config(String),

    #[error("not found: {0}")]
    NotFound(String),

    #[error("invalid input: {0}")]
    Invalid(String),

    #[error("integrity: {0}")]
    Integrity(String),
}

pub type Result<T, E = HippoError> = std::result::Result<T, E>;
