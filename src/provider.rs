//! The provider abstraction: *what* an LLM backend can do, with no detail of
//! *how*. The agent loop will depend only on this module, never on reqwest or
//! any specific API. Swapping in a new backend later means adding a new
//! `impl LlmProvider` somewhere else — nothing here changes.

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// One turn in the conversation. `role` is "user" or "assistant".
///
/// `#[derive(...)]` asks the compiler to generate code for us:
/// - `Serialize` / `Deserialize` (from serde) turn this struct to/from JSON.
/// - `Clone` lets us make an owned copy; `Debug` lets us print it with `{:?}`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// Everything needed to ask a model for one completion.
#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub max_tokens: u32,
    pub messages: Vec<Message>,
}

/// What we get back. For M0 we only care about the text; later this grows
/// to carry tool-call requests, token usage, a stop reason, etc.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
}

/// The core abstraction. Any type that can turn a request into a response is
/// an `LlmProvider`. `&self` means "borrow myself, don't consume me", so one
/// provider can be reused for many requests.
pub trait LlmProvider {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>;
}
