//! The provider abstraction and the neutral data types the whole program
//! speaks. These types are provider-neutral on purpose: each provider
//! (`ollama.rs`, `anthropic.rs`) translates them to and from its own JSON wire
//! format, so the agent loop never sees an API-specific shape.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::Value;

/// A tool we advertise to the model: its name, what it does, and a JSON Schema
/// describing its arguments. `Value` is serde_json's "any JSON" type.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

/// The model's request to call one tool. `arguments` is arbitrary JSON that
/// should match the tool's schema.
#[derive(Debug, Clone)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}

/// One turn of the conversation. Which fields are set depends on the role:
/// - user turn:   role "user",      `content` set
/// - assistant:   role "assistant", `content` and/or `tool_calls` set
/// - tool result: role "tool",      `content` set, `tool_name` set
#[derive(Debug, Clone)]
pub struct Message {
    pub role: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub tool_name: Option<String>,
}

// Small constructors so callers don't fill in every field by hand. `impl Into<String>`
// lets you pass either a &str or a String.
impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: "user".into(), content: content.into(), tool_calls: Vec::new(), tool_name: None }
    }

    pub fn assistant(content: impl Into<String>, tool_calls: Vec<ToolCall>) -> Self {
        Self { role: "assistant".into(), content: content.into(), tool_calls, tool_name: None }
    }

    pub fn tool_result(tool_name: impl Into<String>, content: impl Into<String>) -> Self {
        Self {
            role: "tool".into(),
            content: content.into(),
            tool_calls: Vec::new(),
            tool_name: Some(tool_name.into()),
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompletionRequest {
    pub model: String,
    pub max_tokens: u32,
    /// Sampling temperature. Lower (near 0) = more deterministic and literal;
    /// higher = more random/creative.
    pub temperature: f32,
    /// Optional system prompt. Providers place it where their API expects it
    /// (Ollama: a system-role message; Anthropic: a top-level `system` field).
    pub system: Option<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
}

/// Token/timing metrics for one model call, when the provider reports them.
#[derive(Debug, Clone)]
pub struct Usage {
    /// Tokens the model generated (Ollama's `eval_count`).
    pub output_tokens: u32,
    /// Nanoseconds spent generating them (Ollama's `eval_duration`).
    pub eval_duration_ns: u64,
}

/// An assistant turn: free text, tool calls, or both. `tool_calls` is empty
/// when the model answered in plain language.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
    pub usage: Option<Usage>,
}

// `Send + Sync`: the agent is handed to a background task (so the TUI can keep
// redrawing while a model call runs), and everything it owns must be safe to
// move to another thread. A provider is shared behind `Box<dyn LlmProvider>`,
// so the trait object itself has to carry those guarantees.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>;

    /// Like `complete`, but calls `on_token` with each chunk of generated text
    /// as it arrives, so a frontend can show the answer appearing live.
    ///
    /// The default implementation just runs `complete` and emits the whole
    /// answer in one chunk — so a provider that can't stream still works
    /// through this method. Providers that *can* stream (Ollama) override it.
    ///
    /// `on_token` takes the chunk by value (an owned `String`): it avoids a
    /// borrow-lifetime tangle with the async desugaring, at the cost of one
    /// allocation per chunk — negligible. `+ Send` because the whole turn runs
    /// on a background task.
    async fn complete_streaming(
        &self,
        request: &CompletionRequest,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<CompletionResponse> {
        let response = self.complete(request).await?;
        if !response.text.is_empty() {
            on_token(response.text.clone());
        }
        Ok(response)
    }
}
