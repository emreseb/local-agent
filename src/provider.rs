//! The provider abstraction and the neutral data types the whole program
//! speaks. These types are provider-neutral on purpose: each provider
//! (`ollama.rs`, `anthropic.rs`) translates them to and from its own JSON wire
//! format, so the agent loop never sees an API-specific shape.

use anyhow::Result;
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
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
}

/// An assistant turn: free text, tool calls, or both. `tool_calls` is empty
/// when the model answered in plain language.
#[derive(Debug, Clone)]
pub struct CompletionResponse {
    pub text: String,
    pub tool_calls: Vec<ToolCall>,
}

pub trait LlmProvider {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse>;
}
