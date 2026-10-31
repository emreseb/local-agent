//! Ollama provider with tool-calling support. This file owns the *translation*
//! between our neutral types (Message, ToolSpec, ToolCall) and Ollama's exact
//! `/api/chat` JSON. Everything with a `Wire` prefix is private to this module
//! and mirrors Ollama's shapes precisely; the rest of the program never sees them.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider, Message, ToolCall};

pub struct OllamaProvider {
    base_url: String,
    client: reqwest::blocking::Client,
}

impl OllamaProvider {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::blocking::Client::new(),
        }
    }
}

// ---- request wire types ----

#[derive(Serialize)]
struct WireRequest {
    model: String,
    messages: Vec<WireMessage>,
    stream: bool,
    options: WireOptions,
    // `skip_serializing_if` omits the field entirely when empty, keeping plain
    // chat requests clean (no empty "tools": []).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
}

/// Ollama runtime options go in their own object. Temperature is the one we
/// care about now; many more (top_p, num_predict, seed, ...) live here too.
#[derive(Serialize)]
struct WireOptions {
    temperature: f32,
}

#[derive(Serialize)]
struct WireMessage {
    role: String,
    content: String,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tool_calls: Vec<WireToolCall>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_name: Option<String>,
}

#[derive(Serialize)]
struct WireTool {
    // JSON needs a field literally called "type", which is a Rust keyword, so
    // we store it as `kind` and rename on the wire.
    #[serde(rename = "type")]
    kind: String,
    function: WireFunction,
}

#[derive(Serialize)]
struct WireFunction {
    name: String,
    description: String,
    parameters: Value,
}

// Tool calls appear in BOTH directions — in the history we send and in the
// response we receive — so this type is both Serialize and Deserialize.
#[derive(Serialize, Deserialize)]
struct WireToolCall {
    function: WireToolCallFn,
}

#[derive(Serialize, Deserialize)]
struct WireToolCallFn {
    name: String,
    arguments: Value,
}

// ---- response wire types ----

#[derive(Deserialize)]
struct WireResponse {
    message: WireResponseMessage,
}

#[derive(Deserialize)]
struct WireResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

/// Neutral `Message` -> Ollama wire message.
fn to_wire_message(m: &Message) -> WireMessage {
    WireMessage {
        role: m.role.clone(),
        content: m.content.clone(),
        tool_calls: m
            .tool_calls
            .iter()
            .map(|c| WireToolCall {
                function: WireToolCallFn {
                    name: c.name.clone(),
                    arguments: c.arguments.clone(),
                },
            })
            .collect(),
        tool_name: m.tool_name.clone(),
    }
}

impl LlmProvider for OllamaProvider {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        // Ollama takes the system prompt as the first system-role message.
        let mut messages: Vec<WireMessage> = Vec::new();
        if let Some(system) = &request.system {
            messages.push(WireMessage {
                role: "system".to_string(),
                content: system.clone(),
                tool_calls: Vec::new(),
                tool_name: None,
            });
        }
        messages.extend(request.messages.iter().map(to_wire_message));

        let body = WireRequest {
            model: request.model.clone(),
            messages,
            stream: false,
            options: WireOptions {
                temperature: request.temperature,
            },
            tools: request
                .tools
                .iter()
                .map(|t| WireTool {
                    kind: "function".to_string(),
                    function: WireFunction {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        parameters: t.parameters.clone(),
                    },
                })
                .collect(),
        };

        let url = format!("{}/api/chat", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .context("failed to reach Ollama — is `ollama serve` running?")?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {detail}");
        }

        let parsed: WireResponse = resp
            .json()
            .context("failed to parse the Ollama response as JSON")?;

        // Wire -> neutral: flatten Ollama's nested `function` back into our
        // flat ToolCall.
        let tool_calls = parsed
            .message
            .tool_calls
            .into_iter()
            .map(|c| ToolCall {
                name: c.function.name,
                arguments: c.function.arguments,
            })
            .collect();

        Ok(CompletionResponse {
            text: parsed.message.content,
            tool_calls,
        })
    }
}
