//! Anthropic provider (async), with tool-calling.
//!
//! The Messages API models a tool round-trip differently from Ollama: tool
//! calls and results are *content blocks* inside messages, correlated by id.
//! An assistant turn carries `tool_use` blocks ({id, name, input}); the tool
//! results come back in the *next user turn* as `tool_result` blocks that
//! reference those ids. So this file's main job is translating our flat,
//! neutral `Message` list into that block structure (and back).

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider, Message, ToolCall};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    api_key: String,
    client: reqwest::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::Client::new(),
        }
    }
}

#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    temperature: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<WireMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
}

/// A tool advertised to Anthropic. `input_schema` is the JSON Schema (our
/// neutral `ToolSpec::parameters`).
#[derive(Serialize)]
struct WireTool {
    name: String,
    description: String,
    input_schema: Value,
}

/// A message whose `content` is an array of blocks (text / tool_use / tool_result).
#[derive(Serialize)]
struct WireMessage {
    role: String,
    content: Value,
}

#[derive(Deserialize)]
struct WireResponse {
    content: Vec<ContentBlock>,
}

/// One block of an assistant response. Unknown block types are ignored.
#[derive(Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse { id: String, name: String, input: Value },
    #[serde(other)]
    Other,
}

/// Translate our neutral messages into Anthropic's block format. Consecutive
/// tool-result messages are coalesced into a single user turn, as the API wants
/// all results for one assistant turn's tool calls together.
fn to_wire_messages(messages: &[Message]) -> Vec<WireMessage> {
    let mut out: Vec<WireMessage> = Vec::new();

    for m in messages {
        match m.role.as_str() {
            "user" => out.push(WireMessage {
                role: "user".into(),
                content: json!([{ "type": "text", "text": m.content }]),
            }),

            "assistant" => {
                let mut blocks: Vec<Value> = Vec::new();
                if !m.content.is_empty() {
                    blocks.push(json!({ "type": "text", "text": m.content }));
                }
                for tc in &m.tool_calls {
                    blocks.push(json!({
                        "type": "tool_use",
                        "id": tc.id.clone().unwrap_or_default(),
                        "name": tc.name,
                        "input": tc.arguments,
                    }));
                }
                out.push(WireMessage {
                    role: "assistant".into(),
                    content: Value::Array(blocks),
                });
            }

            // A tool result goes in a user turn as a `tool_result` block.
            "tool" => {
                let block = json!({
                    "type": "tool_result",
                    "tool_use_id": m.tool_call_id.clone().unwrap_or_default(),
                    "content": m.content,
                });
                // Append to the previous turn if it's already a tool-result turn.
                if let Some(last) = out.last_mut()
                    && last.role == "user"
                        && let Value::Array(arr) = &mut last.content {
                            let all_results = !arr.is_empty()
                                && arr.iter().all(|b| {
                                    b.get("type").and_then(Value::as_str) == Some("tool_result")
                                });
                            if all_results {
                                arr.push(block);
                                continue;
                            }
                        }
                out.push(WireMessage {
                    role: "user".into(),
                    content: json!([block]),
                });
            }

            _ => {}
        }
    }

    out
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let body = WireRequest {
            model: &request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            system: request.system.clone(),
            messages: to_wire_messages(&request.messages),
            tools: request
                .tools
                .iter()
                .map(|t| WireTool {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    input_schema: t.parameters.clone(),
                })
                .collect(),
        };

        let resp = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .await
            .context("failed to reach the Anthropic API")?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            anyhow::bail!("Anthropic API returned {status}: {detail}");
        }

        let parsed: WireResponse = resp
            .json()
            .await
            .context("failed to parse the Anthropic response as JSON")?;

        let mut text = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        for block in parsed.content {
            match block {
                ContentBlock::Text { text: t } => text.push_str(&t),
                ContentBlock::ToolUse { id, name, input } => tool_calls.push(ToolCall {
                    name,
                    arguments: input,
                    id: Some(id),
                }),
                ContentBlock::Other => {}
            }
        }

        Ok(CompletionResponse {
            text,
            tool_calls,
            // Anthropic reports usage differently; not wired to our metrics yet.
            usage: None,
        })
    }
}
