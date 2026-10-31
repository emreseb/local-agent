//! Anthropic provider. NOTE: tool-calling is deferred here on purpose.
//! Anthropic represents tools differently from Ollama (it uses `tool_use` and
//! `tool_result` *content blocks* and a `stop_reason: "tool_use"`), so wiring
//! it up is its own step. For now this provider sends only the text of each
//! turn, ignores `request.tools`, and always returns an empty `tool_calls` —
//! so `--provider anthropic` still works for plain chat. This is exactly the
//! seam the `LlmProvider` trait exists to hide.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    api_key: String,
    client: reqwest::blocking::Client,
}

impl AnthropicProvider {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            client: reqwest::blocking::Client::new(),
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
}

#[derive(Serialize)]
struct WireMessage {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct WireResponse {
    content: Vec<ContentBlock>,
}

#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

impl LlmProvider for AnthropicProvider {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        // Text-only translation for now: map each turn to {role, content}.
        let messages: Vec<WireMessage> = request
            .messages
            .iter()
            .map(|m| WireMessage {
                role: m.role.clone(),
                content: m.content.clone(),
            })
            .collect();

        let body = WireRequest {
            model: &request.model,
            max_tokens: request.max_tokens,
            temperature: request.temperature,
            system: request.system.clone(),
            messages,
        };

        let resp = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .context("failed to reach the Anthropic API")?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().unwrap_or_default();
            anyhow::bail!("Anthropic API returned {status}: {detail}");
        }

        let parsed: WireResponse = resp
            .json()
            .context("failed to parse the Anthropic response as JSON")?;

        let text = parsed
            .content
            .into_iter()
            .filter(|block| block.kind == "text")
            .map(|block| block.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResponse {
            text,
            tool_calls: Vec::new(),
        })
    }
}
