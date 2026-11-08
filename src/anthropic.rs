//! Anthropic provider (async). Tool-calling is still deferred here (see the
//! earlier note): it sends only the text of each turn, ignores `request.tools`,
//! and returns an empty `tool_calls`, so `--provider anthropic` works for plain
//! chat. Now uses the async `reqwest::Client`.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider};

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

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
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
            usage: None,
        })
    }
}
