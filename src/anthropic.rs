//! One concrete `LlmProvider` that talks to Anthropic's Messages API over
//! HTTP (blocking, no async yet). The "wire" structs below mirror the exact
//! JSON the API expects and returns — kept private to this module so the rest
//! of the program never sees API-specific shapes.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider, Message};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

/// Holds what the provider needs to make calls: the key and a reusable client.
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

/// The JSON body we send. `<'a>` is a lifetime: this struct borrows its data
/// (it doesn't own copies), so it can't outlive the request it was built from.
#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    max_tokens: u32,
    messages: &'a [Message],
}

/// The JSON body we receive. We only pull out the `content` array.
#[derive(Deserialize)]
struct WireResponse {
    content: Vec<ContentBlock>,
}

/// Each block is tagged with a `type`. `type` is a Rust keyword, so we map the
/// JSON field "type" onto a field named `kind` with `#[serde(rename = ...)]`.
#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    #[serde(default)]
    text: String,
}

impl LlmProvider for AnthropicProvider {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let body = WireRequest {
            model: &request.model,
            max_tokens: request.max_tokens,
            messages: &request.messages,
        };

        // `?` means: if this returns an error, stop and return it from the
        // function. `.context(...)` attaches a human-friendly explanation.
        let resp = self
            .client
            .post(API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", API_VERSION)
            .json(&body)
            .send()
            .context("failed to reach the Anthropic API")?;

        // A 4xx/5xx is still a "successful" HTTP round-trip as far as reqwest
        // is concerned, so we check the status ourselves and surface the body.
        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().unwrap_or_default();
            anyhow::bail!("Anthropic API returned {status}: {detail}");
        }

        let parsed: WireResponse = resp
            .json()
            .context("failed to parse the Anthropic response as JSON")?;

        // Join every text block into one string. `into_iter` consumes the
        // vector so we can move each `text` out without cloning.
        let text = parsed
            .content
            .into_iter()
            .filter(|block| block.kind == "text")
            .map(|block| block.text)
            .collect::<Vec<_>>()
            .join("");

        Ok(CompletionResponse { text })
    }
}
