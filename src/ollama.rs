//! A second `LlmProvider`, this time talking to a local Ollama server over
//! HTTP (default http://localhost:11434). No API key, no cost. Note how this
//! file implements the *same* `LlmProvider` trait as `anthropic.rs` — the rest
//! of the program can't tell them apart.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider, Message};

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

/// Ollama's /api/chat request body. `stream: false` is essential: without it
/// Ollama streams back many JSON lines instead of one object, and our blocking
/// `.json()` parse would choke on them.
#[derive(Serialize)]
struct WireRequest<'a> {
    model: &'a str,
    messages: &'a [Message],
    stream: bool,
}

/// The non-streamed response. The reply text lives under `message.content`.
#[derive(Deserialize)]
struct WireResponse {
    message: WireMessage,
}

#[derive(Deserialize)]
struct WireMessage {
    content: String,
}

impl LlmProvider for OllamaProvider {
    fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        // Ollama uses `options.num_predict` for a token cap rather than a
        // top-level `max_tokens`; we leave that for a later milestone, so
        // `request.max_tokens` is intentionally unused here for now.
        let body = WireRequest {
            model: &request.model,
            messages: &request.messages,
            stream: false,
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

        Ok(CompletionResponse {
            text: parsed.message.content,
        })
    }
}
