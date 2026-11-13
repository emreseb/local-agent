//! Ollama provider with tool-calling support (async). Owns the translation
//! between our neutral types and Ollama's `/api/chat` JSON. Now uses the async
//! `reqwest::Client`, so the network calls `.await` instead of blocking.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::provider::{CompletionRequest, CompletionResponse, LlmProvider, Message, ToolCall, Usage};

pub struct OllamaProvider {
    base_url: String,
    client: reqwest::Client,
}

impl OllamaProvider {
    pub fn new(base_url: String) -> Self {
        Self {
            base_url,
            client: reqwest::Client::new(),
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
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<WireTool>,
}

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
    // Ollama reports these; default to 0 if a response ever omits them.
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    eval_duration: u64,
}

#[derive(Deserialize)]
struct WireResponseMessage {
    #[serde(default)]
    content: String,
    #[serde(default)]
    tool_calls: Vec<WireToolCall>,
}

/// One object in a streaming response: Ollama emits these newline-delimited,
/// one per generated chunk, with `done: true` on the last (which also carries
/// the eval counts).
#[derive(Deserialize)]
struct WireStreamChunk {
    message: WireResponseMessage,
    #[serde(default)]
    done: bool,
    #[serde(default)]
    eval_count: u32,
    #[serde(default)]
    eval_duration: u64,
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

/// Build the request body. Shared by the streaming and non-streaming paths —
/// the only difference between them is the `stream` flag.
fn build_body(request: &CompletionRequest, stream: bool) -> WireRequest {
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

    WireRequest {
        model: request.model.clone(),
        messages,
        stream,
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
    }
}

/// Turn Ollama's wire tool-calls into neutral ones.
fn to_tool_calls(wire: Vec<WireToolCall>) -> Vec<ToolCall> {
    wire.into_iter()
        .map(|c| ToolCall {
            name: c.function.name,
            arguments: c.function.arguments,
            id: None,
        })
        .collect()
}

#[async_trait]
impl LlmProvider for OllamaProvider {
    async fn complete(&self, request: &CompletionRequest) -> Result<CompletionResponse> {
        let body = build_body(request, false);
        let url = format!("{}/api/chat", self.base_url);
        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("failed to reach Ollama — is `ollama serve` running?")?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {detail}");
        }

        let parsed: WireResponse = resp
            .json()
            .await
            .context("failed to parse the Ollama response as JSON")?;

        Ok(CompletionResponse {
            text: parsed.message.content,
            tool_calls: to_tool_calls(parsed.message.tool_calls),
            usage: Some(Usage {
                output_tokens: parsed.eval_count,
                eval_duration_ns: parsed.eval_duration,
            }),
        })
    }

    async fn complete_streaming(
        &self,
        request: &CompletionRequest,
        on_token: &mut (dyn FnMut(String) + Send),
    ) -> Result<CompletionResponse> {
        let body = build_body(request, true);
        let url = format!("{}/api/chat", self.base_url);
        let mut resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .context("failed to reach Ollama — is `ollama serve` running?")?;

        let status = resp.status();
        if !status.is_success() {
            let detail = resp.text().await.unwrap_or_default();
            anyhow::bail!("Ollama returned {status}: {detail}");
        }

        // The body arrives as newline-delimited JSON, one object per chunk. The
        // network doesn't respect those boundaries, so we buffer raw bytes and
        // parse each complete line (a line is whole JSON, hence valid UTF-8).
        let mut content = String::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();
        let mut usage: Option<Usage> = None;
        let mut buf: Vec<u8> = Vec::new();

        while let Some(chunk) = resp
            .chunk()
            .await
            .context("error while streaming from Ollama")?
        {
            buf.extend_from_slice(&chunk);
            while let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                // Drain up to and including the newline; parse the line before it.
                let line: Vec<u8> = buf.drain(..=nl).collect();
                handle_chunk(
                    &line,
                    on_token,
                    &mut content,
                    &mut tool_calls,
                    &mut usage,
                )?;
            }
        }
        // Defensive: a final object not followed by a newline (Ollama normally
        // terminates every line, but don't drop data if it doesn't).
        if !buf.is_empty() {
            handle_chunk(&buf, on_token, &mut content, &mut tool_calls, &mut usage)?;
        }

        Ok(CompletionResponse {
            text: content,
            tool_calls,
            usage,
        })
    }
}

/// Parse one streamed line and fold it into the accumulators. Empty/blank lines
/// (keep-alives) are ignored.
fn handle_chunk(
    line: &[u8],
    on_token: &mut (dyn FnMut(String) + Send),
    content: &mut String,
    tool_calls: &mut Vec<ToolCall>,
    usage: &mut Option<Usage>,
) -> Result<()> {
    let text = String::from_utf8_lossy(line);
    let text = text.trim();
    if text.is_empty() {
        return Ok(());
    }

    let parsed: WireStreamChunk =
        serde_json::from_str(text).context("failed to parse an Ollama stream chunk")?;

    let delta = parsed.message.content;
    if !delta.is_empty() {
        content.push_str(&delta);
        on_token(delta);
    }
    tool_calls.extend(to_tool_calls(parsed.message.tool_calls));
    if parsed.done {
        *usage = Some(Usage {
            output_tokens: parsed.eval_count,
            eval_duration_ns: parsed.eval_duration,
        });
    }
    Ok(())
}
