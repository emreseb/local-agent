//! M0 entry point: a one-shot CLI. Parse a prompt, pick a provider, send it,
//! print the reply. No agent loop, no tools, no async yet. Everything depends
//! on the `LlmProvider` trait, so adding Ollama alongside Anthropic touched
//! only this wiring — `provider.rs` did not change at all.

mod anthropic;
mod ollama;
mod provider;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use crate::anthropic::AnthropicProvider;
use crate::ollama::OllamaProvider;
use crate::provider::{CompletionRequest, LlmProvider, Message};

/// Which backend to talk to. `ValueEnum` lets clap accept `--provider ollama`
/// or `--provider anthropic` and reject anything else.
#[derive(Clone, Copy, ValueEnum)]
enum Provider {
    Anthropic,
    Ollama,
}

/// A local AI agent — M0: send a single prompt and print the response.
#[derive(Parser)]
#[command(name = "local-agent", version, about)]
struct Cli {
    /// The prompt to send to the model.
    prompt: String,

    /// Which backend to use. Defaults to Ollama (local, free).
    #[arg(long, value_enum, default_value_t = Provider::Ollama)]
    provider: Provider,

    /// Which model to use. If omitted, a sensible default per provider is used.
    #[arg(long)]
    model: Option<String>,

    /// Maximum number of tokens in the reply (used by Anthropic).
    #[arg(long, default_value_t = 1024)]
    max_tokens: u32,
}

fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    // Pick a default model based on the provider when the user didn't name one.
    let model = cli.model.unwrap_or_else(|| match cli.provider {
        Provider::Anthropic => "claude-haiku-5-5".to_string(),
        Provider::Ollama => "llama3.1:latest".to_string(),
    });

    // Build the chosen provider. The type is `Box<dyn LlmProvider>`: a trait
    // object. Both arms produce *different* concrete types, but boxing them
    // behind the trait lets them share one variable. Calls on it are
    // dynamically dispatched (decided at runtime). The API key is only read
    // when Anthropic is actually selected, so Ollama needs no key.
    let provider: Box<dyn LlmProvider> = match cli.provider {
        Provider::Anthropic => {
            let api_key = std::env::var("ANTHROPIC_API_KEY")
                .context("ANTHROPIC_API_KEY is not set — put it in a .env file or export it")?;
            Box::new(AnthropicProvider::new(api_key))
        }
        Provider::Ollama => {
            let base_url = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            Box::new(OllamaProvider::new(base_url))
        }
    };

    let request = CompletionRequest {
        model,
        max_tokens: cli.max_tokens,
        messages: vec![Message {
            role: "user".to_string(),
            content: cli.prompt,
        }],
    };

    let response = provider.complete(&request)?;
    println!("{}", response.text);

    Ok(())
}
