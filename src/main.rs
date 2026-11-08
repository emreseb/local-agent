//! Entry point: parse args, pick a provider, register tools, run the agent
//! loop. The loop (in `agent.rs`) handles both plain chat and tool calls, so
//! there's a single path here regardless of whether the model uses a tool.

mod agent;
mod anthropic;
mod judge;
mod ollama;
mod provider;
mod tools;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use crate::anthropic::AnthropicProvider;
use crate::judge::RuleJudge;
use crate::ollama::OllamaProvider;
use crate::provider::LlmProvider;
use crate::tools::{EditFile, ListDirectory, ReadFile, RunShellCommand, ToolRegistry, WriteFile};

/// Which backend to talk to.
#[derive(Clone, Copy, ValueEnum)]
enum Provider {
    Anthropic,
    Ollama,
}

/// A local AI agent — sends a prompt and runs tools the model asks for.
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

    /// Sampling temperature: lower = more deterministic and literal (0.0–1.0).
    #[arg(long, default_value_t = 0.2)]
    temperature: f32,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    let model = cli.model.unwrap_or_else(|| match cli.provider {
        Provider::Anthropic => "claude-haiku-5-5".to_string(),
        Provider::Ollama => "qwen2.5:7b".to_string(),
    });

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

    // Register the tools the agent may use. To add a tool: write an `impl Tool`
    // in tools.rs, then add one `register(...)` line here.
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(ReadFile));
    registry.register(Box::new(WriteFile));
    registry.register(Box::new(RunShellCommand));
    registry.register(Box::new(ListDirectory));
    registry.register(Box::new(EditFile));

    // The Judge gates tool calls: allow reads, confirm writes/shell, deny destructive.
    let judge = RuleJudge;

    // `provider.as_ref()` turns the Box<dyn LlmProvider> into a &dyn LlmProvider.
    agent::run(provider.as_ref(), &model, cli.max_tokens, cli.temperature, &registry, &judge, &cli.prompt).await?;

    Ok(())
}
