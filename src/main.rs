//! Entry point: parse args, build the agent, then run one prompt or an
//! interactive REPL. Pass a prompt for one-shot mode; pass none to chat.

mod agent;
mod anthropic;
mod judge;
mod ollama;
mod provider;
mod tools;

use std::io::{self, BufRead, Write};

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};

use crate::agent::{Agent, TurnStats};
use crate::anthropic::AnthropicProvider;
use crate::judge::RuleJudge;
use crate::ollama::OllamaProvider;
use crate::provider::LlmProvider;
use crate::tools::{
    Browse, EditFile, ListDirectory, ReadFile, RunShellCommand, ToolRegistry, WriteFile,
};

/// Which backend to talk to.
#[derive(Clone, Copy, ValueEnum)]
enum Provider {
    Anthropic,
    Ollama,
}

/// A local AI agent. Give it a prompt for one-shot mode, or no prompt to open
/// an interactive REPL.
#[derive(Parser)]
#[command(name = "local-agent", version, about)]
struct Cli {
    /// The prompt to send. Omit it to start an interactive chat (REPL).
    prompt: Option<String>,

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

    // Register the tools the agent may use.
    let mut registry = ToolRegistry::new();
    registry.register(Box::new(ReadFile));
    registry.register(Box::new(WriteFile));
    registry.register(Box::new(RunShellCommand));
    registry.register(Box::new(ListDirectory));
    registry.register(Box::new(EditFile));
    registry.register(Box::new(Browse));

    let mut agent = Agent::new(
        provider,
        model,
        cli.max_tokens,
        cli.temperature,
        registry,
        Box::new(RuleJudge),
    );

    match cli.prompt {
        // A prompt on the command line -> one-shot, print stats, exit.
        Some(prompt) => {
            let stats = agent.send(&prompt).await?;
            eprintln!("\x1b[2m{}\x1b[0m", format_stats(&stats));
        }
        // No prompt -> interactive chat.
        None => repl(agent).await?,
    }

    Ok(())
}

/// Interactive read-eval-print loop. Each line is a prompt, and the
/// conversation carries over between lines. Commands: `clear`, `exit`.
async fn repl(mut agent: Agent) -> Result<()> {
    println!("local-agent REPL — type a prompt and press Enter.");
    println!("commands: 'clear' (forget context), 'exit' (quit).\n");

    let stdin = io::stdin();
    loop {
        print!("\x1b[1m» \x1b[0m");
        io::stdout().flush()?;

        let mut line = String::new();
        // read_line returns 0 bytes at end-of-input (Ctrl-D) — time to leave.
        if stdin.lock().read_line(&mut line)? == 0 {
            println!();
            break;
        }

        let input = line.trim();
        match input {
            "" => continue,
            "exit" | "quit" | ":q" => break,
            "clear" | ":clear" => {
                agent.clear();
                println!("\x1b[2m(conversation cleared)\x1b[0m");
                continue;
            }
            _ => {}
        }

        match agent.send(input).await {
            Ok(stats) => eprintln!("\x1b[2m{}\x1b[0m", format_stats(&stats)),
            Err(e) => eprintln!("\x1b[31merror: {e:#}\x1b[0m"),
        }
    }

    println!("bye!");
    Ok(())
}

/// One-line, dim stats summary printed after each turn.
fn format_stats(s: &TurnStats) -> String {
    format!(
        "[{:.1}s · {} tok · {:.1} tok/s · {} call{}]",
        s.elapsed.as_secs_f64(),
        s.output_tokens,
        s.tokens_per_second(),
        s.model_calls,
        if s.model_calls == 1 { "" } else { "s" },
    )
}
