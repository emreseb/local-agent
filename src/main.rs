//! Entry point: parse args, build the agent, then run one prompt or an
//! interactive REPL. Pass a prompt for one-shot mode; pass none to chat.

mod agent;
mod anthropic;
mod judge;
mod ollama;
mod provider;
mod tools;
mod tui;
mod ui;

use std::io::{self, BufRead, Write};
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use tokio::sync::Mutex;

use crate::agent::Agent;
use crate::anthropic::AnthropicProvider;
use crate::judge::{AutoApprove, ConfirmAll, Judge, LocalModelJudge, RuleJudge};
use crate::ollama::OllamaProvider;
use crate::provider::LlmProvider;
use crate::tools::{
    Browse, EditFile, Git, HttpFetch, ListDirectory, ReadFile, RunShellCommand, ToolRegistry,
    WriteFile,
};
use crate::tui::{Action, ChannelUi, ToUi, TuiUi};
use crate::ui::CliUi;

/// Which backend to talk to.
#[derive(Clone, Copy, ValueEnum)]
enum Provider {
    Anthropic,
    Ollama,
}

/// Which judge decides tool permissions.
#[derive(Clone, Copy, ValueEnum)]
enum JudgeKind {
    /// Hand-written rules — instant, free, deterministic (the default).
    Rule,
    /// Ask a local LLM to classify each call — slower, experimental.
    Model,
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

    /// Launch the full-screen TUI instead of the plain CLI/REPL.
    #[arg(long)]
    tui: bool,

    /// Auto-approve every action the Judge would merely ask about (destructive
    /// commands are still denied). Convenient, less safe.
    #[arg(long, conflicts_with = "careful")]
    yolo: bool,

    /// Confirm *every* action, even reads. Maximum paranoia.
    #[arg(long)]
    careful: bool,

    /// Which judge decides tool permissions: 'rule' (default) or 'model'.
    #[arg(long, value_enum, default_value_t = JudgeKind::Rule)]
    judge: JudgeKind,

    /// Model for the `--judge model` judge (defaults to the main Ollama model).
    #[arg(long)]
    judge_model: Option<String>,
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
    registry.register(Box::new(HttpFetch::new()));
    registry.register(Box::new(Git));

    // The Judge needs the workspace root to tell in-project actions from ones
    // that reach outside it. Grab a pretty version of it for the TUI footer
    // before the path is moved into the Judge.
    let workspace = std::env::current_dir().context("could not determine the current directory")?;
    let dir = pretty_dir(&workspace);

    // Pick the base judge, then wrap it per the autonomy flags.
    let mut judge: Box<dyn Judge> = match cli.judge {
        JudgeKind::Rule => Box::new(RuleJudge::new(workspace)),
        JudgeKind::Model => {
            let base_url = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let judge_model = cli.judge_model.clone().unwrap_or_else(|| model.clone());
            Box::new(LocalModelJudge::new(
                Box::new(OllamaProvider::new(base_url)),
                judge_model,
            ))
        }
    };
    if cli.yolo {
        judge = Box::new(AutoApprove(judge));
    } else if cli.careful {
        judge = Box::new(ConfirmAll(judge));
    }

    let mut agent = Agent::new(provider, model, cli.max_tokens, cli.temperature, registry, judge);

    if cli.tui {
        run_tui(agent, dir).await?;
    } else {
        match cli.prompt {
            // A prompt on the command line -> one-shot, print stats, exit.
            Some(prompt) => {
                let mut ui = CliUi::default();
                let stats = agent.send(&prompt, &mut ui).await?;
                eprintln!("\x1b[2m{}\x1b[0m", stats.summary());
            }
            // No prompt -> interactive chat.
            None => repl(agent).await?,
        }
    }

    Ok(())
}

/// The full-screen TUI loop. Each turn runs on a background task so the UI can
/// keep animating while the model thinks; the two sides talk over a channel.
///
/// The agent lives behind `Arc<Mutex<_>>`: the background task locks it for the
/// duration of a turn, and because the state is shared (not moved into the
/// task), a Ctrl-C that aborts the task still leaves the agent intact for the
/// next prompt. The terminal is restored when `TuiUi` drops.
async fn run_tui(agent: Agent, dir: String) -> Result<()> {
    let model = agent.model().to_string();
    let agent = Arc::new(Mutex::new(agent));
    let mut tui = TuiUi::new(dir, model)?;

    loop {
        match tui.read_prompt()? {
            Action::Quit => break,
            Action::Clear => {
                agent.lock().await.clear();
                tui.clear_log();
            }
            Action::Prompt(prompt) => {
                // One channel per turn: the agent task sends events, the UI loop
                // receives them.
                let (tx, rx) = std::sync::mpsc::channel::<ToUi>();
                let agent = Arc::clone(&agent);

                let handle = tokio::spawn(async move {
                    let mut ui = ChannelUi::new(tx.clone());
                    let result = agent.lock().await.send(&prompt, &mut ui).await;
                    // Signal completion down the same channel the UI is draining.
                    let _ = tx.send(ToUi::Done(result.map_err(|e| format!("{e:#}"))));
                });

                // Drive the screen until the turn finishes (or is interrupted).
                tui.run_turn(&rx, &handle)?;
            }
        }
    }
    Ok(())
}

/// Interactive read-eval-print loop. Each line is a prompt, and the
/// conversation carries over between lines. Commands: `clear`, `exit`.
async fn repl(mut agent: Agent) -> Result<()> {
    println!("local-agent REPL — type a prompt and press Enter.");
    println!("commands: 'clear' (forget context), 'exit' (quit).\n");

    let mut ui = CliUi::default();
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

        match agent.send(input, &mut ui).await {
            Ok(stats) => eprintln!("\x1b[2m{}\x1b[0m", stats.summary()),
            Err(e) => eprintln!("\x1b[31merror: {e:#}\x1b[0m"),
        }
    }

    println!("bye!");
    Ok(())
}

/// The working directory as a string, with `$HOME` shortened to `~`.
fn pretty_dir(path: &Path) -> String {
    let full = path.display().to_string();
    if let Ok(home) = std::env::var("HOME") {
        if let Some(rest) = full.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    full
}
