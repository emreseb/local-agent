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
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Parser, ValueEnum};
use serde::Deserialize;
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
#[derive(Clone, Copy, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Provider {
    Anthropic,
    Ollama,
}

/// Which judge decides tool permissions.
#[derive(Clone, Copy, ValueEnum, Deserialize)]
#[serde(rename_all = "lowercase")]
enum JudgeKind {
    /// Hand-written rules — instant, free, deterministic (the default).
    Rule,
    /// Ask a local LLM to classify each call — slower, experimental.
    Model,
}

/// Defaults loaded from a config file (TOML). Every field is optional; a CLI
/// flag always wins over the file, and the file wins over the built-in default.
#[derive(Deserialize, Default)]
struct FileConfig {
    provider: Option<Provider>,
    model: Option<String>,
    max_tokens: Option<u32>,
    temperature: Option<f32>,
    judge: Option<JudgeKind>,
    judge_model: Option<String>,
    system: Option<String>,
}

/// A local AI agent. Give it a prompt for one-shot mode, or no prompt to open
/// an interactive REPL.
#[derive(Parser)]
#[command(name = "local-agent", version, about)]
struct Cli {
    /// The prompt to send. Omit it to start an interactive chat (REPL).
    prompt: Option<String>,

    /// Which backend to use. Defaults to Ollama (local, free).
    #[arg(long, value_enum)]
    provider: Option<Provider>,

    /// Which model to use. If omitted, a sensible default per provider is used.
    #[arg(long)]
    model: Option<String>,

    /// Maximum number of tokens in the reply (used by Anthropic).
    #[arg(long)]
    max_tokens: Option<u32>,

    /// Sampling temperature: lower = more deterministic and literal (0.0–1.0).
    #[arg(long)]
    temperature: Option<f32>,

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
    #[arg(long, value_enum)]
    judge: Option<JudgeKind>,

    /// Model for the `--judge model` judge (defaults to the main Ollama model).
    #[arg(long)]
    judge_model: Option<String>,

    /// Save the conversation to this file after every turn, and resume from it
    /// on startup if it already exists.
    #[arg(long)]
    session: Option<PathBuf>,

    /// Override the system prompt entirely (replaces the built-in grounding).
    #[arg(long)]
    system: Option<String>,

    /// Path to a TOML config file. Defaults to
    /// $XDG_CONFIG_HOME/local-agent/config.toml (or ~/.config/...).
    #[arg(long)]
    config: Option<PathBuf>,

    /// List the available tools and exit.
    #[arg(long)]
    list_tools: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    dotenvy::dotenv().ok();
    let cli = Cli::parse();

    // Load the config file, then resolve each setting: CLI flag wins over the
    // file, the file wins over the built-in default.
    let config = load_config(cli.config.as_deref())?;
    let provider_kind = cli.provider.or(config.provider).unwrap_or(Provider::Ollama);
    let max_tokens = cli.max_tokens.or(config.max_tokens).unwrap_or(1024);
    let temperature = cli.temperature.or(config.temperature).unwrap_or(0.2);
    let judge_kind = cli.judge.or(config.judge).unwrap_or(JudgeKind::Rule);
    let judge_model = cli.judge_model.clone().or(config.judge_model.clone());
    let system_override = cli.system.clone().or(config.system.clone());

    let model = cli.model.clone().or(config.model.clone()).unwrap_or_else(|| match provider_kind {
        Provider::Anthropic => "claude-haiku-5-5".to_string(),
        Provider::Ollama => "qwen2.5:7b".to_string(),
    });

    let provider: Box<dyn LlmProvider> = match provider_kind {
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

    // `--list-tools`: print the registered tools and exit.
    if cli.list_tools {
        for spec in registry.specs() {
            println!("{:<18} {}", spec.name, spec.description);
        }
        return Ok(());
    }

    // The Judge needs the workspace root to tell in-project actions from ones
    // that reach outside it. Grab a pretty version of it for the TUI footer
    // before the path is moved into the Judge.
    let workspace = std::env::current_dir().context("could not determine the current directory")?;
    let dir = pretty_dir(&workspace);

    // Pick the base judge, then wrap it per the autonomy flags.
    let mut judge: Box<dyn Judge> = match judge_kind {
        JudgeKind::Rule => Box::new(RuleJudge::new(workspace)),
        JudgeKind::Model => {
            let base_url = std::env::var("OLLAMA_HOST")
                .unwrap_or_else(|_| "http://localhost:11434".to_string());
            let jmodel = judge_model.unwrap_or_else(|| model.clone());
            Box::new(LocalModelJudge::new(
                Box::new(OllamaProvider::new(base_url)),
                jmodel,
            ))
        }
    };
    if cli.yolo {
        judge = Box::new(AutoApprove(judge));
    } else if cli.careful {
        judge = Box::new(ConfirmAll(judge));
    }

    let mut agent = Agent::new(provider, model, max_tokens, temperature, registry, judge);

    // A custom system prompt, if one was given.
    if let Some(system) = system_override {
        agent.set_system(system);
    }

    // Wire up the session file (if any) and resume a prior conversation.
    agent.set_session(cli.session.clone());
    agent.load_session()?;

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
    let mut tui = TuiUi::new(dir, model, history_path())?;

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
    println!("commands: 'clear' (forget context), 'tools' (list tools), 'exit' (quit).\n");

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
            "tools" | ":tools" => {
                for (name, desc) in agent.tools() {
                    println!("\x1b[2m{name:<18} {desc}\x1b[0m");
                }
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

/// Load the config file: the explicit `--config` path if given, else the
/// default location. A missing file yields empty defaults (not an error).
fn load_config(explicit: Option<&Path>) -> Result<FileConfig> {
    let path = match explicit {
        Some(p) => Some(p.to_path_buf()),
        None => default_config_path(),
    };
    if let Some(path) = path
        && path.exists() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("could not read config '{}'", path.display()))?;
            return toml::from_str(&text)
                .with_context(|| format!("could not parse config '{}'", path.display()));
        }
    Ok(FileConfig::default())
}

/// `$XDG_CONFIG_HOME/local-agent/config.toml`, falling back to
/// `~/.config/local-agent/config.toml`.
fn default_config_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".config")))?;
    Some(base.join("local-agent").join("config.toml"))
}

/// Where the TUI stores its prompt history between runs:
/// `$XDG_DATA_HOME/local-agent/history` (or `~/.local/share/...`).
fn history_path() -> Option<PathBuf> {
    let base = std::env::var_os("XDG_DATA_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local").join("share")))?;
    Some(base.join("local-agent").join("history"))
}

/// The working directory as a string, with `$HOME` shortened to `~`.
fn pretty_dir(path: &Path) -> String {
    let full = path.display().to_string();
    if let Ok(home) = std::env::var("HOME")
        && let Some(rest) = full.strip_prefix(&home) {
            return format!("~{rest}");
        }
    full
}
