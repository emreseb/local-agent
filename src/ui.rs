//! The frontend abstraction. The agent emits `Event`s to a `Ui` instead of
//! printing, so one agent loop can drive either the plain CLI (`CliUi`) or the
//! ratatui TUI (`TuiUi`, in tui.rs). The `Ui` also owns confirmations, since
//! "allow this?" is asked differently in a plain terminal vs a full-screen TUI.

use std::io::{self, Write};

use anyhow::Result;

/// Something the agent wants the frontend to show.
pub enum Event {
    /// The user's prompt (so a TUI can echo it into the log).
    User(String),
    /// The model's final answer.
    Assistant(String),
    /// The model asked to call a tool.
    ToolCall { name: String, args: String },
    /// A tool's output (full text; the frontend truncates for display).
    ToolResult(String),
    /// The Judge refused a call outright.
    ToolDenied(String),
}

/// A frontend: it shows events and answers confirmations.
pub trait Ui {
    fn emit(&mut self, event: Event);
    /// Ask the human to approve a tool call. Returns true only for an explicit yes.
    fn confirm(&mut self, name: &str, args: &str) -> Result<bool>;
}

/// The plain-terminal frontend: prints events, reads y/N from stdin. This
/// reproduces exactly what the agent used to print itself.
pub struct CliUi;

impl Ui for CliUi {
    fn emit(&mut self, event: Event) {
        match event {
            // The user sees their own typing in the terminal — don't echo it.
            Event::User(_) => {}
            Event::Assistant(text) => println!("{text}"),
            Event::ToolCall { name, args } => println!("\x1b[2m↳ {name}({args})\x1b[0m"),
            Event::ToolResult(result) => {
                let preview: String = result.chars().take(200).collect();
                println!("\x1b[2m  → {}\x1b[0m", preview.replace('\n', " "));
            }
            Event::ToolDenied(reason) => println!("\x1b[31m  ✗ denied: {reason}\x1b[0m"),
        }
    }

    fn confirm(&mut self, name: &str, args: &str) -> Result<bool> {
        print!("\x1b[33m⚠ allow {name}({args})? [y/N] \x1b[0m");
        io::stdout().flush()?;
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
    }
}
