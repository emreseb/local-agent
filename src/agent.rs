//! The agent: owns the provider, tools, judge, config, and the running
//! conversation. `send` runs one user turn (the tool loop) and reports progress
//! through a `Ui` (which prints, or draws the TUI) instead of printing itself.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};

use crate::judge::{Judge, Verdict};
use crate::provider::{CompletionRequest, LlmProvider, Message};
use crate::tools::ToolRegistry;
use crate::ui::{Event, Ui};

/// Safety valve: never loop forever if a model keeps calling tools.
const MAX_STEPS: usize = 10;

/// Metrics for one `send` — one user turn, which may span several model calls.
pub struct TurnStats {
    pub elapsed: Duration,
    pub output_tokens: u32,
    pub eval_duration_ns: u64,
    pub model_calls: usize,
}

impl TurnStats {
    /// Generation speed from the model's own eval time, not wall clock.
    pub fn tokens_per_second(&self) -> f64 {
        if self.eval_duration_ns == 0 {
            0.0
        } else {
            self.output_tokens as f64 / (self.eval_duration_ns as f64 / 1e9)
        }
    }

    /// One-line summary for a status bar or footer. Shared by every frontend so
    /// the CLI and the TUI report a turn the same way.
    pub fn summary(&self) -> String {
        format!(
            "[{:.1}s · {} tok · {:.1} tok/s · {} call{}]",
            self.elapsed.as_secs_f64(),
            self.output_tokens,
            self.tokens_per_second(),
            self.model_calls,
            if self.model_calls == 1 { "" } else { "s" },
        )
    }
}

pub struct Agent {
    provider: Box<dyn LlmProvider>,
    model: String,
    max_tokens: u32,
    temperature: f32,
    registry: ToolRegistry,
    judge: Box<dyn Judge>,
    system: String,
    messages: Vec<Message>,
    /// If set, the conversation is saved here after every turn and can be
    /// reloaded with `load_session` to resume later.
    session: Option<PathBuf>,
}

impl Agent {
    pub fn new(
        provider: Box<dyn LlmProvider>,
        model: String,
        max_tokens: u32,
        temperature: f32,
        registry: ToolRegistry,
        judge: Box<dyn Judge>,
    ) -> Self {
        let system = build_system_prompt(&registry);
        Self {
            provider,
            model,
            max_tokens,
            temperature,
            registry,
            judge,
            system,
            messages: Vec::new(),
            session: None,
        }
    }

    /// Forget the conversation so far (the REPL `clear` command).
    pub fn clear(&mut self) {
        self.messages.clear();
        self.save_session();
    }

    /// The model name, for display in a status bar / footer.
    pub fn model(&self) -> &str {
        &self.model
    }

    /// Replace the built-in system prompt (the `--system` flag / config key).
    pub fn set_system(&mut self, system: String) {
        self.system = system;
    }

    /// Point the agent at a session file. With one set, every turn is saved
    /// there; `load_session` reads it back.
    pub fn set_session(&mut self, path: Option<PathBuf>) {
        self.session = path;
    }

    /// Load a saved conversation from the session file, if one is set and
    /// exists. A missing file is not an error — it's just a fresh session.
    pub fn load_session(&mut self) -> Result<()> {
        if let Some(path) = &self.session {
            if path.exists() {
                let data = std::fs::read_to_string(path)
                    .with_context(|| format!("could not read session file '{}'", path.display()))?;
                self.messages = serde_json::from_str(&data)
                    .with_context(|| format!("could not parse session file '{}'", path.display()))?;
            }
        }
        Ok(())
    }

    /// Write the conversation to the session file. Best-effort: a failure here
    /// should never crash a turn, so it's logged to stderr and swallowed.
    fn save_session(&self) {
        if let Some(path) = &self.session {
            match serde_json::to_string_pretty(&self.messages) {
                Ok(data) => {
                    if let Err(e) = std::fs::write(path, data) {
                        eprintln!("\x1b[2m(could not save session: {e})\x1b[0m");
                    }
                }
                Err(e) => eprintln!("\x1b[2m(could not serialize session: {e})\x1b[0m"),
            }
        }
    }

    /// Run one user turn: send the conversation, run any tools the model asks
    /// for (gated by the Judge), loop until it answers in plain text. Progress
    /// goes to `ui`; returns timing/token stats for the whole turn.
    pub async fn send(&mut self, user_input: &str, ui: &mut dyn Ui) -> Result<TurnStats> {
        let start = Instant::now();
        let specs = self.registry.specs();

        ui.emit(Event::User(user_input.to_string()));
        self.messages.push(Message::user(user_input));

        let mut output_tokens = 0u32;
        let mut eval_duration_ns = 0u64;
        let mut model_calls = 0usize;

        for _ in 0..MAX_STEPS {
            let request = CompletionRequest {
                model: self.model.clone(),
                max_tokens: self.max_tokens,
                temperature: self.temperature,
                system: Some(self.system.clone()),
                messages: self.messages.clone(),
                tools: specs.clone(),
            };

            // Stream the reply: each chunk becomes a `Token` event for live
            // display, then `AssistantEnd` tells the frontend the message is
            // whole. The closure borrows `ui` only for the duration of the call.
            let response = self
                .provider
                .complete_streaming(&request, &mut |tok| {
                    ui.emit(Event::Token(tok));
                })
                .await?;
            ui.emit(Event::AssistantEnd);
            model_calls += 1;
            if let Some(usage) = &response.usage {
                output_tokens += usage.output_tokens;
                eval_duration_ns += usage.eval_duration_ns;
            }

            // No tool calls -> the model is done. The text already streamed, so
            // we only need to record it in history and return stats.
            if response.tool_calls.is_empty() {
                self.messages
                    .push(Message::assistant(response.text, Vec::new()));
                self.save_session();
                return Ok(TurnStats {
                    elapsed: start.elapsed(),
                    output_tokens,
                    eval_duration_ns,
                    model_calls,
                });
            }

            // Record the assistant's tool-call turn...
            self.messages.push(Message::assistant(
                response.text.clone(),
                response.tool_calls.clone(),
            ));

            // ...ask the Judge about each call, act, and feed the result back.
            for call in &response.tool_calls {
                let args = call.arguments.to_string();
                ui.emit(Event::ToolCall {
                    name: call.name.clone(),
                    args: args.clone(),
                });

                let result = match self.judge.verdict(call).await {
                    Verdict::Allow => self.registry.dispatch(&call.name, &call.arguments).await,
                    Verdict::Deny(reason) => {
                        ui.emit(Event::ToolDenied(reason.clone()));
                        format!("denied: {reason}")
                    }
                    Verdict::Confirm => {
                        if ui.confirm(&call.name, &args)? {
                            self.registry.dispatch(&call.name, &call.arguments).await
                        } else {
                            "denied: the user declined this action".to_string()
                        }
                    }
                };

                ui.emit(Event::ToolResult(result.clone()));
                self.messages.push(Message::tool_result(
                    call.name.clone(),
                    call.id.clone(),
                    result,
                ));
            }
        }

        anyhow::bail!("gave up after {MAX_STEPS} tool-calling steps");
    }
}

/// Build the system prompt from the live registry, so the tool list the model
/// is told about always matches the tools that actually exist.
fn build_system_prompt(registry: &ToolRegistry) -> String {
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| ".".to_string());
    let tool_list = registry.names().join(", ");

    format!(
        "You are a helpful assistant running in {cwd} with these tools: {tool_list}. \
         When the user refers to a file, use read_file to read it and answer from its actual contents. \
         Pass file paths exactly as the user gives them; relative paths resolve from {cwd}. \
         Answer concisely."
    )
}
