//! The agent loop. Sends a system prompt + the conversation + the tool specs;
//! if the model asks for tools, it consults the Judge, runs the allowed ones
//! (asking the human for any that need confirmation), feeds the results back,
//! and repeats until the model answers in plain text.

use anyhow::Result;
use std::io::{self, Write};

use crate::judge::{Judge, Verdict};
use crate::provider::{CompletionRequest, LlmProvider, Message, ToolCall};
use crate::tools::ToolRegistry;

/// Safety valve: never loop forever if a model keeps calling tools.
const MAX_STEPS: usize = 10;

pub fn run(
    provider: &dyn LlmProvider,
    model: &str,
    max_tokens: u32,
    temperature: f32,
    registry: &ToolRegistry,
    judge: &dyn Judge,
    user_input: &str,
) -> Result<()> {
    let system = build_system_prompt(registry);
    let specs = registry.specs();

    let mut messages = vec![Message::user(user_input)];

    for _ in 0..MAX_STEPS {
        let request = CompletionRequest {
            model: model.to_string(),
            max_tokens,
            temperature,
            system: Some(system.clone()),
            messages: messages.clone(),
            tools: specs.clone(),
        };

        let response = provider.complete(&request)?;

        // No tool calls -> the model is done. Print the answer and stop.
        if response.tool_calls.is_empty() {
            println!("{}", response.text);
            return Ok(());
        }

        // Record the assistant's tool-call turn...
        messages.push(Message::assistant(response.text.clone(), response.tool_calls.clone()));

        // ...then, for each call: ask the Judge, act on its verdict, and feed
        // the result (or the refusal) back to the model as a tool turn.
        for call in &response.tool_calls {
            println!("\x1b[2m↳ {}({})\x1b[0m", call.name, call.arguments);

            let result = match judge.verdict(call) {
                Verdict::Allow => registry.dispatch(&call.name, &call.arguments),
                Verdict::Deny(reason) => {
                    println!("\x1b[31m  ✗ denied: {reason}\x1b[0m");
                    format!("denied: {reason}")
                }
                Verdict::Confirm => {
                    if confirm(call)? {
                        registry.dispatch(&call.name, &call.arguments)
                    } else {
                        "denied: the user declined this action".to_string()
                    }
                }
            };

            // Show what the tool ACTUALLY returned (first 200 chars, one line),
            // so a mistake by the model is instantly distinguishable from a
            // mistake by the tool.
            let preview: String = result.chars().take(200).collect();
            println!("\x1b[2m  → {}\x1b[0m", preview.replace('\n', " "));

            messages.push(Message::tool_result(call.name.clone(), result));
        }
    }

    anyhow::bail!("gave up after {MAX_STEPS} tool-calling steps");
}

/// Ask the human to approve a tool call. Returns true only for an explicit yes.
fn confirm(call: &ToolCall) -> Result<bool> {
    print!(
        "\x1b[33m⚠ allow {}({})? [y/N] \x1b[0m",
        call.name, call.arguments
    );
    io::stdout().flush()?; // print! doesn't flush on its own; force the prompt out.

    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim().to_lowercase().as_str(), "y" | "yes"))
}

/// Build the system prompt from the live registry, so the tool list the model
/// is told about always matches the tools that actually exist.
///
/// Kept short and positive: small models follow one clear instruction far
/// better than a long list of "never do X" rules.
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
