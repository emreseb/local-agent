//! The agent loop: the model's three-step tool cycle turned into an actual
//! loop. Send the conversation plus the tool descriptions; if the model asks
//! to call tools, run them and append the results; repeat. Stop when the model
//! replies with plain text and no tool calls.

use anyhow::Result;

use crate::provider::{CompletionRequest, LlmProvider, Message, ToolSpec};
use crate::tools::Tool;

/// Safety valve: never loop forever if a model keeps calling tools.
const MAX_STEPS: usize = 10;

pub fn run(
    provider: &dyn LlmProvider,
    model: &str,
    max_tokens: u32,
    tools: &[Box<dyn Tool>],
    user_input: &str,
) -> Result<()> {
    // Describe our tools once, up front, in the neutral ToolSpec form.
    let specs: Vec<ToolSpec> = tools
        .iter()
        .map(|t| ToolSpec {
            name: t.name().to_string(),
            description: t.description().to_string(),
            parameters: t.parameters(),
        })
        .collect();

    // The running conversation: starts with the user's prompt, grows as the
    // model calls tools and we feed results back in.
    let mut messages = vec![Message::user(user_input)];

    for _ in 0..MAX_STEPS {
        let request = CompletionRequest {
            model: model.to_string(),
            max_tokens,
            messages: messages.clone(),
            tools: specs.clone(),
        };

        let response = provider.complete(&request)?;

        // No tool calls -> the model is done. Print the answer and stop.
        if response.tool_calls.is_empty() {
            println!("{}", response.text);
            return Ok(());
        }

        // Otherwise, record the assistant's tool-call turn in the history...
        messages.push(Message::assistant(response.text.clone(), response.tool_calls.clone()));

        // ...then run each requested tool and append its output as a tool turn.
        for call in &response.tool_calls {
            // Dim trace line so you can watch the agent work (\x1b[2m = dim).
            println!("\x1b[2m↳ {}({})\x1b[0m", call.name, call.arguments);

            let result = match tools.iter().find(|t| t.name() == call.name) {
                Some(tool) => match tool.run(&call.arguments) {
                    Ok(output) => output,
                    // Feed errors back to the model rather than crashing — it
                    // can often recover (e.g. fix a wrong path and retry).
                    Err(e) => format!("error: {e}"),
                },
                None => format!("error: unknown tool '{}'", call.name),
            };

            messages.push(Message::tool_result(call.name.clone(), result));
        }
    }

    anyhow::bail!("gave up after {MAX_STEPS} tool-calling steps");
}
