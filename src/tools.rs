//! Tools the agent can call, and the registry that owns them.
//!
//! A `Tool` describes itself to the model and runs when asked. The
//! `ToolRegistry` holds the set of tools and is the single authority on which
//! tools exist: it builds the specs we advertise, looks tools up by name, and
//! dispatches calls — rejecting any name it doesn't recognise.

use anyhow::{Context, Result};
use serde_json::{json, Value};

use crate::provider::ToolSpec;

pub trait Tool {
    /// The name the model uses to call this tool (must be stable and unique).
    fn name(&self) -> &str;

    /// A one-line description so the model knows *when* to reach for it.
    fn description(&self) -> &str;

    /// JSON Schema for the arguments — always an object with `properties`.
    fn parameters(&self) -> Value;

    /// Execute with the model-supplied `arguments`, returning text to feed back
    /// to the model. Returning `Err` is fine — the registry reports it to the model.
    fn run(&self, arguments: &Value) -> Result<String>;
}

/// Owns the agent's tools and answers "does this tool exist?" in one place.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    /// Add a tool to the registry.
    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

    /// Look up a tool by name (`None` if nothing is registered under it).
    pub fn get(&self, name: &str) -> Option<&dyn Tool> {
        self.tools
            .iter()
            .find(|t| t.name() == name)
            .map(|boxed| boxed.as_ref())
    }

    /// The names of every registered tool — used to tell the model (in the
    /// system prompt) exactly which tools it may call.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.name()).collect()
    }

    /// Neutral specs to advertise to the model on every request.
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools
            .iter()
            .map(|t| ToolSpec {
                name: t.name().to_string(),
                description: t.description().to_string(),
                parameters: t.parameters(),
            })
            .collect()
    }

    /// Run a tool by name. This is the single place that decides whether a tool
    /// exists: an unknown name never executes anything — it returns an error
    /// string (listing the real tools) that we feed back to the model so it can
    /// correct itself. So "send_money" simply bounces here.
    pub fn dispatch(&self, name: &str, arguments: &Value) -> String {
        match self.get(name) {
            Some(tool) => match tool.run(arguments) {
                Ok(output) => output,
                Err(e) => format!("error: {e}"),
            },
            None => format!(
                "error: no tool named '{name}' exists. Available tools: {}",
                self.names().join(", ")
            ),
        }
    }
}

// Idiomatic: when a type has a no-arg `new`, also provide `Default`.
impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Example tool: read a UTF-8 text file and return its contents.
pub struct ReadFile;

impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }

    fn description(&self) -> &str {
        "Read the contents of a UTF-8 text file at the given path and return it."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)."
                }
            },
            "required": ["path"]
        })
    }

    fn run(&self, arguments: &Value) -> Result<String> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .context("read_file requires a string argument 'path'")?;

        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("could not read file '{path}'"))?;

        Ok(contents)
    }
}

pub struct WriteFile;

impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }

    fn description(&self) -> &str {
        "Write the given contents to a UTF-8 text file at the given path."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)."
                },
                "contents": {
                    "type": "string",
                    "description": "Contents to write to the file."
                }
            },
            "required": ["path", "contents"]
        })
    }

    fn run(&self, arguments: &Value) -> Result<String> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .context("write_file requires a string argument 'path'")?;

        let contents = arguments
            .get("contents")
            .and_then(|v| v.as_str())
            .context("write_file requires a string argument 'contents'")?;

        std::fs::write(path, contents)
            .with_context(|| format!("could not write file '{path}'"))?;

        Ok(format!("Wrote {} bytes to '{path}'", contents.len()))
    }
}

/// Run a shell command and return its combined output. The powerful,
/// dangerous one — which is why the Judge gates it before it ever runs here.
pub struct RunShellCommand;

impl Tool for RunShellCommand {
    fn name(&self) -> &str {
        "run_shell_command"
    }

    fn description(&self) -> &str {
        "Run a shell command and return its stdout and stderr."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run."
                }
            },
            "required": ["command"]
        })
    }

    fn run(&self, arguments: &Value) -> Result<String> {
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .context("run_shell_command requires a string argument 'command'")?;

        // `sh -c "<command>"` runs it through the shell, so pipes and the like work.
        let output = std::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .with_context(|| format!("failed to launch command: {command}"))?;

        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);

        let mut result = String::new();
        if !stdout.is_empty() {
            result.push_str(&stdout);
        }
        if !stderr.is_empty() {
            result.push_str("\n[stderr]\n");
            result.push_str(&stderr);
        }
        if result.is_empty() {
            result = format!("(no output; exit status: {})", output.status);
        }

        Ok(result)
    }
}
