//! Tools the agent can call, and the registry that owns them (async).
//!
//! A `Tool` describes itself to the model and runs when asked. `run` is async
//! because it does I/O — reading files, spawning processes — using tokio's
//! async equivalents of `std::fs` / `std::process`. The `ToolRegistry` is the
//! single authority on which tools exist and dispatches calls to them.

use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

use crate::provider::ToolSpec;

#[async_trait]
pub trait Tool {
    /// The name the model uses to call this tool (must be stable and unique).
    fn name(&self) -> &str;

    /// A one-line description so the model knows *when* to reach for it.
    fn description(&self) -> &str;

    /// JSON Schema for the arguments — always an object with `properties`.
    fn parameters(&self) -> Value;

    /// Execute with the model-supplied `arguments`, returning text to feed back
    /// to the model. Async because it waits on I/O.
    async fn run(&self, arguments: &Value) -> Result<String>;
}

/// Owns the agent's tools and answers "does this tool exist?" in one place.
pub struct ToolRegistry {
    tools: Vec<Box<dyn Tool>>,
}

impl ToolRegistry {
    pub fn new() -> Self {
        Self { tools: Vec::new() }
    }

    pub fn register(&mut self, tool: Box<dyn Tool>) {
        self.tools.push(tool);
    }

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

    /// Run a tool by name. The single place that decides whether a tool exists:
    /// an unknown name never executes anything — it returns an error string
    /// (listing the real tools) that we feed back to the model.
    pub async fn dispatch(&self, name: &str, arguments: &Value) -> String {
        match self.get(name) {
            Some(tool) => match tool.run(arguments).await {
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

impl Default for ToolRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Read a UTF-8 text file and return its contents.
pub struct ReadFile;

#[async_trait]
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

    async fn run(&self, arguments: &Value) -> Result<String> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .context("read_file requires a string argument 'path'")?;

        let contents = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("could not read file '{path}'"))?;

        Ok(contents)
    }
}

/// Write the given contents to a UTF-8 text file.
pub struct WriteFile;

#[async_trait]
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

    async fn run(&self, arguments: &Value) -> Result<String> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .context("write_file requires a string argument 'path'")?;

        let contents = arguments
            .get("contents")
            .and_then(|v| v.as_str())
            .context("write_file requires a string argument 'contents'")?;

        tokio::fs::write(path, contents)
            .await
            .with_context(|| format!("could not write file '{path}'"))?;

        Ok(format!("Wrote {} bytes to '{path}'", contents.len()))
    }
}

/// Run a shell command and return its combined output. The powerful, dangerous
/// one — which is why the Judge gates it before it ever runs here.
pub struct RunShellCommand;

#[async_trait]
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

    async fn run(&self, arguments: &Value) -> Result<String> {
        let command = arguments
            .get("command")
            .and_then(|v| v.as_str())
            .context("run_shell_command requires a string argument 'command'")?;

        // `sh -c "<command>"` runs it through the shell, so pipes and the like work.
        let output = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
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

/// List the entries in a directory. Read-only, so the Judge lets it run freely.
pub struct ListDirectory;

#[async_trait]
impl Tool for ListDirectory {
    fn name(&self) -> &str {
        "list_directory"
    }

    fn description(&self) -> &str {
        "List the files and subdirectories in a directory (defaults to the current directory)."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list. Defaults to '.' (the current directory)."
                }
            },
            "required": []
        })
    }

    async fn run(&self, arguments: &Value) -> Result<String> {
        // `path` is optional here — fall back to the current directory.
        let path = arguments.get("path").and_then(|v| v.as_str()).unwrap_or(".");

        let mut dir = tokio::fs::read_dir(path)
            .await
            .with_context(|| format!("could not read directory '{path}'"))?;

        let mut names: Vec<String> = Vec::new();
        // read_dir yields entries one at a time; `next_entry().await` gets each.
        while let Some(entry) = dir
            .next_entry()
            .await
            .with_context(|| format!("error while reading directory '{path}'"))?
        {
            let name = entry.file_name().to_string_lossy().into_owned();
            let is_dir = entry.file_type().await.map(|t| t.is_dir()).unwrap_or(false);
            // Trailing slash marks directories, like `ls -F`.
            names.push(if is_dir { format!("{name}/") } else { name });
        }

        names.sort();
        if names.is_empty() {
            Ok(format!("(empty directory: {path})"))
        } else {
            Ok(names.join("\n"))
        }
    }
}

/// Replace one exact, unique string in a file. Safer and more precise than
/// overwriting the whole file: it refuses if `old_string` is missing or
/// appears more than once, and reports that back so the model can retry.
pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }

    fn description(&self) -> &str {
        "Replace an exact string in a file with new text. old_string must appear exactly once."
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit."
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to replace. Must match exactly and appear once."
                },
                "new_string": {
                    "type": "string",
                    "description": "The text to put in its place."
                }
            },
            "required": ["path", "old_string", "new_string"]
        })
    }

    async fn run(&self, arguments: &Value) -> Result<String> {
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .context("edit_file requires a string argument 'path'")?;
        let old_string = arguments
            .get("old_string")
            .and_then(|v| v.as_str())
            .context("edit_file requires a string argument 'old_string'")?;
        let new_string = arguments
            .get("new_string")
            .and_then(|v| v.as_str())
            .context("edit_file requires a string argument 'new_string'")?;

        if old_string.is_empty() {
            anyhow::bail!("edit_file: old_string must not be empty");
        }

        let contents = tokio::fs::read_to_string(path)
            .await
            .with_context(|| format!("could not read file '{path}'"))?;

        // Require a single, unambiguous match — the same discipline a good edit
        // tool uses, so the model can't accidentally change the wrong spot.
        let count = contents.matches(old_string).count();
        if count == 0 {
            anyhow::bail!("edit_file: old_string was not found in '{path}'");
        }
        if count > 1 {
            anyhow::bail!(
                "edit_file: old_string appears {count} times in '{path}'; \
                 include more surrounding text to make it unique"
            );
        }

        let updated = contents.replacen(old_string, new_string, 1);
        tokio::fs::write(path, &updated)
            .await
            .with_context(|| format!("could not write file '{path}'"))?;

        Ok(format!("Replaced 1 occurrence in '{path}'"))
    }
}
