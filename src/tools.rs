//! Tools the agent can call. A `Tool` is anything that can describe itself to
//! the model and run when asked. **This is the file you copy to add more
//! tools:** write one more `impl Tool` and register it in `main.rs` — nothing
//! else in the program changes.

use anyhow::{Context, Result};
use serde_json::{json, Value};

pub trait Tool {
    /// The name the model uses to call this tool (must be stable and unique).
    fn name(&self) -> &str;

    /// A one-line description so the model knows *when* to reach for it.
    fn description(&self) -> &str;

    /// JSON Schema for the arguments — always an object with `properties`.
    fn parameters(&self) -> Value;

    /// Execute with the model-supplied `arguments`, returning text to feed back
    /// to the model. Returning `Err` is fine — the loop reports it to the model.
    fn run(&self, arguments: &Value) -> Result<String>;
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
        // The `json!` macro builds a serde_json::Value from JSON-looking syntax.
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
        // Pull "path" out of the JSON the model sent. `.as_str()` returns None
        // if it's missing or not a string; `.context(...)` turns that into a
        // clear error instead of a silent default.
        let path = arguments
            .get("path")
            .and_then(|v| v.as_str())
            .context("read_file requires a string argument 'path'")?;

        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("could not read file '{path}'"))?;

        Ok(contents)
    }
}
