//! The Judge: the permission layer. Before the agent runs a tool the model
//! asked for, it consults a Judge, which returns a verdict.
//!
//! `RuleJudge` is a *graduated, workspace-aware* policy — it asks only where
//! risk is real:
//!   - reads  -> always allow
//!   - writes -> allow inside the workspace, confirm outside it
//!   - shell  -> allow a single read-only command, deny the destructive, confirm the rest
//!   - browse -> allow read-only/navigation, confirm interactions
//! Later it could be swapped for a model-based judge behind this same trait.

use std::path::{Component, Path, PathBuf};

use crate::provider::ToolCall;

/// A Judge's decision about one tool call.
pub enum Verdict {
    /// Run it without asking.
    Allow,
    /// Ask the human first.
    Confirm,
    /// Refuse outright; the reason is fed back to the model so it can adapt.
    Deny(String),
}

// `Send + Sync` so the agent that owns the judge can move to a background task.
pub trait Judge: Send + Sync {
    fn verdict(&self, call: &ToolCall) -> Verdict;
}

/// The graduated policy. Holds the workspace root so it can tell in-project
/// actions from ones that reach outside it.
pub struct RuleJudge {
    workspace: PathBuf,
}

impl RuleJudge {
    pub fn new(workspace: PathBuf) -> Self {
        Self { workspace }
    }

    /// Is `path` (resolved against the workspace) inside the workspace? Resolved
    /// lexically, so `..` is honored and escaping paths are caught — and it
    /// works for files that don't exist yet (a new write target).
    fn is_within_workspace(&self, path: &str) -> bool {
        normalize(&self.workspace, Path::new(path)).starts_with(&self.workspace)
    }
}

impl Judge for RuleJudge {
    fn verdict(&self, call: &ToolCall) -> Verdict {
        match call.name.as_str() {
            // Reads can't damage anything.
            "read_file" | "list_directory" => Verdict::Allow,

            // Writes: safe inside the project (git can undo), ask outside it.
            "write_file" | "edit_file" => {
                match call.arguments.get("path").and_then(|v| v.as_str()) {
                    Some(path) if self.is_within_workspace(path) => Verdict::Allow,
                    _ => Verdict::Confirm,
                }
            }

            // Shell and browse look at the actual command, not just the tool name.
            "run_shell_command" => classify_shell(arg_str(call, "command")),
            "browse" => classify_browse(arg_str(call, "command")),

            // Registered but unclassified: ask, to be safe.
            _ => Verdict::Confirm,
        }
    }
}

/// `--yolo`: auto-approve anything the inner judge would merely *ask* about.
/// Deny still means deny — yolo trusts the human's taste, not a blank cheque for
/// the genuinely destructive. A decorator: it wraps another judge and bends only
/// the `Confirm` verdict.
pub struct AutoApprove(pub Box<dyn Judge>);

impl Judge for AutoApprove {
    fn verdict(&self, call: &ToolCall) -> Verdict {
        match self.0.verdict(call) {
            Verdict::Confirm => Verdict::Allow,
            other => other,
        }
    }
}

/// `--careful`: the opposite decorator — turn every `Allow` into a `Confirm`, so
/// nothing runs without a yes. `Deny` still denies.
pub struct ConfirmAll(pub Box<dyn Judge>);

impl Judge for ConfirmAll {
    fn verdict(&self, call: &ToolCall) -> Verdict {
        match self.0.verdict(call) {
            Verdict::Allow => Verdict::Confirm,
            other => other,
        }
    }
}

/// Pull a string argument out of a tool call, or "" if missing.
fn arg_str<'a>(call: &'a ToolCall, key: &str) -> &'a str {
    call.arguments.get(key).and_then(|v| v.as_str()).unwrap_or("")
}

/// Programs that only read and can't modify anything.
const READ_ONLY_PROGRAMS: &[&str] = &[
    "ls", "cat", "pwd", "echo", "grep", "rg", "find", "head", "tail", "wc", "which", "tree",
    "file", "stat", "date", "whoami", "env", "du", "df", "ps",
];

/// git subcommands that only read.
const READ_ONLY_GIT: &[&str] = &["status", "diff", "log", "show", "branch", "remote"];

/// agent-browser subcommands that only read/navigate, not interact.
const READ_ONLY_BROWSE: &[&str] = &[
    "open", "snapshot", "get", "screenshot", "pdf", "close", "list", "tabs", "wait",
];

fn classify_shell(command: &str) -> Verdict {
    if is_destructive(command) {
        return Verdict::Deny(format!("refusing to run a destructive command: {command}"));
    }

    // Compound commands (pipes, chaining, redirection, substitution) could hide
    // anything, so we never auto-allow them — we can't reason about the whole
    // thing by the first word alone.
    if command.contains(|ch| matches!(ch, ';' | '|' | '&' | '>' | '<' | '`')) || command.contains("$(")
    {
        return Verdict::Confirm;
    }

    let mut words = command.split_whitespace();
    let program = words.next().unwrap_or("");

    let read_only = if program == "git" {
        READ_ONLY_GIT.contains(&words.next().unwrap_or(""))
    } else {
        READ_ONLY_PROGRAMS.contains(&program)
    };

    if read_only {
        Verdict::Allow
    } else {
        Verdict::Confirm
    }
}

fn classify_browse(command: &str) -> Verdict {
    let sub = command.split_whitespace().next().unwrap_or("");
    if READ_ONLY_BROWSE.contains(&sub) {
        Verdict::Allow
    } else {
        Verdict::Confirm
    }
}

/// A conservative blocklist for the worst offenders. A blocklist NEVER catches
/// everything — that's why Confirm (a human) is the real safety net.
fn is_destructive(command: &str) -> bool {
    let c = command.to_lowercase();
    c.contains("rm -rf")
        || c.contains("mkfs")
        || c.contains("dd if=")
        || c.contains(":(){") // fork bomb
        || c.contains("> /dev/sd")
}

/// Resolve `path` against `base` and collapse `.`/`..` lexically (no filesystem
/// access, so it works for paths that don't exist yet).
fn normalize(base: &Path, path: &Path) -> PathBuf {
    let mut result = if path.is_relative() {
        base.to_path_buf()
    } else {
        PathBuf::new()
    };

    for component in path.components() {
        match component {
            Component::RootDir => result.push("/"),
            Component::CurDir => {}
            Component::ParentDir => {
                result.pop();
            }
            Component::Normal(c) => result.push(c),
            Component::Prefix(_) => {} // Windows-only; irrelevant here
        }
    }
    result
}
