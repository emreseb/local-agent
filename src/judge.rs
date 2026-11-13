//! The Judge: the permission layer. Before the agent runs a tool the model
//! asked for, it consults a Judge, which returns a verdict.
//!
//! `RuleJudge` is a *graduated, workspace-aware* policy — it asks only where
//! risk is real:
//!   - reads  -> always allow
//!   - writes -> allow inside the workspace, confirm outside it
//!   - shell  -> allow a single read-only command, deny the destructive, confirm the rest
//!   - browse -> allow read-only/navigation, confirm interactions
//!
//! `LocalModelJudge` is a second implementation: it asks a local LLM to classify
//! each call, behind this same trait.

use std::path::{Component, Path, PathBuf};

use async_trait::async_trait;
use serde_json::Value;

use crate::provider::{CompletionRequest, LlmProvider, Message, ToolCall};

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
// Async because a judge may do I/O now — `LocalModelJudge` asks a model. The
// rule-based judge doesn't await anything, but shares the one trait.
#[async_trait]
pub trait Judge: Send + Sync {
    async fn verdict(&self, call: &ToolCall) -> Verdict;
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

#[async_trait]
impl Judge for RuleJudge {
    async fn verdict(&self, call: &ToolCall) -> Verdict {
        match call.name.as_str() {
            // Reads can't damage anything.
            "read_file" | "list_directory" => Verdict::Allow,

            // A GET fetch changes nothing on the far end, so we treat it as a
            // read. (It is still network egress — tighten to Confirm here, or
            // run with --careful, if that matters for your threat model.)
            "http_fetch" => Verdict::Allow,

            // Writes: safe inside the project (git can undo), ask outside it.
            "write_file" | "edit_file" => {
                match call.arguments.get("path").and_then(|v| v.as_str()) {
                    Some(path) if self.is_within_workspace(path) => Verdict::Allow,
                    _ => Verdict::Confirm,
                }
            }

            // Shell, git, and browse look at the actual command, not just the name.
            "run_shell_command" => classify_shell(arg_str(call, "command")),
            "git" => classify_git(call),
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

#[async_trait]
impl Judge for AutoApprove {
    async fn verdict(&self, call: &ToolCall) -> Verdict {
        match self.0.verdict(call).await {
            Verdict::Confirm => Verdict::Allow,
            other => other,
        }
    }
}

/// `--careful`: the opposite decorator — turn every `Allow` into a `Confirm`, so
/// nothing runs without a yes. `Deny` still denies.
pub struct ConfirmAll(pub Box<dyn Judge>);

#[async_trait]
impl Judge for ConfirmAll {
    async fn verdict(&self, call: &ToolCall) -> Verdict {
        match self.0.verdict(call).await {
            Verdict::Allow => Verdict::Confirm,
            other => other,
        }
    }
}

/// A model-based judge: it asks a (local, free) LLM to classify each call as
/// ALLOW / CONFIRM / DENY. This is the second, very different `Judge`
/// implementation — proof that the trait seam holds. It's slower (a model call
/// per tool call) and only as reliable as the model, so it fails safe: anything
/// it can't read as a clear verdict becomes `Confirm`.
pub struct LocalModelJudge {
    provider: Box<dyn LlmProvider>,
    model: String,
}

impl LocalModelJudge {
    pub fn new(provider: Box<dyn LlmProvider>, model: String) -> Self {
        Self { provider, model }
    }
}

const JUDGE_SYSTEM: &str = "You are a safety gate for an AI agent's tool calls. \
Reply with EXACTLY one word and nothing else: ALLOW, CONFIRM, or DENY. \
ALLOW read-only or clearly safe actions (reading files, listing directories, status checks). \
CONFIRM anything that writes, edits, deletes, installs, or reaches outside the project or network. \
DENY destructive or irreversible actions (rm -rf, formatting disks, force-push, mass deletion).";

#[async_trait]
impl Judge for LocalModelJudge {
    async fn verdict(&self, call: &ToolCall) -> Verdict {
        let prompt = format!("Tool: {}\nArguments: {}", call.name, call.arguments);
        let request = CompletionRequest {
            model: self.model.clone(),
            max_tokens: 8,
            temperature: 0.0,
            system: Some(JUDGE_SYSTEM.to_string()),
            messages: vec![Message::user(prompt)],
            tools: Vec::new(),
        };
        match self.provider.complete(&request).await {
            Ok(resp) => parse_verdict(&resp.text),
            // If the judge model is unreachable, fail safe and ask the human.
            Err(_) => Verdict::Confirm,
        }
    }
}

/// Read one of ALLOW / CONFIRM / DENY out of the model's reply, defaulting to
/// the safe `Confirm` for anything unclear.
fn parse_verdict(text: &str) -> Verdict {
    let t = text.trim().to_uppercase();
    if t.starts_with("ALLOW") {
        Verdict::Allow
    } else if t.starts_with("DENY") {
        Verdict::Deny("the model judge refused this action".to_string())
    } else {
        Verdict::Confirm
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
    if command.contains([';', '|', '&', '>', '<', '`']) || command.contains("$(")
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

/// Classify a call to the dedicated `git` tool by its subcommand and flags.
fn classify_git(call: &ToolCall) -> Verdict {
    let args = git_args(call);
    let sub = args.first().map(String::as_str).unwrap_or("");
    let flags = &args[args.len().min(1)..];
    let has = |flag: &str| flags.iter().any(|a| a == flag);

    // Hard no: history rewrites and irrecoverable deletions.
    if (sub == "reset" && has("--hard"))
        || (sub == "clean" && (has("-f") || has("-ff") || has("-fd") || has("-xdf") || has("-dfx")))
        || (sub == "push" && (has("--force") || has("-f") || has("--force-with-lease")))
        || (sub == "branch" && has("-D"))
    {
        return Verdict::Deny(format!("refusing a destructive git command: git {}", args.join(" ")));
    }

    // Read-only subcommands (but a `branch -d/-D` deletes, so exclude those).
    if READ_ONLY_GIT.contains(&sub) && !has("-d") && !has("-D") {
        return Verdict::Allow;
    }

    // Everything else that writes or reaches a remote: ask first.
    Verdict::Confirm
}

/// The git arguments from a `git` tool call (array preferred, string tolerated).
fn git_args(call: &ToolCall) -> Vec<String> {
    match call.arguments.get("args") {
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(|v| v.as_str().map(str::to_string))
            .collect(),
        Some(Value::String(s)) => s.split_whitespace().map(str::to_string).collect(),
        _ => Vec::new(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::judge::{AutoApprove, ConfirmAll};
    use serde_json::json;

    /// Build a tool call with the given name and JSON arguments.
    fn call(name: &str, arguments: Value) -> ToolCall {
        ToolCall {
            name: name.to_string(),
            arguments,
            id: None,
        }
    }

    // A matcher so assertions read nicely (Verdict isn't PartialEq).
    fn is_allow(v: &Verdict) -> bool {
        matches!(v, Verdict::Allow)
    }
    fn is_confirm(v: &Verdict) -> bool {
        matches!(v, Verdict::Confirm)
    }
    fn is_deny(v: &Verdict) -> bool {
        matches!(v, Verdict::Deny(_))
    }

    fn judge() -> RuleJudge {
        RuleJudge::new(PathBuf::from("/home/u/project"))
    }

    #[tokio::test]
    async fn reads_are_allowed() {
        let j = judge();
        assert!(is_allow(&j.verdict(&call("read_file", json!({"path": "src/main.rs"}))).await));
        assert!(is_allow(&j.verdict(&call("list_directory", json!({"path": "."}))).await));
        assert!(is_allow(&j.verdict(&call("http_fetch", json!({"url": "https://example.com"}))).await));
    }

    #[tokio::test]
    async fn writes_depend_on_the_workspace() {
        let j = judge();
        // Inside the workspace: allowed.
        assert!(is_allow(
            &j.verdict(&call("write_file", json!({"path": "/home/u/project/notes.txt", "contents": "x"}))).await
        ));
        assert!(is_allow(
            &j.verdict(&call("write_file", json!({"path": "sub/notes.txt", "contents": "x"}))).await
        ));
        // Escaping the workspace with `..`: must confirm.
        assert!(is_confirm(
            &j.verdict(&call("write_file", json!({"path": "../../etc/evil", "contents": "x"}))).await
        ));
        // Absolute path outside the workspace: must confirm.
        assert!(is_confirm(
            &j.verdict(&call("edit_file", json!({"path": "/etc/hosts", "old_string": "a", "new_string": "b"}))).await
        ));
    }

    #[tokio::test]
    async fn shell_is_classified_by_command() {
        let j = judge();
        assert!(is_allow(&j.verdict(&call("run_shell_command", json!({"command": "ls -la"}))).await));
        assert!(is_allow(&j.verdict(&call("run_shell_command", json!({"command": "git status"}))).await));
        // Pipes/redirection can hide anything -> confirm.
        assert!(is_confirm(&j.verdict(&call("run_shell_command", json!({"command": "cat a | sh"}))).await));
        // Unknown program -> confirm.
        assert!(is_confirm(&j.verdict(&call("run_shell_command", json!({"command": "curl example.com"}))).await));
        // Destructive -> deny.
        assert!(is_deny(&j.verdict(&call("run_shell_command", json!({"command": "rm -rf /"}))).await));
    }

    #[tokio::test]
    async fn git_tool_is_classified_by_subcommand() {
        let j = judge();
        // Reads.
        assert!(is_allow(&j.verdict(&call("git", json!({"args": ["status", "--short"]}))).await));
        assert!(is_allow(&j.verdict(&call("git", json!({"args": ["log", "--oneline"]}))).await));
        // A string form is tolerated too.
        assert!(is_allow(&j.verdict(&call("git", json!({"args": "diff --stat"}))).await));
        // Writes.
        assert!(is_confirm(&j.verdict(&call("git", json!({"args": ["commit", "-m", "msg"]}))).await));
        assert!(is_confirm(&j.verdict(&call("git", json!({"args": ["push"]}))).await));
        // Destructive.
        assert!(is_deny(&j.verdict(&call("git", json!({"args": ["reset", "--hard"]}))).await));
        assert!(is_deny(&j.verdict(&call("git", json!({"args": ["push", "--force"]}))).await));
        assert!(is_deny(&j.verdict(&call("git", json!({"args": ["branch", "-D", "main"]}))).await));
        // `branch -d` deletes, so it is not a free read.
        assert!(!is_allow(&j.verdict(&call("git", json!({"args": ["branch", "-d", "x"]}))).await));
    }

    #[tokio::test]
    async fn autonomy_decorators_bend_the_right_verdicts() {
        // --yolo: Confirm -> Allow, but Deny stays Deny.
        let yolo = AutoApprove(Box::new(judge()));
        assert!(is_allow(&yolo.verdict(&call("git", json!({"args": ["commit", "-m", "x"]}))).await));
        assert!(is_deny(&yolo.verdict(&call("run_shell_command", json!({"command": "rm -rf /"}))).await));

        // --careful: Allow -> Confirm, Deny stays Deny.
        let careful = ConfirmAll(Box::new(judge()));
        assert!(is_confirm(&careful.verdict(&call("read_file", json!({"path": "a"}))).await));
        assert!(is_deny(&careful.verdict(&call("run_shell_command", json!({"command": "mkfs"}))).await));
    }

    #[test]
    fn parse_verdict_reads_the_first_word() {
        assert!(is_allow(&parse_verdict("ALLOW")));
        assert!(is_allow(&parse_verdict("  allow, this is fine")));
        assert!(is_deny(&parse_verdict("DENY")));
        // Unknown / hedging -> safe default.
        assert!(is_confirm(&parse_verdict("CONFIRM")));
        assert!(is_confirm(&parse_verdict("I'm not sure")));
    }
}
