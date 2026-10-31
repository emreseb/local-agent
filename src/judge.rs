//! The Judge: the permission layer. Before the agent runs a tool the model
//! asked for, it consults a Judge, which returns a verdict. This is where
//! "the model can only call registered tools" (the registry's job) becomes
//! "...and dangerous ones need a human's OK".
//!
//! `RuleJudge` is a hand-written policy — no model, no cost, instant. Later it
//! could be swapped for a tiny local model or Jev behind this same trait (see
//! PLAN.md); the loop wouldn't change, exactly like the provider swap.

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

pub trait Judge {
    fn verdict(&self, call: &ToolCall) -> Verdict;
}

/// A fixed rule set: allow reads, confirm writes and shell, deny the obviously
/// destructive.
pub struct RuleJudge;

impl Judge for RuleJudge {
    fn verdict(&self, call: &ToolCall) -> Verdict {
        match call.name.as_str() {
            // Reading can't damage anything.
            "read_file" | "list_directory" => Verdict::Allow,

            // Writing changes the disk — ask first.
            "write_file" => Verdict::Confirm,

            // Shell is all-powerful: block the obviously destructive, ask for the rest.
            "run_shell_command" => {
                let command = call
                    .arguments
                    .get("command")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                if is_destructive(command) {
                    Verdict::Deny(format!("refusing to run a destructive command: {command}"))
                } else {
                    Verdict::Confirm
                }
            }

            // Registered but unclassified: ask, to be safe.
            _ => Verdict::Confirm,
        }
    }
}

/// A conservative blocklist for the worst offenders. A blocklist NEVER catches
/// everything — which is the whole point of `Confirm`: the human in the loop is
/// the real safety net, not this function.
fn is_destructive(command: &str) -> bool {
    let c = command.to_lowercase();
    c.contains("rm -rf")
        || c.contains("mkfs")
        || c.contains("dd if=")
        || c.contains(":(){") // fork bomb
        || c.contains("> /dev/sd")
}
