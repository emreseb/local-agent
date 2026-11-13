# local-agent

A local, terminal-based AI agent written in Rust — "Claude CLI, but mine." It
runs a tool-using agent loop against a **local, free** model (Ollama) by default,
with an optional Anthropic backend. Built as a learn-by-doing Rust project; see
[PLAN.md](PLAN.md) for the milestone history and the concepts behind each step.

## Highlights

- **Provider-agnostic** agent loop behind an `LlmProvider` trait — Ollama
  (streaming, tool calls) and Anthropic (tool calls) both plug into the same loop.
- **Tools**: `read_file`, `write_file`, `edit_file`, `list_directory`,
  `run_shell_command`, `http_fetch`, `git`, and `browse` (via
  [agent-browser](https://agent-browser.dev)).
- **A Judge** gates every tool call — allow / confirm / deny. Two implementations:
  a fast rule-based one (default) and a model-based one (`--judge model`).
- **Three frontends over one loop**: one-shot CLI, interactive REPL, and a
  full-screen **TUI** with streaming tokens, a live spinner + timer, scrollback,
  prompt history, and in-app confirmations.
- **Sessions** save and resume conversations; a **config file** sets your defaults.

## Build & run

Requires a recent Rust toolchain (edition 2024) and, for the default backend,
[Ollama](https://ollama.com) running locally with a model pulled:

```bash
ollama pull qwen2.5:7b
cargo build --release
```

```bash
# One-shot
cargo run -- "List the files here and summarize the project."

# Interactive REPL
cargo run

# Full-screen TUI
cargo run -- --tui
```

## Usage

```
local-agent [OPTIONS] [PROMPT]
```

Omit `PROMPT` for the REPL; add `--tui` for the full-screen UI.

| Option | Meaning |
| ------ | ------- |
| `--provider <ollama\|anthropic>` | Backend. Default: `ollama` (local, free). |
| `--model <NAME>` | Model id. Default: `qwen2.5:7b` / `claude-haiku-5-5`. |
| `--temperature <F>` | Sampling temperature (default `0.2`). |
| `--max-tokens <N>` | Max reply tokens (default `1024`). |
| `--judge <rule\|model>` | Permission judge. Default `rule`. |
| `--judge-model <NAME>` | Model for `--judge model` (default: the main model). |
| `--yolo` | Auto-approve anything the judge would merely ask about. Deny still denies. |
| `--careful` | Confirm *every* action, even reads. |
| `--system <TEXT>` | Replace the built-in system prompt. |
| `--session <FILE>` | Save/resume the conversation in this file. |
| `--config <FILE>` | TOML config path (see below). |
| `--list-tools` | Print the available tools and exit. |
| `--tui` | Launch the full-screen TUI. |

Environment: `OLLAMA_HOST` (default `http://localhost:11434`), and
`ANTHROPIC_API_KEY` (from the environment or a `.env` file) when using
`--provider anthropic`.

## The Judge

Every tool call is classified before it runs:

- **Allow** — read-only / safe (reading files, listing dirs, `http_fetch`,
  read-only git and shell).
- **Confirm** — writes, commits, and anything reaching outside the workspace;
  you're asked y/n.
- **Deny** — the genuinely destructive (`rm -rf`, `git reset --hard`,
  force-push, …); fed back to the model so it can adapt.

`--yolo` turns confirmations into allows; `--careful` turns allows into
confirms; both leave denials intact. `--judge model` instead asks a local LLM to
classify each call (slower, experimental, fails safe to Confirm).

## Config file

Defaults come from `$XDG_CONFIG_HOME/local-agent/config.toml` (or
`~/.config/local-agent/config.toml`), overridable with `--config`. Any CLI flag
wins over the file. All keys are optional:

```toml
provider    = "ollama"
model       = "qwen2.5:7b"
temperature = 0.2
max_tokens  = 1024
judge       = "rule"
system      = "You are a concise, careful coding assistant."
```

## TUI keys

`Enter` send · `↑`/`↓` prompt history · `←`/`→`/`Home`/`End` edit · `PgUp`/`PgDn`
and the mouse wheel scroll the log · `?` toggles help · `clear` resets ·
`Ctrl-C` interrupts a running turn · `Esc` quits. Prompt history persists in
`$XDG_DATA_HOME/local-agent/history`.

## Architecture

The loop depends only on four traits, each with at least one real implementation:

- `LlmProvider` — `ollama.rs`, `anthropic.rs`
- `Tool` — `tools.rs`
- `Judge` — `judge.rs` (`RuleJudge`, `LocalModelJudge`, and the `--yolo`/`--careful` decorators)
- `Ui` — `ui.rs` (`CliUi`), `tui.rs` (`TuiUi`)

A turn runs on a background tokio task and streams events to the frontend over a
channel, so the TUI stays responsive while the model works. Run `cargo test` for
the judge's unit tests.
