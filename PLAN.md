# LocalAgent — Plan

A local, terminal-based AI agent (think "Claude CLI, but mine") written in Rust.
Primary goal: **learn Rust by building something real.** Secondary goal: a genuinely
useful local agent that can run tools on this machine.

Approach: **learn-by-doing.** Build in milestones, meeting one hard Rust concept at a
time. Nothing is thrown away between milestones — the trait design lets ambition arrive
late without rework.

---

## Where we are now  *(snapshot)*

**M0, M1, the async conversion, M2, M3, and most of M4 are done.** The one open
plan item is `JevJudge` (blocked — see below). The agent runs fully **local and
free** against Ollama (`qwen2.5:7b` by default, `OLLAMA_HOST` honored); Anthropic
is there behind the same trait, now **with tool-calling** (untested live — no key).

What works today:

- **Four swappable boundaries**, each a trait with ≥1 real impl: `LlmProvider`
  (Ollama streaming + tools / Anthropic tools), `Tool` (8 of them), `Judge`
  (`RuleJudge` + `LocalModelJudge` + decorators), and a fourth that emerged in
  practice — `Ui` (frontend: `CliUi` / `TuiUi`).
- **Agent loop**: send → stream reply → tool calls → Judge verdict → execute/confirm/deny
  → feed back → loop, with per-turn metrics (elapsed, tokens, real tok/s).
- **Three frontends over one loop**: one-shot CLI, interactive REPL, and a full-screen
  ratatui TUI — all driven by the same `Agent::send`, because it reports through `Ui`.
- **Streaming**: Ollama streams token-by-token; the provider trait has a
  `complete_streaming` whose default lets non-streaming providers work unchanged.
- **Tools**: `read_file`, `write_file`, `edit_file`, `list_directory`,
  `run_shell_command`, `http_fetch`, `git`, and `browse` (a thin wrapper over
  [agent-browser](https://agent-browser.dev)).
- **Judge**: `RuleJudge` is graduated and workspace-aware; `LocalModelJudge`
  asks a local LLM (`--judge model`); `--yolo` / `--careful` decorators bend the
  verdicts. Covered by unit tests (`cargo test`).
- **Sessions & config**: `--session <file>` saves/resumes a conversation; a TOML
  config file sets defaults; `--system` overrides the prompt; `--list-tools` prints
  the toolset.
- **TUI**: streaming tokens, scrollback (PgUp/PgDn + mouse wheel, auto-follow), prompt
  history (↑/↓, persisted to disk), line editing with a live cursor, a live spinner with
  step-aware status ("thinking…" / "running read_file…") and a seconds timer, an in-TUI
  confirm prompt, Ctrl-C interrupt, a `?` help overlay, and a `model · dir` footer.

See [README.md](README.md) for usage. What's left: `JevJudge` (needs paid
TypeSafe access with no free tier — `LocalModelJudge` is the free stand-in that
proves the same trait seam), and wiring Anthropic usage into the metrics.

---

## Vision

An agent is a small loop around three boundaries (plus a frontend):

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Frontend  │────▶│   Agent loop     │────▶│ LlmProvider  │  (trait)
│ CLI · TUI   │◀────│  (orchestrator)  │◀────│  Ollama…     │
└─────────────┘     └────────┬─────────┘     └──────────────┘
   (Ui trait)                │  every tool call
                            ▼
                     ┌──────────────┐        ┌──────────────┐
                     │  Judge trait │───────▶│ Rules / tiny │
                     │ allow/conf./ │        │ local / Jev  │
                     │    deny      │        └──────────────┘
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │  Tool trait  │  shell / fs / browse / …
                     └──────────────┘
```

The core loop never changes:
**send conversation → model streams text and/or tool calls → for each tool call,
ask the Judge (allow / confirm / deny) → execute allowed calls → append results → loop
until the model is done.**

---

## The four traits

Everything plugs into these. Getting them right early is most of the architecture.

- `LlmProvider` — generates. `complete(req)` plus `complete_streaming(req, on_token)`.
- `Tool` — does things. `name()`, `description()`, `parameters()` (JSON schema), async `run()`.
- `Judge` — adjudicates the risky boundary. `verdict(call) -> Allow | Confirm | Deny(reason)`.
- `Ui` — **the fourth trait, discovered while building.** The frontend. The agent `emit`s
  `Event`s and asks `confirm` through it, so one loop drives the CLI, the REPL, and the TUI.

All four are `Send` (`+ Sync` for the first three), because a turn runs on a background
tokio task so the TUI can keep redrawing while the model works. The agent loop depends
only on these traits — never directly on the HTTP client or the TUI.

### Judge progression (free → cloud)

Deciding allow/confirm/deny is a **classification** problem, so it doesn't need a big model.

| Impl              | What                                                | Cost          | Milestone | Status |
| ----------------- | --------------------------------------------------- | ------------- | --------- | ------ |
| `RuleJudge`       | hand-written rules (deny `rm -rf`, confirm writes)  | free, instant | M1        | ✅ shipped (graduated, workspace-aware) |
| `LocalModelJudge` | local Ollama model → ALLOW/CONFIRM/DENY             | free, offline | M2–M3     | ✅ shipped (`--judge model`) |
| `JevJudge`        | TypeSafe's cloud decision model                     | ~pennies      | M4 (opt.) | 🚫 blocked (paid, no free tier) — `LocalModelJudge` is the free stand-in |

Plus two decorators — `AutoApprove` (`--yolo`) and `ConfirmAll` (`--careful`) —
that wrap any judge and bend its verdicts. Develop against `RuleJudge` for free.

---

## Crate shortlist (the Rust syllabus)

| Concern          | Crate                           | Introduced | Status |
| ---------------- | ------------------------------- | ---------- | ------ |
| JSON             | `serde`, `serde_json`           | M0         | ✅ in use |
| HTTP             | `reqwest` (async, `rustls`)     | M0→async   | ✅ (streams via `Response::chunk`) |
| Errors           | `anyhow` (app)                  | M0         | ✅ (`thiserror` not needed yet — no lib crate) |
| CLI args         | `clap` (derive)                 | M0         | ✅ in use |
| Async runtime    | `tokio`                         | Async MS   | ✅ `#[tokio::main]`, `fs`, `process`, `sync::Mutex` |
| Async traits     | `async-trait`                   | M1/async   | ✅ on `LlmProvider`, `Tool`, `Judge` |
| TUI              | `ratatui` + `crossterm`         | M3         | ✅ in use (incl. mouse capture) |
| Config           | `toml` + `dotenvy` + env        | M0–M4      | ✅ config file at `~/.config/local-agent/config.toml` |

---

## Milestones

### M0 — Skeleton that talks  ✅

- [x] `cargo new`, project layout, `clap` one-shot CLI (`agent "prompt"`)
- [x] `LlmProvider` trait + Anthropic implementation
- [x] Ollama implementation (local, free) behind the same trait — `--provider` flag
- [x] Runtime provider selection via `Box<dyn LlmProvider>` (trait objects, pulled forward from M1)
- [x] Request/Response structs with `serde`
- [x] Read API key from env / `.env` (only when Anthropic is selected)
- [x] **Goal:** see a real model reply printed in the terminal ✅
- Concepts: structs, enums, `match`, `Option`/`Result`, `?`, basic ownership

### M1 — The agent loop + first tool + the Judge  ✅

- [x] `Tool` trait + `run_shell_command` (also read_file, write_file)
- [x] Full round-trip: model requests a tool → execute → feed result back → loop
- [x] `Judge` trait + `RuleJudge` — allow reads, confirm writes/shell, deny destructive
- [x] Tool registry (`Box<dyn Tool>`), interactive confirm prompt
- [x] **Goal:** the model completes a task using the shell, gated by the Judge ✅
- Concepts: traits, trait objects, dynamic dispatch, the borrow checker on history

### Async milestone — convert to tokio  ✅

- [x] Move `LlmProvider` + `Tool` (and registry) to async via `async-trait`; `Judge` stays sync
- [x] Switch to async `reqwest`; `#[tokio::main]`; `tokio::fs` / `tokio::process` in tools
- [x] **Goal:** understand *why* async exists, having felt the blocking version ✅
- Concepts: `async`/`await`, `tokio` runtime, futures, `async-trait` for `dyn` traits

### M2 — Broaden the tools  ✅

- [x] `read_file`, `write_file`, `edit_file`, `list_directory`
- [x] `browse` — wraps [agent-browser](https://agent-browser.dev) as one tool, cheat
      sheet in its description *(bonus; not in the original plan)*
- [x] Workspace-aware, graduated `RuleJudge` — path jailing + shell/browse classification
- [x] `http_fetch` — GET-only web fetch, truncated, treated as a read
- [x] dedicated `git` tool — array args (no shell), classified by subcommand
- [x] `LocalModelJudge` via Ollama (`--judge model`)
- [x] **Goal:** genuinely useful; trait design makes each tool ~30 lines ✅
- Concepts: `tokio::fs`, `tokio::process`, repeated patterns, allow-list design, unit tests

### M3 — The TUI  ✅

- [x] `ratatui` frontend (`--tui`) via the `Ui` trait — the agent loop is untouched
- [x] Scrollable conversation log, input box, in-TUI confirm prompt
- [x] **Channels between the UI task and the agent task** — the turn runs on a background
      tokio task and streams events over a channel, so the UI never freezes
- [x] Live spinner with **step-aware status** ("thinking…" / "running <tool>…") + seconds timer
- [x] **Streaming tokens** appear live (pulled forward from M4)
- [x] Scrollback (PgUp/PgDn + mouse wheel, auto-follow), prompt history (↑/↓), line
      editing with a visible cursor, `?` help overlay, Ctrl-C interrupt, `model · dir` footer
- [x] **Goal:** agent loop underneath is untouched; only the frontend changed ✅
- Concepts: event loops, disjoint borrows, `Send` bounds, `Arc`/`Mutex` + channels

### M4 — Polish  *(mostly done)*

- [x] **Streaming responses** — Ollama streams token-by-token; `complete_streaming` on the
      provider trait, default impl keeps non-streaming providers working ✅
- [x] Conversation persistence — `--session <file>` saves/resumes as JSON after each turn
- [x] Config file — TOML at `~/.config/local-agent/config.toml`; CLI flags override it
- [x] Anthropic **tool-calling** — tool_use/tool_result block translation + id threading
      *(to spec; untested live, no API key)*
- [x] (optional) autonomy flags: `--yolo` (auto-allow) / `--careful` (confirm everything)
- [x] Second `Judge` implementation proving the abstraction — `LocalModelJudge` (free,
      local), standing in for the blocked paid `JevJudge`
- [ ] `JevJudge` behind `Judge` — 🚫 blocked: paid, no free tier
- [ ] Wire Anthropic token usage into the metrics
- [x] **Goal:** both the provider and judge abstractions proven with a second real impl ✅

### Extras (beyond the original plan)

- [x] `--system` to override the system prompt (also a config key)
- [x] `--list-tools` and a REPL `tools` command
- [x] TUI prompt history persisted to `~/.local/share/local-agent/history`
- [x] Unit tests for the Judge (`cargo test`)
- [x] `README.md`

---

## Rust concept ramp (one scary thing per stage)

1. structs, enums, `match`, `Option`/`Result`, `?`  — M0 ✅
2. ownership & borrowing (conversation history is the first fight) — M0–M1 ✅
3. traits & `Box<dyn Trait>` — M1 ✅ *(done four times: provider, tool, judge, ui)*
4. `async`/`await`, tokio tasks — async milestone ✅
5. disjoint borrows — splitting `&mut self` into field borrows for a draw closure — M3 ✅
6. `Arc`/`Mutex` + channels — a turn on a background task, events over a channel — M3 ✅
7. `Send` bounds — what the compiler demands to move the agent across threads — M3 ✅
8. streaming I/O — reading `reqwest` body chunks + newline-delimited JSON — M3/M4 ✅

---

## Cost notes

- **Anthropic API:** pay per token. Optional — the default is local Ollama (free).
- **Judge:** free by default (`RuleJudge`, then a local Ollama model). Never required to pay.
- **Jev (optional, M4):** no free tier, but very cheap (~$0.042 / M input tokens, output free
  per third-party reports; ~$0.20 for a few thousand decisions). **Verify on TypeSafe's own
  console before creating a key** — sources disagreed on version and access flow.

---

## Open questions / decisions to revisit

- [x] First concrete `LlmProvider`s: Anthropic + Ollama (both shipped at M0; default is Ollama)
- [x] Default local model: **`qwen2.5:7b`** — chosen after `llama3.1:8b` proved unreliable
      at multi-turn tool use. Lesson: single-turn chat and multi-turn tool-use are
      different skills; a bigger-sounding model isn't automatically better at the latter.
- [x] `LocalModelJudge` shipped using the main Ollama model by default (`--judge-model` to
      pick another). Open: which *small* model classifies best/fastest — not yet benchmarked.
- [ ] Whether a purpose-built open decision model exists worth using instead of a general tiny model
- [x] Anthropic tool-calling: implemented in M4 (untested live — no API key)
