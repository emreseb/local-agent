# LocalAgent — Plan

A local, terminal-based AI agent (think "Claude CLI, but mine") written in Rust.
Primary goal: **learn Rust by building something real.** Secondary goal: a genuinely
useful local agent that can run tools on this machine.

Approach: **learn-by-doing.** Build in milestones, meeting one hard Rust concept at a
time. Nothing is thrown away between milestones — the trait design lets ambition arrive
late without rework.

---

## Vision

An agent is a small loop around three boundaries:

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Frontend  │────▶│   Agent loop     │────▶│ LlmProvider  │  (trait)
│ CLI → TUI   │◀────│  (orchestrator)  │◀────│  Anthropic…  │
└─────────────┘     └────────┬─────────┘     └──────────────┘
                             │  every tool call
                             ▼
                     ┌──────────────┐        ┌──────────────┐
                     │  Judge trait │───────▶│ Rules / tiny │
                     │ allow/conf./ │        │ local / Jev  │
                     │    deny      │        └──────────────┘
                     └──────┬───────┘
                            ▼
                     ┌──────────────┐
                     │  Tool trait  │  shell / fs / http / git
                     └──────────────┘
```

The core loop never changes:
**send conversation → model replies with text and/or tool calls → for each tool call,
ask the Judge (allow / confirm / deny) → execute allowed calls → append results → loop
until the model is done.**

---

## The three (then four) traits

Everything plugs into these. Getting them right early is most of the architecture.

- `LlmProvider` — generates. `complete(req) -> Response`; later a streaming variant.
- `Tool` — does things. `name()`, `schema()` (JSON schema for the model), `run(input)`.
- `Judge` — adjudicates the risky boundary. `verdict(call, ctx) -> Allow | Confirm | Deny(reason)`.

The agent loop depends only on these traits — never directly on the HTTP client or the TUI.

### Judge progression (free → cloud)

Deciding allow/confirm/deny is a **classification** problem, so it doesn't need a big model.

| Impl              | What                                             | Cost        | Milestone |
|-------------------|--------------------------------------------------|-------------|-----------|
| `RuleJudge`       | hand-written rules (deny `rm -rf`, confirm writes) | free, instant | M1        |
| `LocalModelJudge` | tiny instruct model via Ollama → ALLOW/CONFIRM/DENY | free, offline | M2–M3     |
| `JevJudge`        | TypeSafe's cloud decision model                  | ~pennies     | M4 (opt.) |

Develop entirely against `RuleJudge` for free; swap in the others only to compare behavior.

---

## Crate shortlist (the Rust syllabus)

| Concern            | Crate                         | Introduced |
|--------------------|-------------------------------|-----------|
| JSON               | `serde`, `serde_json`         | M0        |
| HTTP               | `reqwest` (blocking first)    | M0        |
| Errors             | `thiserror` (lib) + `anyhow` (app) | M0   |
| CLI args           | `clap` (derive)               | M0        |
| Async runtime      | `tokio`                       | Async MS  |
| Async traits       | `async-trait` (or native)     | M1/async  |
| TUI                | `ratatui` + `crossterm`       | M3        |
| Config / secrets   | `dotenvy` + hand-rolled config | M0–M4    |

---

## Milestones

### M0 — Skeleton that talks  *(synchronous, no async yet)*
- [x] `cargo new`, project layout, `clap` one-shot CLI (`agent "prompt"`)
- [x] `LlmProvider` trait + Anthropic implementation (blocking `reqwest`)
- [x] Ollama implementation (local, free) behind the same trait — `--provider` flag
- [x] Runtime provider selection via `Box<dyn LlmProvider>` (trait objects, pulled forward from M1)
- [x] Request/Response structs with `serde`
- [x] Read API key from env / `.env` (only when Anthropic is selected)
- [x] **Goal:** see a real model reply printed in the terminal ✅ (via local Ollama llama3.1)
- Concepts: structs, enums, `match`, `Option`/`Result`, `?`, basic ownership

### M1 — The agent loop + first tool + the Judge
- [ ] `Tool` trait + `run_shell_command` only
- [ ] Full round-trip: model requests a tool → execute → feed result back → loop
- [ ] `Judge` trait + `RuleJudge` stub wired into the loop (the permission layer)
- [ ] Tool registry (`Box<dyn Tool>`)
- [ ] **Goal:** the model completes a multi-step task using the shell, gated by the Judge
- Concepts: traits, trait objects, dynamic dispatch, the borrow checker on history

### Async milestone — convert to tokio
- [ ] Move `LlmProvider`/`Tool`/`Judge` to async
- [ ] Switch to async `reqwest`
- [ ] **Goal:** understand *why* async exists, having felt the blocking version
- Concepts: `async`/`await`, `tokio` tasks, futures

### M2 — Broaden the tools
- [ ] `read_file`, `write_file`, `edit_file`, `list_dir`
- [ ] `http_fetch`
- [ ] `git` (status/diff/commit)
- [ ] (optional) `LocalModelJudge` via Ollama
- [ ] **Goal:** genuinely useful; trait design makes each tool ~30 lines
- Concepts: `std::fs`, `std::process`, repeated patterns

### M3 — The TUI
- [ ] Replace CLI frontend with `ratatui`: scrollback, input box, spinner on tool calls
- [ ] Tool-approval prompt in the UI (Judge's `Confirm` verdict)
- [ ] Channels between the UI task and the agent task
- [ ] **Goal:** agent loop underneath is untouched; only the frontend changed
- Concepts: event loops, `Arc`/`Mutex`, channels

### M4 — Polish
- [ ] Streaming responses
- [ ] Conversation persistence (save/resume sessions)
- [ ] Config file
- [ ] Provider #2 behind `LlmProvider` (proves the abstraction)
- [ ] `JevJudge` behind `Judge` (proves the abstraction) — verify TypeSafe pricing/access first
- [ ] **Goal:** both abstractions proven with a second real implementation

---

## Rust concept ramp (one scary thing per stage)

1. structs, enums, `match`, `Option`/`Result`, `?`  — M0
2. ownership & borrowing (conversation history is the first fight) — M0–M1
3. traits & `Box<dyn Trait>` — M1
4. `async`/`await`, tokio tasks — async milestone
5. `Arc`/`Mutex` + channels — M3

---

## Cost notes

- **Anthropic API:** pay per token; needed from M0. Keep prompts small while testing.
- **Judge:** free by default (`RuleJudge`, then a local Ollama model). Never required to pay.
- **Jev (optional, M4):** no free tier, but very cheap (~$0.042 / M input tokens, output free
  per third-party reports; ~$0.20 for a few thousand decisions). **Verify on TypeSafe's own
  console before creating a key** — sources disagreed on version and access flow.

---

## Open questions / decisions to revisit

- [x] First concrete `LlmProvider`s: Anthropic + Ollama (both shipped at M0; default is Ollama)
- [ ] Which tiny local model for `LocalModelJudge` (pick at M2)
- [ ] Whether a purpose-built open decision model exists worth using instead of a general tiny model
