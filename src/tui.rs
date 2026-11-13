//! The full-screen TUI frontend (ratatui).
//!
//! The important idea here is concurrency. A turn can take seconds (the model is
//! thinking), and we want the screen to stay alive the whole time — a spinner
//! ticking, tokens appearing live. So the agent runs on a background tokio task
//! and talks to the UI through a **channel**:
//!
//! ```text
//!   agent task  ──ToUi events──▶  UI loop (this file, on the main task)
//!              ◀──confirm reply──
//! ```
//!
//! `ChannelUi` is the agent's end — it implements the same `Ui` trait as
//! `CliUi`, but instead of printing it just sends messages down the channel.
//! `TuiUi` is the screen's end: it owns the terminal, drains the channel,
//! animates the spinner, streams the reply, and answers confirmations.
//!
//! Beyond that, this file carries the UI's own state — scroll position, the
//! input line and its cursor, prompt history, a help overlay — all modelled as
//! plain data and redrawn from scratch each frame.

use std::io;
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use anyhow::Result;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::{
    event::{
        self, DisableMouseCapture, EnableMouseCapture, Event as CEvent, KeyCode, KeyEventKind,
        KeyModifiers, MouseEventKind,
    },
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Clear, Paragraph, Wrap};
use ratatui::{Frame, Terminal};
use tokio::task::JoinHandle;

use crate::agent::TurnStats;
use crate::ui::{Event, Ui};

type Term = Terminal<CrosstermBackend<io::Stdout>>;

/// The spinner frames, cycled while a turn is running.
const SPINNER: &[char] = &['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// How often the UI loop wakes to re-animate while waiting (milliseconds).
const TICK_MS: u64 = 80;

/// Lines scrolled per mouse-wheel notch.
const WHEEL_LINES: i32 = 3;

/// What `read_prompt` decided the user wants next.
pub enum Action {
    Quit,
    Clear,
    Prompt(String),
}

/// A message from the agent task to the UI loop.
pub enum ToUi {
    /// Something to show in the log (user echo, token, tool call, result, …).
    Event(Event),
    /// A tool call needs human approval. The UI sends the yes/no back on `reply`.
    Confirm {
        name: String,
        args: String,
        reply: Sender<bool>,
    },
    /// The turn finished: stats on success, a message string on error.
    Done(Result<TurnStats, String>),
}

/// The agent's view of the frontend: it just forwards everything down a channel.
/// Because it's only a `Sender`, it's cheap to move onto the background task.
pub struct ChannelUi {
    tx: Sender<ToUi>,
}

impl ChannelUi {
    pub fn new(tx: Sender<ToUi>) -> Self {
        Self { tx }
    }
}

impl Ui for ChannelUi {
    fn emit(&mut self, event: Event) {
        // If the UI loop is gone there's nothing to show and nothing to do.
        let _ = self.tx.send(ToUi::Event(event));
    }

    fn confirm(&mut self, name: &str, args: &str) -> Result<bool> {
        // Make a one-shot reply channel, hand its sender to the UI, then block
        // this turn until the UI collects a yes/no. The agent has nothing else
        // to do until the human answers anyway.
        let (reply_tx, reply_rx) = mpsc::channel();
        self.tx
            .send(ToUi::Confirm {
                name: name.to_string(),
                args: args.to_string(),
                reply: reply_tx,
            })
            .map_err(|_| anyhow::anyhow!("the UI closed while asking for confirmation"))?;
        reply_rx
            .recv()
            .map_err(|_| anyhow::anyhow!("the UI closed while asking for confirmation"))
    }
}

/// Which "mode" the screen is drawn in — decides the prompt box and footer text.
enum View<'a> {
    /// Waiting for the user to type. Prompt box shows their input + cursor.
    Idle,
    /// A turn is running. Prompt box shows the spinner, activity, and timer.
    Running {
        frame: char,
        activity: &'a str,
        elapsed: u64,
    },
    /// Asking to approve a tool call. Footer shows the question.
    Confirm(&'a str, &'a str),
}

pub struct TuiUi {
    terminal: Term,
    /// The working directory, shown in the footer so it's always visible.
    dir: String,
    /// The model name, shown alongside the directory.
    model: String,
    /// Committed log lines.
    lines: Vec<Line<'static>>,
    /// The reply currently streaming in (not yet committed to `lines`).
    stream: String,
    /// The input line, as chars so the cursor can sit on any boundary safely.
    input: Vec<char>,
    /// Cursor position within `input`, as a char index.
    cursor: usize,
    /// Previously submitted prompts, for ↑/↓ recall.
    history: Vec<String>,
    /// Where the history is persisted between runs (if anywhere).
    history_path: Option<PathBuf>,
    /// Where we are when browsing history (None = editing a fresh line).
    hist_idx: Option<usize>,
    status: String,
    /// Auto-scroll to the bottom as new lines arrive?
    follow: bool,
    /// First visible log line when *not* following (absolute index).
    top: u16,
    /// Is the help overlay showing?
    show_help: bool,
}

impl TuiUi {
    pub fn new(dir: String, model: String, history_path: Option<PathBuf>) -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;

        // Load prior prompt history, if a file is set and readable.
        let history = history_path
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .map(|text| text.lines().map(str::to_string).collect())
            .unwrap_or_default();

        let mut ui = Self {
            terminal,
            dir,
            model,
            lines: Vec::new(),
            stream: String::new(),
            input: Vec::new(),
            cursor: 0,
            history,
            history_path,
            hist_idx: None,
            status: "type a prompt · Enter to send · '?' for keys · Esc quits".to_string(),
            follow: true,
            top: 0,
            show_help: false,
        };
        ui.push_line(Line::styled(
            "local-agent TUI",
            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
        ));
        Ok(ui)
    }

    pub fn set_status(&mut self, status: String) {
        self.status = status;
    }

    pub fn clear_log(&mut self) {
        self.lines.clear();
        self.stream.clear();
        self.follow = true;
        self.push_line(Line::styled(
            "(conversation cleared)",
            Style::new().fg(Color::DarkGray),
        ));
    }

    pub fn push_error(&mut self, message: String) {
        self.push_line(Line::styled(message, Style::new().fg(Color::Red)));
    }

    /// Idle input loop: draw, read a key, until the user submits a line (or asks
    /// to quit / clear). Blocks on each event — when idle there's nothing to
    /// animate, so there's no reason to spin.
    pub fn read_prompt(&mut self) -> Result<Action> {
        loop {
            self.draw(View::Idle)?;

            match event::read()? {
                CEvent::Mouse(m) => {
                    self.handle_mouse(m.kind);
                    continue;
                }
                CEvent::Key(key) => {
                    if key.kind != KeyEventKind::Press {
                        continue;
                    }

                    // Ctrl-C / Ctrl-D quit.
                    if key.modifiers.contains(KeyModifiers::CONTROL)
                        && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
                    {
                        return Ok(Action::Quit);
                    }

                    // Scroll keys work here too.
                    if self.handle_scroll_key(key.code) {
                        continue;
                    }

                    match key.code {
                        KeyCode::Esc => {
                            if self.show_help {
                                self.show_help = false;
                            } else {
                                return Ok(Action::Quit);
                            }
                        }
                        // '?' opens help, but only on an empty line — otherwise
                        // it's just a character you're typing.
                        KeyCode::Char('?') if self.input.is_empty() => {
                            self.show_help = !self.show_help;
                        }
                        KeyCode::Char(c) => self.insert_char(c),
                        KeyCode::Backspace => self.backspace(),
                        KeyCode::Delete => self.delete(),
                        KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
                        KeyCode::Right => self.cursor = (self.cursor + 1).min(self.input.len()),
                        KeyCode::Home => self.cursor = 0,
                        KeyCode::End => self.cursor = self.input.len(),
                        KeyCode::Up => self.history_prev(),
                        KeyCode::Down => self.history_next(),
                        KeyCode::Enter => {
                            let prompt: String = self.input.iter().collect();
                            let prompt = prompt.trim().to_string();
                            self.push_history(&prompt);
                            self.clear_input();
                            // A new turn: snap back to the bottom to watch it.
                            self.follow = true;
                            self.show_help = false;
                            match prompt.as_str() {
                                "" => continue,
                                "exit" | "quit" => return Ok(Action::Quit),
                                "clear" => return Ok(Action::Clear),
                                _ => return Ok(Action::Prompt(prompt)),
                            }
                        }
                        _ => {}
                    }
                }
                _ => {}
            }
        }
    }

    /// Drive the screen while a turn runs on `handle` in the background. Unlike
    /// `read_prompt`, this never blocks on input: it polls with a short timeout
    /// so the spinner keeps ticking and streamed tokens appear as they arrive.
    /// Returns `false` if the user interrupted the turn with Ctrl-C.
    pub fn run_turn(&mut self, rx: &Receiver<ToUi>, handle: &JoinHandle<()>) -> Result<bool> {
        let start = Instant::now();
        let mut activity = String::from("thinking…");
        loop {
            // 1. Drain whatever the agent has sent since last time.
            loop {
                match rx.try_recv() {
                    Ok(ToUi::Event(e)) => {
                        // Track what the agent is doing, for the status line.
                        match &e {
                            Event::ToolCall { name, .. } => activity = format!("running {name}…"),
                            Event::ToolResult(_) | Event::ToolDenied(_) => {
                                activity = "thinking…".to_string()
                            }
                            _ => {}
                        }
                        self.push_event(e);
                    }
                    Ok(ToUi::Confirm { name, args, reply }) => {
                        let ok = self.ask_confirm(&name, &args)?;
                        let _ = reply.send(ok);
                    }
                    Ok(ToUi::Done(result)) => {
                        match result {
                            Ok(stats) => self.status = stats.summary(),
                            Err(e) => self.push_error(format!("error: {e}")),
                        }
                        self.draw(View::Idle)?;
                        return Ok(true);
                    }
                    Err(mpsc::TryRecvError::Empty) => break,
                    // The task vanished without a Done — treat it as finished.
                    Err(mpsc::TryRecvError::Disconnected) => return Ok(true),
                }
            }

            // 2. Animate. Deriving the frame from elapsed time keeps it smooth
            //    no matter how the loop's timing wobbles.
            let elapsed = start.elapsed();
            let frame =
                SPINNER[(elapsed.as_millis() / TICK_MS as u128) as usize % SPINNER.len()];
            self.draw(View::Running {
                frame,
                activity: &activity,
                elapsed: elapsed.as_secs(),
            })?;

            // 3. Wait up to one tick for input: Ctrl-C to interrupt, or scrolling.
            if event::poll(Duration::from_millis(TICK_MS))? {
                match event::read()? {
                    CEvent::Mouse(m) => self.handle_mouse(m.kind),
                    CEvent::Key(key) if key.kind == KeyEventKind::Press => {
                        if key.modifiers.contains(KeyModifiers::CONTROL)
                            && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
                        {
                            // Abort the task. State lives in the shared
                            // Arc<Mutex<Agent>> back in main, so the agent itself
                            // survives the interrupt.
                            handle.abort();
                            self.push_line(Line::styled(
                                "(interrupted)",
                                Style::new().fg(Color::Red),
                            ));
                            self.status = "interrupted".to_string();
                            self.draw(View::Idle)?;
                            return Ok(false);
                        }
                        self.handle_scroll_key(key.code);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Ask for approval mid-turn and block until the user presses y/n. The turn
    /// is paused waiting on us, so blocking here is exactly right.
    fn ask_confirm(&mut self, name: &str, args: &str) -> Result<bool> {
        self.draw(View::Confirm(name, args))?;
        loop {
            match event::read()? {
                CEvent::Mouse(m) => {
                    self.handle_mouse(m.kind);
                    self.draw(View::Confirm(name, args))?;
                }
                CEvent::Key(key) if key.kind == KeyEventKind::Press => match key.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        self.set_status("allowed".to_string());
                        return Ok(true);
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                        self.set_status("declined".to_string());
                        return Ok(false);
                    }
                    _ => {}
                },
                _ => {}
            }
        }
    }

    // ---- input editing ----

    fn insert_char(&mut self, c: char) {
        self.input.insert(self.cursor, c);
        self.cursor += 1;
        self.hist_idx = None;
    }

    fn backspace(&mut self) {
        if self.cursor > 0 {
            self.input.remove(self.cursor - 1);
            self.cursor -= 1;
            self.hist_idx = None;
        }
    }

    fn delete(&mut self) {
        if self.cursor < self.input.len() {
            self.input.remove(self.cursor);
            self.hist_idx = None;
        }
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.cursor = 0;
        self.hist_idx = None;
    }

    fn set_input(&mut self, s: &str) {
        self.input = s.chars().collect();
        self.cursor = self.input.len();
    }

    // ---- prompt history ----

    fn push_history(&mut self, prompt: &str) {
        if !prompt.is_empty() && self.history.last().map(String::as_str) != Some(prompt) {
            self.history.push(prompt.to_string());
        }
    }

    /// ↑ : step toward older prompts.
    fn history_prev(&mut self) {
        if self.history.is_empty() {
            return;
        }
        let idx = match self.hist_idx {
            None => self.history.len() - 1,
            Some(i) => i.saturating_sub(1),
        };
        self.hist_idx = Some(idx);
        let entry = self.history[idx].clone();
        self.set_input(&entry);
    }

    /// ↓ : step toward newer prompts, then back to a blank fresh line.
    fn history_next(&mut self) {
        match self.hist_idx {
            Some(i) if i + 1 < self.history.len() => {
                self.hist_idx = Some(i + 1);
                let entry = self.history[i + 1].clone();
                self.set_input(&entry);
            }
            Some(_) => {
                self.hist_idx = None;
                self.clear_input();
            }
            None => {}
        }
    }

    // ---- scrolling ----

    /// Handle a key that scrolls the log. Returns true if it was one.
    fn handle_scroll_key(&mut self, code: KeyCode) -> bool {
        let page = self.viewport().saturating_sub(1).max(1) as i32;
        match code {
            KeyCode::PageUp => {
                self.scroll(-page);
                true
            }
            KeyCode::PageDown => {
                self.scroll(page);
                true
            }
            _ => false,
        }
    }

    fn handle_mouse(&mut self, kind: MouseEventKind) {
        match kind {
            MouseEventKind::ScrollUp => self.scroll(-WHEEL_LINES),
            MouseEventKind::ScrollDown => self.scroll(WHEEL_LINES),
            _ => {}
        }
    }

    /// Move the log view by `delta` lines (negative = up). Reaching the bottom
    /// re-enables follow mode so new output stays visible.
    fn scroll(&mut self, delta: i32) {
        let max_top = self.max_top();
        let current = if self.follow { max_top } else { self.top.min(max_top) };
        let next = (current as i32 + delta).clamp(0, max_top as i32) as u16;
        if next >= max_top {
            self.follow = true;
        } else {
            self.follow = false;
            self.top = next;
        }
    }

    /// Height of the log pane's interior, in lines (whole height − input − footer
    /// − the log's own top/bottom border).
    fn viewport(&self) -> u16 {
        self.terminal
            .size()
            .map(|s| s.height.saturating_sub(6))
            .unwrap_or(10)
    }

    /// The number of *rendered* rows the log occupies — i.e. counting wrapping,
    /// not just logical lines. This is what the scroll offset is measured in, so
    /// a screen of long wrapped lines scrolls correctly. Word-wrapping can break
    /// a line a little earlier than a hard character split, so this is a close
    /// estimate rather than exact, which is fine for positioning the view.
    fn content_rows(&self) -> usize {
        let width = self
            .terminal
            .size()
            .map(|s| s.width.saturating_sub(2)) // minus the log's left/right border
            .unwrap_or(80)
            .max(1) as usize;
        // How many rows one logical line of display-width `w` wraps to.
        let rows_for = |w: usize| w.max(1).div_ceil(width);

        let mut rows = 0;
        for line in &self.lines {
            rows += rows_for(line.width());
        }
        for l in self.stream.lines() {
            rows += rows_for(l.chars().count());
        }
        rows
    }

    fn max_top(&self) -> u16 {
        (self.content_rows() as u16).saturating_sub(self.viewport())
    }

    // ---- log building ----

    /// Turn one agent `Event` into log content.
    fn push_event(&mut self, event: Event) {
        match event {
            Event::User(text) => self.push_line(Line::from(vec![
                Span::styled("» ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(text),
            ])),
            // A streamed chunk: grow the in-progress reply (shown live by draw).
            Event::Token(tok) => self.stream.push_str(&tok),
            // The reply is complete: commit it as permanent lines.
            Event::AssistantEnd => self.commit_stream(),
            Event::ToolCall { name, args } => {
                // Commit any streamed preamble before the tool line, so order is
                // right even if a chunk arrives late.
                self.commit_stream();
                self.push_line(Line::styled(
                    format!("↳ {name}({args})"),
                    Style::new().fg(Color::DarkGray),
                ));
            }
            Event::ToolResult(result) => {
                let preview: String = result.chars().take(200).collect();
                self.push_line(Line::styled(
                    format!("  → {}", preview.replace('\n', " ")),
                    Style::new().fg(Color::DarkGray),
                ));
            }
            Event::ToolDenied(reason) => self.push_line(Line::styled(
                format!("  ✗ denied: {reason}"),
                Style::new().fg(Color::Red),
            )),
        }
    }

    /// Move the streaming buffer into the permanent log, split into lines.
    fn commit_stream(&mut self) {
        if self.stream.is_empty() {
            return;
        }
        let text = std::mem::take(&mut self.stream);
        for l in text.lines() {
            self.push_line(Line::raw(l.to_string()));
        }
    }

    fn push_line(&mut self, line: Line<'static>) {
        self.lines.push(line);
    }

    // ---- drawing ----

    fn draw(&mut self, view: View) -> Result<()> {
        let location = format!("{} · {}", self.model, self.dir);
        let (prompt_text, footer_text) = match view {
            View::Idle => (
                format!("» {}", self.input.iter().collect::<String>()),
                format!("{location}   {}", self.status),
            ),
            View::Running {
                frame,
                activity,
                elapsed,
            } => (
                format!("{frame} {activity}  {elapsed}s  (Ctrl-C to interrupt)"),
                location,
            ),
            View::Confirm(name, args) => (
                format!("» {}", self.input.iter().collect::<String>()),
                format!("⚠ allow {name}({args})?   [y]es / [n]o"),
            ),
        };

        // Only show a cursor (and let it be placed) while the user is typing.
        let cursor = matches!(view, View::Idle).then_some(self.cursor as u16);

        // The log shows committed lines plus the still-streaming reply.
        let mut display: Vec<Line<'static>> = self.lines.clone();
        for l in self.stream.lines() {
            display.push(Line::raw(l.to_string()));
        }

        let max_top = self.max_top();
        let scroll_top = if self.follow { max_top } else { self.top.min(max_top) };
        let show_help = self.show_help;

        // Destructure so the closure can hold `&mut terminal` without borrowing
        // the owned locals we just built.
        let Self { terminal, .. } = self;
        terminal.draw(|frame| {
            render(
                frame,
                &display,
                scroll_top,
                &prompt_text,
                &footer_text,
                cursor,
                show_help,
            )
        })?;
        Ok(())
    }
}

/// Restore the terminal no matter how we leave (normal return, `?`, or panic),
/// and persist the prompt history. Both are best-effort.
impl Drop for TuiUi {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(
            self.terminal.backend_mut(),
            DisableMouseCapture,
            LeaveAlternateScreen
        );
        let _ = self.terminal.show_cursor();

        if let Some(path) = &self.history_path {
            // Keep the last 500 prompts so the file can't grow without bound.
            let start = self.history.len().saturating_sub(500);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            let _ = std::fs::write(path, self.history[start..].join("\n"));
        }
    }
}

/// Draw the three panes (and, if open, the help overlay). A free function so the
/// closure borrows only the data, not the whole `TuiUi`.
fn render(
    frame: &mut Frame,
    display: &[Line<'static>],
    scroll_top: u16,
    prompt_text: &str,
    footer_text: &str,
    cursor: Option<u16>,
    show_help: bool,
) {
    let [log_area, input_area, footer_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    let log = Paragraph::new(Text::from(display.to_vec()))
        .block(Block::bordered().title(" conversation "))
        .wrap(Wrap { trim: false })
        .scroll((scroll_top, 0));
    frame.render_widget(log, log_area);

    let prompt =
        Paragraph::new(prompt_text.to_string()).block(Block::bordered().title(" prompt "));
    frame.render_widget(prompt, input_area);

    let footer = Paragraph::new(footer_text.to_string()).style(Style::new().fg(Color::DarkGray));
    frame.render_widget(footer, footer_area);

    // Place a real terminal cursor after the "» " prefix, clamped to the box.
    if let Some(col) = cursor {
        let x = input_area.x + 3 + col;
        let max_x = input_area.x + input_area.width.saturating_sub(2);
        frame.set_cursor_position((x.min(max_x), input_area.y + 1));
    }

    if show_help {
        let area = centered_rect(52, 12, frame.area());
        frame.render_widget(Clear, area);
        let help = Paragraph::new(help_text()).block(Block::bordered().title(" keys "));
        frame.render_widget(help, area);
    }
}

/// A rectangle of the given size, centered in `area`.
fn centered_rect(width: u16, height: u16, area: Rect) -> Rect {
    let x = area.x + area.width.saturating_sub(width) / 2;
    let y = area.y + area.height.saturating_sub(height) / 2;
    Rect {
        x,
        y,
        width: width.min(area.width),
        height: height.min(area.height),
    }
}

fn help_text() -> Text<'static> {
    let row = |keys: &str, what: &str| {
        Line::from(vec![
            Span::styled(format!("  {keys:<14}"), Style::new().fg(Color::Cyan)),
            Span::raw(what.to_string()),
        ])
    };
    Text::from(vec![
        row("Enter", "send the prompt"),
        row("↑ / ↓", "previous / next prompt"),
        row("← → Home End", "move the cursor"),
        row("PgUp / PgDn", "scroll the log"),
        row("mouse wheel", "scroll the log"),
        row("clear", "reset the conversation"),
        row("?", "toggle this help"),
        row("Ctrl-C", "interrupt / quit"),
        row("Esc", "close help / quit"),
    ])
}
