//! The full-screen TUI frontend (ratatui). It implements the same `Ui` trait
//! as `CliUi`, so the agent loop drives it without knowing it's a TUI at all.
//! Layout: a scrollable conversation/log pane, an input box, and a one-line
//! metrics/status footer.

use std::io;

use anyhow::Result;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::{
    event::{self, Event as CEvent, KeyCode, KeyEventKind, KeyModifiers},
    execute,
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
};
use ratatui::layout::{Constraint, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::ui::{Event, Ui};

type Term = Terminal<CrosstermBackend<io::Stdout>>;

/// What `read_prompt` decided the user wants next.
pub enum Action {
    Quit,
    Clear,
    Prompt(String),
}

pub struct TuiUi {
    terminal: Term,
    lines: Vec<Line<'static>>,
    input: String,
    status: String,
}

impl TuiUi {
    pub fn new() -> Result<Self> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;

        let mut ui = Self {
            terminal,
            lines: Vec::new(),
            input: String::new(),
            status: "type a prompt · Enter to send · 'clear' resets · Esc quits".to_string(),
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
        self.push_line(Line::styled(
            "(conversation cleared)",
            Style::new().fg(Color::DarkGray),
        ));
    }

    pub fn push_error(&mut self, message: String) {
        self.push_line(Line::styled(message, Style::new().fg(Color::Red)));
        let _ = self.draw();
    }

    /// Input loop: draw, read a key, update the input buffer, until the user
    /// submits a line (or asks to quit / clear).
    pub fn read_prompt(&mut self) -> Result<Action> {
        loop {
            self.draw()?;

            let CEvent::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            // Ctrl-C / Ctrl-D quit.
            if key.modifiers.contains(KeyModifiers::CONTROL)
                && matches!(key.code, KeyCode::Char('c') | KeyCode::Char('d'))
            {
                return Ok(Action::Quit);
            }

            match key.code {
                KeyCode::Esc => return Ok(Action::Quit),
                KeyCode::Backspace => {
                    self.input.pop();
                }
                KeyCode::Char(c) => self.input.push(c),
                KeyCode::Enter => {
                    let prompt = self.input.trim().to_string();
                    self.input.clear();
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
    }

    fn push_line(&mut self, line: Line<'static>) {
        self.lines.push(line);
    }

    fn draw(&mut self) -> Result<()> {
        // Destructuring gives disjoint borrows: `terminal` (&mut) and the three
        // render inputs (&) at once, which a single `self.` chain would not.
        let Self {
            terminal,
            lines,
            input,
            status,
        } = self;
        terminal.draw(|frame| render(frame, lines, input, status))?;
        Ok(())
    }
}

impl Ui for TuiUi {
    fn emit(&mut self, event: Event) {
        match event {
            Event::User(text) => self.push_line(Line::from(vec![
                Span::styled("» ", Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                Span::raw(text),
            ])),
            Event::Assistant(text) => {
                if text.is_empty() {
                    self.push_line(Line::raw(""));
                } else {
                    for l in text.lines() {
                        self.push_line(Line::raw(l.to_string()));
                    }
                }
            }
            Event::ToolCall { name, args } => self.push_line(Line::styled(
                format!("↳ {name}({args})"),
                Style::new().fg(Color::DarkGray),
            )),
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
        // Redraw so logs appear as the turn progresses.
        let _ = self.draw();
    }

    fn confirm(&mut self, name: &str, args: &str) -> Result<bool> {
        self.status = format!("⚠ allow {name}({args})?   [y]es / [n]o");
        let _ = self.draw();

        loop {
            if let CEvent::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        self.status = "allowed".to_string();
                        return Ok(true);
                    }
                    KeyCode::Char('n') | KeyCode::Char('N') | KeyCode::Esc => {
                        self.status = "declined".to_string();
                        return Ok(false);
                    }
                    _ => {}
                }
            }
        }
    }
}

/// Restore the terminal no matter how we leave (normal return, `?`, or panic).
impl Drop for TuiUi {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
        let _ = execute!(self.terminal.backend_mut(), LeaveAlternateScreen);
        let _ = self.terminal.show_cursor();
    }
}

/// Draw the three panes. A free function so the closure borrows only the data,
/// not the whole `TuiUi`.
fn render(frame: &mut Frame, lines: &[Line<'static>], input: &str, status: &str) {
    let [log_area, input_area, footer_area] = Layout::vertical([
        Constraint::Min(1),
        Constraint::Length(3),
        Constraint::Length(1),
    ])
    .areas(frame.area());

    // Log pane, auto-scrolled so the newest lines stay visible. (Scroll is
    // counted in logical lines, so very long wrapped lines are approximate.)
    let visible = log_area.height.saturating_sub(2);
    let scroll = (lines.len() as u16).saturating_sub(visible);
    let log = Paragraph::new(Text::from(lines.to_vec()))
        .block(Block::bordered().title(" conversation "))
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(log, log_area);

    let prompt = Paragraph::new(format!("» {input}")).block(Block::bordered().title(" prompt "));
    frame.render_widget(prompt, input_area);

    let footer = Paragraph::new(status.to_string()).style(Style::new().fg(Color::DarkGray));
    frame.render_widget(footer, footer_area);
}
