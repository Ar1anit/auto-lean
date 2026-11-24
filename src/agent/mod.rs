//! The proof-search loop: generate → guard → verify → repair.
//!
//! [`Solver::solve`] takes a Lean file with `sorry` holes and iterates:
//!
//! 1. ask the model for the completed file,
//! 2. pull the Lean code out of its reply ([`parse_reply`]),
//! 3. run the anti-cheat guard; a cheating answer is sent back with a strict
//!    correction and **never reaches Lean**,
//! 4. compile with Lean; on failure the exact diagnostics go back to the model,
//! 5. stop as soon as Lean exits with code 0 for a candidate the guard accepted
//!    (and, unless disabled, the axiom audit comes back clean).

pub mod llm;
pub mod prompts;

use std::collections::BTreeSet;
use std::time::{Duration, Instant};

use thiserror::Error;

use crate::lean::{
    AuditOutcome, GuardReport, LeanChecker, LeanDiagnostic, LeanError, LeanOutput, Spec, SpecError,
    TaintedTheorem, Target, Violation,
};
use llm::{LlmProvider, Message};

/// Default number of attempts the model gets.
pub const DEFAULT_MAX_ITERS: usize = 5;

/// Longest `NOTE:` line kept from a reply.
const MAX_NOTE_CHARS: usize = 600;

/// Tuning knobs for [`Solver`].
#[derive(Debug, Clone)]
pub struct SolveConfig {
    /// How many candidates the model may submit.
    pub max_iters: usize,
    /// Run `#print axioms` on every target once a candidate compiles.
    pub axiom_audit: bool,
    /// Compile the spec itself first, so a file that is broken for reasons no
    /// proof can fix is reported immediately instead of burning model calls.
    pub check_spec: bool,
    /// Toolchain description passed to the model, e.g. `Lean (version 4.36.0)`.
    pub lean_version: Option<String>,
}

impl Default for SolveConfig {
    fn default() -> Self {
        Self {
            max_iters: DEFAULT_MAX_ITERS,
            axiom_audit: true,
            check_spec: true,
            lean_version: None,
        }
    }
}

/// Why a candidate was not accepted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rejection {
    /// The reply contained no Lean file.
    NoCode,
    /// The anti-cheat guard refused the candidate; Lean was not run.
    Guard(Vec<Violation>),
    /// Lean reported errors.
    Compile {
        diagnostics: Vec<LeanDiagnostic>,
        exit_code: Option<i32>,
        /// Lean's raw output, for failures without positioned diagnostics.
        output: String,
    },
    /// Lean had to be killed after this long.
    Timeout(Duration),
    /// Lean accepted the file but warned that a declaration uses `sorry`.
    UsesSorry(Vec<LeanDiagnostic>),
    /// A target depends on axioms beyond the standard three.
    Axioms(Vec<TaintedTheorem>),
    /// Lean could not report on every target after compiling.
    AuditInconclusive(String),
}

impl Rejection {
    /// One line describing the rejection.
    pub fn summary(&self) -> String {
        match self {
            Rejection::NoCode => "the reply contained no Lean code".to_string(),
            Rejection::Guard(violations) => match violations.as_slice() {
                [only] => only.to_string(),
                many => format!("{} guard violations, first: {}", many.len(), many[0]),
            },
            Rejection::Compile {
                diagnostics,
                exit_code,
                ..
            } => {
                let errors = diagnostics.iter().filter(|d| d.is_error()).count();
                match errors {
                    0 => format!("Lean failed with exit code {exit_code:?}"),
                    1 => "1 error".to_string(),
                    n => format!("{n} errors"),
                }
            }
            Rejection::Timeout(elapsed) => format!("timed out after {:.0}s", elapsed.as_secs_f64()),
            Rejection::UsesSorry(_) => "a declaration still uses `sorry`".to_string(),
            Rejection::Axioms(tainted) => format!(
                "non-standard axioms: {}",
                tainted
                    .iter()
                    .map(|theorem| format!("`{}` uses {}", theorem.name, theorem.axioms.join(", ")))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
            Rejection::AuditInconclusive(reason) => format!("axiom audit failed: {reason}"),
        }
    }

    /// The message sent back to the model.
    fn feedback(&self, candidate: &str) -> String {
        match self {
            Rejection::NoCode => prompts::no_code_feedback(),
            Rejection::Guard(violations) => prompts::guard_feedback(violations),
            Rejection::Compile {
                diagnostics,
                exit_code,
                output,
            } => prompts::compile_feedback(candidate, diagnostics, *exit_code, output),
            Rejection::Timeout(elapsed) => prompts::timeout_feedback(*elapsed),
            Rejection::UsesSorry(diagnostics) => prompts::sorry_feedback(diagnostics),
            Rejection::Axioms(tainted) => prompts::axiom_feedback(tainted),
            Rejection::AuditInconclusive(reason) => prompts::audit_inconclusive_feedback(reason),
        }
    }
}

/// One round of the loop.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Attempt {
    /// 1-based.
    pub iteration: usize,
    /// The Lean file extracted from the reply, if any.
    pub candidate: Option<String>,
    /// A `NOTE:` the model added, e.g. a claimed counterexample. Unverified.
    pub note: Option<String>,
    /// `None` means the candidate was verified.
    pub rejection: Option<Rejection>,
    /// Targets (indices into [`SolveReport::targets`]) Lean reported errors in.
    pub failing_targets: Vec<usize>,
    /// Errors that could not be attributed to a target.
    pub unattributed_errors: usize,
}

/// The outcome of [`Solver::solve`].
#[derive(Debug, Clone)]
pub struct SolveReport {
    pub verified: bool,
    /// The verified Lean file.
    pub proof: Option<String>,
    /// The declarations that had to be proven.
    pub targets: Vec<Target>,
    pub attempts: Vec<Attempt>,
    /// Axioms the verified proofs depend on, when the audit ran.
    pub axioms: Option<Vec<String>>,
    /// How many times Lean was invoked.
    pub lean_runs: usize,
}

impl SolveReport {
    /// The attempt that got furthest: among candidates Lean actually compiled,
    /// the one with the fewest failing targets (the latest on a tie); otherwise
    /// the last attempt.
    pub fn best_attempt(&self) -> Option<&Attempt> {
        self.attempts
            .iter()
            .filter(|attempt| matches!(attempt.rejection, Some(Rejection::Compile { .. })))
            .min_by_key(|attempt| {
                let unknown = usize::from(attempt.unattributed_errors > 0) * self.targets.len();
                (
                    attempt.failing_targets.len() + unknown,
                    std::cmp::Reverse(attempt.iteration),
                )
            })
            .or(self.attempts.last())
    }

    /// The most recent `NOTE:` from the model.
    pub fn model_note(&self) -> Option<&str> {
        self.attempts
            .iter()
            .rev()
            .find_map(|attempt| attempt.note.as_deref())
    }
}

/// Progress notifications emitted while solving.
#[derive(Debug)]
pub enum SolveEvent<'a> {
    Started {
        targets: &'a [Target],
        max_iters: usize,
    },
    /// A model request is about to be sent.
    Generating {
        iteration: usize,
    },
    Generated {
        iteration: usize,
        elapsed: Duration,
    },
    GuardPassed {
        iteration: usize,
    },
    LeanPassed {
        iteration: usize,
        elapsed: Duration,
    },
    AuditPassed {
        iteration: usize,
        axioms: &'a [String],
    },
    /// The candidate was rejected; `lean_time` is set if Lean ran.
    Rejected {
        iteration: usize,
        rejection: &'a Rejection,
        failing_targets: &'a [usize],
        lean_time: Option<Duration>,
    },
    Verified {
        iteration: usize,
    },
}

/// Receives [`SolveEvent`]s. Implemented for closures.
pub trait Observer {
    fn on_event(&mut self, event: &SolveEvent<'_>);
}

impl<F: FnMut(&SolveEvent<'_>)> Observer for F {
    fn on_event(&mut self, event: &SolveEvent<'_>) {
        self(event);
    }
}

/// An [`Observer`] that ignores everything.
#[derive(Debug, Clone, Copy, Default)]
pub struct Silent;

impl Observer for Silent {
    fn on_event(&mut self, _event: &SolveEvent<'_>) {}
}

/// Errors that end a solve before a verdict is reached. Running out of attempts
/// is not one of them; that is a [`SolveReport`] with `verified == false`.
#[derive(Debug, Error)]
pub enum SolveError {
    #[error(transparent)]
    Spec(#[from] SpecError),

    #[error(
        "nothing to prove: the file contains no `sorry` (use `auto-lean verify` to check a finished file)"
    )]
    NothingToProve,

    #[error("the input does not compile on its own, before any proof attempt:\n{0}")]
    BrokenSpec(String),

    #[error(transparent)]
    Lean(#[from] LeanError),

    #[error("the language model request failed: {0:#}")]
    Llm(anyhow::Error),
}

/// What a model reply contained.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reply {
    /// The Lean file, if the reply had one.
    pub code: Option<String>,
    /// A `NOTE:` line outside the code, e.g. a claimed counterexample.
    pub note: Option<String>,
}

/// Extracts the Lean file (and an optional `NOTE:`) from a model reply.
///
/// The file is taken from a Markdown code fence when there is one: the last
/// block tagged `lean`, otherwise the last untagged block. Fences inside the
/// block, as found in Lean doc comments, are kept as content. A reply without
/// any fence is used as is if it looks like Lean.
pub fn parse_reply(reply: &str) -> Reply {
    struct Open<'a> {
        fence: usize,
        language: String,
        /// Fence lengths of code blocks opened inside this one.
        nested: Vec<usize>,
        lines: Vec<&'a str>,
    }

    let text = strip_reasoning(reply);
    let mut blocks: Vec<(String, String)> = Vec::new();
    let mut prose: Vec<&str> = Vec::new();
    let mut open: Option<Open<'_>> = None;

    for line in text.lines() {
        let trimmed = line.trim_start();
        let ticks = trimmed.chars().take_while(|&c| c == '`').count();
        let info = trimmed[ticks..].trim();
        let is_fence = ticks >= 3 && !info.contains('`');

        match open.as_mut() {
            None if is_fence => {
                open = Some(Open {
                    fence: ticks,
                    language: info
                        .split_whitespace()
                        .next()
                        .unwrap_or("")
                        .to_ascii_lowercase(),
                    nested: Vec::new(),
                    lines: Vec::new(),
                });
            }
            None => prose.push(line),
            Some(block) => {
                let closes_nested = block.nested.last().is_some_and(|&inner| ticks >= inner);
                if is_fence && !info.is_empty() {
                    // A tagged fence inside the block opens a nested one.
                    block.nested.push(ticks);
                    block.lines.push(line);
                } else if is_fence && closes_nested {
                    block.nested.pop();
                    block.lines.push(line);
                } else if is_fence && block.nested.is_empty() && ticks >= block.fence {
                    if let Some(done) = open.take() {
                        blocks.push((done.language, done.lines.join("\n")));
                    }
                } else {
                    block.lines.push(line);
                }
            }
        }
    }
    // A reply cut off mid-block still carries a candidate.
    if let Some(unterminated) = open {
        blocks.push((unterminated.language, unterminated.lines.join("\n")));
    }

    let note = prose.iter().rev().find_map(|line| {
        // Tolerate Markdown decoration such as `**NOTE:** ...` or `> NOTE: ...`.
        let line = line.trim().trim_start_matches(['*', '>', '_', ' ']);
        let note = line.strip_prefix("NOTE:")?.trim_matches(['*', '_', ' ']);
        (!note.is_empty()).then(|| note.chars().take(MAX_NOTE_CHARS).collect::<String>())
    });

    let fenced = blocks
        .iter()
        .rev()
        .find(|(language, _)| language == "lean" || language == "lean4")
        .or_else(|| {
            blocks
                .iter()
                .rev()
                .find(|(language, _)| language.is_empty())
        })
        .map(|(_, content)| content.clone());
    let code = match fenced {
        Some(content) => Some(content),
        None if blocks.is_empty() && looks_like_lean(&text) => Some(
            prose
                .iter()
                .filter(|line| !line.trim().starts_with("NOTE:"))
                .copied()
                .collect::<Vec<_>>()
                .join("\n"),
        ),
        None => None,
    };

    Reply {
        code: code
            .map(|code| format!("{}\n", code.trim_end()))
            .filter(|code| !code.trim().is_empty()),
        note,
    }
}

/// Removes `<think>...</think>` sections that local reasoning models emit inline.
fn strip_reasoning(reply: &str) -> String {
    const OPEN: &str = "<think>";
    const CLOSE: &str = "</think>";
    let mut text = reply.to_string();
    while let Some(start) = text.find(OPEN) {
        match text[start..].find(CLOSE) {
            Some(length) => text.replace_range(start..start + length + CLOSE.len(), ""),
            None => text.truncate(start),
        }
    }
    text
}

fn looks_like_lean(text: &str) -> bool {
    ["theorem ", "lemma ", "def ", ":= by"]
        .iter()
        .any(|marker| text.contains(marker))
}

/// Result of checking one candidate.
#[derive(Debug, Default)]
struct Evaluation {
    rejection: Option<Rejection>,
    failing_targets: Vec<usize>,
    unattributed_errors: usize,
    axioms: Option<Vec<String>>,
    lean_runs: usize,
    lean_time: Option<Duration>,
}

impl Evaluation {
    fn rejected(mut self, rejection: Rejection) -> Self {
        self.rejection = Some(rejection);
        self
    }
}

/// Runs the generate → guard → verify → repair loop.
pub struct Solver<'a> {
    llm: &'a dyn LlmProvider,
    lean: &'a dyn LeanChecker,
    config: SolveConfig,
}

impl<'a> Solver<'a> {
    pub fn new(llm: &'a dyn LlmProvider, lean: &'a dyn LeanChecker) -> Self {
        Self {
            llm,
            lean,
            config: SolveConfig::default(),
        }
    }

    pub fn with_config(mut self, config: SolveConfig) -> Self {
        self.config = config;
        self
    }

    /// Tries to prove every `sorry` in `spec_source`.
    pub async fn solve(
        &self,
        spec_source: &str,
        observer: &mut dyn Observer,
    ) -> Result<SolveReport, SolveError> {
        let spec = Spec::parse(spec_source)?;
        let targets = spec.targets();
        if targets.is_empty() {
            return Err(SolveError::NothingToProve);
        }

        let mut lean_runs = 0;
        if self.config.check_spec {
            let output = self.lean.check(spec.source()).await?;
            lean_runs += 1;
            if !output.success() {
                return Err(SolveError::BrokenSpec(describe_failure(&output)));
            }
        }

        observer.on_event(&SolveEvent::Started {
            targets: &targets,
            max_iters: self.config.max_iters,
        });

        let system = prompts::system_prompt(self.config.lean_version.as_deref());
        let mut messages = vec![Message::user(prompts::initial_prompt(
            spec.source(),
            &targets,
        ))];
        let mut attempts = Vec::new();

        for iteration in 1..=self.config.max_iters {
            observer.on_event(&SolveEvent::Generating { iteration });
            let asked = Instant::now();
            let raw = self
                .llm
                .complete(&system, &messages)
                .await
                .map_err(SolveError::Llm)?;
            observer.on_event(&SolveEvent::Generated {
                iteration,
                elapsed: asked.elapsed(),
            });

            let reply = parse_reply(&raw);
            let evaluation = self
                .evaluate(&spec, reply.code.as_deref(), iteration, observer)
                .await?;
            lean_runs += evaluation.lean_runs;

            let Some(rejection) = evaluation.rejection else {
                observer.on_event(&SolveEvent::Verified { iteration });
                attempts.push(Attempt {
                    iteration,
                    candidate: reply.code.clone(),
                    note: reply.note,
                    rejection: None,
                    failing_targets: Vec::new(),
                    unattributed_errors: 0,
                });
                return Ok(SolveReport {
                    verified: true,
                    proof: reply.code,
                    targets,
                    attempts,
                    axioms: evaluation.axioms,
                    lean_runs,
                });
            };

            observer.on_event(&SolveEvent::Rejected {
                iteration,
                rejection: &rejection,
                failing_targets: &evaluation.failing_targets,
                lean_time: evaluation.lean_time,
            });
            // Providers reject empty turns, so keep the transcript well-formed.
            let said = if raw.trim().is_empty() {
                "(empty reply)".to_string()
            } else {
                raw
            };
            messages.push(Message::assistant(said));
            messages.push(Message::user(
                rejection.feedback(reply.code.as_deref().unwrap_or_default()),
            ));
            attempts.push(Attempt {
                iteration,
                candidate: reply.code,
                note: reply.note,
                rejection: Some(rejection),
                failing_targets: evaluation.failing_targets,
                unattributed_errors: evaluation.unattributed_errors,
            });
        }

        Ok(SolveReport {
            verified: false,
            proof: None,
            targets,
            attempts,
            axioms: None,
            lean_runs,
        })
    }

    /// Guard first, then Lean, then the axiom audit.
    async fn evaluate(
        &self,
        spec: &Spec,
        candidate: Option<&str>,
        iteration: usize,
        observer: &mut dyn Observer,
    ) -> Result<Evaluation, SolveError> {
        let mut evaluation = Evaluation::default();
        let Some(candidate) = candidate else {
            return Ok(evaluation.rejected(Rejection::NoCode));
        };

        // A cheating candidate is turned away here and never compiled.
        let guard = spec.check(candidate);
        if !guard.passed() {
            return Ok(evaluation.rejected(Rejection::Guard(guard.violations)));
        }
        observer.on_event(&SolveEvent::GuardPassed { iteration });

        let output = self.lean.check(candidate).await?;
        evaluation.lean_runs += 1;
        evaluation.lean_time = Some(output.duration);
        if output.timed_out {
            return Ok(evaluation.rejected(Rejection::Timeout(output.duration)));
        }
        let diagnostics = output.diagnostics();
        if !output.success() {
            let (failing, unattributed) =
                attribute_errors(&diagnostics, &guard, spec.targets().len());
            evaluation.failing_targets = failing;
            evaluation.unattributed_errors = unattributed;
            return Ok(evaluation.rejected(Rejection::Compile {
                diagnostics,
                exit_code: output.exit_code,
                output: output.combined(),
            }));
        }
        // Belt and braces: the guard bans the token, Lean reports the effect.
        let sorries: Vec<LeanDiagnostic> = diagnostics
            .into_iter()
            .filter(LeanDiagnostic::mentions_sorry)
            .collect();
        if !sorries.is_empty() {
            return Ok(evaluation.rejected(Rejection::UsesSorry(sorries)));
        }
        observer.on_event(&SolveEvent::LeanPassed {
            iteration,
            elapsed: output.duration,
        });

        if self.config.axiom_audit {
            if let Some(audit) = spec.axiom_audit() {
                let audited = format!("{candidate}{}", audit.commands());
                let output = self.lean.check(&audited).await?;
                evaluation.lean_runs += 1;
                let outcome = if output.timed_out {
                    AuditOutcome::Inconclusive("Lean timed out while listing axioms".to_string())
                } else {
                    audit.evaluate(&output.stdout)
                };
                match outcome {
                    AuditOutcome::Clean(axioms) => {
                        observer.on_event(&SolveEvent::AuditPassed {
                            iteration,
                            axioms: &axioms,
                        });
                        evaluation.axioms = Some(axioms);
                    }
                    AuditOutcome::Tainted(tainted) => {
                        return Ok(evaluation.rejected(Rejection::Axioms(tainted)));
                    }
                    AuditOutcome::Inconclusive(reason) => {
                        return Ok(evaluation.rejected(Rejection::AuditInconclusive(reason)));
                    }
                }
            }
        }
        Ok(evaluation)
    }
}

/// Works out which targets Lean's errors fall into. If Lean failed without a
/// single positioned error, every target counts as unproven.
fn attribute_errors(
    diagnostics: &[LeanDiagnostic],
    guard: &GuardReport,
    target_count: usize,
) -> (Vec<usize>, usize) {
    let mut failing = BTreeSet::new();
    let mut unattributed = 0;
    let mut errors = 0;
    for diagnostic in diagnostics.iter().filter(|d| d.is_error()) {
        errors += 1;
        match guard.target_at(diagnostic.line, diagnostic.column) {
            Some(target) => {
                failing.insert(target);
            }
            None => unattributed += 1,
        }
    }
    if errors == 0 {
        failing.extend(0..target_count);
    }
    (failing.into_iter().collect(), unattributed)
}

/// Renders a failed Lean run for an error message.
fn describe_failure(output: &LeanOutput) -> String {
    if output.timed_out {
        return format!("Lean timed out after {:.0}s", output.duration.as_secs_f64());
    }
    let errors: Vec<String> = output
        .diagnostics()
        .iter()
        .filter(|d| d.is_error())
        .map(|d| prompts::clip(&d.to_string(), 12, 1_500))
        .collect();
    if errors.is_empty() {
        let raw = output.combined();
        match raw.trim() {
            "" => format!(
                "Lean exited with {:?} and printed nothing",
                output.exit_code
            ),
            text => prompts::clip(text, 20, 2_000),
        }
    } else {
        errors.join("\n")
    }
}

/// Test doubles shared by this crate's unit tests.
#[cfg(test)]
pub(crate) mod testing {
    use std::collections::VecDeque;
    use std::sync::Mutex;
    use std::time::Duration;

    use async_trait::async_trait;

    use crate::lean::{LeanChecker, LeanError, LeanOutput};

    /// A stand-in for Lean that replays scripted results and records every
    /// source it was asked to compile.
    #[derive(Debug, Default)]
    pub(crate) struct ScriptedLean {
        outputs: Mutex<VecDeque<LeanOutput>>,
        sources: Mutex<Vec<String>>,
    }

    impl ScriptedLean {
        pub(crate) fn new(outputs: impl IntoIterator<Item = LeanOutput>) -> Self {
            Self {
                outputs: Mutex::new(outputs.into_iter().collect()),
                sources: Mutex::new(Vec::new()),
            }
        }

        /// Everything "compiled" so far.
        pub(crate) fn sources(&self) -> Vec<String> {
            self.sources.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl LeanChecker for ScriptedLean {
        async fn check(&self, source: &str) -> Result<LeanOutput, LeanError> {
            self.sources.lock().unwrap().push(source.to_string());
            self.outputs
                .lock()
                .unwrap()
                .pop_front()
                .ok_or(LeanError::Unusable {
                    binary: "scripted lean".to_string(),
                    detail: "no scripted results left".to_string(),
                })
        }
    }

    fn output(stdout: &str, exit_code: Option<i32>, timed_out: bool) -> LeanOutput {
        LeanOutput {
            stdout: stdout.to_string(),
            stderr: String::new(),
            exit_code,
            timed_out,
            duration: Duration::from_millis(250),
        }
    }

    /// Lean exiting with code 0.
    pub(crate) fn accepted(stdout: &str) -> LeanOutput {
        output(stdout, Some(0), false)
    }

    /// Lean exiting with code 1.
    pub(crate) fn rejected(stdout: &str) -> LeanOutput {
        output(stdout, Some(1), false)
    }

    /// Lean killed by the timeout.
    pub(crate) fn hung() -> LeanOutput {
        output("", None, true)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::{ScriptedLean, accepted, hung, rejected};
    use super::*;
    use crate::agent::llm::{MockProvider, Role};
    use crate::lean::ViolationKind;

    const SPEC: &str = "\
def double (n : Nat) : Nat := n + n

theorem double_eq (n : Nat) : double n = 2 * n := by
  sorry
";

    const HEADER: &str = "\
def double (n : Nat) : Nat := n + n

theorem double_eq (n : Nat) : double n = 2 * n := by
";

    fn fenced(proof: &str) -> String {
        format!("```lean\n{HEADER}{proof}\n```")
    }

    const SORRY_WARNING: &str = "Candidate.lean:3:8: warning: declaration uses `sorry`\n";
    const RFL_ERROR: &str = "\
Candidate.lean:4:2: error: Tactic `rfl` failed: The left-hand side
  double n
is not definitionally equal to the right-hand side
  2 * n

n : Nat
⊢ double n = 2 * n
";
    const AXIOMS: &str = "'double_eq' depends on axioms: [propext]\n";

    fn kinds(events: &[String]) -> Vec<&str> {
        events.iter().map(String::as_str).collect()
    }

    async fn run(
        llm: &MockProvider,
        lean: &ScriptedLean,
        config: SolveConfig,
    ) -> (Result<SolveReport, SolveError>, Vec<String>) {
        let mut events = Vec::new();
        let mut observer = |event: &SolveEvent<'_>| {
            events.push(match event {
                SolveEvent::Started { .. } => "started".to_string(),
                SolveEvent::Generating { iteration } => format!("ask {iteration}"),
                SolveEvent::Generated { .. } => "reply".to_string(),
                SolveEvent::GuardPassed { .. } => "guard ok".to_string(),
                SolveEvent::LeanPassed { .. } => "lean ok".to_string(),
                SolveEvent::AuditPassed { .. } => "audit ok".to_string(),
                SolveEvent::Rejected { rejection, .. } => match rejection {
                    Rejection::Guard(_) => "guard rejected".to_string(),
                    Rejection::Compile { .. } => "lean rejected".to_string(),
                    other => format!("rejected: {}", other.summary()),
                },
                SolveEvent::Verified { iteration } => format!("verified {iteration}"),
            });
        };
        let result = Solver::new(llm, lean)
            .with_config(config)
            .solve(SPEC, &mut observer)
            .await;
        (result, events)
    }

    /// The loop end to end: the first answer cheats with `sorry`, the second
    /// fails to compile, the third is verified.
    #[tokio::test]
    async fn repairs_a_cheat_and_a_compile_error_then_verifies() {
        let cheat = fenced("  sorry");
        let wrong = fenced("  rfl");
        let right = format!(
            "Unfolding first makes it linear.\n\n{}",
            fenced("  unfold double\n  omega")
        );
        let llm = MockProvider::new([cheat.clone(), wrong.clone(), right]);
        let lean = ScriptedLean::new([
            accepted(SORRY_WARNING), // the spec itself
            rejected(RFL_ERROR),     // attempt 2
            accepted(""),            // attempt 3
            accepted(AXIOMS),        // axiom audit of attempt 3
        ]);

        let (result, events) = run(&llm, &lean, SolveConfig::default()).await;
        let report = result.unwrap();

        assert!(report.verified);
        assert_eq!(report.attempts.len(), 3);
        let proof = format!("{HEADER}  unfold double\n  omega\n");
        assert_eq!(report.proof.as_deref(), Some(proof.as_str()));
        assert_eq!(report.axioms, Some(vec!["propext".to_string()]));
        assert_eq!(report.targets.len(), 1);

        // Attempt 1 was stopped by the guard ...
        let Some(Rejection::Guard(violations)) = &report.attempts[0].rejection else {
            panic!(
                "attempt 1 should be a guard rejection: {:?}",
                report.attempts[0]
            );
        };
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].kind, ViolationKind::ForbiddenToken);
        assert_eq!(violations[0].line, Some(4));

        // ... and never reached Lean: only the spec, attempts 2 and 3, and the
        // audit were compiled.
        let compiled = lean.sources();
        assert_eq!(compiled.len(), 4);
        assert_eq!(report.lean_runs, 4);
        assert_eq!(compiled[0], SPEC);
        assert_eq!(compiled[1], format!("{HEADER}  rfl\n"));
        assert_eq!(compiled[2], proof);
        assert_eq!(
            compiled[3],
            format!("{proof}\n\n#print axioms _root_.double_eq\n")
        );
        assert!(
            compiled
                .iter()
                .all(|source| source == SPEC || !source.contains("sorry"))
        );

        // Attempt 2 failed in Lean, inside the one target.
        let Some(Rejection::Compile {
            diagnostics,
            exit_code,
            ..
        }) = &report.attempts[1].rejection
        else {
            panic!(
                "attempt 2 should be a compile failure: {:?}",
                report.attempts[1]
            );
        };
        assert_eq!(*exit_code, Some(1));
        assert_eq!((diagnostics[0].line, diagnostics[0].column), (4, 2));
        assert_eq!(report.attempts[1].failing_targets, [0]);
        assert_eq!(report.attempts[2].rejection, None);

        // The conversation grew by one exchange per failure.
        let calls = llm.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(
            calls
                .iter()
                .map(|call| call.messages.len())
                .collect::<Vec<_>>(),
            [1, 3, 5]
        );
        assert!(calls.iter().all(|call| call.system == calls[0].system));
        assert!(calls[0].messages[0].content.contains(SPEC.trim_end()));

        // After the cheat: a strict correction that says Lean was not run.
        let correction = &calls[1].messages[2];
        assert_eq!(calls[1].messages[1], Message::assistant(cheat));
        assert_eq!(correction.role, Role::User);
        assert!(
            correction
                .content
                .starts_with("REJECTED by the anti-cheat guard. Lean was not run")
        );
        assert!(
            correction
                .content
                .contains("line 4, column 2: forbidden token `sorry`")
        );

        // After the compile error: Lean's diagnostic, verbatim, goal state included.
        let repair = &calls[2].messages[4];
        assert_eq!(calls[2].messages[3], Message::assistant(wrong));
        assert!(
            repair
                .content
                .contains("Error at line 4, column 2:\n  4 |   rfl\n"),
            "{}",
            repair.content
        );
        assert!(
            repair
                .content
                .contains(RFL_ERROR.split_once("error: ").unwrap().1.trim_end())
        );

        assert_eq!(
            kinds(&events),
            [
                "started",
                "ask 1",
                "reply",
                "guard rejected",
                "ask 2",
                "reply",
                "guard ok",
                "lean rejected",
                "ask 3",
                "reply",
                "guard ok",
                "lean ok",
                "audit ok",
                "verified 3",
            ]
        );
    }

    #[tokio::test]
    async fn gives_up_after_max_iters_and_reports_the_best_attempt() {
        let llm = MockProvider::new([
            fenced("  rfl"),
            format!(
                "{}\nNOTE: `double` is fine; I just cannot find the lemma.",
                fenced("  simp")
            ),
            fenced("  sorry"),
        ]);
        let lean = ScriptedLean::new([
            rejected(RFL_ERROR),
            rejected("Candidate.lean:4:2: error: simp made no progress\n"),
        ]);
        let config = SolveConfig {
            max_iters: 3,
            check_spec: false,
            ..SolveConfig::default()
        };

        let (result, _) = run(&llm, &lean, config).await;
        let report = result.unwrap();
        assert!(!report.verified);
        assert_eq!(report.proof, None);
        assert_eq!(report.attempts.len(), 3);
        assert_eq!(llm.remaining(), 0);
        assert_eq!(report.lean_runs, 2);

        // The last attempt cheated, so the best one is the latest that compiled.
        assert_eq!(report.best_attempt().unwrap().iteration, 2);
        assert_eq!(
            report.model_note(),
            Some("`double` is fine; I just cannot find the lemma.")
        );
        assert_eq!(
            report.attempts[1].rejection.as_ref().unwrap().summary(),
            "1 error"
        );
    }

    #[tokio::test]
    async fn stops_early_when_the_spec_itself_is_broken() {
        let llm = MockProvider::new(Vec::<String>::new());
        let lean = ScriptedLean::new([rejected(
            "Candidate.lean:1:0: error: unknown module prefix 'Mathlib'\n",
        )]);
        let (result, events) = run(&llm, &lean, SolveConfig::default()).await;

        let error = result.unwrap_err();
        assert!(matches!(error, SolveError::BrokenSpec(_)));
        assert!(
            error
                .to_string()
                .contains("1:0: error: unknown module prefix 'Mathlib'"),
            "{error}"
        );
        assert!(
            llm.calls().is_empty(),
            "no model call for a spec that cannot compile"
        );
        assert!(events.is_empty());
    }

    #[tokio::test]
    async fn refuses_files_without_holes_or_with_forbidden_tokens() {
        let llm = MockProvider::new(Vec::<String>::new());
        let lean = ScriptedLean::default();
        let solver = Solver::new(&llm, &lean);

        let complete = solver
            .solve("theorem t : True := trivial\n", &mut Silent)
            .await;
        assert!(matches!(complete, Err(SolveError::NothingToProve)));

        let axiomatic = solver
            .solve(
                "axiom a : False\ntheorem t : False := by sorry\n",
                &mut Silent,
            )
            .await;
        assert!(matches!(axiomatic, Err(SolveError::Spec(_))));
        assert!(lean.sources().is_empty());
    }

    #[tokio::test]
    async fn feeds_back_timeouts_and_missing_code() {
        let llm = MockProvider::new([
            "I think this needs induction, let me explain...".to_string(),
            fenced("  decide"),
            fenced("  unfold double\n  omega"),
        ]);
        let lean = ScriptedLean::new([hung(), accepted("")]);
        let config = SolveConfig {
            check_spec: false,
            axiom_audit: false,
            ..SolveConfig::default()
        };

        let (result, events) = run(&llm, &lean, config).await;
        let report = result.unwrap();
        assert!(report.verified);
        assert_eq!(report.axioms, None);
        assert_eq!(report.attempts[0].rejection, Some(Rejection::NoCode));
        assert!(matches!(
            report.attempts[1].rejection,
            Some(Rejection::Timeout(_))
        ));
        assert_eq!(report.lean_runs, 2);

        let calls = llm.calls();
        assert!(
            calls[1].messages[2]
                .content
                .contains("did not contain a Lean file")
        );
        assert!(calls[2].messages[4].content.contains("was stopped"));
        assert!(kinds(&events).contains(&"rejected: the reply contained no Lean code"));
    }

    #[tokio::test]
    async fn rejects_proofs_that_lean_says_still_use_sorry() {
        // `stop` expands to `repeat sorry`: no forbidden token, exit code 0.
        let llm = MockProvider::new([fenced("  stop"), fenced("  unfold double\n  omega")]);
        let lean = ScriptedLean::new([accepted(SORRY_WARNING), accepted("")]);
        let config = SolveConfig {
            check_spec: false,
            axiom_audit: false,
            ..SolveConfig::default()
        };

        let (result, _) = run(&llm, &lean, config).await;
        let report = result.unwrap();
        assert!(report.verified);
        assert!(matches!(
            report.attempts[0].rejection,
            Some(Rejection::UsesSorry(_))
        ));
        assert!(
            llm.calls()[1].messages[2]
                .content
                .contains("still depends on `sorry`")
        );
    }

    #[tokio::test]
    async fn rejects_proofs_with_foreign_axioms_or_missing_targets() {
        let llm = MockProvider::new([
            fenced("  exact cheat n"),
            fenced("  exact other n"),
            fenced("  unfold double\n  omega"),
        ]);
        let lean = ScriptedLean::new([
            accepted(""),
            accepted("'double_eq' depends on axioms: [propext, cheat]\n"),
            accepted(""),
            rejected("Candidate.lean:7:14: error: Unknown constant `double_eq`\n"),
            accepted(""),
            accepted("'double_eq' does not depend on any axioms\n"),
        ]);
        let config = SolveConfig {
            check_spec: false,
            ..SolveConfig::default()
        };

        let (result, _) = run(&llm, &lean, config).await;
        let report = result.unwrap();
        assert!(report.verified);
        assert_eq!(report.axioms, Some(Vec::new()));
        assert_eq!(report.lean_runs, 6);

        let Some(Rejection::Axioms(tainted)) = &report.attempts[0].rejection else {
            panic!("expected an axiom rejection: {:?}", report.attempts[0]);
        };
        assert_eq!(tainted[0].axioms, ["cheat"]);
        assert!(matches!(
            report.attempts[1].rejection,
            Some(Rejection::AuditInconclusive(_))
        ));
        assert!(
            llm.calls()[1].messages[2]
                .content
                .contains("`double_eq` depends on cheat")
        );
    }

    #[tokio::test]
    async fn propagates_model_and_lean_failures() {
        let lean = ScriptedLean::new([accepted("")]);
        let out_of_replies = MockProvider::new(Vec::<String>::new());
        let (result, _) = run(&out_of_replies, &lean, SolveConfig::default()).await;
        let error = result.unwrap_err();
        assert!(matches!(error, SolveError::Llm(_)));
        assert!(error.to_string().contains("no scripted replies left"));

        let llm = MockProvider::new([fenced("  rfl")]);
        let no_lean = ScriptedLean::default();
        let config = SolveConfig {
            check_spec: false,
            ..SolveConfig::default()
        };
        let (result, _) = run(&llm, &no_lean, config).await;
        assert!(matches!(result, Err(SolveError::Lean(_))));
    }

    // -- reply parsing ---------------------------------------------------------

    #[test]
    fn extracts_code_from_a_lean_fence() {
        let reply = parse_reply("Here you go:\n```lean\ntheorem t : True := trivial\n```\nDone.");
        assert_eq!(reply.code.as_deref(), Some("theorem t : True := trivial\n"));
        assert_eq!(reply.note, None);

        let lean4 = parse_reply("```lean4\n  indented stays\n```");
        assert_eq!(lean4.code.as_deref(), Some("  indented stays\n"));
    }

    #[test]
    fn prefers_the_last_lean_block() {
        let reply = parse_reply(
            "First try:\n```lean\ntheorem t : True := by simp\n```\nBetter:\n```lean\ntheorem t : True := trivial\n```\n```text\nnot lean\n```",
        );
        assert_eq!(reply.code.as_deref(), Some("theorem t : True := trivial\n"));

        let untagged = parse_reply("```\ntheorem t : True := trivial\n```");
        assert_eq!(
            untagged.code.as_deref(),
            Some("theorem t : True := trivial\n")
        );

        assert_eq!(parse_reply("```python\nprint('hi')\n```").code, None);
    }

    #[test]
    fn accepts_bare_lean_and_rejects_prose() {
        let bare = parse_reply("theorem t : True := by\n  trivial\nNOTE: easy");
        assert_eq!(
            bare.code.as_deref(),
            Some("theorem t : True := by\n  trivial\n")
        );
        assert_eq!(bare.note.as_deref(), Some("easy"));

        assert_eq!(parse_reply("I could not find a proof, sorry.").code, None);
        assert_eq!(parse_reply("").code, None);
        assert_eq!(parse_reply("```lean\n\n```").code, None);
    }

    #[test]
    fn keeps_fences_nested_in_doc_comments() {
        let reply = parse_reply(
            "````lean\n/-- Example:\n```lean\n#eval f 1\n```\n-/\ndef f (n : Nat) := n\n````",
        );
        assert_eq!(
            reply.code.as_deref(),
            Some("/-- Example:\n```lean\n#eval f 1\n```\n-/\ndef f (n : Nat) := n\n")
        );

        // The same with an ordinary three-backtick outer fence.
        let nested = parse_reply(
            "```lean\n/-- Usage:\n```lean\nf 1\n```\n-/\ndef f := 1\n```\ntrailing prose",
        );
        assert_eq!(
            nested.code.as_deref(),
            Some("/-- Usage:\n```lean\nf 1\n```\n-/\ndef f := 1\n")
        );
    }

    #[test]
    fn tolerates_truncated_and_reasoning_laden_replies() {
        let truncated = parse_reply("```lean\ntheorem t : True := by\n  trivial");
        assert_eq!(
            truncated.code.as_deref(),
            Some("theorem t : True := by\n  trivial\n")
        );

        let reasoning = parse_reply(
            "<think>\nmaybe\n```lean\ntheorem wrong : False := sorry\n```\n</think>\n```lean\ntheorem t : True := trivial\n```",
        );
        assert_eq!(
            reasoning.code.as_deref(),
            Some("theorem t : True := trivial\n")
        );

        assert_eq!(
            parse_reply("<think>never finished ```lean\ntheorem t").code,
            None
        );
    }

    #[test]
    fn captures_notes_outside_the_code() {
        let reply = parse_reply(
            "```lean\n-- NOTE: inside the code, not a note\ntheorem t : True := trivial\n```\n**NOTE:** false for lo = hi = 2147483648 (the sum wraps to 0).",
        );
        assert_eq!(
            reply.note.as_deref(),
            Some("false for lo = hi = 2147483648 (the sum wraps to 0).")
        );
        assert!(reply.code.unwrap().contains("-- NOTE: inside the code"));
    }

    #[test]
    fn picks_the_best_attempt_by_failing_targets() {
        let attempt = |iteration, failing: &[usize], compiled: bool| Attempt {
            iteration,
            candidate: None,
            note: None,
            rejection: Some(if compiled {
                Rejection::Compile {
                    diagnostics: vec![],
                    exit_code: Some(1),
                    output: String::new(),
                }
            } else {
                Rejection::NoCode
            }),
            failing_targets: failing.to_vec(),
            unattributed_errors: 0,
        };
        let target = |name: &str| Target {
            kind: "theorem".into(),
            name: Some(name.into()),
            line: 1,
            holes: 1,
        };
        let mut report = SolveReport {
            verified: false,
            proof: None,
            targets: vec![target("a"), target("b")],
            attempts: vec![
                attempt(1, &[0, 1], true),
                attempt(2, &[1], true),
                attempt(3, &[0], true),
                attempt(4, &[], false),
            ],
            axioms: None,
            lean_runs: 3,
        };
        assert_eq!(report.best_attempt().unwrap().iteration, 3);

        report.attempts = vec![attempt(1, &[], false), attempt(2, &[], false)];
        assert_eq!(report.best_attempt().unwrap().iteration, 2);
        report.attempts.clear();
        assert!(report.best_attempt().is_none());
    }
}
