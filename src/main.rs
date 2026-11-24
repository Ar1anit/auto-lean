//! Command-line interface: `auto-lean verify | solve | prove-code`.

use std::fmt::Display;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use clap::{Args, Parser, Subcommand, ValueEnum};
use console::{StyledObject, style};
use serde::Serialize;

use auto_lean::agent::llm::{
    AnthropicProvider, FoundryProvider, LlmProvider, MockProvider, OpenAiProvider,
};
use auto_lean::agent::prompts::{clip, count};
use auto_lean::agent::{
    DEFAULT_MAX_ITERS, Rejection, SolveConfig, SolveEvent, SolveReport, Solver,
};
use auto_lean::lean::guard;
use auto_lean::lean::{
    AuditOutcome, LeanDiagnostic, LeanOutput, LeanRunner, Severity, Spec, Target, Violation,
};
use auto_lean::translator::{SourceLanguage, SpecProblem, TranslateEvent, Translation, Translator};

/// Environment variable naming the script the `mock` provider replays.
const MOCK_SCRIPT_ENV: &str = "AUTO_LEAN_MOCK_SCRIPT";

/// Lines of a diagnostic shown before the rest is elided (see `--verbose`).
const EXCERPT_LINES: usize = 14;

const EXIT_NOT_VERIFIED: u8 = 1;
const EXIT_ERROR: u8 = 2;

#[derive(Parser)]
#[command(
    name = "auto-lean",
    version,
    about = "Prove Lean 4 theorems and verify pure Python/Rust functions with an LLM in a compiler-feedback loop",
    after_help = "Exit status: 0 = verified, 1 = not verified, 2 = error.\n\
                  Environment: ANTHROPIC_API_KEY, OPENAI_API_KEY, OPENAI_BASE_URL, \
                  FOUNDRY_RESOURCE_ENDPOINT, FOUNDRY_API_KEY, AUTO_LEAN_LEAN_BIN."
)]
struct Cli {
    #[command(subcommand)]
    command: Command,

    /// Path to the `lean` binary [env: AUTO_LEAN_LEAN_BIN] [default: lean]
    #[arg(long, global = true, value_name = "PATH")]
    lean_bin: Option<PathBuf>,

    /// Seconds before a single Lean invocation is stopped
    #[arg(long, global = true, value_name = "SECS", default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    timeout: u64,

    /// Show Lean's diagnostics in full instead of an excerpt
    #[arg(long, short, global = true)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Run the anti-cheat guard and `lean` on a file and report the result
    Verify {
        /// Lean file to check
        path: PathBuf,
        /// Original file with `sorry` holes: also check that its statements were
        /// kept intact and that the proofs only use standard axioms
        #[arg(long, value_name = "PATH")]
        spec: Option<PathBuf>,
        /// Print the report as JSON
        #[arg(long)]
        json: bool,
    },

    /// Prove every `sorry` in a Lean file
    Solve {
        /// Lean file containing theorems with `sorry`
        path: PathBuf,
        #[command(flatten)]
        options: ProverOptions,
    },

    /// Translate a pure Python (.py) or Rust (.rs) function to Lean 4 and prove its invariants
    ProveCode {
        /// Source file containing the function
        path: PathBuf,
        /// Stop after generating the Lean spec (theorems still end in `sorry`)
        #[arg(long)]
        spec_only: bool,
        #[command(flatten)]
        options: ProverOptions,
    },
}

#[derive(Args)]
struct ProverOptions {
    /// Maximum number of proof attempts
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MAX_ITERS, value_parser = parse_iterations)]
    max_iters: usize,

    /// Which model API to use
    #[arg(long, value_enum, default_value_t = ProviderKind::Anthropic)]
    provider: ProviderKind,

    /// Model name [default: claude-opus-5-5 for anthropic, gpt-6.2-astra for openai; required for foundry, where it is the deployment name]
    #[arg(long, value_name = "NAME")]
    model: Option<String>,

    /// Write the resulting Lean file here instead of printing it
    #[arg(long, value_name = "PATH")]
    out: Option<PathBuf>,

    /// Skip the final `#print axioms` check of the proven theorems
    #[arg(long)]
    no_axiom_audit: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum ProviderKind {
    /// Claude via the Anthropic API (ANTHROPIC_API_KEY)
    Anthropic,
    /// OpenAI, or any OpenAI-compatible server via OPENAI_BASE_URL (OPENAI_API_KEY)
    Openai,
    /// A deployment in Microsoft Foundry, Claude or OpenAI-compatible; name it with --model (FOUNDRY_RESOURCE_ENDPOINT, FOUNDRY_API_KEY)
    Foundry,
    /// Replay canned replies from the file named by AUTO_LEAN_MOCK_SCRIPT (offline demos and tests)
    Mock,
}

fn parse_iterations(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if n >= 1 => Ok(n),
        _ => Err("must be a whole number of at least 1".to_string()),
    }
}

#[tokio::main]
async fn main() -> ExitCode {
    let cli = Cli::parse();
    match run(cli).await {
        Ok(code) => code,
        Err(error) => {
            eprintln!("{} {error:#}", red("error:").bold());
            ExitCode::from(EXIT_ERROR)
        }
    }
}

async fn run(cli: Cli) -> Result<ExitCode> {
    let mut runner = LeanRunner::new().with_timeout(Duration::from_secs(cli.timeout));
    if let Some(binary) = cli.lean_bin {
        runner = runner.with_binary(binary);
    }
    let app = App {
        runner,
        verbose: cli.verbose,
    };

    match cli.command {
        Command::Verify { path, spec, json } => app.verify(&path, spec.as_deref(), json).await,
        Command::Solve { path, options } => app.solve(&path, &options).await,
        Command::ProveCode {
            path,
            spec_only,
            options,
        } => app.prove_code(&path, spec_only, &options).await,
    }
}

struct App {
    runner: LeanRunner,
    verbose: bool,
}

/// Everything `verify` found, in the shape printed by `--json`.
#[derive(Serialize)]
struct VerifyReport {
    file: String,
    status: VerifyStatus,
    guard: GuardSection,
    lean: LeanSection,
    #[serde(skip_serializing_if = "Option::is_none")]
    axioms: Option<AuditOutcome>,
}

#[derive(Serialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum VerifyStatus {
    /// The guard passed and Lean accepted the file.
    Verified,
    /// Lean accepted the file, but it cheats.
    Rejected,
    /// Lean did not accept the file.
    Failed,
}

#[derive(Serialize)]
struct GuardSection {
    passed: bool,
    /// Whether the file was compared against an original spec.
    against_spec: bool,
    violations: Vec<Violation>,
}

#[derive(Serialize)]
struct LeanSection {
    passed: bool,
    exit_code: Option<i32>,
    timed_out: bool,
    seconds: f64,
    diagnostics: Vec<LeanDiagnostic>,
}

impl App {
    // -- verify -------------------------------------------------------------

    async fn verify(&self, path: &Path, spec_path: Option<&Path>, json: bool) -> Result<ExitCode> {
        let source = read(path)?;
        let spec = match spec_path {
            Some(spec_path) => Some(
                Spec::parse(&read(spec_path)?)
                    .with_context(|| format!("{} cannot be used as a spec", spec_path.display()))?,
            ),
            None => None,
        };

        // The guard is a pure text check; it runs whether or not Lean succeeds.
        let violations = match &spec {
            Some(spec) => spec.check(&source).violations,
            None => guard::scan(&source),
        };

        let output = self.runner.check_file(path).await?;
        let diagnostics = output.diagnostics();
        let uses_sorry = diagnostics.iter().any(LeanDiagnostic::mentions_sorry);

        // With a spec at hand, also make sure the proven theorems exist and rest
        // on standard axioms only.
        let mut axioms = None;
        if let Some(audit) = spec.as_ref().and_then(Spec::axiom_audit) {
            if output.success() && violations.is_empty() {
                let audited = format!("{source}{}", audit.commands());
                let audit_output = self.runner.check_source(&audited).await?;
                axioms = Some(audit.evaluate(&audit_output.stdout));
            }
        }

        let cheats = !violations.is_empty()
            || uses_sorry
            || matches!(
                axioms,
                Some(AuditOutcome::Tainted(_) | AuditOutcome::Inconclusive(_))
            );
        let status = if !output.success() {
            VerifyStatus::Failed
        } else if cheats {
            VerifyStatus::Rejected
        } else {
            VerifyStatus::Verified
        };

        let report = VerifyReport {
            file: path.display().to_string(),
            status,
            guard: GuardSection {
                passed: violations.is_empty(),
                against_spec: spec.is_some(),
                violations,
            },
            lean: LeanSection {
                passed: output.success(),
                exit_code: output.exit_code,
                timed_out: output.timed_out,
                seconds: output.duration.as_secs_f64(),
                diagnostics,
            },
            axioms,
        };

        if json {
            println!("{}", serde_json::to_string_pretty(&report)?);
        } else {
            self.print_verify_report(&report, &output);
        }
        Ok(match status {
            VerifyStatus::Verified => ExitCode::SUCCESS,
            _ => ExitCode::from(EXIT_NOT_VERIFIED),
        })
    }

    fn print_verify_report(&self, report: &VerifyReport, output: &LeanOutput) {
        let ok = |text: &'static str| style(text).green().bold();
        let bad = |text: &'static str| style(text).red().bold();
        let label = |text: &'static str| style(format!("{text:<8}")).bold();

        // Guard.
        let guard = &report.guard;
        let scope = if guard.against_spec {
            "forbidden tokens, original statements, added text"
        } else {
            "forbidden tokens"
        };
        if guard.passed {
            println!("  {}{}  {}", label("guard"), ok("PASS"), style(scope).dim());
        } else {
            println!(
                "  {}{}  {}",
                label("guard"),
                bad("FAIL"),
                count(guard.violations.len(), "violation")
            );
            for violation in &guard.violations {
                println!("            {violation}");
                if let Some(expected) = &violation.expected {
                    for line in clip(expected, 6, 600).lines() {
                        println!("              {}", style(line).dim());
                    }
                }
            }
        }

        // Lean.
        let lean = &report.lean;
        let errors = lean.diagnostics.iter().filter(|d| d.is_error()).count();
        let warnings = lean
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Warning)
            .count();
        let outcome = if lean.timed_out {
            format!("timed out after {:.0}s", lean.seconds)
        } else {
            let code = lean
                .exit_code
                .map_or_else(|| "killed".to_string(), |code| format!("exit code {code}"));
            format!(
                "{code} in {:.2}s, {}, {}",
                lean.seconds,
                count(errors, "error"),
                count(warnings, "warning")
            )
        };
        let verdict = if lean.passed { ok("PASS") } else { bad("FAIL") };
        println!("  {}{verdict}  {outcome}", label("lean"));
        for diagnostic in lean
            .diagnostics
            .iter()
            .filter(|d| d.severity != Severity::Info)
        {
            let position = format!("{}:{}", diagnostic.line, diagnostic.column);
            let severity = match diagnostic.severity {
                Severity::Error => style(diagnostic.severity.to_string()).red(),
                _ => style(diagnostic.severity.to_string()).yellow(),
            };
            let message = self.excerpt(&diagnostic.message);
            let mut lines = message.lines();
            println!(
                "            {position} {severity}: {}",
                lines.next().unwrap_or("")
            );
            for line in lines {
                println!("              {line}");
            }
        }
        if !lean.passed && lean.diagnostics.is_empty() {
            for line in self.excerpt(output.combined().trim()).lines() {
                println!("            {line}");
            }
        }

        // Axioms.
        match &report.axioms {
            Some(AuditOutcome::Clean(axioms)) => {
                println!(
                    "  {}{}  {}",
                    label("axioms"),
                    ok("PASS"),
                    describe_axioms(axioms)
                );
            }
            Some(AuditOutcome::Tainted(tainted)) => {
                println!("  {}{}  non-standard axioms", label("axioms"), bad("FAIL"));
                for theorem in tainted {
                    println!(
                        "            `{}` depends on {}",
                        theorem.name,
                        theorem.axioms.join(", ")
                    );
                }
            }
            Some(AuditOutcome::Inconclusive(reason)) => {
                println!("  {}{}  {reason}", label("axioms"), bad("FAIL"));
            }
            None => {}
        }

        println!();
        match report.status {
            VerifyStatus::Verified => println!("{}  {}", ok("VERIFIED"), report.file),
            VerifyStatus::Rejected => println!(
                "{}  {} compiles, but does not count as proven",
                bad("REJECTED"),
                report.file
            ),
            VerifyStatus::Failed => {
                println!("{}  Lean does not accept {}", bad("FAILED"), report.file)
            }
        }
    }

    // -- solve --------------------------------------------------------------

    async fn solve(&self, path: &Path, options: &ProverOptions) -> Result<ExitCode> {
        let spec = read(path)?;
        let (llm, model) = build_provider(options.provider, options.model.clone())?;
        let lean_version = concise_version(&self.runner.version().await?);

        header("lean", &lean_version);
        header("model", &model);
        header("input", path.display());

        let config = SolveConfig {
            max_iters: options.max_iters,
            axiom_audit: !options.no_axiom_audit,
            check_spec: true,
            lean_version: Some(lean_version),
        };
        let started = Instant::now();
        let report = Solver::new(llm.as_ref(), &self.runner)
            .with_config(config)
            .solve(&spec, &mut |event: &SolveEvent<'_>| {
                self.on_solve_event(event, options.max_iters)
            })
            .await?;
        self.finish(&report, started.elapsed(), options)
    }

    // -- prove-code ---------------------------------------------------------

    async fn prove_code(
        &self,
        path: &Path,
        spec_only: bool,
        options: &ProverOptions,
    ) -> Result<ExitCode> {
        let language = SourceLanguage::from_path(path)?;
        let source = read(path)?;
        let file_name = path.file_name().map_or_else(
            || path.display().to_string(),
            |name| name.to_string_lossy().into_owned(),
        );
        let (llm, model) = build_provider(options.provider, options.model.clone())?;
        let lean_version = concise_version(&self.runner.version().await?);

        header("lean", &lean_version);
        header("model", &model);
        header("input", format!("{} ({})", path.display(), language.name()));

        let started = Instant::now();
        section("Translating to Lean 4");
        let translation = Translator::new(llm.as_ref(), &self.runner)
            .with_lean_version(Some(lean_version.clone()))
            .translate(
                &source,
                language,
                &file_name,
                &mut |event: &TranslateEvent<'_>| {
                    self.on_translate_event(event);
                },
            )
            .await?;
        show_spec(&translation);

        if spec_only {
            emit(&translation.spec, options.out.as_deref(), "spec")?;
            eprintln!(
                "\n{} for {}; run without --spec-only to prove it",
                green("SPEC READY").bold(),
                count(translation.targets.len(), "theorem")
            );
            return Ok(ExitCode::SUCCESS);
        }

        section("Proving");
        let config = SolveConfig {
            max_iters: options.max_iters,
            axiom_audit: !options.no_axiom_audit,
            // Translation has just compiled this exact file.
            check_spec: false,
            lean_version: Some(lean_version),
        };
        let report = Solver::new(llm.as_ref(), &self.runner)
            .with_config(config)
            .solve(&translation.spec, &mut |event: &SolveEvent<'_>| {
                self.on_solve_event(event, options.max_iters);
            })
            .await?;
        self.finish(&report, started.elapsed(), options)
    }

    // -- progress -----------------------------------------------------------

    fn on_translate_event(&self, event: &TranslateEvent<'_>) {
        match event {
            TranslateEvent::Generating {
                attempt,
                max_attempts,
            } => {
                eprint!(
                    "{} asking the model ... ",
                    dim(format!("[{attempt}/{max_attempts}]"))
                );
                let _ = std::io::stderr().flush();
            }
            TranslateEvent::Generated { elapsed, .. } => {
                eprintln!("reply in {:.1}s", elapsed.as_secs_f64());
            }
            TranslateEvent::Rejected { problem, .. } => {
                step("spec", false, problem_headline(problem));
                for line in self.problem_details(problem) {
                    detail(line);
                }
            }
            TranslateEvent::Validated {
                targets, lean_time, ..
            } => step(
                "spec",
                true,
                format!(
                    "{}, type-checks in {:.2}s",
                    count(targets.len(), "theorem"),
                    lean_time.as_secs_f64()
                ),
            ),
        }
    }

    fn problem_details(&self, problem: &SpecProblem) -> Vec<String> {
        match problem {
            SpecProblem::Forbidden(violations) => {
                violations.iter().map(ToString::to_string).collect()
            }
            SpecProblem::Shape(problems) | SpecProblem::TypeFidelity(problems) => problems.clone(),
            SpecProblem::Compile {
                diagnostics,
                output,
                ..
            } => {
                let errors = self.error_lines(diagnostics);
                if errors.is_empty() {
                    self.excerpt(output.trim())
                        .lines()
                        .map(str::to_string)
                        .collect()
                } else {
                    errors
                }
            }
            SpecProblem::NoCode | SpecProblem::Timeout(_) => Vec::new(),
        }
    }

    fn on_solve_event(&self, event: &SolveEvent<'_>, max_iters: usize) {
        match event {
            SolveEvent::Started { .. } | SolveEvent::Verified { .. } => {}
            SolveEvent::Generating { iteration } => {
                eprint!(
                    "{} asking the model ... ",
                    dim(format!("[{iteration}/{max_iters}]"))
                );
                let _ = std::io::stderr().flush();
            }
            SolveEvent::Generated { elapsed, .. } => {
                eprintln!("reply in {:.1}s", elapsed.as_secs_f64());
            }
            SolveEvent::GuardPassed { .. } => step(
                "guard",
                true,
                "original statements intact, no forbidden tokens",
            ),
            SolveEvent::LeanPassed { elapsed, .. } => {
                step(
                    "lean",
                    true,
                    format!("exit code 0 in {:.2}s", elapsed.as_secs_f64()),
                );
            }
            SolveEvent::AuditPassed { axioms, .. } => step("axioms", true, describe_axioms(axioms)),
            SolveEvent::Rejected {
                rejection,
                lean_time,
                ..
            } => {
                let timing =
                    lean_time.map_or_else(String::new, |t| format!(" in {:.2}s", t.as_secs_f64()));
                match rejection {
                    Rejection::NoCode => step("reply", false, rejection.summary()),
                    Rejection::Guard(violations) => {
                        step(
                            "guard",
                            false,
                            format!("{}; Lean was not run", count(violations.len(), "violation")),
                        );
                        for violation in violations {
                            detail(violation);
                        }
                    }
                    Rejection::Compile {
                        diagnostics,
                        output,
                        ..
                    } => {
                        step("lean", false, format!("{}{timing}", rejection.summary()));
                        let errors = self.error_lines(diagnostics);
                        if errors.is_empty() {
                            self.excerpt(output.trim()).lines().for_each(detail);
                        } else {
                            errors.iter().for_each(detail);
                        }
                    }
                    Rejection::Timeout(_) => step("lean", false, rejection.summary()),
                    Rejection::UsesSorry(_) => step("lean", false, rejection.summary()),
                    Rejection::Axioms(_) | Rejection::AuditInconclusive(_) => {
                        step("axioms", false, rejection.summary());
                    }
                }
            }
        }
    }

    /// Lean's errors as display lines: position and message, long tactic states
    /// cut down unless `--verbose` was given.
    fn error_lines(&self, diagnostics: &[LeanDiagnostic]) -> Vec<String> {
        diagnostics
            .iter()
            .filter(|d| d.is_error())
            .flat_map(|diagnostic| {
                let message = self.excerpt(&diagnostic.message);
                let mut lines = message.lines();
                let first = format!(
                    "{}:{} {}",
                    diagnostic.line,
                    diagnostic.column,
                    lines.next().unwrap_or("")
                );
                std::iter::once(first)
                    .chain(lines.map(|line| format!("  {line}")))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn excerpt(&self, text: &str) -> String {
        if self.verbose {
            text.to_string()
        } else {
            clip(text, EXCERPT_LINES, 2_000)
        }
    }

    // -- verdict ------------------------------------------------------------

    fn finish(
        &self,
        report: &SolveReport,
        elapsed: Duration,
        options: &ProverOptions,
    ) -> Result<ExitCode> {
        let rounds = count(report.attempts.len(), "iteration");
        let seconds = elapsed.as_secs_f64();
        eprintln!();

        if let Some(proof) = &report.proof {
            eprintln!(
                "{} {} in {rounds} ({seconds:.1}s)",
                green("VERIFIED").bold(),
                describe_targets(&report.targets)
            );
            emit(proof, options.out.as_deref(), "verified proof")?;
            return Ok(ExitCode::SUCCESS);
        }

        eprintln!(
            "{} no accepted proof after {rounds} ({seconds:.1}s)",
            red("NOT VERIFIED").bold()
        );
        self.explain_failure(report);
        Ok(ExitCode::from(EXIT_NOT_VERIFIED))
    }

    /// Says which theorems resisted, based on the attempt that got furthest.
    fn explain_failure(&self, report: &SolveReport) {
        let Some(best) = report.best_attempt() else {
            return;
        };
        eprintln!();
        match &best.rejection {
            Some(Rejection::Compile { diagnostics, .. }) => {
                eprintln!("Closest attempt (iteration {}):", best.iteration);
                for (index, target) in report.targets.iter().enumerate() {
                    let failed = best.failing_targets.contains(&index);
                    let mark = if failed { red("✗") } else { green("✓") };
                    let verdict = if failed {
                        "not proven"
                    } else {
                        "proof accepted by Lean"
                    };
                    eprintln!("  {mark} {}: {verdict}", target.label());
                }
                if best.unattributed_errors > 0 {
                    eprintln!(
                        "  {} {} outside the target declarations",
                        red("✗"),
                        count(best.unattributed_errors, "error")
                    );
                }
                eprintln!();
                eprintln!("Lean's objections:");
                for line in self.error_lines(diagnostics) {
                    eprintln!("  {line}");
                }
            }
            Some(other) => eprintln!(
                "Last attempt (iteration {}): {}",
                best.iteration,
                other.summary()
            ),
            None => {}
        }
        if let Some(note) = report.model_note() {
            eprintln!();
            eprintln!("{} {note}", yellow("Model's note (not verified):"));
        }
        eprintln!();
        eprintln!(
            "This means either a statement is false as written, which points at a bug in the code \
             or in the proposed invariant, or the prover needs more attempts (--max-iters)."
        );
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn read(path: &Path) -> Result<String> {
    std::fs::read_to_string(path).with_context(|| format!("failed to read {}", path.display()))
}

/// Writes the result to `--out`, or prints it to stdout.
fn emit(content: &str, out: Option<&Path>, what: &str) -> Result<()> {
    match out {
        Some(path) => {
            std::fs::write(path, content)
                .with_context(|| format!("failed to write {}", path.display()))?;
            eprintln!("{} written to {}", capitalize(what), path.display());
        }
        None => {
            print!("{content}");
            if !content.ends_with('\n') {
                println!();
            }
        }
    }
    Ok(())
}

fn capitalize(text: &str) -> String {
    let mut chars = text.chars();
    match chars.next() {
        Some(first) => first.to_uppercase().chain(chars).collect(),
        None => String::new(),
    }
}

fn build_provider(
    kind: ProviderKind,
    model: Option<String>,
) -> Result<(Box<dyn LlmProvider>, String)> {
    Ok(match kind {
        ProviderKind::Anthropic => {
            let provider = AnthropicProvider::from_env(model)?;
            let label = format!("anthropic · {}", provider.model());
            (Box::new(provider), label)
        }
        ProviderKind::Openai => {
            let provider = OpenAiProvider::from_env(model)?;
            let label = format!("openai · {} ({})", provider.model(), provider.base_url());
            (Box::new(provider), label)
        }
        ProviderKind::Foundry => {
            let provider = FoundryProvider::from_env(model)?;
            let label = format!(
                "foundry · {} via the {} ({})",
                provider.deployment(),
                provider.api().label(),
                provider.endpoint()
            );
            (Box::new(provider), label)
        }
        ProviderKind::Mock => {
            let script = std::env::var_os(MOCK_SCRIPT_ENV)
                .filter(|value| !value.is_empty())
                .map(PathBuf::from)
                .ok_or_else(|| {
                    anyhow!("--provider mock replays a script; set {MOCK_SCRIPT_ENV} to its path")
                })?;
            let provider = MockProvider::from_script_file(&script)?;
            let label = format!("mock · replaying {}", script.display());
            (Box::new(provider), label)
        }
    })
}

/// `Lean (version 4.36.0, arm64-apple-darwin, commit ..., Release)` -> `Lean 4.36.0`.
fn concise_version(full: &str) -> String {
    full.split_once("version ")
        .and_then(|(_, rest)| rest.split([',', ')']).next())
        .map(str::trim)
        .filter(|version| !version.is_empty())
        .map_or_else(
            || full.trim().to_string(),
            |version| format!("Lean {version}"),
        )
}

fn describe_axioms(axioms: &[String]) -> String {
    if axioms.is_empty() {
        "no axioms used".to_string()
    } else {
        format!("only standard axioms ({})", axioms.join(", "))
    }
}

fn describe_targets(targets: &[Target]) -> String {
    match targets {
        [only] => only.label(),
        many => {
            let same_kind = many.iter().all(|target| target.kind == many[0].kind);
            let noun = if same_kind {
                many[0].kind.as_str()
            } else {
                "declaration"
            };
            format!(
                "{} ({})",
                count(many.len(), noun),
                many.iter()
                    .map(|target| target.name.clone().unwrap_or_else(|| target.label()))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

fn problem_headline(problem: &SpecProblem) -> String {
    match problem {
        SpecProblem::NoCode => "the reply contained no Lean code".to_string(),
        SpecProblem::Forbidden(_) => "forbidden tokens in the spec".to_string(),
        SpecProblem::Shape(_) => "the spec does not have the required shape".to_string(),
        SpecProblem::TypeFidelity(_) => "the model violates the type-fidelity rule".to_string(),
        SpecProblem::Compile { diagnostics, .. } => format!(
            "Lean rejected the spec ({})",
            count(diagnostics.iter().filter(|d| d.is_error()).count(), "error")
        ),
        SpecProblem::Timeout(_) => problem.summary(),
    }
}

/// Prints the generated spec so the reader can see what is about to be proven.
fn show_spec(translation: &Translation) {
    eprintln!();
    for line in translation.spec.trim_end().lines() {
        if line.is_empty() {
            eprintln!();
        } else {
            eprintln!("    {}", dim(line));
        }
    }
}

fn header(name: &str, value: impl Display) {
    eprintln!("{} {value}", dim(format!("{name:<6}")));
}

fn section(title: &str) {
    eprintln!("\n{}", style(title).bold().for_stderr());
}

/// One line of the per-iteration log: `      guard  ✓ ...`.
fn step(name: &str, passed: bool, message: impl Display) {
    let mark = if passed { green("✓") } else { red("✗") };
    eprintln!("      {name:<6} {mark} {message}");
}

fn detail(line: impl Display) {
    eprintln!("               {line}");
}

fn green<D>(value: D) -> StyledObject<D> {
    style(value).for_stderr().green()
}

fn red<D>(value: D) -> StyledObject<D> {
    style(value).for_stderr().red()
}

fn yellow<D>(value: D) -> StyledObject<D> {
    style(value).for_stderr().yellow()
}

fn dim<D>(value: D) -> StyledObject<D> {
    style(value).for_stderr().dim()
}

#[cfg(test)]
mod tests {
    use super::*;
    use auto_lean::agent::llm::{DEFAULT_ANTHROPIC_MODEL, DEFAULT_OPENAI_MODEL};
    use auto_lean::lean::{LEAN_BIN_ENV, STANDARD_AXIOMS};
    use clap::CommandFactory;

    #[test]
    fn the_cli_definition_is_consistent() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_the_documented_invocations() {
        let cli = Cli::try_parse_from(["auto-lean", "verify", "proof.lean"]).unwrap();
        assert!(matches!(
            cli.command,
            Command::Verify {
                json: false,
                spec: None,
                ..
            }
        ));
        assert_eq!(cli.timeout, 30);

        let cli = Cli::try_parse_from([
            "auto-lean",
            "solve",
            "demo.lean",
            "--max-iters",
            "8",
            "--provider",
            "openai",
            "--model",
            "gpt-test",
            "--out",
            "proved.lean",
        ])
        .unwrap();
        let Command::Solve { path, options } = cli.command else {
            panic!("expected solve");
        };
        assert_eq!(path, PathBuf::from("demo.lean"));
        assert_eq!(options.max_iters, 8);
        assert!(options.provider == ProviderKind::Openai);
        assert_eq!(options.model.as_deref(), Some("gpt-test"));
        assert_eq!(options.out, Some(PathBuf::from("proved.lean")));

        let cli = Cli::try_parse_from([
            "auto-lean",
            "solve",
            "demo.lean",
            "--provider",
            "foundry",
            "--model",
            "claude-opus-5-5",
        ])
        .unwrap();
        let Command::Solve { options, .. } = cli.command else {
            panic!("expected solve");
        };
        assert!(options.provider == ProviderKind::Foundry);
        assert_eq!(options.model.as_deref(), Some("claude-opus-5-5"));

        let cli =
            Cli::try_parse_from(["auto-lean", "prove-code", "pricing.py", "--spec-only"]).unwrap();
        let Command::ProveCode {
            spec_only, options, ..
        } = cli.command
        else {
            panic!("expected prove-code");
        };
        assert!(spec_only);
        assert_eq!(options.max_iters, DEFAULT_MAX_ITERS);
        assert!(options.provider == ProviderKind::Anthropic);
        assert!(!options.no_axiom_audit);
    }

    #[test]
    fn rejects_invalid_arguments() {
        assert!(Cli::try_parse_from(["auto-lean", "solve", "x.lean", "--max-iters", "0"]).is_err());
        assert!(
            Cli::try_parse_from(["auto-lean", "solve", "x.lean", "--provider", "gemini"]).is_err()
        );
        assert!(Cli::try_parse_from(["auto-lean", "solve"]).is_err());
        assert!(Cli::try_parse_from(["auto-lean", "verify", "x.lean", "--timeout", "0"]).is_err());
    }

    #[test]
    fn default_models_match_the_help_text() {
        let help = Cli::command()
            .find_subcommand_mut("solve")
            .unwrap()
            .render_long_help()
            .to_string();
        assert!(help.contains(DEFAULT_ANTHROPIC_MODEL), "{help}");
        assert!(help.contains(DEFAULT_OPENAI_MODEL), "{help}");
        assert!(help.contains("- foundry:"), "{help}");
        assert!(help.contains("required for foundry"), "{help}");
        assert!(
            Cli::command()
                .render_long_help()
                .to_string()
                .contains(LEAN_BIN_ENV)
        );
    }

    #[test]
    fn describes_results() {
        let target = |name: &str| Target {
            kind: "theorem".into(),
            name: Some(name.into()),
            line: 1,
            holes: 1,
        };
        assert_eq!(describe_targets(&[target("a")]), "theorem `a`");
        assert_eq!(
            describe_targets(&[target("a"), target("b")]),
            "2 theorems (a, b)"
        );
        let definition = Target {
            kind: "def".into(),
            name: Some("f".into()),
            line: 1,
            holes: 1,
        };
        assert_eq!(
            describe_targets(&[target("a"), definition]),
            "2 declarations (a, f)"
        );
        assert_eq!(describe_axioms(&[]), "no axioms used");
        assert_eq!(
            describe_axioms(&["propext".into()]),
            "only standard axioms (propext)"
        );
        assert_eq!(capitalize("verified proof"), "Verified proof");
        assert_eq!(
            concise_version(
                "Lean (version 4.36.0, arm64-apple-darwin24.6.0, commit 611c117bfaf7, Release)"
            ),
            "Lean 4.36.0"
        );
        assert_eq!(
            concise_version("Lean (version 4.37.0-rc1)"),
            "Lean 4.37.0-rc1"
        );
        assert_eq!(concise_version("something else\n"), "something else");
        assert!(STANDARD_AXIOMS.contains(&"propext"));
    }
}
