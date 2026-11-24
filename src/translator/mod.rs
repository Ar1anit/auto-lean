//! Mode 2: from a Python or Rust function to a Lean 4 spec, and on to a proof.
//!
//! [`Translator::translate`] asks the model for a faithful Lean model of the
//! function plus two or three invariants stated as `theorem ... := by sorry`.
//! The answer is not trusted:
//!
//! 1. it is linted without running anything ([`lint`]): no forbidden tokens, no
//!    commands that execute code, the expected shape, and the strict
//!    type-fidelity rule ([`check_type_fidelity`]);
//! 2. it must parse and type-check in Lean, where `sorry` is tolerated for this
//!    one step only.
//!
//! A rejected spec goes back to the model with the reason, at most
//! [`MAX_TRANSLATION_RETRIES`] times. [`prove_code`] then hands the validated
//! spec to the same [`Solver`] loop that Mode 1 uses.

pub mod prompts;

use std::collections::BTreeSet;
use std::path::Path;
use std::sync::LazyLock;
use std::time::{Duration, Instant};

use regex::Regex;
use thiserror::Error;

use crate::agent::llm::{LlmProvider, Message};
use crate::agent::{Observer, Silent, SolveConfig, SolveError, SolveReport, Solver, parse_reply};
use crate::lean::{LeanChecker, LeanDiagnostic, LeanError, Spec, SpecError, Target, Violation};

/// How many times a rejected translation is sent back for correction.
pub const MAX_TRANSLATION_RETRIES: usize = 2;
/// A generated spec must state at least this many theorems...
pub const MIN_THEOREMS: usize = 2;
/// ...and at most this many.
pub const MAX_THEOREMS: usize = 3;

/// Lean's fixed-width integer types.
const FIXED_WIDTH_TYPES: &[&str] = &[
    "UInt8", "UInt16", "UInt32", "UInt64", "USize", "Int8", "Int16", "Int32", "Int64", "ISize",
    "BitVec",
];

static RUST_FN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^>]*>)?\s*\(([^)]*)\)\s*(?:->\s*([^{;]+))?")
        .expect("valid regex")
});
static RUST_INT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\b[ui](?:8|16|32|64|128|size)\b").expect("valid regex"));
static PYTHON_DEF: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_][A-Za-z0-9_]*)[ \t]*\(([^)]*)\)[ \t]*(?:->[ \t]*([^:\n]+))?:")
        .expect("valid regex")
});
static PYTHON_INT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\bint\b").expect("valid regex"));

/// Languages [`Translator`] understands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceLanguage {
    Python,
    Rust,
}

impl SourceLanguage {
    /// Detects the language from a file extension (`.py` or `.rs`).
    pub fn from_path(path: &Path) -> Result<Self, TranslateError> {
        let extension = path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("py") => Ok(SourceLanguage::Python),
            Some("rs") => Ok(SourceLanguage::Rust),
            _ => Err(TranslateError::UnsupportedFile(path.display().to_string())),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            SourceLanguage::Python => "Python",
            SourceLanguage::Rust => "Rust",
        }
    }

    /// The Markdown code-fence tag for this language.
    pub fn fence(self) -> &'static str {
        match self {
            SourceLanguage::Python => "python",
            SourceLanguage::Rust => "rust",
        }
    }
}

/// Why a generated spec was sent back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecProblem {
    /// The reply contained no Lean file.
    NoCode,
    /// The spec uses `axiom`, `unsafe` or another forbidden token.
    Forbidden(Vec<Violation>),
    /// Wrong structure: theorem count, a `sorry` outside a theorem, an import...
    Shape(Vec<String>),
    /// A type in the model does not match the source type.
    TypeFidelity(Vec<String>),
    /// Lean could not parse or type-check the spec.
    Compile {
        diagnostics: Vec<LeanDiagnostic>,
        exit_code: Option<i32>,
        output: String,
    },
    /// Lean had to be killed after this long.
    Timeout(Duration),
}

impl SpecProblem {
    /// One line describing the problem.
    pub fn summary(&self) -> String {
        match self {
            SpecProblem::NoCode => "the reply contained no Lean code".to_string(),
            SpecProblem::Forbidden(violations) => violations
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join("; "),
            SpecProblem::Shape(problems) => problems.join("; "),
            SpecProblem::TypeFidelity(problems) => {
                format!("type fidelity: {}", problems.join("; "))
            }
            SpecProblem::Compile {
                diagnostics,
                exit_code,
                ..
            } => match diagnostics.iter().find(|d| d.is_error()) {
                Some(first) => format!(
                    "Lean rejected the spec: {}:{}: {}",
                    first.line,
                    first.column,
                    first.summary()
                ),
                None => format!("Lean failed on the spec with exit code {exit_code:?}"),
            },
            SpecProblem::Timeout(elapsed) => {
                format!(
                    "Lean timed out on the spec after {:.0}s",
                    elapsed.as_secs_f64()
                )
            }
        }
    }
}

/// A validated Lean spec for a source function.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Translation {
    /// The Lean file: model definition plus theorems ending in `sorry`.
    pub spec: String,
    /// The theorems to prove.
    pub targets: Vec<Target>,
    /// How many model calls it took (1 means the first answer was valid).
    pub attempts: usize,
    /// How many times Lean was invoked.
    pub lean_runs: usize,
}

/// Progress notifications emitted while translating.
#[derive(Debug)]
pub enum TranslateEvent<'a> {
    Generating {
        attempt: usize,
        max_attempts: usize,
    },
    Generated {
        attempt: usize,
        elapsed: Duration,
    },
    Rejected {
        attempt: usize,
        problem: &'a SpecProblem,
    },
    Validated {
        attempt: usize,
        targets: &'a [Target],
        lean_time: Duration,
    },
}

/// Receives [`TranslateEvent`]s. Implemented for closures and [`Silent`].
pub trait TranslateObserver {
    fn on_event(&mut self, event: &TranslateEvent<'_>);
}

impl<F: FnMut(&TranslateEvent<'_>)> TranslateObserver for F {
    fn on_event(&mut self, event: &TranslateEvent<'_>) {
        self(event);
    }
}

impl TranslateObserver for Silent {
    fn on_event(&mut self, _event: &TranslateEvent<'_>) {}
}

/// Errors that end a translation.
#[derive(Debug, Error)]
pub enum TranslateError {
    #[error("cannot tell the language of `{0}`: expected a .py or .rs file")]
    UnsupportedFile(String),

    #[error("the model reports that this code cannot be translated faithfully: {0}")]
    Unsupported(String),

    #[error(
        "no valid Lean spec after {attempts} attempts; the last one was rejected because: {last}"
    )]
    Exhausted { attempts: usize, last: String },

    #[error(transparent)]
    Lean(#[from] LeanError),

    #[error("the language model request failed: {0:#}")]
    Llm(anyhow::Error),
}

/// A function found in the source file.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SourceFunction {
    name: String,
    /// Integer types named in its signature (`u32`, `i64`, `int`, ...).
    integer_types: BTreeSet<String>,
}

fn source_functions(source: &str, language: SourceLanguage) -> Vec<SourceFunction> {
    let (definition, integer) = match language {
        SourceLanguage::Rust => (&*RUST_FN, &*RUST_INT),
        SourceLanguage::Python => (&*PYTHON_DEF, &*PYTHON_INT),
    };
    definition
        .captures_iter(source)
        .map(|captures| {
            let signature = format!(
                "{} {}",
                captures.get(2).map_or("", |m| m.as_str()),
                captures.get(3).map_or("", |m| m.as_str())
            );
            SourceFunction {
                name: captures[1].to_string(),
                integer_types: integer
                    .find_iter(&signature)
                    .map(|found| found.as_str().to_string())
                    .collect(),
            }
        })
        .collect()
}

/// `apply_discount`, `applyDiscount` and `Pricing.apply_discount` name the same function.
fn same_function(lean_name: &str, source_name: &str) -> bool {
    let normalize = |name: &str| name.replace('_', "").to_ascii_lowercase();
    let short = lean_name.rsplit('.').next().unwrap_or(lean_name);
    normalize(short) == normalize(source_name)
}

fn lean_integer_type(rust: &str) -> &'static str {
    match rust {
        "u8" => "UInt8",
        "u16" => "UInt16",
        "u32" => "UInt32",
        "u64" => "UInt64",
        "usize" => "USize",
        "i8" => "Int8",
        "i16" => "Int16",
        "i32" => "Int32",
        "i64" => "Int64",
        "isize" => "ISize",
        _ => "BitVec",
    }
}

/// Enforces the strict type-fidelity rule on the model of each source function:
/// the signature of its Lean `def` must use the Lean type with exactly the range
/// of the source type. Returns one message per violation.
pub fn check_type_fidelity(source: &str, language: SourceLanguage, spec: &Spec) -> Vec<String> {
    let definitions = spec.definitions();
    let mut problems = Vec::new();

    for function in source_functions(source, language) {
        let Some(definition) = definitions
            .iter()
            .find(|definition| same_function(&definition.name, &function.name))
        else {
            continue;
        };
        let uses = |lean_type: &str| definition.signature.iter().any(|word| word == lean_type);
        let model = &definition.name;

        match language {
            SourceLanguage::Rust => {
                for unbounded in ["Nat", "Int"] {
                    if uses(unbounded) {
                        problems.push(format!(
                            "`{model}` uses the unbounded `{unbounded}` in its signature, but every Rust \
                             integer has a fixed width; an unbounded model cannot overflow and hides \
                             overflow bugs"
                        ));
                    }
                }
                for rust_type in &function.integer_types {
                    let expected = lean_integer_type(rust_type);
                    // `BitVec` is the sanctioned fallback for signed integers.
                    let fallback = rust_type.starts_with('i') && uses("BitVec");
                    if !uses(expected) && !fallback {
                        problems.push(format!(
                            "Rust `{rust_type}` in the signature of `{}` must be `{expected}` in `{model}`",
                            function.name
                        ));
                    }
                }
            }
            SourceLanguage::Python => {
                if uses("Nat") {
                    problems.push(format!(
                        "`{model}` uses `Nat` in its signature, but a Python `int` can be negative; use `Int`"
                    ));
                }
                for fixed in FIXED_WIDTH_TYPES {
                    if uses(fixed) {
                        problems.push(format!(
                            "`{model}` uses the fixed-width `{fixed}` in its signature, but a Python \
                             `int` is unbounded; use `Int`"
                        ));
                    }
                }
                if function.integer_types.contains("int") && !uses("Int") {
                    problems.push(format!(
                        "Python `int` in the signature of `{}` must be `Int` in `{model}`",
                        function.name
                    ));
                }
            }
        }
    }
    problems
}

/// Structural requirements on a generated spec.
fn check_shape(source: &str, language: SourceLanguage, spec: &Spec) -> Vec<String> {
    let mut problems: Vec<String> = spec
        .executable_constructs()
        .iter()
        .map(|violation| format!("line {}", violation))
        .collect();

    let targets = spec.targets();
    let is_theorem = |target: &&Target| target.kind == "theorem" || target.kind == "lemma";
    for stray in targets.iter().filter(|target| !is_theorem(target)) {
        problems.push(format!(
            "`sorry` may only stand for the proof of a theorem, but the {} contains one",
            stray.label()
        ));
    }
    let theorems = targets.iter().filter(is_theorem).count();
    if !(MIN_THEOREMS..=MAX_THEOREMS).contains(&theorems) {
        problems.push(format!(
            "expected {MIN_THEOREMS} or {MAX_THEOREMS} theorems ending in `:= by sorry`, found {theorems}"
        ));
    }

    let functions = source_functions(source, language);
    let definitions = spec.definitions();
    let modelled = functions.iter().any(|function| {
        definitions
            .iter()
            .any(|definition| same_function(&definition.name, &function.name))
    });
    if let (false, Some(function)) = (modelled, functions.first()) {
        problems.push(format!(
            "the file must contain a `def` named after the source function `{}`",
            function.name
        ));
    }
    problems
}

/// Checks a generated spec without running anything: forbidden tokens, shape,
/// and type fidelity. Lean is only invoked on specs that pass.
pub fn lint(source: &str, language: SourceLanguage, lean_spec: &str) -> Result<Spec, SpecProblem> {
    let spec = Spec::parse(lean_spec)
        .map_err(|SpecError::ForbiddenTokens(violations)| SpecProblem::Forbidden(violations))?;
    let shape = check_shape(source, language, &spec);
    if !shape.is_empty() {
        return Err(SpecProblem::Shape(shape));
    }
    let fidelity = check_type_fidelity(source, language, &spec);
    if !fidelity.is_empty() {
        return Err(SpecProblem::TypeFidelity(fidelity));
    }
    Ok(spec)
}

/// The reason given on an `UNSUPPORTED:` line, if the reply has one.
fn unsupported_reason(reply: &str) -> Option<String> {
    reply.lines().find_map(|line| {
        let reason = line
            .trim()
            .trim_start_matches(['*', '>', ' '])
            .strip_prefix("UNSUPPORTED:")?;
        Some(reason.trim().trim_end_matches('*').trim().to_string())
    })
}

/// Turns source code into a validated Lean spec.
pub struct Translator<'a> {
    llm: &'a dyn LlmProvider,
    lean: &'a dyn LeanChecker,
    max_retries: usize,
    lean_version: Option<String>,
}

impl<'a> Translator<'a> {
    pub fn new(llm: &'a dyn LlmProvider, lean: &'a dyn LeanChecker) -> Self {
        Self {
            llm,
            lean,
            max_retries: MAX_TRANSLATION_RETRIES,
            lean_version: None,
        }
    }

    /// Toolchain description passed to the model.
    pub fn with_lean_version(mut self, lean_version: Option<String>) -> Self {
        self.lean_version = lean_version;
        self
    }

    pub fn with_max_retries(mut self, max_retries: usize) -> Self {
        self.max_retries = max_retries;
        self
    }

    /// Translates `source`, retrying while the result is rejected.
    pub async fn translate(
        &self,
        source: &str,
        language: SourceLanguage,
        file_name: &str,
        observer: &mut dyn TranslateObserver,
    ) -> Result<Translation, TranslateError> {
        let system = prompts::system_prompt(language, self.lean_version.as_deref());
        let mut messages = vec![Message::user(prompts::translation_prompt(
            language, file_name, source,
        ))];
        let max_attempts = self.max_retries + 1;
        let mut lean_runs = 0;
        let mut last_problem = String::new();

        for attempt in 1..=max_attempts {
            observer.on_event(&TranslateEvent::Generating {
                attempt,
                max_attempts,
            });
            let asked = Instant::now();
            let raw = self
                .llm
                .complete(&system, &messages)
                .await
                .map_err(TranslateError::Llm)?;
            observer.on_event(&TranslateEvent::Generated {
                attempt,
                elapsed: asked.elapsed(),
            });

            let reply = parse_reply(&raw);
            if reply.code.is_none() {
                if let Some(reason) = unsupported_reason(&raw) {
                    return Err(TranslateError::Unsupported(reason));
                }
            }
            let code = reply.code.unwrap_or_default();

            let problem = match self
                .validate(source, language, &code, &mut lean_runs)
                .await?
            {
                Ok((targets, lean_time)) => {
                    observer.on_event(&TranslateEvent::Validated {
                        attempt,
                        targets: &targets,
                        lean_time,
                    });
                    return Ok(Translation {
                        spec: code,
                        targets,
                        attempts: attempt,
                        lean_runs,
                    });
                }
                Err(problem) => problem,
            };

            observer.on_event(&TranslateEvent::Rejected {
                attempt,
                problem: &problem,
            });
            let said = if raw.trim().is_empty() {
                "(empty reply)".to_string()
            } else {
                raw
            };
            messages.push(Message::assistant(said));
            messages.push(Message::user(prompts::rejection_feedback(&problem, &code)));
            last_problem = problem.summary();
        }

        Err(TranslateError::Exhausted {
            attempts: max_attempts,
            last: last_problem,
        })
    }

    /// Lints the spec, then makes sure Lean accepts it. `sorry` is fine here and
    /// nowhere else: at this point the theorems are only being stated.
    async fn validate(
        &self,
        source: &str,
        language: SourceLanguage,
        code: &str,
        lean_runs: &mut usize,
    ) -> Result<Result<(Vec<Target>, Duration), SpecProblem>, TranslateError> {
        if code.trim().is_empty() {
            return Ok(Err(SpecProblem::NoCode));
        }
        // The lint also rules out commands that would execute code, so nothing
        // the model wrote is compiled before it passes.
        let spec = match lint(source, language, code) {
            Ok(spec) => spec,
            Err(problem) => return Ok(Err(problem)),
        };

        let output = self.lean.check(code).await?;
        *lean_runs += 1;
        if output.timed_out {
            return Ok(Err(SpecProblem::Timeout(output.duration)));
        }
        if !output.success() {
            return Ok(Err(SpecProblem::Compile {
                diagnostics: output.diagnostics(),
                exit_code: output.exit_code,
                output: output.combined(),
            }));
        }
        Ok(Ok((spec.targets(), output.duration)))
    }
}

/// The result of the whole Mode 2 pipeline.
#[derive(Debug, Clone)]
pub struct CodeProof {
    pub translation: Translation,
    pub solve: SolveReport,
}

/// Errors of [`prove_code`].
#[derive(Debug, Error)]
pub enum ProveCodeError {
    #[error(transparent)]
    Translate(#[from] TranslateError),
    #[error(transparent)]
    Solve(#[from] SolveError),
}

/// Mode 2 end to end: translate `source` into a Lean spec, validate it, and run
/// the proof-search loop on it.
#[allow(clippy::too_many_arguments)]
pub async fn prove_code(
    llm: &dyn LlmProvider,
    lean: &dyn LeanChecker,
    source: &str,
    language: SourceLanguage,
    file_name: &str,
    config: SolveConfig,
    on_translate: &mut dyn TranslateObserver,
    on_solve: &mut dyn Observer,
) -> Result<CodeProof, ProveCodeError> {
    let translation = Translator::new(llm, lean)
        .with_lean_version(config.lean_version.clone())
        .translate(source, language, file_name, on_translate)
        .await?;

    // Translation has just compiled this exact file; no need to do it twice.
    let config = SolveConfig {
        check_spec: false,
        ..config
    };
    let solve = Solver::new(llm, lean)
        .with_config(config)
        .solve(&translation.spec, on_solve)
        .await?;
    Ok(CodeProof { translation, solve })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::llm::MockProvider;
    use crate::agent::testing::{ScriptedLean, accepted, hung, rejected};

    const PRICING_PY: &str = include_str!("../../examples/pricing.py");
    const PRICING_REPLAY: &str = include_str!("../../tests/fixtures/pricing.replay.md");
    const MIDPOINT_RS: &str = "pub fn midpoint(lo: u32, hi: u32) -> u32 {\n    (lo + hi) / 2\n}\n";

    const SORRY_WARNINGS: &str = "\
Candidate.lean:7:8: warning: declaration uses `sorry`
Candidate.lean:10:8: warning: declaration uses `sorry`
";

    fn midpoint_spec(int: &str) -> String {
        format!(
            "```lean\ndef midpoint (lo hi : {int}) : {int} :=\n  (lo + hi) / 2\n\n\
             theorem midpoint_ge_lo (lo hi : {int}) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by sorry\n\n\
             theorem midpoint_le_hi (lo hi : {int}) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by sorry\n```"
        )
    }

    fn code_of(reply: &str) -> String {
        parse_reply(reply).code.expect("fixture contains Lean code")
    }

    /// Mode 2 end to end with a scripted model: translate `pricing.py`, validate
    /// the spec, hand it to the solver, and get a verified proof back.
    #[tokio::test]
    async fn translates_pricing_validates_the_spec_and_proves_it() {
        let llm = MockProvider::from_script(PRICING_REPLAY);
        assert_eq!(llm.remaining(), 2, "fixture: one translation, one proof");
        let lean = ScriptedLean::new([
            accepted(SORRY_WARNINGS), // the generated spec
            accepted(""),             // the proof
            accepted(
                "'applyDiscount_nonneg' depends on axioms: [propext, Quot.sound]\n\
                 'applyDiscount_le_price' depends on axioms: [propext, Quot.sound]\n",
            ),
        ]);

        let mut translate_events = Vec::new();
        let mut on_translate = |event: &TranslateEvent<'_>| {
            translate_events.push(match event {
                TranslateEvent::Generating {
                    attempt,
                    max_attempts,
                } => format!("ask {attempt}/{max_attempts}"),
                TranslateEvent::Generated { .. } => "reply".to_string(),
                TranslateEvent::Rejected { problem, .. } => {
                    format!("rejected: {}", problem.summary())
                }
                TranslateEvent::Validated { targets, .. } => {
                    format!("valid: {} theorems", targets.len())
                }
            });
        };

        let config = SolveConfig {
            lean_version: Some("Lean (version 4.36.0)".to_string()),
            ..SolveConfig::default()
        };
        let proof = prove_code(
            &llm,
            &lean,
            PRICING_PY,
            SourceLanguage::Python,
            "pricing.py",
            config,
            &mut on_translate,
            &mut Silent,
        )
        .await
        .unwrap();

        // Step 1: the translation was accepted on the first try.
        assert_eq!(translate_events, ["ask 1/3", "reply", "valid: 2 theorems"]);
        let translation = &proof.translation;
        assert_eq!(translation.attempts, 1);
        assert!(
            translation
                .spec
                .contains("def applyDiscount (priceCents discountPct : Int) : Int")
        );
        assert_eq!(translation.spec.matches("sorry").count(), 2);
        let names: Vec<_> = translation
            .targets
            .iter()
            .filter_map(|t| t.name.as_deref())
            .collect();
        assert_eq!(names, ["applyDiscount_nonneg", "applyDiscount_le_price"]);

        // Step 2: the solver proved exactly that spec.
        let solve = &proof.solve;
        assert!(solve.verified);
        assert_eq!(solve.attempts.len(), 1);
        assert_eq!(solve.targets, translation.targets);
        let verified = solve.proof.as_deref().unwrap();
        assert!(!verified.contains("sorry"));
        assert!(
            Spec::parse(&translation.spec)
                .unwrap()
                .check(verified)
                .passed()
        );
        assert_eq!(
            solve.axioms,
            Some(vec!["propext".to_string(), "Quot.sound".to_string()])
        );

        // Lean saw the spec once (validation, with `sorry`), then the proof,
        // then the proof again for the axiom audit.
        let compiled = lean.sources();
        assert_eq!(compiled.len(), 3);
        assert_eq!(compiled[0], translation.spec);
        assert_eq!(compiled[1], verified);
        assert!(compiled[2].ends_with(
            "#print axioms _root_.applyDiscount_nonneg\n#print axioms _root_.applyDiscount_le_price\n"
        ));
        assert_eq!(translation.lean_runs + solve.lean_runs, 3);

        // The model was asked twice, in two different roles.
        let calls = llm.calls();
        assert_eq!(calls.len(), 2);
        assert!(calls[0].system.contains(prompts::TYPE_FIDELITY_RULE));
        assert!(calls[0].system.contains("Lean (version 4.36.0)"));
        assert!(
            calls[0].messages[0]
                .content
                .contains("def apply_discount(price_cents: int, discount_pct: int) -> int:")
        );
        assert!(calls[0].messages[0].content.contains("`pricing.py`"));
        assert!(calls[1].system.contains("proof engineer"));
        assert!(
            calls[1].messages[0]
                .content
                .contains(translation.spec.trim_end())
        );
    }

    #[tokio::test]
    async fn rejects_nat_for_a_rust_u32_without_running_lean() {
        let llm = MockProvider::new([midpoint_spec("Nat"), midpoint_spec("UInt32")]);
        let lean = ScriptedLean::new([accepted(SORRY_WARNINGS)]);
        let mut problems = Vec::new();
        let mut observer = |event: &TranslateEvent<'_>| {
            if let TranslateEvent::Rejected { problem, .. } = event {
                problems.push((*problem).clone());
            }
        };

        let translation = Translator::new(&llm, &lean)
            .translate(
                MIDPOINT_RS,
                SourceLanguage::Rust,
                "midpoint.rs",
                &mut observer,
            )
            .await
            .unwrap();

        assert_eq!(translation.attempts, 2);
        assert!(translation.spec.contains("(lo hi : UInt32) : UInt32"));
        // The `Nat` model never reached Lean.
        assert_eq!(lean.sources(), std::slice::from_ref(&translation.spec));

        let [SpecProblem::TypeFidelity(messages)] = problems.as_slice() else {
            panic!("expected one type-fidelity rejection, got {problems:?}");
        };
        assert!(
            messages.iter().any(|m| m.contains("unbounded `Nat`")),
            "{messages:?}"
        );
        assert!(
            messages
                .iter()
                .any(|m| m.contains("Rust `u32`") && m.contains("`UInt32`")),
            "{messages:?}"
        );

        let feedback = &llm.calls()[1].messages[2].content;
        assert!(feedback.contains("strict type-fidelity rule"));
        assert!(feedback.contains("must be `UInt32` in `midpoint`"));
    }

    #[tokio::test]
    async fn retries_when_lean_rejects_the_spec() {
        let broken = "```lean\ndef midpoint (lo hi : UInt32) : UInt32 :=\n  (lo + hi) / 2\n\n\
                      theorem a (lo hi : UInt32) : lo ≤ midpoint lo hi hi := by sorry\n\n\
                      theorem b (lo hi : UInt32) : midpoint lo hi ≤ hi := by sorry\n```";
        let llm = MockProvider::new([broken.to_string(), midpoint_spec("UInt32")]);
        let lean = ScriptedLean::new([
            rejected(
                "Candidate.lean:4:35: error: Function expected at\n  midpoint lo hi\nbut this term has type\n  UInt32\n",
            ),
            accepted(SORRY_WARNINGS),
        ]);

        let translation = Translator::new(&llm, &lean)
            .translate(
                MIDPOINT_RS,
                SourceLanguage::Rust,
                "midpoint.rs",
                &mut Silent,
            )
            .await
            .unwrap();
        assert_eq!(translation.attempts, 2);
        assert_eq!(translation.lean_runs, 2);

        let calls = llm.calls();
        assert_eq!(calls[1].messages.len(), 3);
        let feedback = &calls[1].messages[2].content;
        assert!(
            feedback.contains("Error at line 4, column 35:"),
            "{feedback}"
        );
        assert!(
            feedback
                .contains("  4 | theorem a (lo hi : UInt32) : lo ≤ midpoint lo hi hi := by sorry"),
            "{feedback}"
        );
        assert!(
            feedback.contains(
                "Function expected at\n  midpoint lo hi\nbut this term has type\n  UInt32"
            )
        );
    }

    #[tokio::test]
    async fn gives_up_after_two_retries() {
        let llm = MockProvider::new([
            "no code here",
            "still nothing",
            "```lean\n-- empty\n```",
            "unused",
        ]);
        let lean = ScriptedLean::default();
        let error = Translator::new(&llm, &lean)
            .translate(
                MIDPOINT_RS,
                SourceLanguage::Rust,
                "midpoint.rs",
                &mut Silent,
            )
            .await
            .unwrap_err();

        let TranslateError::Exhausted { attempts, last } = &error else {
            panic!("expected exhaustion, got {error:?}");
        };
        assert_eq!(*attempts, MAX_TRANSLATION_RETRIES + 1);
        assert!(last.contains("expected 2 or 3 theorems"), "{last}");
        assert_eq!(llm.remaining(), 1, "exactly three model calls");
        assert!(lean.sources().is_empty());
    }

    #[tokio::test]
    async fn reports_unsupported_code_and_lean_timeouts() {
        let llm = MockProvider::new(["UNSUPPORTED: the function reads a file, which is not pure."]);
        let lean = ScriptedLean::default();
        let error = Translator::new(&llm, &lean)
            .translate(
                "def f(): return open('x').read()",
                SourceLanguage::Python,
                "f.py",
                &mut Silent,
            )
            .await
            .unwrap_err();
        assert!(
            matches!(&error, TranslateError::Unsupported(reason) if reason.starts_with("the function reads a file"))
        );

        let llm = MockProvider::new([midpoint_spec("UInt32")]);
        let lean = ScriptedLean::new([hung()]);
        let error = Translator::new(&llm, &lean)
            .with_max_retries(0)
            .translate(
                MIDPOINT_RS,
                SourceLanguage::Rust,
                "midpoint.rs",
                &mut Silent,
            )
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("Lean timed out on the spec"),
            "{error}"
        );
    }

    // -- lint -------------------------------------------------------------------

    fn shape_problems(lean: &str) -> Vec<String> {
        match lint(MIDPOINT_RS, SourceLanguage::Rust, lean) {
            Err(SpecProblem::Shape(problems)) => problems,
            other => panic!("expected shape problems, got {other:?}"),
        }
    }

    #[test]
    fn lint_accepts_a_faithful_spec() {
        let spec = lint(
            MIDPOINT_RS,
            SourceLanguage::Rust,
            &code_of(&midpoint_spec("UInt32")),
        )
        .unwrap();
        assert_eq!(spec.targets().len(), 2);

        // The first reply of the replay fixture is the translation of pricing.py.
        let translation = PRICING_REPLAY
            .split(crate::agent::llm::MOCK_SCRIPT_SEPARATOR)
            .next()
            .unwrap();
        assert!(lint(PRICING_PY, SourceLanguage::Python, &code_of(translation)).is_ok());
    }

    #[test]
    fn lint_requires_two_or_three_sorry_theorems() {
        let model = "def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2\n";
        let theorem =
            |n: usize| format!("theorem t{n} (x : UInt32) : midpoint x x = x := by sorry\n");

        let one = format!("{model}{}", theorem(1));
        assert_eq!(
            shape_problems(&one),
            ["expected 2 or 3 theorems ending in `:= by sorry`, found 1"]
        );
        let four: String = std::iter::once(model.to_string())
            .chain((1..=4).map(theorem))
            .collect();
        assert_eq!(
            shape_problems(&four),
            ["expected 2 or 3 theorems ending in `:= by sorry`, found 4"]
        );
        let three: String = std::iter::once(model.to_string())
            .chain((1..=3).map(theorem))
            .collect();
        assert!(lint(MIDPOINT_RS, SourceLanguage::Rust, &three).is_ok());

        // Theorems that are already proven do not count as invariants to prove.
        let proven = format!("{model}theorem a : True := trivial\ntheorem b : True := trivial\n");
        assert_eq!(
            shape_problems(&proven),
            ["expected 2 or 3 theorems ending in `:= by sorry`, found 0"]
        );
    }

    #[test]
    fn lint_rejects_incomplete_models_and_code_execution() {
        let theorems = "theorem a (x : UInt32) : midpoint x x = x := by sorry\n\
                        theorem b (x : UInt32) : midpoint x x ≤ x := by sorry\n";

        let hole = format!("def midpoint (lo hi : UInt32) : UInt32 := sorry\n{theorems}");
        assert_eq!(
            shape_problems(&hole),
            [
                "`sorry` may only stand for the proof of a theorem, but the def `midpoint` contains one"
            ]
        );

        let renamed = format!(
            "def mid (lo hi : UInt32) : UInt32 := (lo + hi) / 2\n{}",
            theorems.replace("midpoint", "mid")
        );
        assert_eq!(
            shape_problems(&renamed),
            ["the file must contain a `def` named after the source function `midpoint`"]
        );

        // Compiling this would run a shell command on the user's machine.
        let model = "def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2\n";
        let evil =
            format!("import Lean\n{model}#eval IO.Process.run {{ cmd := \"sh\" }}\n{theorems}");
        assert_eq!(
            shape_problems(&evil),
            [
                "line 1:0: `import` is not allowed",
                "line 3:0: `#eval` commands are not allowed"
            ]
        );

        let axiom = format!("{model}axiom cheat : False\n{theorems}");
        assert!(matches!(
            lint(MIDPOINT_RS, SourceLanguage::Rust, &axiom),
            Err(SpecProblem::Forbidden(_))
        ));
    }

    #[test]
    fn type_fidelity_for_rust_demands_exact_fixed_width_types() {
        let check = |rust: &str, lean: &str| {
            check_type_fidelity(rust, SourceLanguage::Rust, &Spec::parse(lean).unwrap())
        };

        assert!(
            check(
                MIDPOINT_RS,
                "def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2"
            )
            .is_empty()
        );
        // Unbounded models hide overflow.
        assert_eq!(
            check(
                MIDPOINT_RS,
                "def midpoint (lo hi : Nat) : Nat := (lo + hi) / 2"
            )
            .len(),
            2
        );
        assert_eq!(
            check(
                MIDPOINT_RS,
                "def midpoint (lo hi : Int) : Int := (lo + hi) / 2"
            )
            .len(),
            2
        );
        // The wrong width is just as wrong.
        let wider = check(
            MIDPOINT_RS,
            "def midpoint (lo hi : UInt64) : UInt64 := (lo + hi) / 2",
        );
        assert_eq!(
            wider,
            ["Rust `u32` in the signature of `midpoint` must be `UInt32` in `midpoint`"]
        );
        // Mixed signatures: every source type must be represented.
        let mixed = "pub fn scale(x: u8, factor: u64) -> u64 { x as u64 * factor }";
        assert!(
            check(
                mixed,
                "def scale (x : UInt8) (factor : UInt64) : UInt64 := x.toUInt64 * factor"
            )
            .is_empty()
        );
        assert_eq!(
            check(
                mixed,
                "def scale (x : UInt64) (factor : UInt64) : UInt64 := x * factor"
            )
            .len(),
            1
        );
        // Signed integers may use `BitVec`.
        let signed = "fn neg(x: i32) -> i32 { -x }";
        assert!(check(signed, "def neg (x : Int32) : Int32 := -x").is_empty());
        assert!(check(signed, "def neg (x : BitVec 32) : BitVec 32 := -x").is_empty());
        assert_eq!(check(signed, "def neg (x : Int) : Int := -x").len(), 2);
        // Only the signature is inspected: `.toNat` in a body or a helper with
        // a `Nat` fuel parameter is fine.
        let helper = "def midpoint (lo hi : UInt32) : UInt32 := go 3 lo hi\nwhere go (fuel : Nat) (a b : UInt32) : UInt32 := a";
        assert!(check(MIDPOINT_RS, helper).is_empty());
        let pattern = "def midpoint : UInt32 → UInt32 → UInt32\n  | lo, hi => UInt32.ofNat ((lo.toNat + hi.toNat) / 2)";
        assert!(check(MIDPOINT_RS, pattern).is_empty());
    }

    #[test]
    fn type_fidelity_for_python_demands_unbounded_int() {
        let check = |lean: &str| {
            check_type_fidelity(
                PRICING_PY,
                SourceLanguage::Python,
                &Spec::parse(lean).unwrap(),
            )
        };

        assert!(check("def applyDiscount (priceCents discountPct : Int) : Int := 0").is_empty());
        assert!(check("def apply_discount (price_cents discount_pct : Int) : Int := 0").is_empty());
        let nat = check("def applyDiscount (priceCents discountPct : Nat) : Nat := 0");
        assert!(nat.iter().any(|m| m.contains("can be negative")), "{nat:?}");
        let fixed = check("def applyDiscount (priceCents discountPct : UInt32) : UInt32 := 0");
        assert!(fixed.iter().any(|m| m.contains("unbounded")), "{fixed:?}");
        // Other definitions are not held to the function's signature.
        assert!(
            check(
                "def clampPct (pct : Nat) : Nat := pct\ndef applyDiscount (p d : Int) : Int := 0"
            )
            .is_empty()
        );
    }

    #[test]
    fn finds_functions_and_their_integer_types() {
        let rust = source_functions(
            "/// Docs\npub fn midpoint(lo: u32, hi: u32) -> u32 { (lo + hi) / 2 }\nfn helper<T>(x: T, n: usize) -> i64 { 0 }",
            SourceLanguage::Rust,
        );
        assert_eq!(rust.len(), 2);
        assert_eq!(rust[0].name, "midpoint");
        assert_eq!(rust[0].integer_types, BTreeSet::from(["u32".to_string()]));
        assert_eq!(
            rust[1].integer_types,
            BTreeSet::from(["i64".to_string(), "usize".to_string()])
        );

        let python = source_functions(PRICING_PY, SourceLanguage::Python);
        assert_eq!(python.len(), 1);
        assert_eq!(python[0].name, "apply_discount");
        assert_eq!(python[0].integer_types, BTreeSet::from(["int".to_string()]));
        // `print` contains "int" but is not the type.
        let untyped = source_functions("def show(x):\n    print(x)\n", SourceLanguage::Python);
        assert!(untyped[0].integer_types.is_empty());
    }

    #[test]
    fn detects_the_language_from_the_extension() {
        assert_eq!(
            SourceLanguage::from_path(Path::new("examples/pricing.py")).unwrap(),
            SourceLanguage::Python
        );
        assert_eq!(
            SourceLanguage::from_path(Path::new("src/MIDPOINT.RS")).unwrap(),
            SourceLanguage::Rust
        );
        for other in ["demo.lean", "Makefile", "archive.tar.gz"] {
            let error = SourceLanguage::from_path(Path::new(other)).unwrap_err();
            assert!(
                error.to_string().contains("expected a .py or .rs file"),
                "{error}"
            );
        }
    }
}
