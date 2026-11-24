//! Parser for the diagnostics the `lean` binary prints.
//!
//! Lean reports problems as
//!
//! ```text
//! path/to/File.lean:12:4: error: unsolved goals
//! case cons
//! α : Type
//! ih : myAppend xs [] = xs
//! ⊢ myAppend (x :: xs) [] = x :: xs
//! ```
//!
//! A header line carries the position and severity; everything up to the next
//! header belongs to the message. That body is preserved verbatim, because the
//! goal state it contains is the most useful thing to hand back to the model.
//!
//! Recent Lean versions decorate the header with an end position
//! (`12:4-12:9`) or an error name (`error(lean.unknownIdentifier)`); both are
//! understood. Output of `#eval`, `#check` and `#print` carries no position at all
//! and cannot be told apart from message text, so it is kept with whatever
//! diagnostic precedes it.

use std::fmt;
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;

static HEADER: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+)(?:-(?P<end_line>\d+):(?P<end_col>\d+))?: (?P<severity>error|warning|info|information|trace)(?:\((?P<name>[^)\s]*)\))?:(?: (?P<message>.*))?$",
    )
    .expect("valid regex")
});

static USES_SORRY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"declaration uses\s+[`'‘]?sorry").expect("valid regex"));

/// How serious a diagnostic is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Severity {
    Error,
    Warning,
    Info,
}

impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Info => "info",
        })
    }
}

/// One message from the Lean compiler.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LeanDiagnostic {
    /// File the message refers to, as Lean printed it.
    pub file: String,
    /// 1-based line.
    pub line: usize,
    /// 0-based column, counted in code points (Lean's convention).
    pub column: usize,
    pub severity: Severity,
    /// Named error kind such as `lean.unknownIdentifier`, when Lean provides one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// The full message, including multi-line tactic states.
    pub message: String,
}

impl LeanDiagnostic {
    pub fn is_error(&self) -> bool {
        self.severity == Severity::Error
    }

    /// The first line of the message, e.g. `unsolved goals`.
    pub fn summary(&self) -> &str {
        self.message.lines().next().unwrap_or("")
    }

    /// Whether this is Lean's "declaration uses `sorry`" warning.
    pub fn mentions_sorry(&self) -> bool {
        USES_SORRY.is_match(&self.message)
    }

    /// Goal states contained in the message (see [`parse_goals`]).
    pub fn goals(&self) -> Vec<Goal> {
        parse_goals(&self.message)
    }
}

impl fmt::Display for LeanDiagnostic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}:{}: {}: {}",
            self.line, self.column, self.severity, self.message
        )
    }
}

/// A proof goal as Lean prints it: hypotheses above the turnstile, target below.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Goal {
    /// The `case` tag, if the goal has one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub case: Option<String>,
    /// Hypotheses, one entry per binder group (`x y : Nat`).
    pub hypotheses: Vec<String>,
    /// The proposition after `⊢`.
    pub target: String,
}

impl fmt::Display for Goal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(case) = &self.case {
            writeln!(f, "case {case}")?;
        }
        for hypothesis in &self.hypotheses {
            writeln!(f, "{hypothesis}")?;
        }
        write!(f, "⊢ {}", self.target)
    }
}

/// Parses everything Lean printed into diagnostics.
///
/// Lean writes diagnostics to stdout; pass stdout followed by stderr to also pick
/// up anything a wrapper printed. Text before the first header is ignored.
pub fn parse_diagnostics(output: &str) -> Vec<LeanDiagnostic> {
    let mut diagnostics: Vec<LeanDiagnostic> = Vec::new();

    for line in output.lines() {
        let line = line.strip_suffix('\r').unwrap_or(line);
        if let Some(header) = HEADER.captures(line) {
            let severity = match &header["severity"] {
                "error" => Severity::Error,
                "warning" => Severity::Warning,
                _ => Severity::Info,
            };
            diagnostics.push(LeanDiagnostic {
                file: header["file"].to_string(),
                line: header["line"].parse().unwrap_or(0),
                column: header["col"].parse().unwrap_or(0),
                severity,
                name: header
                    .name("name")
                    .map(|name| name.as_str().to_string())
                    .filter(|name| !name.is_empty()),
                message: header
                    .name("message")
                    .map_or_else(String::new, |message| message.as_str().to_string()),
            });
        } else if let Some(current) = diagnostics.last_mut() {
            if !current.message.is_empty() {
                current.message.push('\n');
            }
            current.message.push_str(line);
        }
    }

    for diagnostic in &mut diagnostics {
        let trimmed = diagnostic.message.trim_end().len();
        diagnostic.message.truncate(trimmed);
    }
    diagnostics
}

/// Extracts the goal states from a message such as `unsolved goals ...`.
///
/// Goals are separated by blank lines. Within a goal, lines before the `⊢` are an
/// optional `case` tag followed by hypotheses; the target may continue over
/// several indented lines. Text that is not part of a goal (the leading
/// `unsolved goals`, a trailing hint) is skipped.
pub fn parse_goals(message: &str) -> Vec<Goal> {
    let mut goals = Vec::new();

    for block in message.split("\n\n") {
        let mut case = None;
        let mut hypotheses: Vec<String> = Vec::new();
        let mut target: Option<String> = None;

        for line in block.lines() {
            if let Some(target) = target.as_mut() {
                // The target has started; indented lines continue it.
                if line.starts_with(' ') {
                    target.push('\n');
                    target.push_str(line);
                    continue;
                }
                break;
            }
            if let Some(rest) = line.strip_prefix("⊢ ") {
                target = Some(rest.to_string());
            } else if let Some(tag) = line.strip_prefix("case ") {
                case = Some(tag.trim().to_string());
                hypotheses.clear();
            } else if line.starts_with(' ') && !hypotheses.is_empty() {
                // A long hypothesis wrapped onto the next line.
                let last = hypotheses.len() - 1;
                hypotheses[last].push('\n');
                hypotheses[last].push_str(line);
            } else if is_hypothesis(line) {
                hypotheses.push(line.to_string());
            } else {
                // Prose such as "unsolved goals" or "error: ..." resets the scan.
                hypotheses.clear();
                case = None;
            }
        }

        if let Some(target) = target {
            goals.push(Goal {
                case,
                hypotheses,
                target,
            });
        }
    }
    goals
}

/// A hypothesis line looks like `names : type` (or `name : type := value`).
fn is_hypothesis(line: &str) -> bool {
    let Some((names, _)) = line.split_once(" : ") else {
        return false;
    };
    !names.is_empty()
        && !names.starts_with(' ')
        && names
            .split(' ')
            .all(|name| !name.is_empty() && !name.ends_with(':') && !name.contains('('))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real output of Lean 4.36.0 (paths shortened).
    const OUTPUT: &str = "\
/tmp/x/Demo.lean:8:17: error: unsolved goals
case cons
α : Type
x : α
xs : List α
ih : myAppend xs [] = xs
⊢ myAppend (x :: xs) [] = x :: xs
/tmp/x/Demo.lean:10:60: error: unsolved goals
case left
p q : Prop
hp : p
hq : q
⊢ p

case right
p q : Prop
hp : p
hq : q
⊢ q
/tmp/x/Demo.lean:13:8: warning: declaration uses `sorry`
/tmp/x/Demo.lean:16:8: error(lean.unknownIdentifier): Unknown constant `Nat.this_lemma_does_not_exist`
/tmp/x/Demo.lean:19:8: error: Type mismatch
  rfl
has type
  ?m.17 = ?m.17
but is expected to have type
  n = n + 1
";

    #[test]
    fn parses_positions_severities_and_names() {
        let diagnostics = parse_diagnostics(OUTPUT);
        let heads: Vec<_> = diagnostics
            .iter()
            .map(|d| (d.line, d.column, d.severity, d.summary().to_string()))
            .collect();
        assert_eq!(
            heads,
            vec![
                (8, 17, Severity::Error, "unsolved goals".to_string()),
                (10, 60, Severity::Error, "unsolved goals".to_string()),
                (
                    13,
                    8,
                    Severity::Warning,
                    "declaration uses `sorry`".to_string()
                ),
                (
                    16,
                    8,
                    Severity::Error,
                    "Unknown constant `Nat.this_lemma_does_not_exist`".to_string()
                ),
                (19, 8, Severity::Error, "Type mismatch".to_string()),
            ]
        );
        assert!(diagnostics.iter().all(|d| d.file == "/tmp/x/Demo.lean"));
        assert_eq!(
            diagnostics[3].name.as_deref(),
            Some("lean.unknownIdentifier")
        );
        assert_eq!(diagnostics[0].name, None);
        assert!(diagnostics[2].mentions_sorry());
        assert!(!diagnostics[0].mentions_sorry());
    }

    #[test]
    fn preserves_multi_line_messages() {
        let diagnostics = parse_diagnostics(OUTPUT);
        assert_eq!(
            diagnostics[0].message,
            "unsolved goals\ncase cons\nα : Type\nx : α\nxs : List α\nih : myAppend xs [] = xs\n⊢ myAppend (x :: xs) [] = x :: xs"
        );
        // Indentation and blank lines inside a message survive.
        assert_eq!(
            diagnostics[4].message,
            "Type mismatch\n  rfl\nhas type\n  ?m.17 = ?m.17\nbut is expected to have type\n  n = n + 1"
        );
        assert!(diagnostics[1].message.contains("⊢ p\n\ncase right"));
    }

    #[test]
    fn extracts_unsolved_goals() {
        let diagnostics = parse_diagnostics(OUTPUT);
        assert_eq!(
            diagnostics[0].goals(),
            vec![Goal {
                case: Some("cons".into()),
                hypotheses: vec![
                    "α : Type".into(),
                    "x : α".into(),
                    "xs : List α".into(),
                    "ih : myAppend xs [] = xs".into(),
                ],
                target: "myAppend (x :: xs) [] = x :: xs".into(),
            }]
        );

        let two = diagnostics[1].goals();
        assert_eq!(two.len(), 2);
        assert_eq!(two[0].case.as_deref(), Some("left"));
        assert_eq!(two[0].target, "p");
        assert_eq!(two[1].case.as_deref(), Some("right"));
        assert_eq!(two[1].hypotheses, ["p q : Prop", "hp : p", "hq : q"]);
        assert_eq!(two[1].target, "q");

        // Messages without a turnstile contain no goals.
        assert!(diagnostics[4].goals().is_empty());
    }

    #[test]
    fn keeps_wrapped_targets_together() {
        let output = "\
F.lean:30:205: error: unsolved goals
a b c : Nat
⊢ a + b + c + 100000000 + 200000000 + 300000000 =
    c + b + a + 1
";
        let goals = parse_diagnostics(output)[0].goals();
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].case, None);
        assert_eq!(goals[0].hypotheses, ["a b c : Nat"]);
        assert_eq!(
            goals[0].target,
            "a + b + c + 100000000 + 200000000 + 300000000 =\n    c + b + a + 1"
        );
        assert_eq!(
            goals[0].to_string(),
            "a b c : Nat\n⊢ a + b + c + 100000000 + 200000000 + 300000000 =\n    c + b + a + 1"
        );
    }

    #[test]
    fn finds_goals_inside_other_tactic_failures() {
        // `grind` and friends print the goal first and diagnostics afterwards.
        let message = "\
`grind` failed
case grind.1
lo hi : UInt32
h : lo ≤ hi
h_1 : ¬lo ≤ (lo + hi) / 2
⊢ False
[grind] Goal diagnostics
  [facts] Asserted facts
    [prop] lo ≤ hi";
        let goals = parse_goals(message);
        assert_eq!(goals.len(), 1);
        assert_eq!(goals[0].case.as_deref(), Some("grind.1"));
        assert_eq!(goals[0].hypotheses.len(), 3);
        assert_eq!(goals[0].target, "False");
    }

    #[test]
    fn keeps_blank_lines_and_hints_in_one_message() {
        let output = "\
F.lean:4:41: warning: Variable name `h` is not explicitly referenced.

Hint: The binding can be removed (if unused) or named `_`:
  [apply] _h

Note: This linter can be disabled with `set_option linter.unusedVariables false`
F.lean:9:0: error: boom
";
        let diagnostics = parse_diagnostics(output);
        assert_eq!(diagnostics.len(), 2);
        assert!(
            diagnostics[0]
                .message
                .ends_with("linter.unusedVariables false`")
        );
        assert!(diagnostics[0].message.contains("\n\nHint:"));
        assert_eq!(diagnostics[1].message, "boom");
    }

    #[test]
    fn understands_header_variants() {
        let output = "\
C:\\Users\\me\\proof.lean:3:2-3:9: error: range header
F.lean:1:0: info: hello
F.lean:2:0: error:
message on the next line
F.lean:5:1: warning: declaration uses 'sorry'
";
        let diagnostics = parse_diagnostics(output);
        assert_eq!(diagnostics.len(), 4);
        assert_eq!(diagnostics[0].file, "C:\\Users\\me\\proof.lean");
        assert_eq!((diagnostics[0].line, diagnostics[0].column), (3, 2));
        assert_eq!(diagnostics[1].severity, Severity::Info);
        assert_eq!(diagnostics[2].message, "message on the next line");
        // Older Lean versions quote `sorry` with apostrophes.
        assert!(diagnostics[3].mentions_sorry());
    }

    #[test]
    fn ignores_output_without_diagnostics() {
        assert!(parse_diagnostics("").is_empty());
        // `#eval` output carries no position and precedes any diagnostic here.
        assert!(parse_diagnostics("2\n'foo' does not depend on any axioms\n").is_empty());
        // Prose that merely mentions a position is not a header.
        assert!(parse_diagnostics("see Foo.lean:3:4 for details: error\n").is_empty());
    }

    #[test]
    fn displays_like_lean_without_the_path() {
        let diagnostic = &parse_diagnostics(OUTPUT)[3];
        assert_eq!(
            diagnostic.to_string(),
            "16:8: error: Unknown constant `Nat.this_lemma_does_not_exist`"
        );
    }
}
