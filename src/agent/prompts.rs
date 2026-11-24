//! Prompt templates for the proof-search loop.
//!
//! The system prompt states the contract once; every later user message is
//! feedback produced by this module from the guard's findings or from Lean's own
//! diagnostics, quoted verbatim.

use std::fmt::Write as _;
use std::time::Duration;

use crate::lean::{
    LeanDiagnostic, STANDARD_AXIOMS, TaintedTheorem, Target, Violation, ViolationKind,
};

/// Diagnostics beyond this many are summarised by a count.
const MAX_DIAGNOSTICS: usize = 8;
/// Tactic states can be enormous (`grind` prints hundreds of lines).
const MAX_MESSAGE_LINES: usize = 60;
const MAX_MESSAGE_CHARS: usize = 4_000;
const MAX_RAW_OUTPUT_LINES: usize = 40;

/// The contract between the loop and the model.
pub fn system_prompt(lean_version: Option<&str>) -> String {
    let toolchain = lean_version.unwrap_or("a recent Lean 4 release");
    format!(
        "\
You are an expert Lean 4 proof engineer working inside an automated verification loop.

You receive a Lean 4 file in which some proofs are the placeholder `sorry`. Return the same \
file with every `sorry` replaced by a complete proof. Your answer is checked mechanically: \
first by a guard that compares it with the original file, then by the Lean compiler. If a \
check fails you are shown the exact findings and can try again.

## Environment
- Toolchain: {toolchain}.
- The file is compiled on its own with `lean`, so only Lean's core library is available. \
There is no Mathlib, and you cannot add imports.
- Tactics that work here include `rfl`, `simp`, `simp only [..]`, `simp_all`, `omega` (linear \
arithmetic over `Int` and `Nat`, including `/` and `%` by literals), `decide`, `grind`, \
`induction`, `cases`, `rcases`, `split`, `constructor`, `exact`, `apply`, `refine`, `intro`, \
`rw`, `unfold`, `calc`, `funext`, `contradiction` and `exfalso`.
- Mathlib tactics do not exist here: `linarith`, `nlinarith`, `norm_num`, `ring`, \
`positivity`, `field_simp`, `aesop`, `tauto`, `bv_decide`.

## Rules the guard enforces
1. The original file is frozen. Reproduce every definition, theorem statement and other \
line exactly and in the original order. Do not weaken, strengthen, rename, reorder or delete \
anything, and do not repair a definition even if you believe it is wrong.
2. Replace each `sorry` with a proof. The only other text you may add is helper `theorem`s, \
placed between the original declarations (before the first one that uses them), with names \
that do not occur anywhere in the original file.
3. Never write `sorry`, `admit`, `axiom`, `unsafe`, `implemented_by`, `extern` or \
`native_decide`.
4. Add no other kind of command: no `def`, `instance`, `notation`, `macro`, `open`, \
`namespace`, `section`, `variable`, `attribute`, `import`, and no `#eval` or other `#` \
command. `set_option maxRecDepth` and `set_option maxHeartbeats` are allowed.

## Answer format
Reply with the complete Lean file in a single ```lean code block and nothing else.

One exception: if you are convinced that a theorem is false as stated (for example because \
fixed-width arithmetic overflows), leave its statement unchanged, still return your best \
attempt, and add one line after the code block that starts with `NOTE:` and gives a concrete \
counterexample.
"
    )
}

/// The first user message: the file to prove.
pub fn initial_prompt(spec: &str, targets: &[Target]) -> String {
    let holes: usize = targets.iter().map(|target| target.holes).sum();
    let labels: Vec<String> = targets.iter().map(Target::label).collect();
    format!(
        "Prove every `sorry` in this Lean 4 file ({holes} placeholder{} in {}).\n\n```lean\n{}\n```\n",
        if holes == 1 { "" } else { "s" },
        labels.join(", "),
        spec.trim_end()
    )
}

/// Feedback when a reply contained no Lean code at all.
pub fn no_code_feedback() -> String {
    "Your reply did not contain a Lean file. Reply with the complete file, including all of \
     the original text, in a single ```lean code block."
        .to_string()
}

/// Feedback for a candidate the guard rejected. Lean is not consulted for these.
pub fn guard_feedback(violations: &[Violation]) -> String {
    let mut feedback = String::from(
        "REJECTED by the anti-cheat guard. Lean was not run, because this answer breaks the \
         rules:\n\n",
    );
    for violation in violations {
        match (violation.line, violation.column) {
            (Some(line), Some(column)) => {
                let _ = writeln!(
                    feedback,
                    "- line {line}, column {column}: {}",
                    violation.message
                );
            }
            _ => {
                let _ = writeln!(feedback, "- {}", violation.message);
            }
        }
        if let Some(expected) = &violation.expected {
            feedback.push_str("  Original text (only its `sorry` may be replaced):\n  ```lean\n");
            for line in expected.lines() {
                let _ = writeln!(feedback, "  {line}");
            }
            feedback.push_str("  ```\n");
        }
    }

    feedback.push('\n');
    let has = |kind| violations.iter().any(|violation| violation.kind == kind);
    if has(ViolationKind::ForbiddenToken) {
        feedback.push_str(
            "Placeholders and escape hatches are never accepted: every proof must be complete \
             and checked by Lean's kernel.\n",
        );
    }
    if has(ViolationKind::AlteredSpec) {
        feedback.push_str(
            "The original file is frozen: copy it exactly, and prove the statements as given \
             even if they are hard.\n",
        );
    }
    if has(ViolationKind::DisallowedAddition) || has(ViolationKind::NameCollision) {
        feedback.push_str(
            "Besides proofs you may only add helper `theorem`s with fresh names, placed \
             between the original declarations.\n",
        );
    }
    feedback.push_str("Return the complete corrected file in a single ```lean code block.");
    feedback
}

/// Feedback for a candidate Lean rejected: the diagnostics, verbatim, each with
/// the source line it points at.
pub fn compile_feedback(
    candidate: &str,
    diagnostics: &[LeanDiagnostic],
    exit_code: Option<i32>,
    raw_output: &str,
) -> String {
    let errors = diagnostics.iter().filter(|d| d.is_error()).count();
    let exit = exit_code.map_or_else(
        || "no exit code".to_string(),
        |code| format!("exit code {code}"),
    );

    if errors == 0 {
        // A crash or a wrapper failure: there is nothing structured to show.
        return format!(
            "Lean failed ({exit}) without reporting a positioned error. Its output was:\n\n{}\n\n\
             Return a corrected, complete file in a single ```lean code block.",
            clip(raw_output.trim(), MAX_RAW_OUTPUT_LINES, MAX_MESSAGE_CHARS)
        );
    }

    format!(
        "Lean rejected the file ({exit}) with {}:\n{}\n\
         Use the goal states above to repair the proofs, changing strategy where an approach \
         keeps failing. Keep the original text exactly as it was and return the complete file \
         in a single ```lean code block.",
        count(errors, "error"),
        error_listing(candidate, diagnostics)
    )
}

/// Lists the errors among `diagnostics`, each with the source line it points at
/// and Lean's message quoted verbatim (long tactic states are clipped).
pub fn error_listing(source: &str, diagnostics: &[LeanDiagnostic]) -> String {
    let errors: Vec<&LeanDiagnostic> = diagnostics.iter().filter(|d| d.is_error()).collect();
    let lines: Vec<&str> = source.lines().collect();
    let mut listing = String::new();
    for diagnostic in errors.iter().take(MAX_DIAGNOSTICS) {
        let _ = write!(
            listing,
            "\nError at line {}, column {}:\n",
            diagnostic.line, diagnostic.column
        );
        if let Some(text) = diagnostic
            .line
            .checked_sub(1)
            .and_then(|index| lines.get(index))
        {
            let _ = writeln!(listing, "  {} | {}", diagnostic.line, text);
        }
        listing.push_str(&clip(
            &diagnostic.message,
            MAX_MESSAGE_LINES,
            MAX_MESSAGE_CHARS,
        ));
        listing.push('\n');
    }
    if errors.len() > MAX_DIAGNOSTICS {
        let _ = writeln!(
            listing,
            "\n({} further errors not shown)",
            errors.len() - MAX_DIAGNOSTICS
        );
    }
    listing
}

/// `1 error`, `2 errors`.
pub fn count(n: usize, noun: &str) -> String {
    format!("{n} {noun}{}", if n == 1 { "" } else { "s" })
}

/// Feedback when Lean had to be killed.
pub fn timeout_feedback(elapsed: Duration) -> String {
    format!(
        "Lean did not finish within {:.0} seconds and was stopped. A tactic is probably looping \
         or exploring a search space that is too large (`decide` on a big type, `simp` with a \
         rewriting cycle, an expensive `grind` call). Use a cheaper, more direct proof and \
         return the complete file in a single ```lean code block.",
        elapsed.as_secs_f64()
    )
}

/// Feedback when Lean accepted the file but a declaration still rests on `sorry`.
pub fn sorry_feedback(diagnostics: &[LeanDiagnostic]) -> String {
    let lines: Vec<String> = diagnostics.iter().map(|d| d.line.to_string()).collect();
    format!(
        "Lean compiled the file but reports that the declaration at line {} still depends on \
         `sorry`. Every proof must be complete. Return the complete corrected file in a single \
         ```lean code block.",
        lines.join(", ")
    )
}

/// Feedback when a proof compiles but depends on axioms beyond the standard three.
pub fn axiom_feedback(tainted: &[TaintedTheorem]) -> String {
    let mut feedback = String::from(
        "The file compiles, but the proof is not accepted because of the axioms it depends on:\n\n",
    );
    for theorem in tainted {
        let _ = writeln!(
            feedback,
            "- `{}` depends on {}",
            theorem.name,
            theorem.axioms.join(", ")
        );
    }
    let _ = write!(
        feedback,
        "\nOnly {} are allowed. Prove the statements with tactics that are checked by Lean's \
         kernel and return the complete file in a single ```lean code block.",
        STANDARD_AXIOMS.join(", ")
    );
    feedback
}

/// Feedback when Lean could not report on every target after a successful compile.
pub fn audit_inconclusive_feedback(reason: &str) -> String {
    format!(
        "The file compiles, but the final check of the target declarations failed: {reason}. \
         Make sure every original declaration is present under its original name and return \
         the complete file in a single ```lean code block."
    )
}

/// Shortens long text to its first lines, saying how much was left out.
pub fn clip(text: &str, max_lines: usize, max_chars: usize) -> String {
    let total = text.lines().count();
    let mut clipped = String::new();
    let mut kept = 0;
    for line in text.lines().take(max_lines) {
        if clipped.len() + line.len() > max_chars && kept > 0 {
            break;
        }
        if kept > 0 {
            clipped.push('\n');
        }
        clipped.push_str(line);
        kept += 1;
    }
    if kept < total {
        let _ = write!(clipped, "\n... ({} more lines omitted)", total - kept);
    }
    clipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lean::{FORBIDDEN_TOKENS, Severity, Spec};

    #[test]
    fn system_prompt_states_every_guard_rule() {
        let prompt = system_prompt(Some("Lean (version 4.36.0)"));
        assert!(prompt.contains("Lean (version 4.36.0)"));
        for token in FORBIDDEN_TOKENS {
            assert!(
                prompt.contains(&format!("`{token}`")),
                "{token} is not mentioned"
            );
        }
        for phrase in [
            "frozen",
            "helper `theorem`s",
            "no Mathlib",
            "```lean",
            "NOTE:",
            "native_decide",
        ] {
            assert!(prompt.contains(phrase), "missing {phrase:?}");
        }
        assert!(system_prompt(None).contains("a recent Lean 4 release"));
    }

    #[test]
    fn initial_prompt_embeds_the_spec_and_names_the_targets() {
        let source =
            "theorem a : True := by sorry\n\ntheorem b : True ∧ True := ⟨by sorry, by sorry⟩\n";
        let spec = Spec::parse(source).unwrap();
        let prompt = initial_prompt(source, &spec.targets());
        assert!(
            prompt.contains("3 placeholders in theorem `a`, theorem `b`"),
            "{prompt}"
        );
        assert!(prompt.contains(&format!("```lean\n{}\n```", source.trim_end())));
    }

    #[test]
    fn guard_feedback_is_strict_and_specific() {
        let spec = Spec::parse("def f := 1\ntheorem t : f = 1 := by sorry\n").unwrap();
        let report = spec.check("def f := 1\ntheorem t : f = 1 ∨ True := by\n  sorry\n");
        let feedback = guard_feedback(&report.violations);
        assert!(feedback.starts_with("REJECTED by the anti-cheat guard. Lean was not run"));
        assert!(
            feedback.contains("- line 3, column 2: forbidden token `sorry`"),
            "{feedback}"
        );
        assert!(feedback.contains("original text was changed"), "{feedback}");
        assert!(
            feedback.contains("  theorem t : f = 1 := by sorry"),
            "{feedback}"
        );
        assert!(feedback.contains("The original file is frozen"));
        assert!(feedback.contains("every proof must be complete"));
    }

    #[test]
    fn compile_feedback_quotes_lean_verbatim_with_source_lines() {
        let candidate =
            "theorem t (p q : Prop) (hp : p) : p ∧ q := by\n  constructor\n  · exact hp\n";
        let diagnostics = vec![
            LeanDiagnostic {
                file: "Candidate.lean".into(),
                line: 1,
                column: 44,
                severity: Severity::Error,
                name: None,
                message: "unsolved goals\ncase right\np q : Prop\nhp : p\n⊢ q".into(),
            },
            LeanDiagnostic {
                file: "Candidate.lean".into(),
                line: 2,
                column: 2,
                severity: Severity::Warning,
                name: None,
                message: "unused variable".into(),
            },
        ];
        let feedback = compile_feedback(candidate, &diagnostics, Some(1), "");
        assert!(
            feedback.starts_with("Lean rejected the file (exit code 1) with 1 error:"),
            "{feedback}"
        );
        assert!(feedback.contains(
            "Error at line 1, column 44:\n  1 | theorem t (p q : Prop) (hp : p) : p ∧ q := by\nunsolved goals\ncase right\np q : Prop\nhp : p\n⊢ q\n"
        ));
        assert!(
            !feedback.contains("unused variable"),
            "warnings are noise for repair"
        );
    }

    #[test]
    fn compile_feedback_falls_back_to_raw_output() {
        let feedback =
            compile_feedback("x", &[], Some(134), "Stack overflow detected. Aborting.\n");
        assert!(feedback.contains("exit code 134"));
        assert!(feedback.contains("Stack overflow detected. Aborting."));
    }

    #[test]
    fn long_messages_are_clipped() {
        let message = (1..=200)
            .map(|n| format!("line {n}"))
            .collect::<Vec<_>>()
            .join("\n");
        let clipped = clip(&message, 3, 10_000);
        assert_eq!(
            clipped,
            "line 1\nline 2\nline 3\n... (197 more lines omitted)"
        );
        assert_eq!(clip("short", 3, 10_000), "short");
        // The character budget cuts on line boundaries and always keeps one line.
        assert_eq!(
            clip("aaaa\nbbbb\ncccc", 10, 6),
            "aaaa\n... (2 more lines omitted)"
        );
    }

    #[test]
    fn other_feedback_names_the_problem() {
        assert!(timeout_feedback(Duration::from_secs(30)).contains("within 30 seconds"));
        assert!(no_code_feedback().contains("```lean"));
        let tainted = [TaintedTheorem {
            name: "t".into(),
            axioms: vec!["sorryAx".into()],
        }];
        let feedback = axiom_feedback(&tainted);
        assert!(feedback.contains("- `t` depends on sorryAx"));
        assert!(feedback.contains("propext, Classical.choice, Quot.sound"));
        assert!(audit_inconclusive_feedback("`t` is gone").contains("`t` is gone"));
    }
}
