//! Prompt templates for turning a Python or Rust function into a Lean 4 spec.
//!
//! The most important text in this file is [`TYPE_FIDELITY_RULE`]. A model that
//! translates a Rust `u32` to Lean's unbounded `Nat` produces a spec in which
//! overflow cannot happen, and every overflow bug then "verifies". The rule is
//! part of every translation prompt, and [`super::check_type_fidelity`] enforces
//! the same mapping on the result.

use std::fmt::Write as _;

use super::{MAX_THEOREMS, MIN_THEOREMS, SourceLanguage, SpecProblem};
use crate::agent::prompts::{clip, error_listing};

/// The mapping from source types to Lean types that every translation must obey.
pub const TYPE_FIDELITY_RULE: &str = "\
## Strict type-fidelity rule
Every type in the model must have exactly the value range of the source type.

| Source type                    | Lean 4 type                                              |
|--------------------------------|----------------------------------------------------------|
| Python `int`                   | `Int`                                                    |
| Python `bool`                  | `Bool`                                                   |
| Rust `u8`, `u16`, `u32`, `u64` | `UInt8`, `UInt16`, `UInt32`, `UInt64`                    |
| Rust `usize`                   | `USize`                                                  |
| Rust `i8`, `i16`, `i32`, `i64` | `Int8`, `Int16`, `Int32`, `Int64` (or `BitVec n` when an operation has no counterpart on these types) |
| Rust `isize`                   | `ISize`                                                  |
| Rust `bool`                    | `Bool`                                                   |

- NEVER model a Rust fixed-width integer with the unbounded `Nat` or `Int`. Doing so \
silently assumes that overflow cannot happen and hides exactly the bugs this tool exists to \
find.
- Rust `+`, `-` and `*` become the same operators on the fixed-width Lean type. They wrap \
around on overflow, as Rust does in release builds, and that is the behaviour to verify. \
Rust `/` and `%` become `/` and `%`.
- Do not add a hypothesis whose only purpose is to rule out overflow or wraparound, and do \
not weaken a property to step around it. If the natural property fails because of overflow, \
it has to stay unprovable.
- Python `int` is unbounded, so it is always `Int`: never `Nat`, and never a fixed-width \
type, even when the values are expected to be non-negative.
- Python `//` and `%` round towards negative infinity. For a positive literal divisor they \
agree with Lean's `/` and `%` on `Int`; otherwise use `Int.fdiv` and `Int.fmod`.
";

/// System prompt for the translation step.
pub fn system_prompt(language: SourceLanguage, lean_version: Option<&str>) -> String {
    let toolchain = lean_version.unwrap_or("a recent Lean 4 release");
    let language = language.name();
    format!(
        "\
You translate a pure, deterministic {language} function into a faithful Lean 4 model and \
state the properties its callers rely on. An automated prover then tries to prove those \
properties; a property it cannot prove points at a bug in the code.

## What to write
One self-contained Lean 4 file. It is compiled on its own with `lean` ({toolchain}), so only \
Lean's core library is available: no imports and no Mathlib.

1. The model: a `def` for the function, named after it, plus a `def` for every helper it \
calls. Keep the same parameters in the same order and the same control flow and arithmetic. \
Translate what the code does, not what it was probably meant to do: never repair a suspected \
bug, and never rewrite an expression into one you consider equivalent.
2. The invariants: {MIN_THEOREMS} or {MAX_THEOREMS} `theorem` statements about the model, \
each ending in `:= by sorry`. Do not prove them.

## Choosing the invariants
- State what a caller would expect from the function's name, signature and documentation: \
bounds on the result, non-negativity, monotonicity, idempotence, fixed points, symmetry.
- Each theorem is one universally quantified statement. Prefer several simple theorems to \
one large conjunction.
- Add a hypothesis only for a precondition the function's contract obviously has, such as \
`lo ≤ hi` for the midpoint of a range.
- Keep the statements within reach of Lean's core automation (`omega`, `simp`, `decide`, \
`grind`, `induction`): linear arithmetic, and at most one product of two variables.

{TYPE_FIDELITY_RULE}
## Answer format
Reply with the Lean file in a single ```lean code block and nothing else.
- Use `sorry` only as the proof of the theorems; every definition must be complete.
- Do not use `axiom`, `unsafe`, `partial`, `implemented_by`, `extern`, `import`, or `#eval` \
and other `#` commands.
- If the function cannot be modelled faithfully under these rules (floating point, I/O, \
shared mutable state, loops without an obvious bound), reply with a single line that starts \
with `UNSUPPORTED:` and gives the reason instead.
"
    )
}

/// The user message carrying the source code.
pub fn translation_prompt(language: SourceLanguage, file_name: &str, source: &str) -> String {
    format!(
        "Translate this {} function from `{file_name}` into a Lean 4 model with invariants.\n\n```{}\n{}\n```\n",
        language.name(),
        language.fence(),
        source.trim_end()
    )
}

/// Feedback for a generated spec that was rejected, asking for a corrected one.
pub fn rejection_feedback(problem: &SpecProblem, spec: &str) -> String {
    let mut feedback = match problem {
        SpecProblem::NoCode => {
            "Your reply did not contain a Lean file in a ```lean code block.\n".to_string()
        }
        SpecProblem::Forbidden(violations) => {
            let mut text =
                String::from("The Lean file uses constructs that are not allowed in a spec:\n\n");
            for violation in violations {
                let _ = writeln!(text, "- {violation}");
            }
            text.push_str(
                "\n`sorry` is the only placeholder allowed, and only as the proof of a theorem.\n",
            );
            text
        }
        SpecProblem::Shape(problems) => {
            let mut text = String::from("The Lean file does not have the required shape:\n\n");
            for problem in problems {
                let _ = writeln!(text, "- {problem}");
            }
            text
        }
        SpecProblem::TypeFidelity(problems) => {
            let mut text =
                String::from("The Lean file violates the strict type-fidelity rule:\n\n");
            for problem in problems {
                let _ = writeln!(text, "- {problem}");
            }
            text.push_str(
                "\nEvery type in the model must have exactly the value range of the source type. \
                 Do not replace fixed-width integers by `Nat` or `Int`, and do not add hypotheses \
                 that rule out overflow.\n",
            );
            text
        }
        SpecProblem::Compile {
            diagnostics,
            exit_code,
            output,
        } => {
            let exit = exit_code.map_or_else(
                || "no exit code".to_string(),
                |code| format!("exit code {code}"),
            );
            if diagnostics.iter().any(|d| d.is_error()) {
                format!(
                    "Lean rejected the file ({exit}); it has to parse and type-check before \
                     anything can be proven:\n{}",
                    error_listing(spec, diagnostics)
                )
            } else {
                format!(
                    "Lean failed on the file ({exit}) without a positioned error. Its output was:\n\n{}\n",
                    clip(output.trim(), 40, 4_000)
                )
            }
        }
        SpecProblem::Timeout(elapsed) => format!(
            "Lean did not finish checking the file within {:.0} seconds. Keep the definitions \
             simple and structurally recursive.\n",
            elapsed.as_secs_f64()
        ),
    };
    feedback.push_str(
        "\nReturn the complete corrected Lean file in a single ```lean code block, keeping the \
         model faithful to the source code.",
    );
    feedback
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn the_type_fidelity_rule_is_part_of_every_translation_prompt() {
        for language in [SourceLanguage::Python, SourceLanguage::Rust] {
            let prompt = system_prompt(language, Some("Lean (version 4.36.0)"));
            assert!(prompt.contains(TYPE_FIDELITY_RULE), "{language:?}");
            assert!(prompt.contains(language.name()));
            assert!(prompt.contains("Lean (version 4.36.0)"));
        }
    }

    #[test]
    fn the_rule_maps_every_supported_type() {
        let rule = TYPE_FIDELITY_RULE;
        // Python: unbounded.
        assert!(rule.contains("| Python `int`                   | `Int`"));
        // Rust unsigned: fixed width, never Nat.
        assert!(
            rule.contains(
                "| Rust `u8`, `u16`, `u32`, `u64` | `UInt8`, `UInt16`, `UInt32`, `UInt64`"
            )
        );
        // Rust signed: fixed width, BitVec as the fallback.
        assert!(rule.contains(
            "| Rust `i8`, `i16`, `i32`, `i64` | `Int8`, `Int16`, `Int32`, `Int64` (or `BitVec n`"
        ));
        assert!(
            rule.contains(
                "NEVER model a Rust fixed-width integer with the unbounded `Nat` or `Int`"
            )
        );
        assert!(rule.contains("They wrap around on overflow"));
        assert!(rule.contains("rule out overflow"));
        assert!(rule.contains("it is always `Int`: never `Nat`"));
    }

    #[test]
    fn the_prompt_demands_a_lean_only_answer_with_sorry_theorems() {
        let prompt = system_prompt(SourceLanguage::Rust, None);
        for phrase in [
            "2 or 3 `theorem` statements",
            "`:= by sorry`",
            "no imports and no Mathlib",
            "single ```lean code block",
            "never repair a suspected bug",
            "UNSUPPORTED:",
        ] {
            assert!(prompt.contains(phrase), "missing {phrase:?}");
        }
    }

    #[test]
    fn the_translation_prompt_embeds_the_source() {
        let prompt = translation_prompt(
            SourceLanguage::Rust,
            "midpoint.rs",
            "pub fn midpoint() {}\n\n",
        );
        assert_eq!(
            prompt,
            "Translate this Rust function from `midpoint.rs` into a Lean 4 model with invariants.\n\n```rust\npub fn midpoint() {}\n```\n"
        );
        assert!(
            translation_prompt(SourceLanguage::Python, "p.py", "def f(): pass")
                .contains("```python\n")
        );
    }

    #[test]
    fn feedback_explains_each_kind_of_rejection() {
        let fidelity = SpecProblem::TypeFidelity(vec!["`midpoint` uses `Nat`".into()]);
        let feedback = rejection_feedback(&fidelity, "");
        assert!(feedback.contains("strict type-fidelity rule"));
        assert!(feedback.contains("- `midpoint` uses `Nat`"));
        assert!(feedback.ends_with("keeping the model faithful to the source code."));

        let shape = SpecProblem::Shape(vec!["expected 2 or 3 theorems".into()]);
        assert!(rejection_feedback(&shape, "").contains("- expected 2 or 3 theorems"));
        assert!(rejection_feedback(&SpecProblem::NoCode, "").contains("```lean code block"));
        assert!(
            rejection_feedback(&SpecProblem::Timeout(Duration::from_secs(30)), "")
                .contains("30 seconds")
        );

        let crash = SpecProblem::Compile {
            diagnostics: vec![],
            exit_code: None,
            output: "segfault\n".into(),
        };
        assert!(rejection_feedback(&crash, "").contains("segfault"));
    }
}
