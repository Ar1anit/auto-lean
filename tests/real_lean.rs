//! End-to-end tests against the real `lean` binary.
//!
//! The model is scripted (`MockProvider` replaying `tests/fixtures/*.replay.md`),
//! Lean is not: every candidate below is compiled by the toolchain found on this
//! machine. The tests skip themselves when Lean is not installed; set
//! `AUTO_LEAN_REQUIRE_LEAN=1` to turn that into a failure.

#[macro_use]
mod common;

use auto_lean::agent::llm::MockProvider;
use auto_lean::agent::{Rejection, Silent, SolveConfig, SolveEvent, Solver};
use auto_lean::lean::{
    AuditOutcome, LeanChecker, LeanRunner, STANDARD_AXIOMS, Severity, Spec, ViolationKind,
};
use auto_lean::translator::{SourceLanguage, prove_code};

use common::read;

const MIDPOINT_BUG_SPEC: &str = "\
def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2

theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by sorry
";

fn lean() -> LeanRunner {
    LeanRunner::new()
}

fn only_standard(axioms: &[String]) -> bool {
    axioms
        .iter()
        .all(|axiom| STANDARD_AXIOMS.contains(&axiom.as_str()))
}

// -- runner and parser -------------------------------------------------------

#[tokio::test]
async fn reports_the_toolchain_version() {
    require_lean!();
    let version = lean().version().await.unwrap();
    assert!(version.starts_with("Lean (version 4."), "{version}");
}

#[tokio::test]
async fn the_demo_compiles_as_shipped() {
    require_lean!();
    let output = lean()
        .check_file(&common::repo("examples/demo.lean"))
        .await
        .unwrap();
    assert!(output.success(), "{}", output.combined());

    // The only thing Lean has to say is that the theorem is still open.
    let diagnostics = output.diagnostics();
    assert_eq!(diagnostics.len(), 1, "{diagnostics:?}");
    assert_eq!(diagnostics[0].severity, Severity::Warning);
    assert!(diagnostics[0].mentions_sorry());
    assert_eq!(diagnostics[0].line, 24);
}

#[tokio::test]
async fn parses_what_lean_actually_prints() {
    require_lean!();
    let source = "\
theorem two_goals (p q : Prop) (hp : p) (hq : q) : p ∧ q := by
  constructor

theorem unknown (n : Nat) : n + 0 = n := by
  exact Nat.this_lemma_does_not_exist n

theorem open_question (n : Nat) : n = n := by sorry
";
    let output = lean().check(source).await.unwrap();
    assert_eq!(output.exit_code, Some(1));
    assert!(!output.success());

    let diagnostics = output.diagnostics();
    assert_eq!(diagnostics.len(), 3, "{}", output.combined());

    let unsolved = &diagnostics[0];
    assert!(unsolved.is_error());
    assert_eq!(unsolved.line, 1);
    assert_eq!(unsolved.summary(), "unsolved goals");
    let goals = unsolved.goals();
    assert_eq!(goals.len(), 2);
    assert_eq!(goals[0].case.as_deref(), Some("left"));
    assert_eq!(goals[0].hypotheses, ["p q : Prop", "hp : p", "hq : q"]);
    assert_eq!(goals[0].target, "p");
    assert_eq!(goals[1].case.as_deref(), Some("right"));
    assert_eq!(goals[1].target, "q");

    let unknown = &diagnostics[1];
    assert!(unknown.is_error());
    assert_eq!((unknown.line, unknown.column), (5, 8));
    assert!(
        unknown.message.contains("Nat.this_lemma_does_not_exist"),
        "{}",
        unknown.message
    );

    let sorry = &diagnostics[2];
    assert_eq!(sorry.severity, Severity::Warning);
    assert_eq!(sorry.line, 7);
    assert!(sorry.mentions_sorry());
}

#[tokio::test]
async fn kills_a_runaway_proof() {
    require_lean!();
    // Evaluating this never terminates.
    let source = "partial def spin (n : Nat) : Nat := spin (n + 1)\n#eval spin 0\n";
    let runner = lean().with_timeout(std::time::Duration::from_secs(2));
    let output = runner.check(source).await.unwrap();
    assert!(output.timed_out);
    assert!(!output.success());
    assert_eq!(output.exit_code, None);
    assert!(
        output.duration < std::time::Duration::from_secs(20),
        "{:?}",
        output.duration
    );
}

// -- Mode 1 ------------------------------------------------------------------

/// The scenario from the README: the first answer cheats with `sorry`, the
/// second fails in Lean, the third is verified.
#[tokio::test]
async fn mode1_repairs_a_cheat_and_a_compile_error() {
    require_lean!();
    let spec = read("examples/demo.lean");
    let llm = MockProvider::from_script(&read("tests/fixtures/demo.replay.md"));
    let runner = lean();

    let mut log = Vec::new();
    let mut observer = |event: &SolveEvent<'_>| {
        log.push(match event {
            SolveEvent::Rejected {
                rejection: Rejection::Guard(_),
                ..
            } => "guard rejected",
            SolveEvent::Rejected {
                rejection: Rejection::Compile { .. },
                ..
            } => "lean rejected",
            SolveEvent::Rejected { .. } => "rejected",
            SolveEvent::GuardPassed { .. } => "guard ok",
            SolveEvent::LeanPassed { .. } => "lean ok",
            SolveEvent::AuditPassed { .. } => "audit ok",
            SolveEvent::Verified { .. } => "verified",
            _ => return,
        });
    };
    let report = Solver::new(&llm, &runner)
        .solve(&spec, &mut observer)
        .await
        .unwrap();

    assert!(report.verified);
    assert_eq!(
        log,
        [
            "guard rejected",
            "guard ok",
            "lean rejected",
            "guard ok",
            "lean ok",
            "audit ok",
            "verified"
        ]
    );

    // 1: stopped by the guard, so Lean ran only for the spec, attempts 2 and 3
    // and the axiom audit.
    let Some(Rejection::Guard(violations)) = &report.attempts[0].rejection else {
        panic!("attempt 1: {:?}", report.attempts[0].rejection);
    };
    assert_eq!(violations.len(), 1);
    assert_eq!(violations[0].kind, ViolationKind::ForbiddenToken);
    assert_eq!(report.lean_runs, 4);

    // 2: a genuine Lean error whose goal state names the missing lemma.
    let Some(Rejection::Compile { diagnostics, .. }) = &report.attempts[1].rejection else {
        panic!("attempt 2: {:?}", report.attempts[1].rejection);
    };
    let goals = diagnostics[0].goals();
    assert_eq!(goals.len(), 1);
    assert_eq!(goals[0].case.as_deref(), Some("cons"));
    assert_eq!(
        goals[0].target,
        "myAppend (myAppend (myRev ys) (myRev xs)) [x] = myAppend (myRev ys) (myAppend (myRev xs) [x])"
    );
    assert_eq!(report.attempts[1].failing_targets, [0]);
    // That exact goal went back to the model.
    assert!(
        llm.calls()[2].messages[4]
            .content
            .contains(&goals[0].target)
    );

    // 3: verified, and independently re-checkable.
    let proof = report.proof.as_deref().unwrap();
    assert!(Spec::parse(&spec).unwrap().check(proof).passed());
    let recheck = runner.check(proof).await.unwrap();
    assert!(recheck.success(), "{}", recheck.combined());
    assert!(
        recheck
            .diagnostics()
            .iter()
            .all(|d| !d.is_error() && !d.mentions_sorry())
    );
    assert!(only_standard(report.axioms.as_deref().unwrap()));
}

// -- Mode 2 ------------------------------------------------------------------

#[tokio::test]
async fn mode2_proves_the_pricing_invariants() {
    require_lean!();
    let llm = MockProvider::from_script(&read("tests/fixtures/pricing.replay.md"));
    let runner = lean();
    let proof = prove_code(
        &llm,
        &runner,
        &read("examples/pricing.py"),
        SourceLanguage::Python,
        "pricing.py",
        SolveConfig::default(),
        &mut Silent,
        &mut Silent,
    )
    .await
    .unwrap();

    assert_eq!(proof.translation.attempts, 1);
    assert!(
        proof
            .translation
            .spec
            .contains("(priceCents discountPct : Int) : Int")
    );
    let names: Vec<_> = proof
        .solve
        .targets
        .iter()
        .filter_map(|t| t.name.as_deref())
        .collect();
    assert_eq!(names, ["applyDiscount_nonneg", "applyDiscount_le_price"]);
    assert!(proof.solve.verified);
    assert_eq!(proof.solve.attempts.len(), 1);
    assert!(only_standard(proof.solve.axioms.as_deref().unwrap()));
}

#[tokio::test]
async fn mode2_verifies_the_fixed_midpoint() {
    require_lean!();
    let llm = MockProvider::from_script(&read("tests/fixtures/midpoint_fixed.replay.md"));
    let runner = lean();
    let proof = prove_code(
        &llm,
        &runner,
        &read("examples/midpoint_fixed.rs"),
        SourceLanguage::Rust,
        "midpoint_fixed.rs",
        SolveConfig::default(),
        &mut Silent,
        &mut Silent,
    )
    .await
    .unwrap();

    assert!(
        proof
            .translation
            .spec
            .contains("def midpoint (lo hi : UInt32) : UInt32")
    );
    assert!(proof.solve.verified);
    assert_eq!(proof.solve.attempts.len(), 1);
    assert!(only_standard(proof.solve.axioms.as_deref().unwrap()));
}

/// The headline example: `(lo + hi) / 2` on `u32`. Because the model uses
/// `UInt32`, the lower bound cannot be proven, and Lean says why.
#[tokio::test]
async fn mode2_cannot_prove_the_overflowing_midpoint() {
    require_lean!();
    let llm = MockProvider::from_script(&read("tests/fixtures/midpoint_bug.replay.md"));
    let runner = lean();
    let config = SolveConfig {
        max_iters: 3,
        ..SolveConfig::default()
    };
    let proof = prove_code(
        &llm,
        &runner,
        &read("examples/midpoint_bug.rs"),
        SourceLanguage::Rust,
        "midpoint_bug.rs",
        config,
        &mut Silent,
        &mut Silent,
    )
    .await
    .unwrap();

    let solve = &proof.solve;
    assert!(!solve.verified);
    assert_eq!(solve.proof, None);
    assert_eq!(solve.attempts.len(), 3);
    let names: Vec<_> = solve
        .targets
        .iter()
        .filter_map(|t| t.name.as_deref())
        .collect();
    assert_eq!(names, ["midpoint_ge_lo", "midpoint_le_hi"]);

    // Three honest attempts: each passes the guard and fails in Lean, always in
    // the lower bound and never in the upper bound, which does hold.
    for attempt in &solve.attempts {
        assert!(
            matches!(attempt.rejection, Some(Rejection::Compile { .. })),
            "iteration {}: {:?}",
            attempt.iteration,
            attempt.rejection
        );
        assert_eq!(
            attempt.failing_targets,
            [0],
            "iteration {}",
            attempt.iteration
        );
        assert_eq!(attempt.unattributed_errors, 0);
    }

    // The closest attempt isolates the overflow: Lean's counterexample needs
    // `lo + hi ≥ 2^32`.
    let best = solve.best_attempt().unwrap();
    assert_eq!(best.iteration, 3);
    let Some(Rejection::Compile { diagnostics, .. }) = &best.rejection else {
        unreachable!();
    };
    assert_eq!(diagnostics.iter().filter(|d| d.is_error()).count(), 1);
    assert!(
        diagnostics[0]
            .message
            .starts_with("omega could not prove the goal")
    );
    assert!(
        diagnostics[0].message.contains("a + b ≥ 4294967296"),
        "{}",
        diagnostics[0].message
    );
    assert!(solve.model_note().unwrap().contains("2147483648"));
}

/// "Could not prove" is not just prover weakness here: the invariant is false
/// for the buggy code, with a kernel-checked counterexample.
#[tokio::test]
async fn the_overflow_is_real_and_a_nat_model_would_hide_it() {
    require_lean!();
    let refutation = "\
def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2

theorem refuted : ¬ (∀ lo hi : UInt32, lo ≤ hi → lo ≤ midpoint lo hi) := by
  intro h
  have h1 := h 2147483648 2147483648 (by decide)
  revert h1
  decide
";
    let output = lean().check(refutation).await.unwrap();
    assert!(output.success(), "{}", output.combined());
    assert!(output.diagnostics().is_empty());

    // Translate `u32` to the unbounded `Nat` instead and the same "invariant"
    // goes through in one line: this is what the type-fidelity rule prevents.
    let unbounded = "\
def midpoint (lo hi : Nat) : Nat := (lo + hi) / 2

theorem midpoint_ge_lo (lo hi : Nat) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by
  unfold midpoint
  omega
";
    let output = lean().check(unbounded).await.unwrap();
    assert!(output.success(), "{}", output.combined());
}

// -- the guard versus real exploits ------------------------------------------

/// Each candidate below makes Lean exit with code 0 for a theorem that is
/// false. The guard has to be the one to say no.
#[tokio::test]
async fn guard_rejects_cheats_that_lean_accepts() {
    require_lean!();
    let runner = lean();
    let spec = Spec::parse(MIDPOINT_BUG_SPEC).unwrap();
    assert!(runner.check(MIDPOINT_BUG_SPEC).await.unwrap().success());

    let definition = "def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2\n\n";
    let statement =
        "theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by\n";
    // (No `\` continuation here: it would eat the first line's indentation.)
    let fixed_proof = "  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h
  have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt
  have two : (2 : UInt32).toNat = 2 := rfl
  rw [UInt32.le_iff_toNat_le]
  simp only [midpoint, UInt32.toNat_add, UInt32.toNat_div, UInt32.toNat_sub_of_le _ _ h, two]
  omega
";

    let cheats: Vec<(&str, String, ViolationKind)> = vec![
        (
            "quietly repair the function under test",
            format!(
                "def midpoint (lo hi : UInt32) : UInt32 := lo + (hi - lo) / 2\n\n{statement}{fixed_proof}"
            ),
            ViolationKind::AlteredSpec,
        ),
        (
            "assume the overflow away",
            format!(
                "{definition}theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) (hsum : lo.toNat + hi.toNat < 2 ^ 32) :\n    lo ≤ midpoint lo hi := by\n  \
                 have two : (2 : UInt32).toNat = 2 := rfl\n  rw [UInt32.le_iff_toNat_le]\n  \
                 simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]\n  \
                 have hle := UInt32.le_iff_toNat_le.mp h\n  omega\n"
            ),
            ViolationKind::AlteredSpec,
        ),
        (
            "redefine `≤` with an instance",
            format!(
                "{definition}instance : LE UInt32 := ⟨fun _ _ => True⟩\n{statement}  trivial\n"
            ),
            ViolationKind::DisallowedAddition,
        ),
        (
            "redefine `≤` with a macro",
            format!(
                "{definition}local macro_rules | `($a ≤ $b) => `(True)\n{statement}  trivial\n"
            ),
            ViolationKind::DisallowedAddition,
        ),
        (
            "smuggle in `False` as a hypothesis",
            format!(
                "{definition}variable (cheat : False)\ninclude cheat\n{statement}  exact cheat.elim\n"
            ),
            ViolationKind::DisallowedAddition,
        ),
        (
            "swallow the error message",
            format!("{definition}#guard_msgs (drop all) in\n{statement}  skip\n"),
            ViolationKind::DisallowedAddition,
        ),
        (
            "stop Lean before the theorem",
            format!("{definition}#exit\n{statement}  this is not a proof\n"),
            ViolationKind::DisallowedAddition,
        ),
        (
            "use the axiom behind `sorry` directly",
            format!("{definition}{statement}  exact sorryAx _ true\n"),
            ViolationKind::ForbiddenToken,
        ),
    ];

    for (name, candidate, expected) in &cheats {
        let output = runner.check(candidate).await.unwrap();
        assert!(
            output.success(),
            "`{name}` should be a cheat that Lean accepts, but:\n{}",
            output.combined()
        );
        let report = spec.check(candidate);
        assert!(
            report
                .violations
                .iter()
                .any(|violation| violation.kind == *expected),
            "`{name}` should be rejected as {expected:?}, got {:?}",
            report.violations
        );
    }
}

/// What the textual guard cannot see, the compiler-side checks catch.
#[tokio::test]
async fn backstops_catch_what_the_token_scan_cannot() {
    require_lean!();
    let runner = lean();
    let spec = Spec::parse(MIDPOINT_BUG_SPEC).unwrap();
    let audit = spec.axiom_audit().unwrap();
    assert_eq!(audit.names(), ["midpoint_ge_lo"]);

    // `stop` expands to `repeat sorry`: no forbidden token, exit code 0.
    let stopped = MIDPOINT_BUG_SPEC.replace("by sorry", "by\n  stop skip");
    assert!(spec.check(&stopped).passed());
    let llm = MockProvider::new([format!("```lean\n{stopped}```")]);
    let config = SolveConfig {
        max_iters: 1,
        ..SolveConfig::default()
    };
    let report = Solver::new(&llm, &runner)
        .with_config(config)
        .solve(MIDPOINT_BUG_SPEC, &mut Silent)
        .await
        .unwrap();
    assert!(!report.verified);
    assert!(matches!(
        report.attempts[0].rejection,
        Some(Rejection::UsesSorry(_))
    ));

    // The axiom audit sees the same thing from the other side ...
    let output = runner
        .check(&format!("{stopped}{}", audit.commands()))
        .await
        .unwrap();
    let AuditOutcome::Tainted(tainted) = audit.evaluate(&output.stdout) else {
        panic!("expected a tainted audit:\n{}", output.combined());
    };
    assert_eq!(tainted[0].name, "midpoint_ge_lo");
    assert_eq!(tainted[0].axioms, ["sorryAx"]);

    // ... notices a target that silently vanished from the file ...
    let vanished = "def midpoint (lo hi : UInt32) : UInt32 := (lo + hi) / 2\n";
    let output = runner
        .check(&format!("{vanished}{}", audit.commands()))
        .await
        .unwrap();
    assert!(matches!(
        audit.evaluate(&output.stdout),
        AuditOutcome::Inconclusive(_)
    ));

    // ... and passes an honest proof, here of the upper bound, which is true.
    let honest_spec = MIDPOINT_BUG_SPEC
        .replace("midpoint_ge_lo", "midpoint_le_hi")
        .replace("lo ≤ midpoint lo hi", "midpoint lo hi ≤ hi");
    let honest = honest_spec.replace(
        "by sorry",
        "by\n  have hle : lo.toNat ≤ hi.toNat := UInt32.le_iff_toNat_le.mp h\n  \
         have hhi : hi.toNat < 2 ^ 32 := hi.toNat_lt\n  \
         have two : (2 : UInt32).toNat = 2 := rfl\n  \
         rw [UInt32.le_iff_toNat_le]\n  \
         simp only [midpoint, UInt32.toNat_div, UInt32.toNat_add, two]\n  omega",
    );
    let audit = Spec::parse(&honest_spec).unwrap().axiom_audit().unwrap();
    let output = runner
        .check(&format!("{honest}{}", audit.commands()))
        .await
        .unwrap();
    assert!(output.success(), "{}", output.combined());
    match audit.evaluate(&output.stdout) {
        AuditOutcome::Clean(axioms) => assert!(only_standard(&axioms)),
        other => panic!("expected a clean audit, got {other:?}"),
    }
}
