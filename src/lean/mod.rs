//! Everything that touches Lean itself: running the compiler, reading its
//! diagnostics, and guarding against proofs that cheat.

pub mod guard;
pub mod parser;
pub mod runner;

pub use guard::{
    AuditOutcome, AxiomAudit, Definition, EXTRA_FORBIDDEN_TOKENS, FORBIDDEN_TOKENS, GuardReport,
    STANDARD_AXIOMS, Spec, SpecError, TaintedTheorem, Target, TargetSpan, Violation, ViolationKind,
};
pub use parser::{Goal, LeanDiagnostic, Severity, parse_diagnostics, parse_goals};
pub use runner::{DEFAULT_TIMEOUT, LEAN_BIN_ENV, LeanChecker, LeanError, LeanOutput, LeanRunner};
