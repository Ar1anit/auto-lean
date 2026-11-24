//! `auto-lean`: prove Lean 4 theorems, and verify pure Python/Rust functions,
//! with a language model in a compiler-feedback loop.
//!
//! The crate is organised around one verification engine and two front ends:
//!
//! * [`lean`] runs the local `lean` binary ([`lean::LeanRunner`]), parses its
//!   diagnostics ([`lean::parse_diagnostics`]) and makes sure a candidate proof
//!   does not cheat ([`lean::Spec`]).
//! * [`agent`] drives the generate → guard → verify → repair loop
//!   ([`agent::Solver`]) on top of an [`agent::llm::LlmProvider`].
//! * [`translator`] turns a Python or Rust function into a Lean model with
//!   `sorry`-terminated invariants ([`translator::Translator`]) and hands it to
//!   the same loop ([`translator::prove_code`]).

pub mod agent;
pub mod lean;
pub mod translator;
