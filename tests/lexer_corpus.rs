//! Differential test of the guard's lexer against the real Lean compiler.
//!
//! The guard is only as good as its idea of where comments and strings begin and
//! end: text it wrongly takes for a comment is text it never inspects. This test
//! blanks out everything the lexer calls a comment (`strip_comments`) in each
//! file of a corpus and compiles the result. If the lexer ever mistakes code for
//! a comment, Lean notices the missing code.
//!
//! It needs a directory of Lean files that compile on their own, so it is
//! ignored by default. The sources shipped with the toolchain work well:
//!
//! ```sh
//! AUTO_LEAN_LEXER_CORPUS="$(lean --print-prefix)/src/lean" \
//!   cargo test --test lexer_corpus -- --ignored --nocapture
//! ```
//!
//! `AUTO_LEAN_LEXER_CORPUS_LIMIT` caps the number of files (default 200).

#[macro_use]
mod common;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use auto_lean::lean::guard::strip_comments;
use auto_lean::lean::{LeanChecker, LeanRunner};
use tokio::sync::Semaphore;
use tokio::task::JoinSet;

const CORPUS_ENV: &str = "AUTO_LEAN_LEXER_CORPUS";
const LIMIT_ENV: &str = "AUTO_LEAN_LEXER_CORPUS_LIMIT";
/// Larger files take long to compile and add little.
const MAX_FILE_BYTES: u64 = 40_000;

/// Commands whose doc comment is part of their syntax or their expected output,
/// so removing it legitimately changes the result.
const DOC_DEPENDENT: &[&str] = &[
    "#guard_msgs",
    "add_decl_doc",
    "register_error_explanation",
    "recommended_spelling",
];

fn lean_files(dir: &Path, found: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            lean_files(&path, found);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "lean")
            && entry
                .metadata()
                .is_ok_and(|meta| meta.len() <= MAX_FILE_BYTES)
        {
            found.push(path);
        }
    }
}

/// What Lean says when an attribute refers to a doc comment that is not there.
const MISSING_DOC: &str = "No documentation found";

enum Outcome {
    /// The original does not compile standalone, or relies on doc comments.
    Skipped,
    Agrees,
    /// Lean accepted the original but not the comment-free version.
    Disagrees(String),
}

async fn compare(runner: LeanRunner, path: PathBuf) -> (PathBuf, Outcome) {
    let Ok(source) = std::fs::read_to_string(&path) else {
        return (path, Outcome::Skipped);
    };
    if DOC_DEPENDENT.iter().any(|command| source.contains(command)) {
        return (path, Outcome::Skipped);
    }
    let Ok(original) = runner.check(&source).await else {
        return (path, Outcome::Skipped);
    };
    if !original.success() {
        return (path, Outcome::Skipped);
    }

    let stripped = strip_comments(&source);
    let outcome = match runner.check(&stripped).await {
        Ok(output) if output.success() => Outcome::Agrees,
        Ok(output) => {
            let diagnostics = output.diagnostics();
            let mut errors = diagnostics
                .iter()
                .filter(|diagnostic| diagnostic.is_error());
            match errors.next() {
                // `@[inherit_doc]` needs the doc comment that was just removed.
                Some(first)
                    if first.message.starts_with(MISSING_DOC)
                        && errors.all(|error| error.message.starts_with(MISSING_DOC)) =>
                {
                    Outcome::Skipped
                }
                Some(first) => Outcome::Disagrees(first.to_string()),
                None => Outcome::Disagrees(output.combined()),
            }
        }
        Err(error) => Outcome::Disagrees(error.to_string()),
    };
    (path, outcome)
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "needs AUTO_LEAN_LEXER_CORPUS and a Lean installation"]
async fn lexer_never_mistakes_code_for_a_comment() {
    require_lean!();
    let Some(corpus) = std::env::var_os(CORPUS_ENV).map(PathBuf::from) else {
        eprintln!("{CORPUS_ENV} is not set; nothing to check");
        return;
    };
    let limit: usize = std::env::var(LIMIT_ENV)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(200);

    let mut files = Vec::new();
    lean_files(&corpus, &mut files);
    files.sort();
    assert!(
        !files.is_empty(),
        "no .lean files under {}",
        corpus.display()
    );
    // An evenly spread, reproducible sample.
    let step = files.len().div_ceil(limit).max(1);
    let sample: Vec<PathBuf> = files.into_iter().step_by(step).collect();

    let runner = LeanRunner::new().with_timeout(Duration::from_secs(120));
    let workers = std::thread::available_parallelism().map_or(4, |n| n.get().min(8));
    let permits = Arc::new(Semaphore::new(workers));
    let mut tasks = JoinSet::new();
    for path in sample {
        let runner = runner.clone();
        let permits = permits.clone();
        tasks.spawn(async move {
            let _permit = permits.acquire_owned().await.expect("semaphore is open");
            compare(runner, path).await
        });
    }

    let (mut agreed, mut skipped, mut disagreements) = (0, 0, Vec::new());
    while let Some(result) = tasks.join_next().await {
        match result.expect("comparison task panicked") {
            (_, Outcome::Agrees) => agreed += 1,
            (_, Outcome::Skipped) => skipped += 1,
            (path, Outcome::Disagrees(error)) => disagreements.push((path, error)),
        }
    }

    eprintln!(
        "lexer corpus: {agreed} files agree, {skipped} skipped, {} disagree",
        disagreements.len()
    );
    for (path, error) in &disagreements {
        eprintln!(
            "  {}\n    {}",
            path.display(),
            error.lines().next().unwrap_or("")
        );
    }
    assert!(
        agreed > 0,
        "no file of the corpus could be used as a baseline"
    );
    assert!(
        disagreements.is_empty(),
        "Lean rejects {} comment-free files",
        disagreements.len()
    );
}
