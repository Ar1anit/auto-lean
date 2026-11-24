//! Helpers shared by the integration tests.
#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::OnceLock;

use auto_lean::lean::LEAN_BIN_ENV;

/// Set this to make tests fail, rather than skip, when Lean is missing.
pub const REQUIRE_LEAN_ENV: &str = "AUTO_LEAN_REQUIRE_LEAN";

/// The `lean` binary the tests use: `$AUTO_LEAN_LEAN_BIN`, or `lean` on `PATH`.
pub fn lean_binary() -> PathBuf {
    std::env::var_os(LEAN_BIN_ENV)
        .filter(|value| !value.is_empty())
        .map_or_else(|| PathBuf::from("lean"), PathBuf::from)
}

/// Whether a working Lean toolchain is installed.
///
/// Tests that need the real compiler return early when this is `false`, so the
/// suite stays green on machines without Lean; everything they cover is also
/// exercised with scripted stand-ins in the unit tests.
pub fn lean_available() -> bool {
    static AVAILABLE: OnceLock<bool> = OnceLock::new();
    *AVAILABLE.get_or_init(|| {
        let works = std::process::Command::new(lean_binary())
            .arg("--version")
            .output()
            .is_ok_and(|output| output.status.success());
        if !works {
            assert!(
                std::env::var_os(REQUIRE_LEAN_ENV).is_none(),
                "{REQUIRE_LEAN_ENV} is set, but `{}` does not run",
                lean_binary().display()
            );
            eprintln!("note: `lean` not found; tests that need the real compiler are skipped");
        }
        works
    })
}

/// Returns from the calling test when Lean is not installed.
macro_rules! require_lean {
    () => {
        if !$crate::common::lean_available() {
            return;
        }
    };
}

/// A path inside the repository.
pub fn repo(relative: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(relative)
}

/// The contents of a file inside the repository.
pub fn read(relative: &str) -> String {
    let path = repo(relative);
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("failed to read {}: {error}", path.display()))
}
