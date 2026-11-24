//! Tests of the `auto-lean` binary itself: argument handling, exit codes, and
//! the three commands end to end.
//!
//! The first half needs neither Lean nor a network. The second half replays
//! scripted model answers through the real `lean` binary, with `--provider mock`
//! or over HTTP from a local stand-in for a Microsoft Foundry resource, and
//! skips itself when Lean is not installed.

#[macro_use]
mod common;

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use auto_lean::agent::llm::MOCK_SCRIPT_SEPARATOR;
use common::repo;
use serde_json::{Value, json};

/// Provider settings that must not leak from the developer's shell into a test.
const PROVIDER_ENV: &[&str] = &[
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "FOUNDRY_RESOURCE_ENDPOINT",
    "FOUNDRY_RESOURCE",
    "FOUNDRY_API_KEY",
    "FOUNDRY_AUTH_TOKEN",
    "ANTHROPIC_FOUNDRY_RESOURCE",
    "ANTHROPIC_FOUNDRY_BASE_URL",
    "ANTHROPIC_FOUNDRY_API_KEY",
    "AUTO_LEAN_FOUNDRY_API",
    "AUTO_LEAN_MOCK_SCRIPT",
];

/// Runs the compiled binary from the repository root with a clean environment
/// as far as providers are concerned.
fn auto_lean(args: &[&str]) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_auto-lean"));
    command.current_dir(repo("")).args(args);
    for name in PROVIDER_ENV {
        command.env_remove(name);
    }
    // Keep assertions independent of whether the test runs in a terminal.
    command.env("NO_COLOR", "1");
    command
}

fn replay(fixture: &str, args: &[&str]) -> Command {
    let mut command = auto_lean(args);
    command.env("AUTO_LEAN_MOCK_SCRIPT", repo(fixture));
    command
}

fn run(mut command: Command) -> (i32, String, String) {
    let Output {
        status,
        stdout,
        stderr,
    } = command.output().expect("failed to run auto-lean");
    (
        status.code().expect("auto-lean was killed by a signal"),
        String::from_utf8(stdout).expect("stdout is UTF-8"),
        String::from_utf8(stderr).expect("stderr is UTF-8"),
    )
}

/// A scratch directory under Cargo's target directory, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(name: &str) -> Self {
        let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
            .join(format!("{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("failed to create a scratch directory");
        Self(dir)
    }

    fn path(&self, file: &str) -> PathBuf {
        self.0.join(file)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn text(path: &Path) -> &str {
    path.to_str().expect("test paths are UTF-8")
}

// -- no Lean required --------------------------------------------------------

#[test]
fn help_lists_the_three_commands() {
    let (code, stdout, _) = run(auto_lean(&["--help"]));
    assert_eq!(code, 0);
    for expected in ["verify", "solve", "prove-code", "--lean-bin", "Exit status"] {
        assert!(
            stdout.contains(expected),
            "--help does not mention {expected:?}:\n{stdout}"
        );
    }

    let (code, stdout, _) = run(auto_lean(&["prove-code", "--help"]));
    assert_eq!(code, 0);
    for expected in [
        "--spec-only",
        "--max-iters",
        "--provider",
        "--model",
        "--out",
        "anthropic",
        "openai",
    ] {
        assert!(
            stdout.contains(expected),
            "prove-code --help does not mention {expected:?}:\n{stdout}"
        );
    }

    let (code, stdout, _) = run(auto_lean(&["--version"]));
    assert_eq!(code, 0);
    assert_eq!(
        stdout.trim(),
        format!("auto-lean {}", env!("CARGO_PKG_VERSION"))
    );
}

#[test]
fn usage_errors_exit_with_2() {
    for args in [
        &["solve"][..],
        &["frobnicate", "x.lean"],
        &["solve", "x.lean", "--max-iters", "0"],
        &["solve", "x.lean", "--provider", "gemini"],
    ] {
        let (code, _, stderr) = run(auto_lean(args));
        assert_eq!(code, 2, "{args:?}: {stderr}");
    }
}

#[test]
fn missing_input_files_are_reported() {
    let (code, stdout, stderr) = run(auto_lean(&["verify", "no/such/file.lean"]));
    assert_eq!(code, 2);
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("error: failed to read no/such/file.lean"),
        "{stderr}"
    );

    let (code, _, stderr) = run(auto_lean(&[
        "solve",
        "no/such/file.lean",
        "--provider",
        "mock",
    ]));
    assert_eq!(code, 2);
    assert!(
        stderr.contains("failed to read no/such/file.lean"),
        "{stderr}"
    );
}

#[test]
fn prove_code_only_accepts_python_and_rust() {
    let (code, _, stderr) = run(auto_lean(&["prove-code", "examples/demo.lean"]));
    assert_eq!(code, 2);
    assert!(stderr.contains("expected a .py or .rs file"), "{stderr}");
}

#[test]
fn a_missing_api_key_is_explained() {
    let (code, _, stderr) = run(auto_lean(&["solve", "examples/demo.lean"]));
    assert_eq!(code, 2);
    assert!(stderr.contains("ANTHROPIC_API_KEY is not set"), "{stderr}");

    let (code, _, stderr) = run(auto_lean(&[
        "prove-code",
        "examples/pricing.py",
        "--provider",
        "openai",
    ]));
    assert_eq!(code, 2);
    assert!(stderr.contains("OPENAI_API_KEY is not set"), "{stderr}");

    let (code, _, stderr) = run(auto_lean(&[
        "solve",
        "examples/demo.lean",
        "--provider",
        "mock",
    ]));
    assert_eq!(code, 2);
    assert!(stderr.contains("AUTO_LEAN_MOCK_SCRIPT"), "{stderr}");
}

#[test]
fn a_missing_lean_binary_is_explained() {
    let missing = "/definitely/not/here/lean";
    let (code, stdout, stderr) = run(auto_lean(&[
        "--lean-bin",
        missing,
        "verify",
        "examples/demo.lean",
    ]));
    assert_eq!(code, 2);
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("could not find the Lean compiler"),
        "{stderr}"
    );
    assert!(stderr.contains("elan"), "{stderr}");

    // The same through the environment variable, and for the proving commands.
    let mut command = replay(
        "tests/fixtures/demo.replay.md",
        &["solve", "examples/demo.lean", "--provider", "mock"],
    );
    command.env("AUTO_LEAN_LEAN_BIN", missing);
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 2);
    assert!(
        stdout.is_empty(),
        "nothing may be printed as a proof: {stdout}"
    );
    assert!(
        stderr.contains("could not find the Lean compiler"),
        "{stderr}"
    );
}

// -- with the real Lean ------------------------------------------------------

#[test]
fn verify_rejects_the_unsolved_demo() {
    require_lean!();
    let (code, stdout, _) = run(auto_lean(&["verify", "examples/demo.lean"]));
    assert_eq!(code, 1);
    assert!(stdout.contains("guard   FAIL"), "{stdout}");
    assert!(stdout.contains("25:2: forbidden token `sorry`"), "{stdout}");
    assert!(stdout.contains("lean    PASS"), "{stdout}");
    assert!(stdout.contains("REJECTED"), "{stdout}");

    let (code, stdout, _) = run(auto_lean(&["verify", "examples/demo.lean", "--json"]));
    assert_eq!(code, 1);
    let report: serde_json::Value = serde_json::from_str(&stdout).expect("--json prints JSON");
    assert_eq!(report["status"], "rejected");
    assert_eq!(report["guard"]["passed"], false);
    assert_eq!(report["guard"]["violations"][0]["kind"], "forbidden_token");
    assert_eq!(report["guard"]["violations"][0]["line"], 25);
    assert_eq!(report["lean"]["passed"], true);
    assert_eq!(report["lean"]["exit_code"], 0);
    assert_eq!(report["lean"]["diagnostics"][0]["severity"], "warning");
    assert_eq!(report["lean"]["diagnostics"][0]["line"], 24);
}

#[test]
fn verify_reports_lean_errors_as_failed() {
    require_lean!();
    let scratch = Scratch::new("verify-failed");
    let file = scratch.path("Broken.lean");
    std::fs::write(
        &file,
        "theorem broken (p q : Prop) (hp : p) : p ∧ q := by\n  constructor\n  · exact hp\n",
    )
    .unwrap();

    let (code, stdout, _) = run(auto_lean(&["verify", text(&file)]));
    assert_eq!(code, 1);
    assert!(stdout.contains("guard   PASS"), "{stdout}");
    assert!(stdout.contains("lean    FAIL"), "{stdout}");
    assert!(stdout.contains("1:48 error: unsolved goals"), "{stdout}");
    assert!(stdout.contains("⊢ q"), "{stdout}");
    assert!(stdout.contains("FAILED"), "{stdout}");
}

#[test]
fn solve_replays_the_demo_and_the_result_verifies() {
    require_lean!();
    let scratch = Scratch::new("solve-demo");
    let proof = scratch.path("demo.proved.lean");

    let command = replay(
        "tests/fixtures/demo.replay.md",
        &[
            "solve",
            "examples/demo.lean",
            "--provider",
            "mock",
            "--out",
            text(&proof),
        ],
    );
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.is_empty(), "with --out the proof goes to the file");

    // Step by step: a cheat stopped by the guard, a Lean error, then success.
    for expected in [
        "[1/5] asking the model",
        "guard  ✗ 1 violation; Lean was not run",
        "25:2: forbidden token `sorry`",
        "lean   ✗ 1 error",
        "32:17 unsolved goals",
        "⊢ myAppend (myAppend (myRev ys) (myRev xs)) [x] = myAppend (myRev ys) (myAppend (myRev xs) [x])",
        "lean   ✓ exit code 0",
        "axioms ✓ only standard axioms",
        "VERIFIED theorem `myRev_myRev` in 3 iterations",
    ] {
        assert!(
            stderr.contains(expected),
            "missing {expected:?} in:\n{stderr}"
        );
    }

    let written = std::fs::read_to_string(&proof).unwrap();
    assert!(written.contains("theorem myAppend_assoc"));
    assert!(!written.contains("sorry"));

    // The independent check: against the original, with the axiom audit.
    let (code, stdout, _) = run(auto_lean(&[
        "verify",
        text(&proof),
        "--spec",
        "examples/demo.lean",
    ]));
    assert_eq!(code, 0, "{stdout}");
    for expected in ["guard   PASS", "lean    PASS", "axioms  PASS", "VERIFIED"] {
        assert!(
            stdout.contains(expected),
            "missing {expected:?} in:\n{stdout}"
        );
    }

    // The same file checked against a different statement is rejected.
    let other = scratch.path("Other.lean");
    let weaker = common::read("examples/demo.lean").replace(
        "myRev (myRev xs) = xs",
        "myRev (myRev xs) = myRev (myRev xs)",
    );
    std::fs::write(&other, weaker).unwrap();
    let (code, stdout, _) = run(auto_lean(&["verify", text(&proof), "--spec", text(&other)]));
    assert_eq!(code, 1);
    assert!(stdout.contains("guard   FAIL"), "{stdout}");
    assert!(stdout.contains("original text was changed"), "{stdout}");
    assert!(stdout.contains("REJECTED"), "{stdout}");
}

#[test]
fn solve_prints_the_proof_when_no_output_file_is_given() {
    require_lean!();
    let command = replay(
        "tests/fixtures/demo.replay.md",
        &["solve", "examples/demo.lean", "--provider", "mock"],
    );
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    // stdout carries exactly the Lean file, so it can be redirected.
    assert!(
        stdout.starts_with("/-!\n# auto-lean demo (Mode 1)"),
        "{stdout}"
    );
    assert!(
        stdout
            .trim_end()
            .ends_with("| cons x xs ih => simp [myRev, myRev_myAppend, myAppend, ih]")
    );
    assert!(!stdout.contains("VERIFIED") && stderr.contains("VERIFIED"));
}

#[test]
fn prove_code_catches_the_midpoint_overflow() {
    require_lean!();
    let command = replay(
        "tests/fixtures/midpoint_bug.replay.md",
        &[
            "prove-code",
            "examples/midpoint_bug.rs",
            "--provider",
            "mock",
            "--max-iters",
            "3",
        ],
    );
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 1, "{stderr}");
    assert!(
        stdout.is_empty(),
        "an unproven spec must not be printed as a result: {stdout}"
    );
    for expected in [
        "input  examples/midpoint_bug.rs (Rust)",
        "spec   ✓ 2 theorems",
        "def midpoint (lo hi : UInt32) : UInt32 :=",
        "NOT VERIFIED no accepted proof after 3 iterations",
        "✗ theorem `midpoint_ge_lo`: not proven",
        "✓ theorem `midpoint_le_hi`: proof accepted by Lean",
        "a + b ≥ 4294967296",
        "Model's note (not verified): `midpoint_ge_lo` is false as stated.",
    ] {
        assert!(
            stderr.contains(expected),
            "missing {expected:?} in:\n{stderr}"
        );
    }
}

#[test]
fn prove_code_verifies_the_fixed_midpoint_and_the_pricing_function() {
    require_lean!();
    let command = replay(
        "tests/fixtures/midpoint_fixed.replay.md",
        &[
            "prove-code",
            "examples/midpoint_fixed.rs",
            "--provider",
            "mock",
        ],
    );
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    assert!(
        stderr.contains("VERIFIED 2 theorems (midpoint_ge_lo, midpoint_le_hi) in 1 iteration"),
        "{stderr}"
    );
    assert!(stdout.contains("def midpoint (lo hi : UInt32) : UInt32 :=\n  lo + (hi - lo) / 2"));
    assert!(!stdout.contains("sorry"));

    let command = replay(
        "tests/fixtures/pricing.replay.md",
        &["prove-code", "examples/pricing.py", "--provider", "mock"],
    );
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    assert!(
        stderr.contains("input  examples/pricing.py (Python)"),
        "{stderr}"
    );
    assert!(
        stderr.contains("VERIFIED 2 theorems (applyDiscount_nonneg, applyDiscount_le_price)"),
        "{stderr}"
    );
    assert!(stdout.contains("def applyDiscount (priceCents discountPct : Int) : Int :="));
    assert!(!stdout.contains("sorry"));
}

#[test]
fn spec_only_stops_after_the_translation() {
    require_lean!();
    let scratch = Scratch::new("spec-only");
    let spec = scratch.path("pricing.spec.lean");
    let command = replay(
        "tests/fixtures/pricing.replay.md",
        &[
            "prove-code",
            "examples/pricing.py",
            "--provider",
            "mock",
            "--spec-only",
            "--out",
            text(&spec),
        ],
    );
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    assert!(stdout.is_empty());
    assert!(stderr.contains("SPEC READY for 2 theorems"), "{stderr}");
    assert!(!stderr.contains("Proving"), "{stderr}");

    let written = std::fs::read_to_string(&spec).unwrap();
    assert_eq!(written.matches(":= by sorry").count(), 2);

    // The saved spec is itself a valid input for `solve`: it has holes to fill.
    let (code, stdout, _) = run(auto_lean(&["verify", text(&spec)]));
    assert_eq!(code, 1);
    assert!(stdout.contains("forbidden token `sorry`"), "{stdout}");
    assert!(stdout.contains("lean    PASS"), "{stdout}");
}

#[test]
fn running_out_of_scripted_replies_is_an_error_not_a_verdict() {
    require_lean!();
    // The bug fixture holds three proof attempts; the default asks for five.
    let command = replay(
        "tests/fixtures/midpoint_bug.replay.md",
        &[
            "prove-code",
            "examples/midpoint_bug.rs",
            "--provider",
            "mock",
        ],
    );
    let (code, _, stderr) = run(command);
    assert_eq!(code, 2);
    assert!(
        stderr.contains("the language model request failed"),
        "{stderr}"
    );
    assert!(stderr.contains("no scripted replies left"), "{stderr}");
}

// -- Microsoft Foundry ---------------------------------------------------------

/// A local HTTP server standing in for a Foundry resource: it answers each
/// connection with the next canned response and records what it was sent.
struct FakeFoundry {
    url: String,
    requests: Arc<Mutex<Vec<String>>>,
}

impl FakeFoundry {
    fn start(responses: Vec<String>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("failed to bind a local port");
        let url = format!("http://{}", listener.local_addr().expect("local address"));
        let requests = Arc::new(Mutex::new(Vec::new()));
        let seen = Arc::clone(&requests);
        std::thread::spawn(move || {
            for response in responses {
                let Ok((mut socket, _)) = listener.accept() else {
                    return;
                };
                let _ = socket.set_read_timeout(Some(Duration::from_secs(10)));
                let mut raw = Vec::new();
                let mut buffer = [0u8; 8192];
                while !request_complete(&raw) {
                    match socket.read(&mut buffer) {
                        Ok(0) | Err(_) => break,
                        Ok(read) => raw.extend_from_slice(&buffer[..read]),
                    }
                }
                seen.lock()
                    .expect("request log")
                    .push(String::from_utf8_lossy(&raw).into_owned());
                let _ = socket.write_all(response.as_bytes());
                let _ = socket.flush();
            }
        });
        Self { url, requests }
    }

    fn requests(&self) -> Vec<String> {
        self.requests.lock().expect("request log").clone()
    }

    fn bodies(&self) -> Vec<Value> {
        self.requests()
            .iter()
            .map(|raw| {
                let (_, body) = raw.split_once("\r\n\r\n").expect("request has a body");
                serde_json::from_str(body).expect("request body is JSON")
            })
            .collect()
    }
}

/// Whether the headers and the announced number of body bytes have arrived.
fn request_complete(raw: &[u8]) -> bool {
    let text = String::from_utf8_lossy(raw);
    let Some((head, body)) = text.split_once("\r\n\r\n") else {
        return false;
    };
    let expected = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().unwrap_or(0))
        })
        .unwrap_or(0);
    body.len() >= expected
}

fn header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
    let (head, _) = request.split_once("\r\n\r\n")?;
    head.lines().find_map(|line| {
        let (key, value) = line.split_once(':')?;
        key.eq_ignore_ascii_case(name).then(|| value.trim())
    })
}

fn http_ok(content_type: &str, body: &str) -> String {
    format!(
        "HTTP/1.1 200 OK\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
        body.len()
    )
}

/// A reply as the Anthropic Messages API streams it.
fn messages_stream(text: &str) -> String {
    let events = [
        json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-5-5"}}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}),
        json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}}),
        json!({"type": "message_stop"}),
    ];
    let body: String = events
        .iter()
        .map(|event| {
            format!(
                "event: {}\ndata: {event}\n\n",
                event["type"].as_str().unwrap()
            )
        })
        .collect();
    http_ok("text/event-stream", &body)
}

/// A reply as an OpenAI-compatible Chat Completions API returns it.
fn chat_completion(text: &str) -> String {
    let body = json!({"choices": [{"index": 0, "finish_reason": "stop",
        "message": {"role": "assistant", "content": text}}]});
    http_ok("application/json", &body.to_string())
}

/// The scripted model replies of a replay fixture.
fn replies(fixture: &str) -> Vec<String> {
    common::read(fixture)
        .split(MOCK_SCRIPT_SEPARATOR)
        .map(|reply| reply.trim().to_string())
        .filter(|reply| !reply.is_empty())
        .collect()
}

#[test]
fn foundry_settings_are_checked_up_front() {
    let no_model = ["solve", "examples/demo.lean", "--provider", "foundry"];
    let (code, stdout, stderr) = run(auto_lean(&no_model));
    assert_eq!(code, 2);
    assert!(stdout.is_empty());
    assert!(
        stderr.contains("pass the name of your deployment with --model"),
        "{stderr}"
    );

    let with_model = [
        "solve",
        "examples/demo.lean",
        "--provider",
        "foundry",
        "--model",
        "claude-opus-5-5",
    ];
    let (code, _, stderr) = run(auto_lean(&with_model));
    assert_eq!(code, 2);
    assert!(
        stderr.contains("FOUNDRY_RESOURCE_ENDPOINT is not set"),
        "{stderr}"
    );

    let mut command = auto_lean(&with_model);
    command.env("FOUNDRY_RESOURCE", "my-resource");
    let (code, _, stderr) = run(command);
    assert_eq!(code, 2);
    assert!(stderr.contains("FOUNDRY_API_KEY is not set"), "{stderr}");
    assert!(stderr.contains("FOUNDRY_AUTH_TOKEN"), "{stderr}");

    let mut command = auto_lean(&with_model);
    command
        .env("FOUNDRY_RESOURCE", "not a resource")
        .env("FOUNDRY_API_KEY", "key");
    let (code, _, stderr) = run(command);
    assert_eq!(code, 2);
    assert!(
        stderr.contains("neither a Microsoft Foundry resource name nor an endpoint"),
        "{stderr}"
    );

    let mut command = auto_lean(&with_model);
    command
        .env("FOUNDRY_RESOURCE", "my-resource")
        .env("FOUNDRY_API_KEY", "key")
        .env("AUTO_LEAN_FOUNDRY_API", "gemini");
    let (code, _, stderr) = run(command);
    assert_eq!(code, 2);
    assert!(
        stderr.contains("AUTO_LEAN_FOUNDRY_API must be `anthropic` or `openai`"),
        "{stderr}"
    );
}

/// Mode 1 through a Claude deployment: the Messages API under `/anthropic`,
/// authenticated with the resource's API key.
#[test]
fn solve_through_a_claude_deployment_in_foundry() {
    require_lean!();
    let responses = replies("tests/fixtures/demo.replay.md")
        .iter()
        .map(|reply| messages_stream(reply))
        .collect();
    let server = FakeFoundry::start(responses);

    let mut command = auto_lean(&[
        "solve",
        "examples/demo.lean",
        "--provider",
        "foundry",
        "--model",
        "claude-opus-5-5",
    ]);
    command
        // A URL pasted straight from the docs is reduced to the resource endpoint.
        .env(
            "FOUNDRY_RESOURCE_ENDPOINT",
            format!("{}/anthropic/v1/messages", server.url),
        )
        .env("FOUNDRY_API_KEY", "azure-test-key");
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    let model_line = format!(
        "model  foundry · claude-opus-5-5 via the Anthropic Messages API ({})",
        server.url
    );
    assert!(stderr.contains(&model_line), "{stderr}");
    assert!(
        stderr.contains("guard  ✗ 1 violation; Lean was not run"),
        "{stderr}"
    );
    assert!(
        stderr.contains("VERIFIED theorem `myRev_myRev` in 3 iterations"),
        "{stderr}"
    );
    assert!(stdout.contains("theorem myAppend_assoc"));

    let requests = server.requests();
    assert_eq!(requests.len(), 3);
    for request in &requests {
        assert!(
            request.starts_with("POST /anthropic/v1/messages HTTP/1.1"),
            "{request}"
        );
        assert_eq!(header(request, "api-key"), Some("azure-test-key"));
        assert_eq!(header(request, "anthropic-version"), Some("2026-10-21"));
        assert_eq!(header(request, "x-api-key"), None);
        assert_eq!(header(request, "authorization"), None);
        assert_eq!(header(request, "anthropic-beta"), None);
    }

    let bodies = server.bodies();
    assert_eq!(bodies[0]["model"], "claude-opus-5-5");
    assert_eq!(bodies[0]["stream"], true);
    assert_eq!(bodies[0]["thinking"], json!({"type": "adaptive"}));
    assert!(bodies[0].get("fallbacks").is_none());
    let turns = |body: &Value| body["messages"].as_array().map_or(0, Vec::len);
    assert_eq!(bodies.iter().map(turns).collect::<Vec<_>>(), [1, 3, 5]);
    // The guard's correction and Lean's diagnostics went back over the wire.
    let correction = bodies[1]["messages"][2]["content"].as_str().unwrap();
    assert!(
        correction.starts_with("REJECTED by the anti-cheat guard"),
        "{correction}"
    );
    let repair = bodies[2]["messages"][4]["content"].as_str().unwrap();
    assert!(repair.contains("unsolved goals"), "{repair}");
}

/// Mode 2 through any other deployment: the OpenAI-compatible route, here
/// authenticated with a Microsoft Entra ID token.
#[test]
fn prove_code_through_an_openai_compatible_deployment_in_foundry() {
    require_lean!();
    let responses = replies("tests/fixtures/pricing.replay.md")
        .iter()
        .map(|reply| chat_completion(reply))
        .collect();
    let server = FakeFoundry::start(responses);

    let mut command = auto_lean(&[
        "prove-code",
        "examples/pricing.py",
        "--provider",
        "foundry",
        "--model",
        "gpt-6.2-astra",
    ]);
    command
        .env("FOUNDRY_RESOURCE_ENDPOINT", &server.url)
        .env("FOUNDRY_AUTH_TOKEN", "entra-test-token")
        // A token that is set wins over a key.
        .env("FOUNDRY_API_KEY", "unused-key");
    let (code, stdout, stderr) = run(command);
    assert_eq!(code, 0, "{stderr}");
    let model_line = format!(
        "model  foundry · gpt-6.2-astra via the OpenAI-compatible API ({})",
        server.url
    );
    assert!(stderr.contains(&model_line), "{stderr}");
    assert!(
        stderr.contains("VERIFIED 2 theorems (applyDiscount_nonneg, applyDiscount_le_price)"),
        "{stderr}"
    );
    assert!(stdout.contains("def applyDiscount (priceCents discountPct : Int) : Int :="));

    let requests = server.requests();
    assert_eq!(requests.len(), 2, "one translation and one proof");
    for request in &requests {
        assert!(
            request.starts_with("POST /openai/v1/chat/completions HTTP/1.1"),
            "{request}"
        );
        assert_eq!(
            header(request, "authorization"),
            Some("Bearer entra-test-token")
        );
        assert_eq!(header(request, "api-key"), None);
    }

    let bodies = server.bodies();
    assert_eq!(bodies[0]["model"], "gpt-6.2-astra");
    assert_eq!(bodies[0]["messages"][0]["role"], "system");
    let translation_prompt = bodies[0]["messages"][0]["content"].as_str().unwrap();
    assert!(translation_prompt.contains("Strict type-fidelity rule"));
    let source = bodies[0]["messages"][1]["content"].as_str().unwrap();
    assert!(source.contains("def apply_discount(price_cents: int, discount_pct: int) -> int:"));
    // Nothing but `model` and `messages`: reasoning models reject sampling settings.
    assert_eq!(bodies[0].as_object().unwrap().len(), 2);
}
