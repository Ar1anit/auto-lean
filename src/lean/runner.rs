//! Async wrapper around the local `lean` binary.
//!
//! Candidates are written to a private temporary directory and compiled with a
//! plain `lean <file>`, which is all a dependency-free (`Init`-only) file needs.
//! Lean prints its diagnostics to stdout and signals errors through the exit
//! status; both are captured, together with stderr, for [`parse_diagnostics`].

use std::io;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use async_trait::async_trait;
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt};
use tokio::process::Command;
use tokio::task::JoinHandle;

use super::parser::{LeanDiagnostic, parse_diagnostics};

/// How long a single Lean invocation may run before it is killed.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(30);

/// Environment variable that overrides which `lean` binary is used.
pub const LEAN_BIN_ENV: &str = "AUTO_LEAN_LEAN_BIN";

/// Name of the file candidates are compiled as.
const CANDIDATE_FILE: &str = "Candidate.lean";

/// Output beyond this many bytes per stream is discarded; a looping tactic can
/// otherwise print without bound until the timeout hits.
const MAX_CAPTURE_BYTES: usize = 8 * 1024 * 1024;

/// How long to wait for the output pipes to close once Lean has exited.
const PIPE_GRACE: Duration = Duration::from_secs(2);

/// Reasons Lean could not be run at all. A proof that merely fails to compile is
/// not an error; it is a [`LeanOutput`] with a non-zero exit code.
#[derive(Debug, Error)]
pub enum LeanError {
    #[error(
        "could not find the Lean compiler (`{binary}`). Install Lean 4 with elan \
         (https://lean-lang.org/install/) and make sure `lean` is on your PATH, \
         or point --lean-bin / {LEAN_BIN_ENV} at the binary"
    )]
    NotFound { binary: String },

    #[error("failed to start `{binary}`: {source}")]
    Spawn {
        binary: String,
        #[source]
        source: io::Error,
    },

    #[error("`{binary}` is not usable: {detail}")]
    Unusable { binary: String, detail: String },

    #[error("failed to prepare a temporary file for Lean: {0}")]
    TempFile(#[source] io::Error),

    #[error("failed while waiting for Lean: {0}")]
    Wait(#[source] io::Error),
}

/// Everything one Lean invocation produced.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LeanOutput {
    pub stdout: String,
    pub stderr: String,
    /// `None` when Lean was killed, by a signal or by the timeout.
    pub exit_code: Option<i32>,
    /// Whether the invocation was killed for exceeding the timeout.
    pub timed_out: bool,
    pub duration: Duration,
}

impl LeanOutput {
    /// Lean ran to completion and reported no errors.
    pub fn success(&self) -> bool {
        self.exit_code == Some(0) && !self.timed_out
    }

    /// Diagnostics parsed from stdout and stderr.
    pub fn diagnostics(&self) -> Vec<LeanDiagnostic> {
        parse_diagnostics(&self.combined())
    }

    /// stdout followed by stderr.
    pub fn combined(&self) -> String {
        let mut combined = self.stdout.clone();
        if !self.stderr.is_empty() {
            if !combined.is_empty() && !combined.ends_with('\n') {
                combined.push('\n');
            }
            combined.push_str(&self.stderr);
        }
        combined
    }
}

/// Something that can type-check a standalone Lean file.
///
/// The real implementation is [`LeanRunner`]; tests substitute scripted checkers
/// so the agent loop can be exercised without a Lean installation.
#[async_trait]
pub trait LeanChecker: Send + Sync {
    async fn check(&self, source: &str) -> Result<LeanOutput, LeanError>;
}

/// Runs the `lean` binary as a subprocess with a timeout.
#[derive(Debug, Clone)]
pub struct LeanRunner {
    binary: PathBuf,
    timeout: Duration,
}

impl Default for LeanRunner {
    fn default() -> Self {
        Self::new()
    }
}

impl LeanRunner {
    /// Uses `$AUTO_LEAN_LEAN_BIN` if set, otherwise `lean` from `PATH`.
    pub fn new() -> Self {
        let binary = std::env::var_os(LEAN_BIN_ENV)
            .filter(|value| !value.is_empty())
            .map_or_else(|| PathBuf::from("lean"), PathBuf::from);
        Self {
            binary,
            timeout: DEFAULT_TIMEOUT,
        }
    }

    pub fn with_binary(mut self, binary: impl Into<PathBuf>) -> Self {
        self.binary = binary.into();
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    pub fn binary(&self) -> &Path {
        &self.binary
    }

    pub fn timeout(&self) -> Duration {
        self.timeout
    }

    /// The toolchain's self-description, e.g. `Lean (version 4.36.0, ...)`.
    pub async fn version(&self) -> Result<String, LeanError> {
        let output = self
            .run(&["--version".as_ref()], Duration::from_secs(20))
            .await?;
        let version = output
            .stdout
            .lines()
            .next()
            .unwrap_or("")
            .trim()
            .to_string();
        if !output.success() || version.is_empty() {
            let detail = match output.combined().trim() {
                _ if output.timed_out => "`--version` did not answer in time".to_string(),
                "" => format!("`--version` exited with {:?}", output.exit_code),
                text => text.to_string(),
            };
            return Err(LeanError::Unusable {
                binary: self.binary_name(),
                detail,
            });
        }
        Ok(version)
    }

    /// Compiles an existing file in place with `lean <path>`.
    pub async fn check_file(&self, path: &Path) -> Result<LeanOutput, LeanError> {
        self.run(&[path.as_os_str()], self.timeout).await
    }

    /// Writes `source` to a temporary file and compiles it.
    pub async fn check_source(&self, source: &str) -> Result<LeanOutput, LeanError> {
        let dir = TempDir::create().map_err(LeanError::TempFile)?;
        let file = dir.path().join(CANDIDATE_FILE);
        std::fs::write(&file, source).map_err(LeanError::TempFile)?;
        self.run(&[file.as_os_str()], self.timeout).await
    }

    fn binary_name(&self) -> String {
        self.binary.display().to_string()
    }

    async fn run(
        &self,
        args: &[&std::ffi::OsStr],
        timeout: Duration,
    ) -> Result<LeanOutput, LeanError> {
        let started = Instant::now();
        let mut child = Command::new(&self.binary)
            .args(args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true)
            .spawn()
            .map_err(|source| match source.kind() {
                io::ErrorKind::NotFound => LeanError::NotFound {
                    binary: self.binary_name(),
                },
                _ => LeanError::Spawn {
                    binary: self.binary_name(),
                    source,
                },
            })?;

        // Drain both pipes concurrently so a chatty Lean can never block on a
        // full pipe while we wait for it to exit.
        let stdout = capture(child.stdout.take());
        let stderr = capture(child.stderr.take());

        let (exit_code, timed_out) = match tokio::time::timeout(timeout, child.wait()).await {
            Ok(status) => (status.map_err(LeanError::Wait)?.code(), false),
            Err(_) => {
                // Best effort: the process may have exited in the meantime.
                let _ = child.kill().await;
                (None, true)
            }
        };

        Ok(LeanOutput {
            stdout: collect(stdout).await,
            stderr: collect(stderr).await,
            exit_code,
            timed_out,
            duration: started.elapsed(),
        })
    }
}

#[async_trait]
impl LeanChecker for LeanRunner {
    async fn check(&self, source: &str) -> Result<LeanOutput, LeanError> {
        self.check_source(source).await
    }
}

/// Reads a pipe to the end on a background task, keeping at most
/// [`MAX_CAPTURE_BYTES`] of it.
fn capture<R>(pipe: Option<R>) -> JoinHandle<Vec<u8>>
where
    R: AsyncRead + Unpin + Send + 'static,
{
    tokio::spawn(async move {
        let mut captured = Vec::new();
        let Some(mut pipe) = pipe else {
            return captured;
        };
        let mut buffer = [0u8; 8192];
        loop {
            match pipe.read(&mut buffer).await {
                Ok(0) | Err(_) => break,
                Ok(read) => {
                    let room = MAX_CAPTURE_BYTES.saturating_sub(captured.len());
                    captured.extend_from_slice(&buffer[..read.min(room)]);
                }
            }
        }
        captured
    })
}

/// Waits briefly for a capture task. If a stray grandchild still holds the pipe
/// open after Lean itself is gone, whatever was read so far is abandoned rather
/// than blocking the caller.
async fn collect(task: JoinHandle<Vec<u8>>) -> String {
    let abort = task.abort_handle();
    match tokio::time::timeout(PIPE_GRACE, task).await {
        Ok(Ok(bytes)) => String::from_utf8_lossy(&bytes).into_owned(),
        _ => {
            abort.abort();
            String::new()
        }
    }
}

/// A private scratch directory that is removed when dropped.
#[derive(Debug)]
pub(crate) struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub(crate) fn create() -> io::Result<Self> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let base = std::env::temp_dir();
        for _ in 0..32 {
            let nanos = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map_or(0, |elapsed| elapsed.subsec_nanos());
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = base.join(format!(
                "auto-lean-{}-{unique}-{nanos:08x}",
                std::process::id()
            ));
            let mut builder = std::fs::DirBuilder::new();
            #[cfg(unix)]
            {
                use std::os::unix::fs::DirBuilderExt;
                builder.mode(0o700);
            }
            match builder.create(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(error),
            }
        }
        Err(io::Error::new(
            io::ErrorKind::AlreadyExists,
            "could not create a unique temporary directory",
        ))
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn reports_a_missing_lean_binary_clearly() {
        let runner = LeanRunner::new().with_binary("/definitely/not/here/lean");
        let error = runner
            .check_source("theorem t : True := trivial")
            .await
            .unwrap_err();
        assert!(matches!(error, LeanError::NotFound { .. }));
        let message = error.to_string();
        assert!(
            message.contains("could not find the Lean compiler"),
            "{message}"
        );
        assert!(message.contains("elan"), "{message}");
        assert!(message.contains(LEAN_BIN_ENV), "{message}");

        assert!(matches!(
            runner.version().await,
            Err(LeanError::NotFound { .. })
        ));
    }

    #[test]
    fn combines_streams_for_parsing() {
        let output = LeanOutput {
            stdout: "F.lean:1:0: error: first".into(),
            stderr: "F.lean:2:0: warning: second\n".into(),
            exit_code: Some(1),
            timed_out: false,
            duration: Duration::ZERO,
        };
        assert!(!output.success());
        assert_eq!(
            output.combined(),
            "F.lean:1:0: error: first\nF.lean:2:0: warning: second\n"
        );
        assert_eq!(output.diagnostics().len(), 2);

        let killed = LeanOutput {
            exit_code: Some(0),
            timed_out: true,
            ..output
        };
        assert!(!killed.success());
    }

    #[test]
    fn temp_dirs_are_unique_and_cleaned_up() {
        let first = TempDir::create().unwrap();
        let second = TempDir::create().unwrap();
        assert_ne!(first.path(), second.path());
        let path = first.path().to_path_buf();
        std::fs::write(path.join("x.lean"), "-- scratch").unwrap();
        drop(first);
        assert!(!path.exists());
        assert!(second.path().is_dir());
    }

    // The remaining tests stand in a POSIX shell for Lean: `sh <file>` runs the
    // "candidate" as a script, which exercises the real subprocess plumbing
    // (temporary file, pipes, exit status, timeout) without a Lean installation.
    #[cfg(unix)]
    fn shell() -> LeanRunner {
        LeanRunner::new().with_binary("sh")
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn captures_exit_status_and_both_streams() {
        let script = "\
echo \"$0:3:2: error: unsolved goals\"
echo \"⊢ False\"
echo \"note on stderr\" >&2
exit 1
";
        let output = shell().check_source(script).await.unwrap();
        assert_eq!(output.exit_code, Some(1));
        assert!(!output.success());
        assert!(!output.timed_out);
        assert_eq!(output.stderr, "note on stderr\n");

        let diagnostics = output.diagnostics();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!((diagnostics[0].line, diagnostics[0].column), (3, 2));
        assert!(diagnostics[0].file.ends_with(CANDIDATE_FILE));
        // stderr is appended after stdout, so it lands in the last message.
        assert_eq!(
            diagnostics[0].message,
            "unsolved goals\n⊢ False\nnote on stderr"
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn succeeds_on_exit_code_zero_and_removes_the_file() {
        let output = shell().check_source("echo \"$0\"\nexit 0\n").await.unwrap();
        assert!(output.success());
        let file = PathBuf::from(output.stdout.trim());
        assert!(file.ends_with(CANDIDATE_FILE));
        assert!(!file.exists(), "temporary candidate should be deleted");
        assert!(!file.parent().unwrap().exists());
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn kills_lean_after_the_timeout() {
        let runner = shell().with_timeout(Duration::from_millis(300));
        let started = Instant::now();
        let output = runner
            .check_source("echo started\nexec sleep 30\n")
            .await
            .unwrap();
        assert!(output.timed_out);
        assert!(!output.success());
        assert_eq!(output.exit_code, None);
        assert_eq!(output.stdout, "started\n");
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "took {:?}",
            started.elapsed()
        );
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn checks_existing_files_in_place() {
        let dir = TempDir::create().unwrap();
        let file = dir.path().join("Existing.lean");
        std::fs::write(&file, "echo \"checked $0\"\n").unwrap();
        let output = shell().check_file(&file).await.unwrap();
        assert!(output.success());
        assert!(output.stdout.contains("Existing.lean"));
        assert!(file.exists(), "user files must not be removed");
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn drains_large_output_without_deadlocking() {
        // Far more than a pipe buffer holds, on both streams.
        let script = "\
i=0
while [ $i -lt 4000 ]; do
  echo \"line $i of a very chatty tactic trace ............................................\"
  echo \"err $i ..............................................................\" >&2
  i=$((i+1))
done
";
        let output = shell().check_source(script).await.unwrap();
        assert!(output.success());
        assert_eq!(output.stdout.lines().count(), 4000);
        assert_eq!(output.stderr.lines().count(), 4000);
    }
}
