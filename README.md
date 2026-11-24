# auto-lean

Prove Lean 4 theorems, and formally verify pure Python and Rust functions, with
an LLM in a compiler-feedback loop.

`auto-lean` asks a language model for a proof, turns away anything that cheats,
compiles the rest with your local `lean`, and feeds the compiler's exact
diagnostics back until the proof checks or the attempts run out. Nothing counts
as proven until Lean's kernel has accepted it on your machine.

| Command | What it does |
| --- | --- |
| `auto-lean solve <file.lean>` | **Mode 1.** Fills in every `sorry` of a Lean file. |
| `auto-lean prove-code <file.py\|file.rs>` | **Mode 2.** Translates a pure function into a faithful Lean 4 model with 2–3 invariants, then proves them with the same loop. |
| `auto-lean verify <file.lean>` | Runs the anti-cheat guard and `lean` on a file and reports the result. |

No Mathlib and no Lake project are needed. The examples use only Lean's core
library and compile in about a quarter of a second.

## It finds the overflow

The same function, one line apart. Rust's `u32` is modelled by Lean's `UInt32`,
never by the unbounded `Nat`, so wraparound is part of the model.

<table>
<tr>
<th><code>examples/midpoint_bug.rs</code>: overflow caught</th>
<th><code>examples/midpoint_fixed.rs</code>: verified</th>
</tr>
<tr>
<td>

```rust
pub fn midpoint(lo: u32, hi: u32) -> u32 {
    (lo + hi) / 2
}
```

</td>
<td>

```rust
pub fn midpoint(lo: u32, hi: u32) -> u32 {
    lo + (hi - lo) / 2
}
```

</td>
</tr>
<tr>
<td>

```text
$ auto-lean prove-code examples/midpoint_bug.rs \
    --provider mock --max-iters 3
[...]
Proving
[1/3] asking the model ... reply in 0.0s
      guard  ✓ original statements intact, [...]
      lean   ✗ 1 error in 0.29s
               13:2 omega could not prove the goal:
[...]
NOT VERIFIED no accepted proof after 3 iterations (2.9s)

Closest attempt (iteration 3):
  ✗ theorem `midpoint_ge_lo`: not proven
  ✓ theorem `midpoint_le_hi`: proof accepted by Lean

Lean's objections:
  15:4 omega could not prove the goal:
    a possible counterexample may satisfy the constraints
      [...]
      a + b ≥ 4294967296
      [...]
    where
     a := ↑hi.toNat
     b := ↑lo.toNat
[...]
$ echo $?
1
```

</td>
<td>

```text
$ auto-lean prove-code examples/midpoint_fixed.rs \
    --provider mock
[...]
Proving
[1/5] asking the model ... reply in 0.0s
      guard  ✓ original statements intact, [...]
      lean   ✓ exit code 0 in 0.35s
      axioms ✓ only standard axioms (propext, Quot.sound)

VERIFIED 2 theorems (midpoint_ge_lo, midpoint_le_hi) in 1 iteration (0.9s)
$ echo $?
0
```

</td>
</tr>
</table>

Both runs work on the same two invariants, proposed by the translation step:

```lean
def midpoint (lo hi : UInt32) : UInt32 :=
  (lo + hi) / 2        -- `lo + (hi - lo) / 2` in the fixed version

theorem midpoint_ge_lo (lo hi : UInt32) (h : lo ≤ hi) : lo ≤ midpoint lo hi := by sorry
theorem midpoint_le_hi (lo hi : UInt32) (h : lo ≤ hi) : midpoint lo hi ≤ hi := by sorry
```

For the buggy version Lean accepts the upper bound and rejects the lower one,
and its own counterexample needs `a + b ≥ 4294967296`: the sum has to overflow.
Indeed `midpoint 2147483648 2147483648 = 0`. Had `u32` been translated to `Nat`,
`omega` would have proven the buggy function correct.

> The transcripts are real output, recorded with the bundled replay provider
> (`AUTO_LEAN_MOCK_SCRIPT` set to the matching file in `tests/fixtures/`) so that
> they can be reproduced without an API key; see
> [Try it without an API key](#try-it-without-an-api-key). `[...]` marks text cut
> for width. With a live model the individual attempts and timings differ.

## Install

Prerequisites:

- **Lean 4**, installed with [elan](https://lean-lang.org/install/):

  ```sh
  curl https://elan.lean-lang.org/elan-init.sh -sSf | sh
  lean --version        # Lean (version 4.x.y, ...)
  ```

- **Rust** 1.85 or newer, to build the tool.
- Access to a model: an Anthropic or OpenAI API key, a Microsoft Foundry
  resource, or a local OpenAI-compatible server.

```sh
cargo install --path .          # or: cargo build --release
```

Tested with Lean 4.36.0.

## Usage

```sh
# Mode 1: prove the `sorry` in a Lean file
auto-lean solve examples/demo.lean --out demo.proved.lean

# Check the result independently, against the original statements
auto-lean verify demo.proved.lean --spec examples/demo.lean

# Mode 2: verify a Python or Rust function
auto-lean prove-code examples/pricing.py
auto-lean prove-code examples/pricing.py --spec-only     # stop after the Lean spec
auto-lean prove-code examples/midpoint_fixed.rs --provider openai --model gpt-6.2-sol

# Any provider works for either mode, e.g. a deployment in Microsoft Foundry
auto-lean solve examples/demo.lean --provider foundry --model claude-opus-5-5
```

Progress goes to stderr and the resulting Lean file to stdout (or to `--out`),
so `auto-lean solve x.lean > proved.lean` works.

### `auto-lean verify <path> [--spec <path>] [--json]`

Runs the guard and `lean <path>` and prints both verdicts with Lean's
diagnostics. With `--spec`, it also checks that the original file's statements
are intact and that the proofs rest only on standard axioms. `--json` prints the
same report for scripts.

### `auto-lean solve <path> [options]`

| Option | Default | |
| --- | --- | --- |
| `--max-iters <N>` | `5` | Proof attempts before giving up |
| `--provider <anthropic\|openai\|foundry\|mock>` | `anthropic` | Model API |
| `--model <name>` | `claude-opus-5-5` / `gpt-6.2-astra` | Model to use; required for `foundry`, where it is the deployment name |
| `--out <path>` | stdout | Where to write the verified file |
| `--no-axiom-audit` | off | Skip the final `#print axioms` check |

### `auto-lean prove-code <path> [--spec-only] [options]`

Detects the language from the extension (`.py` or `.rs`), generates the Lean
model and invariants, validates that spec in Lean, and runs the `solve` loop on
it. Takes the same options as `solve`; `--spec-only` stops after the spec.

Global options: `--lean-bin <path>`, `--timeout <secs>` (per Lean run, default
30) and `-v` for unabridged diagnostics.

**Exit status:** `0` verified, `1` not verified, `2` error (missing Lean, missing
credentials, unreadable file, failed model request).

## Environment variables

| Variable | Purpose |
| --- | --- |
| `ANTHROPIC_API_KEY` | Key for `--provider anthropic` (the default) |
| `ANTHROPIC_BASE_URL` | Optional gateway or proxy for the Anthropic API |
| `OPENAI_API_KEY` | Key for `--provider openai`; optional when `OPENAI_BASE_URL` points at a local server |
| `OPENAI_BASE_URL` | Any OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1` (Ollama) or `http://localhost:8000/v1` (vLLM) |
| `FOUNDRY_RESOURCE_ENDPOINT` | For `--provider foundry`: `https://<resource>.services.ai.azure.com`, or just the resource name |
| `FOUNDRY_API_KEY` | The Foundry resource's API key |
| `FOUNDRY_AUTH_TOKEN` | A Microsoft Entra ID token, used instead of the key when set |
| `AUTO_LEAN_LEAN_BIN` | Path to `lean` if it is not on `PATH` |
| `AUTO_LEAN_MOCK_SCRIPT` | Script replayed by `--provider mock` |

```sh
# A local model through Ollama
OPENAI_BASE_URL=http://localhost:11434/v1 \
  auto-lean solve examples/demo.lean --provider openai --model <local-model>
```

Requests to Claude use adaptive thinking at high effort and opt into
server-side refusal fallbacks where the model supports them.

### Microsoft Foundry

`--provider foundry` talks to a deployment in a Microsoft Foundry resource;
`--model` is the deployment name.

```sh
export FOUNDRY_RESOURCE_ENDPOINT=https://<resource>.services.ai.azure.com
export FOUNDRY_API_KEY=<key>

auto-lean prove-code examples/pricing.py --provider foundry --model claude-opus-5-5
auto-lean prove-code examples/pricing.py --provider foundry --model gpt-6.2-astra
```

Foundry serves its catalog through two APIs, and the right one is chosen from
the deployment name: a name containing `claude` goes to the Anthropic Messages
API (`/anthropic/v1/messages`), anything else to the OpenAI-compatible Chat
Completions API (`/openai/v1/chat/completions`). If a deployment's name does not
say what it is, set `AUTO_LEAN_FOUNDRY_API` to `anthropic` or `openai`.

To sign in with Microsoft Entra ID instead of a key (required for the Claude
Mythos models), export a token:

```sh
export FOUNDRY_AUTH_TOKEN=$(az account get-access-token \
  --resource https://ai.azure.com --query accessToken -o tsv)
```

If you already use Anthropic's Foundry SDKs, their variables are picked up too:
`ANTHROPIC_FOUNDRY_RESOURCE` or `ANTHROPIC_FOUNDRY_BASE_URL`, and
`ANTHROPIC_FOUNDRY_API_KEY`.

## How it works

```text
 spec ──▶ model ──▶ guard ──▶ lean ──▶ axiom audit ──▶ VERIFIED
            ▲         │         │           │
            └─────────┴─────────┴───────────┘
             strict correction, or Lean's diagnostics verbatim
```

1. **Generate.** The model returns the complete file with proofs filled in.
2. **Guard.** A cheating answer is sent back with a correction and never
   reaches Lean.
3. **Verify.** `lean` compiles the candidate (30 s timeout). On failure the
   exact diagnostics, goal states included, go back to the model.
4. **Stop** as soon as Lean exits with code 0 for a candidate the guard accepted.

### The anti-cheat guard

Asking a model to make a compiler happy invites it to change the question. The
guard works on a comment-free token stream from a small lexer that mirrors
Lean's own tokenizer, and enforces:

- **Forbidden tokens**: `sorry`, `admit`, `axiom`, `unsafe`, `implemented_by`,
  `extern`, plus `sorryAx` and `native_decide`, as whole words outside comments
  and strings.
- **A frozen spec**: every original definition and theorem statement must
  reappear token for token and in order. "Fixing" the function under test or
  adding a hypothesis is rejected.
- **Inert additions**: besides proofs, a candidate may only add helper
  `theorem`s with fresh names between the original declarations. Commands that
  could change what the frozen statements mean are rejected: `instance`,
  `notation`, `macro`, `variable`, `open`, `namespace`, `attribute`, `import`,
  every `#` command, and `set_option` beyond a few harmless options.
- **Two checks on Lean's side**: a leftover `declaration uses sorry` warning is a
  failure, and `#print axioms` must show nothing beyond `propext`,
  `Classical.choice` and `Quot.sound` for every target theorem.

These rules are not hypothetical. Redefining `≤` with an `instance`, swallowing
the error with `#guard_msgs`, cutting the file short with `#exit`, or smuggling
in `variable (h : False)` each make Lean exit with code 0 for a false theorem;
`tests/real_lean.rs` runs every one of them against the real compiler.

### Strict type fidelity (Mode 2)

| Source type | Lean 4 type |
| --- | --- |
| Python `int` | `Int` |
| Rust `u8` / `u16` / `u32` / `u64` | `UInt8` / `UInt16` / `UInt32` / `UInt64`, never `Nat` |
| Rust `i8` / `i16` / `i32` / `i64` | `Int8` / `Int16` / `Int32` / `Int64` (or `BitVec`) |

The rule is part of the translation prompt and is checked again on the result:
a model whose signature uses `Nat` for a `u32` is sent back before Lean runs.
Generated specs are also refused if they contain `import`, `#eval` or anything
else that would execute code while compiling.

## Try it without an API key

`--provider mock` replays scripted model replies, so the whole pipeline,
including the real Lean runs, works offline:

```sh
export AUTO_LEAN_MOCK_SCRIPT=tests/fixtures/demo.replay.md
auto-lean solve examples/demo.lean --provider mock
```

The replayed model first cheats, then makes a real mistake, then gets it right
(the verified Lean file follows on stdout):

```text
lean   Lean 4.36.0
model  mock · replaying tests/fixtures/demo.replay.md
input  examples/demo.lean
[1/5] asking the model ... reply in 0.0s
      guard  ✗ 1 violation; Lean was not run
               25:2: forbidden token `sorry`: every proof must be complete
[2/5] asking the model ... reply in 0.0s
      guard  ✓ original statements intact, no forbidden tokens
      lean   ✗ 1 error in 0.31s
               32:17 unsolved goals
                 case cons
                 α : Type
                 ys : List α
                 x : α
                 xs : List α
                 ih : myRev (myAppend xs ys) = myAppend (myRev ys) (myRev xs)
                 ⊢ myAppend (myAppend (myRev ys) (myRev xs)) [x] = myAppend (myRev ys) (myAppend (myRev xs) [x])
[3/5] asking the model ... reply in 0.0s
      guard  ✓ original statements intact, no forbidden tokens
      lean   ✓ exit code 0 in 0.27s
      axioms ✓ only standard axioms (propext)

VERIFIED theorem `myRev_myRev` in 3 iterations (1.6s)
```

The other fixtures replay Mode 2 the same way: `pricing.replay.md`,
`midpoint_fixed.replay.md` and `midpoint_bug.replay.md` (use `--max-iters 3`
with the last one; it holds three proof attempts).

## Limits worth knowing

- **`NOT VERIFIED` is not a disproof.** It means no proof was found in the
  allowed attempts: a strong hint when Lean's diagnostics show a counterexample,
  as above, and otherwise a reason to raise `--max-iters` or read the spec.
- **The invariants are proposals.** A verified theorem is only as meaningful as
  its statement, so read the generated spec (`--spec-only`). The translation of
  the function body is the model's reading of your code; types are enforced, the
  body is not.
- **Rust arithmetic is modelled as wrapping**, as in release builds. Where a
  property fails because of wraparound, a debug build would panic instead.
- **Core library only.** Candidates cannot add imports, and Mathlib tactics such
  as `linarith` or `ring` are not available.
- **Lean runs code while compiling.** The guard rejects `#eval`-style commands
  in model output before Lean sees them, but your own input files are compiled
  as they are.
- **The guard is a syntactic check plus an axiom audit**, not a proof that a
  model cannot cheat. For adversarial settings, add an independent checker.

## Development

```sh
cargo test                            # 152 tests
AUTO_LEAN_REQUIRE_LEAN=1 cargo test   # fail, rather than skip, without Lean
```

The unit tests need neither Lean nor a network: the model is a `MockProvider`
and Lean a scripted stand-in. `tests/real_lean.rs` and `tests/cli.rs` run the
same scenarios against the real `lean` binary and skip themselves if it is not
installed.

`tests/lexer_corpus.rs` (ignored by default) checks the guard's lexer against
the compiler itself: it removes everything the lexer calls a comment from each
file of a corpus and expects Lean to accept the result. Over 1,230 files of the
Lean 4.36.0 library, none disagreed.

```sh
AUTO_LEAN_LEXER_CORPUS="$(lean --print-prefix)/src/lean" \
  cargo test --test lexer_corpus -- --ignored --nocapture
```

```text
src/
├── main.rs              CLI: verify, solve, prove-code
├── lib.rs
├── lean/
│   ├── runner.rs        async `lean` subprocess with timeout
│   ├── guard.rs         anti-cheat guard and axiom audit
│   └── parser.rs        diagnostic and goal-state parser
├── agent/
│   ├── mod.rs           generate → guard → verify → repair loop
│   ├── llm.rs           LlmProvider: Anthropic, OpenAI-compatible, Foundry, Mock
│   └── prompts.rs       system and repair prompts
└── translator/
    ├── mod.rs           Python/Rust → Lean spec, validation, hand-off
    └── prompts.rs       translation prompt and the type-fidelity rule
examples/                demo.lean, pricing.py, midpoint_bug.rs, midpoint_fixed.rs
tests/                   real-Lean and CLI tests, replay fixtures
```
