//! Anti-cheat guard for machine-written Lean.
//!
//! A proof-search loop is only trustworthy if the model cannot "win" by changing
//! the question. The guard works on a comment-free token stream produced by a
//! small Lean lexer and enforces three things:
//!
//! 1. **Forbidden tokens.** `sorry`, `admit`, `axiom`, `unsafe`, `implemented_by`
//!    and `extern` (plus equivalents such as `sorryAx` and `native_decide`) are
//!    rejected wherever they appear as whole words outside comments and strings.
//! 2. **Frozen spec.** The original file is cut at every `sorry` and in front of
//!    every declaration. Each resulting piece (definitions and theorem statements
//!    alike) must reappear in the candidate token for token and in order. The
//!    only text a candidate may add is a proof in place of each `sorry`, and
//!    helper `theorem`s between the original declarations.
//! 3. **Inert additions.** Added text is checked against a deny-list of commands
//!    that could change what the frozen statements *mean* without touching their
//!    text: `instance`, `notation`, `variable`, `open`, `#eval`, `#guard_msgs`...
//!
//! A semantic backstop lives in [`AxiomAudit`]: once Lean accepts a candidate,
//! `#print axioms` is run for every target theorem and anything beyond Lean's
//! three standard axioms is rejected.
//!
//! The lexer mirrors Lean's own tokenizer (`Lean.Parser.Basic`) closely, because
//! any disagreement about where a comment or string starts would be a place to
//! hide a cheat from the guard.

use std::collections::HashSet;
use std::fmt;
use std::ops::Range;
use std::sync::LazyLock;

use regex::Regex;
use serde::Serialize;
use thiserror::Error;

/// Tokens that are never allowed in a candidate proof.
pub const FORBIDDEN_TOKENS: &[&str] = &[
    "sorry",
    "admit",
    "axiom",
    "unsafe",
    "implemented_by",
    "extern",
];

/// Further tokens that defeat the kernel or the `sorry` check in the same way and
/// are therefore rejected alongside [`FORBIDDEN_TOKENS`].
pub const EXTRA_FORBIDDEN_TOKENS: &[&str] =
    &["sorryAx", "native_decide", "ofReduceBool", "ofReduceNat"];

/// The axioms every Lean development is allowed to depend on.
pub const STANDARD_AXIOMS: &[&str] = &["propext", "Classical.choice", "Quot.sound"];

/// Keywords that start a declaration. A `sorry` belongs to the declaration it
/// sits in, and helper theorems may be inserted in front of any declaration.
/// (`lemma` joins the list for specs that import Mathlib.)
const DECL_KEYWORDS: &[&str] = &[
    "theorem",
    "def",
    "example",
    "instance",
    "abbrev",
    "opaque",
    "structure",
    "inductive",
    "class",
];

/// Modifiers that may precede a declaration keyword.
const DECL_MODIFIERS: &[&str] = &[
    "private",
    "protected",
    "noncomputable",
    "partial",
    "nonrec",
    "public",
    "meta",
    "local",
    "scoped",
];

/// Commands that can be prefixed to a declaration with `... in`, changing how it
/// is elaborated. They belong to the declaration: nothing may be inserted
/// between such a prefix and the declaration it applies to.
const SCOPING_COMMANDS: &[&str] = &[
    "open",
    "set_option",
    "omit",
    "include",
    "universe",
    "variable",
    "attribute",
    "unseal",
    "seal",
];

/// Commands that may not appear in text a candidate adds to the spec, because they
/// can change the meaning of the frozen statements, run arbitrary code while Lean
/// elaborates, or stop elaboration early.
const DENIED_IN_ADDITIONS: &[&str] = &[
    // declarations other than helper theorems
    "def",
    "abbrev",
    "instance",
    "opaque",
    "inductive",
    "coinductive",
    "structure",
    "class",
    "mutual",
    "deriving",
    // scoping and name resolution
    "namespace",
    "section",
    "end",
    "open",
    "export",
    "variable",
    "universe",
    "include",
    "omit",
    "local",
    "scoped",
    "attribute",
    "import",
    "prelude",
    // syntax extensions
    "notation",
    "infix",
    "infixl",
    "infixr",
    "prefix",
    "postfix",
    "macro",
    "macro_rules",
    "syntax",
    "elab",
    "elab_rules",
    "declare_syntax_cat",
    "binder_predicate",
    "unif_hint",
    "unification_hint",
    "default_instance",
    "simproc",
    "dsimproc",
    "simproc_decl",
    "register_simp_attr",
    "declare_simp_like_tactic",
    // code execution during elaboration
    "initialize",
    "builtin_initialize",
    "run_cmd",
    "run_elab",
    "run_meta",
    "run_tac",
    "by_elab",
];

/// Commands that run code while Lean elaborates a file, or pull other files in.
/// Machine-written specs must not contain them: compiling such a file would
/// execute whatever the model wrote.
const EXECUTABLE_COMMANDS: &[&str] = &[
    "import",
    "initialize",
    "builtin_initialize",
    "run_cmd",
    "run_elab",
    "run_meta",
    "run_tac",
    "by_elab",
];

/// `set_option` names a candidate may use. Everything else is rejected, most
/// importantly `debug.skipKernelTC`.
const ALLOWED_OPTIONS: &[&str] = &[
    "maxHeartbeats",
    "maxRecDepth",
    "synthInstance.maxHeartbeats",
    "synthInstance.maxSize",
];
const ALLOWED_OPTION_PREFIXES: &[&str] = &["linter.", "pp."];

/// How many altered-spec violations to report before giving up; one damaged
/// declaration usually cascades into several mismatches.
const MAX_ALTERED_REPORTS: usize = 3;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// What kind of rule a candidate broke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ViolationKind {
    /// `sorry`, `admit`, `axiom`, ... appears in the file.
    ForbiddenToken,
    /// Part of the original spec is missing or was changed.
    AlteredSpec,
    /// Added text contains a command that is not allowed.
    DisallowedAddition,
    /// A helper theorem reuses a name that already occurs in the spec.
    NameCollision,
}

/// A single guard finding, positioned in the checked file where possible.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Violation {
    pub kind: ViolationKind,
    /// 1-based line in the checked file.
    pub line: Option<usize>,
    /// 0-based column, following Lean's convention.
    pub column: Option<usize>,
    pub message: String,
    /// For [`ViolationKind::AlteredSpec`]: the original text that must be restored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub expected: Option<String>,
}

impl Violation {
    fn at(kind: ViolationKind, token: &Token, message: impl Into<String>) -> Self {
        Self {
            kind,
            line: Some(token.line),
            column: Some(token.col),
            message: message.into(),
            expected: None,
        }
    }
}

impl fmt::Display for Violation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match (self.line, self.column) {
            (Some(line), Some(col)) => write!(f, "{line}:{col}: {}", self.message),
            (Some(line), None) => write!(f, "{line}: {}", self.message),
            _ => write!(f, "{}", self.message),
        }
    }
}

/// A declaration in the spec that contains at least one `sorry`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Target {
    /// `theorem`, `def`, `example`, ...
    pub kind: String,
    /// Fully qualified name, when the declaration has one.
    pub name: Option<String>,
    /// 1-based line of the declaration in the spec.
    pub line: usize,
    /// Number of `sorry` holes inside the declaration.
    pub holes: usize,
}

impl Target {
    /// A short human-readable label such as ``theorem `foo` ``.
    pub fn label(&self) -> String {
        match &self.name {
            Some(name) => format!("{} `{}`", self.kind, name),
            None => format!("{} at spec line {}", self.kind, self.line),
        }
    }
}

/// A `def` or `abbrev` in a spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct Definition {
    /// The name as written, e.g. `midpoint` or `Pricing.applyDiscount`.
    pub name: String,
    /// 1-based line of the keyword.
    pub line: usize,
    /// Identifiers in the signature (binders and result type), in order.
    pub signature: Vec<String>,
}

/// Where a [`Target`] ended up in a candidate that kept the spec intact: its
/// statement, its proof, and the helper theorems placed in front of it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TargetSpan {
    /// Index into [`Spec::targets`].
    pub target: usize,
    /// Position (1-based line, 0-based column) of the region's first token.
    pub start: (usize, usize),
    /// Position of the first token after the region, or `None` if the region
    /// runs to the end of the file.
    pub end: Option<(usize, usize)>,
}

impl TargetSpan {
    /// Whether a position reported by Lean falls inside the region.
    pub fn contains(&self, line: usize, column: usize) -> bool {
        let position = (line, column);
        position >= self.start && self.end.is_none_or(|end| position < end)
    }
}

/// Result of checking a candidate against a [`Spec`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct GuardReport {
    pub violations: Vec<Violation>,
    /// Empty unless the candidate preserved the whole spec.
    pub spans: Vec<TargetSpan>,
}

impl GuardReport {
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }

    /// The target whose region (statement, proof and helper theorems) contains
    /// the given position in the candidate.
    pub fn target_at(&self, line: usize, column: usize) -> Option<usize> {
        self.spans
            .iter()
            .find(|span| span.contains(line, column))
            .map(|span| span.target)
    }
}

/// Why a file cannot be used as a spec.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SpecError {
    #[error(
        "the spec itself contains forbidden tokens ({}); only `sorry` placeholders are allowed",
        .0.iter().map(ToString::to_string).collect::<Vec<_>>().join("; ")
    )]
    ForbiddenTokens(Vec<Violation>),
}

// ---------------------------------------------------------------------------
// Lexer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Identifier or keyword, possibly dotted (`Nat.succ`) or escaped (`«x y»`).
    Word,
    Number,
    /// String literal, including raw strings. Contents are opaque to the guard.
    Str,
    Char,
    /// A bracket, a delimiter, or a run of operator characters.
    Symbol,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    kind: Kind,
    text: String,
    /// 1-based.
    line: usize,
    /// 0-based, counted in characters.
    col: usize,
    /// True when nothing (no whitespace, no comment) separates this token from
    /// the previous one.
    glued: bool,
}

impl Token {
    fn is_word(&self, word: &str) -> bool {
        self.kind == Kind::Word && self.text == word
    }

    fn is_symbol(&self, symbol: &str) -> bool {
        self.kind == Kind::Symbol && self.text == symbol
    }
}

/// `Char.isWhitespace` in Lean: ASCII only.
fn is_lean_whitespace(c: char) -> bool {
    matches!(c, ' ' | '\t' | '\r' | '\n')
}

/// Port of `Lean.isLetterLike` (Lean 4.36).
fn is_letter_like(c: char) -> bool {
    let v = c as u32;
    ((0x3b1..=0x3c9).contains(&v) && v != 0x3bb)
        || ((0x391..=0x3a9).contains(&v) && v != 0x3a0 && v != 0x3a3)
        || (0x3ca..=0x3fb).contains(&v)
        || (0x1f00..=0x1ffe).contains(&v)
        || (0x2100..=0x214f).contains(&v)
        || (0x1d49c..=0x1d59f).contains(&v)
        || ((0x00c0..=0x00ff).contains(&v) && v != 0x00d7 && v != 0x00f7)
        || (0x0100..=0x017f).contains(&v)
}

/// Port of `Lean.isSubScriptAlnum`.
fn is_subscript_alnum(c: char) -> bool {
    let v = c as u32;
    (0x2080..=0x2089).contains(&v)
        || (0x2090..=0x209c).contains(&v)
        || (0x1d62..=0x1d6a).contains(&v)
        || v == 0x2c7c
}

/// Port of `Lean.isIdFirst`.
fn is_id_first(c: char) -> bool {
    c.is_ascii_alphabetic() || c == '_' || is_letter_like(c)
}

/// Port of `Lean.isIdRest`.
fn is_id_rest(c: char) -> bool {
    c.is_ascii_alphanumeric()
        || matches!(c, '_' | '\'' | '!' | '?')
        || is_letter_like(c)
        || is_subscript_alnum(c)
}

/// Brackets and separators. Each is always a token of its own, and no Lean token
/// continues from one of these into `-` or `/`, so a comment may start right
/// after them.
fn is_delimiter(c: char) -> bool {
    matches!(
        c,
        '(' | ')' | '[' | ']' | '{' | '}' | '⟨' | '⟩' | '⟦' | '⟧' | '⦃' | '⦄' | ',' | ';'
    )
}

/// Characters that are glued into operator tokens such as `:=`, `<;>` or `//`.
fn is_operator_char(c: char) -> bool {
    !(is_lean_whitespace(c)
        || is_id_first(c)
        || c.is_ascii_digit()
        || is_delimiter(c)
        || matches!(c, '"' | '\'' | '«'))
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
    line: usize,
    col: usize,
}

impl Lexer {
    fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            line: 1,
            col: 0,
        }
    }

    fn peek(&self, ahead: usize) -> Option<char> {
        self.chars.get(self.pos + ahead).copied()
    }

    fn bump(&mut self) -> Option<char> {
        let c = self.chars.get(self.pos).copied()?;
        self.pos += 1;
        if c == '\n' {
            self.line += 1;
            self.col = 0;
        } else {
            self.col += 1;
        }
        Some(c)
    }

    fn take_while(&mut self, pred: impl Fn(char) -> bool) {
        while self.peek(0).is_some_and(&pred) {
            self.bump();
        }
    }

    fn text_from(&self, start: usize) -> String {
        self.chars[start..self.pos].iter().collect()
    }

    /// Skips a block comment whose opening `/-` has already been consumed.
    /// Mirrors `finishCommentBlock`: comments nest.
    fn skip_block_comment(&mut self) {
        let mut nesting = 1usize;
        while let Some(c) = self.bump() {
            if c == '-' && self.peek(0) == Some('/') {
                self.bump();
                nesting -= 1;
                if nesting == 0 {
                    return;
                }
            } else if c == '/' && self.peek(0) == Some('-') {
                self.bump();
                nesting += 1;
            }
        }
    }

    /// `"..."` with backslash escapes; mirrors `strLitFnAux`.
    fn lex_string(&mut self) {
        self.bump();
        while let Some(c) = self.bump() {
            match c {
                '\\' => {
                    self.bump();
                }
                '"' => return,
                _ => {}
            }
        }
    }

    /// Number of `#`s if the cursor (at an `r`) starts a raw string literal;
    /// mirrors `isRawStrLitStart`.
    fn raw_string_hashes(&self) -> Option<usize> {
        let mut ahead = 1;
        while self.peek(ahead) == Some('#') {
            ahead += 1;
        }
        (self.peek(ahead) == Some('"')).then_some(ahead - 1)
    }

    /// `r"..."` / `r#"..."#`; mirrors `rawStrLitFnAux`.
    fn lex_raw_string(&mut self, hashes: usize) {
        for _ in 0..hashes + 2 {
            self.bump();
        }
        loop {
            match self.bump() {
                None => return,
                Some('"') => {}
                Some(_) => continue,
            }
            if hashes == 0 {
                return;
            }
            let mut closing = 0;
            loop {
                match self.peek(0) {
                    Some('#') => {
                        self.bump();
                        closing += 1;
                        if closing == hashes {
                            return;
                        }
                    }
                    Some('"') => {
                        self.bump();
                        closing = 0;
                    }
                    _ => break,
                }
            }
        }
    }

    /// Tries to lex a character literal at a `'`; mirrors `charLitFnAux`.
    /// Leaves the cursor untouched and returns `false` when it is malformed.
    fn try_char_literal(&mut self) -> bool {
        let saved = (self.pos, self.line, self.col);
        self.bump();
        let well_formed = match self.bump() {
            None => false,
            Some('\\') => match self.bump() {
                None => false,
                Some('x') => {
                    self.bump();
                    self.bump();
                    true
                }
                Some('u') => {
                    for _ in 0..4 {
                        self.bump();
                    }
                    true
                }
                Some(_) => true,
            },
            Some(_) => true,
        };
        if well_formed && self.peek(0) == Some('\'') {
            self.bump();
            true
        } else {
            (self.pos, self.line, self.col) = saved;
            false
        }
    }

    /// Numeric literal; mirrors `numberFnAux` / `decimalNumberFn`. Crucially a
    /// number never swallows a following identifier: `1sorry` is `1` then `sorry`.
    fn lex_number(&mut self) {
        let first = self.bump();
        if first == Some('0') {
            match self.peek(0) {
                Some('b' | 'B') => {
                    self.bump();
                    self.take_while(|c| matches!(c, '0' | '1' | '_'));
                    return;
                }
                Some('o' | 'O') => {
                    self.bump();
                    self.take_while(|c| matches!(c, '0'..='7' | '_'));
                    return;
                }
                Some('x' | 'X') => {
                    self.bump();
                    self.take_while(|c| c.is_ascii_hexdigit() || c == '_');
                    return;
                }
                _ => {}
            }
        }
        self.take_while(|c| c.is_ascii_digit() || c == '_');
        if self.peek(0) == Some('.') && self.peek(1).is_some_and(|c| c.is_ascii_digit()) {
            self.bump();
            self.take_while(|c| c.is_ascii_digit() || c == '_');
        }
        if matches!(self.peek(0), Some('e' | 'E')) {
            let sign = usize::from(matches!(self.peek(1), Some('+' | '-')));
            if self.peek(1 + sign).is_some_and(|c| c.is_ascii_digit()) {
                for _ in 0..=sign {
                    self.bump();
                }
                self.take_while(|c| c.is_ascii_digit() || c == '_');
            }
        }
    }

    /// Identifier, keyword or dotted name; mirrors `identFnAux`.
    fn lex_word(&mut self) {
        loop {
            match self.peek(0) {
                Some('«') => {
                    self.bump();
                    while let Some(c) = self.bump() {
                        if c == '»' {
                            break;
                        }
                    }
                }
                Some(c) if is_id_first(c) => {
                    self.bump();
                    self.take_while(is_id_rest);
                }
                _ => return,
            }
            let continues = self.peek(0) == Some('.')
                && self.peek(1).is_some_and(|c| is_id_first(c) || c == '«');
            if !continues {
                return;
            }
            self.bump();
        }
    }
}

/// Turns Lean source into tokens, dropping whitespace and comments.
fn lex(source: &str) -> Vec<Token> {
    lex_with_comments(source).0
}

/// Like [`lex`], but also returns where the comments were, as ranges of
/// character indices.
fn lex_with_comments(source: &str) -> (Vec<Token>, Vec<Range<usize>>) {
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    let mut comments = Vec::new();
    let mut glued = false;
    // Whether the character right before the cursor belongs to an operator token.
    // Lean decides where tokens end with a longest-match token table we do not
    // have (`<-` followed by `-x`, `//` followed by `-x`), so after an operator
    // character a `--` or `/-` is treated as code rather than a comment. That is
    // the safe direction: text is never hidden from the guard.
    let mut after_operator = false;

    while let Some(c) = lexer.peek(0) {
        if is_lean_whitespace(c) {
            lexer.bump();
            glued = false;
            after_operator = false;
            continue;
        }
        if !after_operator && c == '-' && lexer.peek(1) == Some('-') {
            let start = lexer.pos;
            lexer.take_while(|c| c != '\n');
            comments.push(start..lexer.pos);
            glued = false;
            continue;
        }
        if !after_operator && c == '/' && lexer.peek(1) == Some('-') {
            let start = lexer.pos;
            lexer.bump();
            lexer.bump();
            lexer.skip_block_comment();
            comments.push(start..lexer.pos);
            glued = false;
            continue;
        }

        let (start, line, col) = (lexer.pos, lexer.line, lexer.col);
        let mut operator = false;
        let kind = if c == '"' {
            lexer.lex_string();
            Kind::Str
        } else if c == '\'' {
            if lexer.peek(1) != Some('\'') && lexer.try_char_literal() {
                Kind::Char
            } else {
                lexer.bump();
                Kind::Symbol
            }
        } else if c.is_ascii_digit() {
            lexer.lex_number();
            Kind::Number
        } else if let Some(hashes) = (c == 'r').then(|| lexer.raw_string_hashes()).flatten() {
            lexer.lex_raw_string(hashes);
            Kind::Str
        } else if is_id_first(c) || c == '«' {
            lexer.lex_word();
            Kind::Word
        } else if is_delimiter(c) {
            lexer.bump();
            Kind::Symbol
        } else {
            lexer.bump();
            lexer.take_while(is_operator_char);
            operator = true;
            Kind::Symbol
        };

        tokens.push(Token {
            kind,
            text: lexer.text_from(start),
            line,
            col,
            glued,
        });
        glued = true;
        after_operator = operator;
    }
    (tokens, comments)
}

/// Removes Lean comments from `source`: line comments (`-- ...`), block comments
/// (`/- ... -/`, which nest) and doc comments (`/-- ... -/`, `/-! ... -/`).
///
/// Comments are replaced by spaces and line breaks are kept, so every remaining
/// character stays at its original line and column. Comment markers inside
/// string and character literals are left alone.
pub fn strip_comments(source: &str) -> String {
    let (_, comments) = lex_with_comments(source);
    let mut comments = comments.into_iter().peekable();
    let mut stripped = String::with_capacity(source.len());
    for (index, c) in source.chars().enumerate() {
        while comments.peek().is_some_and(|comment| comment.end <= index) {
            comments.next();
        }
        let in_comment = comments
            .peek()
            .is_some_and(|comment| comment.contains(&index));
        stripped.push(if in_comment && c != '\n' { ' ' } else { c });
    }
    stripped
}

/// Splits a dotted identifier into components, respecting `«...»` escapes.
fn name_parts(word: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut escaped = false;
    for (i, c) in word.char_indices() {
        match c {
            '«' => escaped = true,
            '»' => escaped = false,
            '.' if !escaped => {
                parts.push(&word[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&word[start..]);
    parts
}

fn unescape(part: &str) -> &str {
    part.strip_prefix('«')
        .and_then(|inner| inner.strip_suffix('»'))
        .unwrap_or(part)
}

fn last_component(word: &str) -> &str {
    unescape(name_parts(word).last().copied().unwrap_or(word))
}

fn same_token(a: &Token, b: &Token) -> bool {
    a.kind == b.kind && a.text == b.text
}

fn same_tokens(a: &[Token], b: &[Token]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(x, y)| same_token(x, y))
}

fn common_prefix(a: &[Token], b: &[Token]) -> usize {
    a.iter()
        .zip(b)
        .take_while(|(x, y)| same_token(x, y))
        .count()
}

fn find_from(haystack: &[Token], needle: &[Token], from: usize) -> Option<usize> {
    let last_start = haystack.len().checked_sub(needle.len())?;
    (from..=last_start).find(|&start| same_tokens(&haystack[start..start + needle.len()], needle))
}

// ---------------------------------------------------------------------------
// Forbidden tokens
// ---------------------------------------------------------------------------

fn forbidden_word(part: &str) -> Option<&'static str> {
    FORBIDDEN_TOKENS
        .iter()
        .chain(EXTRA_FORBIDDEN_TOKENS)
        .copied()
        .find(|forbidden| *forbidden == part)
}

fn forbidden_reason(word: &str) -> &'static str {
    match word {
        "sorry" | "admit" | "sorryAx" => "every proof must be complete",
        "axiom" => "new axioms are not allowed",
        "unsafe" | "implemented_by" | "extern" => "it bypasses Lean's kernel",
        _ => "it trusts the compiler instead of the kernel",
    }
}

/// Forbidden words in a token stream, matched as whole identifier components so
/// that `sorry_lemma` is fine while `Foo.sorry` and `(sorry)` are not.
fn forbidden_in(tokens: &[Token]) -> Vec<(usize, &'static str)> {
    let mut found = Vec::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.kind != Kind::Word {
            continue;
        }
        let mut seen = HashSet::new();
        for part in name_parts(&token.text) {
            if let Some(word) = forbidden_word(unescape(part)) {
                if seen.insert(word) {
                    found.push((index, word));
                }
            }
        }
    }
    found
}

fn forbidden_violations(tokens: &[Token]) -> Vec<Violation> {
    forbidden_in(tokens)
        .into_iter()
        .map(|(index, word)| {
            Violation::at(
                ViolationKind::ForbiddenToken,
                &tokens[index],
                format!("forbidden token `{word}`: {}", forbidden_reason(word)),
            )
        })
        .collect()
}

/// Scans a standalone Lean file for forbidden tokens. Comments and string
/// literals are ignored.
pub fn scan(source: &str) -> Vec<Violation> {
    forbidden_violations(&lex(source))
}

// ---------------------------------------------------------------------------
// Spec
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct TargetInfo {
    target: Target,
    /// Index of the chunk that ends at this target's last `sorry`.
    last_header: usize,
}

/// A Lean file with `sorry` holes, prepared for checking candidates against it.
#[derive(Debug, Clone)]
pub struct Spec {
    source: String,
    tokens: Vec<Token>,
    /// Frozen pieces of the spec as ranges into `tokens`. A candidate must contain
    /// them in order; whatever lies between two consecutive pieces is new text.
    chunks: Vec<Range<usize>>,
    targets: Vec<TargetInfo>,
    /// Final components of every identifier in the spec, for name-collision checks.
    names: HashSet<String>,
    /// `lemma` declares a theorem in Mathlib but is an ordinary identifier in
    /// core Lean, so it only counts as a keyword for specs that import something.
    lemma_is_keyword: bool,
}

impl Spec {
    /// Analyses a spec. Fails if the spec contains forbidden tokens other than the
    /// `sorry` placeholders, since no candidate could ever pass the guard then.
    pub fn parse(source: &str) -> Result<Self, SpecError> {
        let tokens = lex(source);

        let offending: Vec<Violation> = forbidden_in(&tokens)
            .into_iter()
            .filter(|&(index, _)| !tokens[index].is_word("sorry"))
            .map(|(index, word)| {
                Violation::at(
                    ViolationKind::ForbiddenToken,
                    &tokens[index],
                    format!("forbidden token `{word}`"),
                )
            })
            .collect();
        if !offending.is_empty() {
            return Err(SpecError::ForbiddenTokens(offending));
        }

        let lemma_is_keyword = tokens.iter().any(|token| token.is_word("import"));

        // Cut the spec into frozen chunks. A cut is made
        //  * at every `sorry`, which is dropped: a proof goes there;
        //  * in front of every declaration, where helper theorems may go.
        let mut chunks = Vec::new();
        let mut targets: Vec<TargetInfo> = Vec::new();
        let mut start = 0;
        // The declaration the scan is inside: its keyword, and its target once
        // a hole has been seen in it.
        let mut declaration: Option<(usize, Option<usize>)> = None;
        let mut depth = 0usize;

        for (index, token) in tokens.iter().enumerate() {
            if token.is_word("sorry") {
                chunks.push(start..index);
                start = index + 1;
                let header = chunks.len() - 1;
                match &mut declaration {
                    Some((_, Some(target))) => {
                        targets[*target].last_header = header;
                        targets[*target].target.holes += 1;
                    }
                    Some((keyword, target)) => {
                        *target = Some(targets.len());
                        targets.push(TargetInfo {
                            target: Target {
                                kind: tokens[*keyword].text.clone(),
                                name: declaration_name(&tokens, *keyword),
                                line: tokens[*keyword].line,
                                holes: 1,
                            },
                            last_header: header,
                        });
                    }
                    None => {
                        declaration = Some((index, Some(targets.len())));
                        targets.push(TargetInfo {
                            target: Target {
                                kind: "declaration".to_string(),
                                name: None,
                                line: token.line,
                                holes: 1,
                            },
                            last_header: header,
                        });
                    }
                }
                continue;
            }
            match token.kind {
                Kind::Symbol if is_opening_bracket(&token.text) => depth += 1,
                Kind::Symbol if is_closing_bracket(&token.text) => depth = depth.saturating_sub(1),
                // Keywords inside brackets (`@[instance]`) do not start anything.
                Kind::Word if depth == 0 && is_declaration_keyword(token, lemma_is_keyword) => {
                    declaration = Some((index, None));
                    if let Some(at) = declaration_start(&tokens, index, start) {
                        chunks.push(start..at);
                        start = at;
                    }
                }
                _ => {}
            }
        }
        chunks.push(start..tokens.len());

        let names = tokens
            .iter()
            .filter(|token| token.kind == Kind::Word)
            .map(|token| last_component(&token.text).to_string())
            .collect();

        Ok(Self {
            source: source.to_string(),
            tokens,
            chunks,
            targets,
            names,
            lemma_is_keyword,
        })
    }

    /// The original spec text.
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Declarations that contain `sorry` and therefore need a proof.
    pub fn targets(&self) -> Vec<Target> {
        self.targets
            .iter()
            .map(|info| info.target.clone())
            .collect()
    }

    /// Number of `sorry` placeholders in the spec.
    pub fn hole_count(&self) -> usize {
        self.targets.iter().map(|info| info.target.holes).sum()
    }

    /// Every `def` and `abbrev` with the identifiers of its signature, i.e. of
    /// everything between the name and the body.
    pub fn definitions(&self) -> Vec<Definition> {
        let tokens = &self.tokens;
        let mut definitions = Vec::new();
        for (index, keyword) in tokens.iter().enumerate() {
            if !(keyword.is_word("def") || keyword.is_word("abbrev")) {
                continue;
            }
            let Some(name) = tokens
                .get(index + 1)
                .filter(|token| token.kind == Kind::Word)
            else {
                continue;
            };
            let mut signature = Vec::new();
            let mut depth = 0usize;
            for token in &tokens[index + 2..] {
                match token.kind {
                    Kind::Symbol if matches!(token.text.as_str(), "(" | "[" | "{" | "⟨" | "⦃") => {
                        depth += 1
                    }
                    Kind::Symbol if matches!(token.text.as_str(), ")" | "]" | "}" | "⟩" | "⦄") =>
                    {
                        depth = depth.saturating_sub(1);
                    }
                    // `:=` starts the body, `|` the first pattern-matching arm.
                    Kind::Symbol
                        if depth == 0 && (token.text == ":=" || token.text.starts_with('|')) =>
                    {
                        break;
                    }
                    Kind::Word if depth == 0 && token.text == "where" => break,
                    Kind::Word if depth == 0 && DECL_KEYWORDS.contains(&token.text.as_str()) => {
                        break;
                    }
                    Kind::Word => signature.push(token.text.clone()),
                    _ => {}
                }
            }
            definitions.push(Definition {
                name: name.text.clone(),
                line: keyword.line,
                signature,
            });
        }
        definitions
    }

    /// Commands in the spec that would run code during elaboration (`#eval`,
    /// `run_cmd`, ...) or load other files (`import`).
    pub fn executable_constructs(&self) -> Vec<Violation> {
        let tokens = &self.tokens;
        let mut found = Vec::new();
        for (index, token) in tokens.iter().enumerate() {
            if token.kind == Kind::Word && EXECUTABLE_COMMANDS.contains(&token.text.as_str()) {
                found.push(Violation::at(
                    ViolationKind::DisallowedAddition,
                    token,
                    format!("`{}` is not allowed", token.text),
                ));
            } else if let Some(command) = hash_command(tokens, index, tokens.len()) {
                found.push(Violation::at(
                    ViolationKind::DisallowedAddition,
                    token,
                    format!("`#{command}` commands are not allowed"),
                ));
            }
        }
        found
    }

    /// Checks a candidate against the spec.
    pub fn check(&self, candidate: &str) -> GuardReport {
        let tokens = lex(candidate);
        let mut violations = forbidden_violations(&tokens);
        let mut spans = Vec::new();

        if let Some(matches) = self.align(&tokens, &mut violations) {
            for pair in matches.windows(2) {
                self.check_addition(&tokens, pair[0].end..pair[1].start, &mut violations);
            }
            spans = self.spans(&tokens, &matches);
        }

        violations.sort_by_key(|v| (v.line.unwrap_or(usize::MAX), v.column.unwrap_or(usize::MAX)));
        violations.dedup();
        GuardReport { violations, spans }
    }

    /// Locates every frozen chunk in the candidate. Returns the matched ranges
    /// only if all of them were found; otherwise records what went wrong.
    fn align(&self, cand: &[Token], violations: &mut Vec<Violation>) -> Option<Vec<Range<usize>>> {
        let last = self.chunks.len() - 1;
        let mut matches = Vec::with_capacity(self.chunks.len());
        let mut pos = 0;
        let mut altered = 0;

        for (index, chunk) in self.chunks.iter().enumerate() {
            let want = &self.tokens[chunk.clone()];
            if want.is_empty() {
                // Nothing to find. An empty final chunk means the last proof runs
                // to the end of the file.
                if index == last {
                    pos = cand.len();
                }
                matches.push(pos..pos);
                continue;
            }

            let found = if index == 0 && index == last {
                // No holes at all: the candidate must be the spec.
                same_tokens(cand, want).then_some(0)
            } else if index == 0 {
                // Nothing may precede the spec's opening text.
                (common_prefix(cand, want) == want.len()).then_some(0)
            } else if index == last {
                // Nothing may follow the spec's closing text.
                cand.len()
                    .checked_sub(want.len())
                    .filter(|&start| start >= pos && same_tokens(&cand[start..], want))
            } else {
                // Leftmost match: it leaves the most room for the chunks after it.
                find_from(cand, want, pos)
            };

            match found {
                Some(start) => {
                    pos = start + want.len();
                    matches.push(start..pos);
                }
                None => {
                    altered += 1;
                    if altered <= MAX_ALTERED_REPORTS {
                        violations.push(self.describe_mismatch(index, cand, pos));
                    }
                    matches.push(pos..pos);
                }
            }
        }
        (altered == 0).then_some(matches)
    }

    /// Explains why chunk `index` could not be found at or after `from`.
    fn describe_mismatch(&self, index: usize, cand: &[Token], from: usize) -> Violation {
        let want = &self.tokens[self.chunks[index].clone()];
        let last = self.chunks.len() - 1;
        let spec_line = want[0].line;
        let expected = Some(self.excerpt(index));
        let altered = |line, column, message: String| Violation {
            kind: ViolationKind::AlteredSpec,
            line,
            column,
            message,
            expected: expected.clone(),
        };

        let trailing = |extra: &Token| {
            altered(
                Some(extra.line),
                Some(extra.col),
                format!(
                    "unexpected text `{}` after the end of the original file; nothing may follow the spec's last line",
                    extra.text
                ),
            )
        };

        // The closing text is present but something was appended after it.
        if index == last && index != 0 {
            let extra = find_from(cand, want, from).and_then(|start| cand.get(start + want.len()));
            if let Some(extra) = extra {
                return trailing(extra);
            }
        }

        // Find where the candidate follows the original text for the longest stretch.
        let anchors: Vec<usize> = if index == 0 {
            vec![0]
        } else {
            (from..cand.len())
                .filter(|&start| same_token(&cand[start], &want[0]))
                .collect()
        };
        let best = anchors
            .into_iter()
            .map(|start| (start, common_prefix(&cand[start.min(cand.len())..], want)))
            .max_by_key(|&(start, matched)| (matched, std::cmp::Reverse(start)));

        match best {
            // A spec without holes was reproduced, then something was added.
            Some((start, matched)) if matched == want.len() => match cand.get(start + matched) {
                Some(extra) => trailing(extra),
                None => altered(None, None, "original text was changed".to_string()),
            },
            // Sharing only the leading keyword (`theorem`) with some other
            // declaration does not make that declaration a modified copy.
            Some((start, matched)) if matched >= 2 || index == 0 => {
                let wanted = &want[matched];
                match cand.get(start + matched) {
                    Some(found) if index == 0 && matched == 0 => altered(
                        Some(found.line),
                        Some(found.col),
                        format!(
                            "the file must start with the original spec text; expected `{}` (spec line {}) but found `{}`",
                            wanted.text, wanted.line, found.text
                        ),
                    ),
                    Some(found) => altered(
                        Some(found.line),
                        Some(found.col),
                        format!(
                            "original text was changed: expected `{}` (spec line {}) but found `{}`",
                            wanted.text, wanted.line, found.text
                        ),
                    ),
                    None => altered(
                        cand.last().map(|token| token.line),
                        None,
                        format!(
                            "the file ends before the original text is complete; `{}` from spec line {} is missing",
                            wanted.text, wanted.line
                        ),
                    ),
                }
            }
            _ => altered(
                None,
                None,
                format!(
                    "original text starting at spec line {spec_line} is missing or was rewritten"
                ),
            ),
        }
    }

    /// The source lines a chunk spans, for showing the model what to restore.
    fn excerpt(&self, index: usize) -> String {
        const MAX_LINES: usize = 12;
        let chunk = &self.tokens[self.chunks[index].clone()];
        let (Some(first), Some(last)) = (chunk.first(), chunk.last()) else {
            return String::new();
        };
        let lines: Vec<&str> = self
            .source
            .lines()
            .skip(first.line - 1)
            .take(last.line - first.line + 1)
            .collect();
        if lines.len() > MAX_LINES {
            format!("{}\n...", lines[..MAX_LINES].join("\n"))
        } else {
            lines.join("\n")
        }
    }

    /// Checks one stretch of added text (a proof or helper theorems).
    fn check_addition(&self, cand: &[Token], gap: Range<usize>, violations: &mut Vec<Violation>) {
        let disallowed = |token: &Token, message: String| {
            Violation::at(ViolationKind::DisallowedAddition, token, message)
        };

        for index in gap.clone() {
            let token = &cand[index];
            let next = (index + 1 < gap.end).then(|| &cand[index + 1]);
            match token.kind {
                Kind::Word if DENIED_IN_ADDITIONS.contains(&token.text.as_str()) => {
                    violations.push(disallowed(
                        token,
                        format!(
                            "`{}` is not allowed in added text: you may only add proofs and helper `theorem`s",
                            token.text
                        ),
                    ));
                }
                Kind::Word if token.text == "set_option" => {
                    let allowed = next.is_some_and(|option| {
                        option.kind == Kind::Word
                            && (ALLOWED_OPTIONS.contains(&option.text.as_str())
                                || ALLOWED_OPTION_PREFIXES
                                    .iter()
                                    .any(|prefix| option.text.starts_with(prefix)))
                    });
                    if !allowed {
                        let name = next.map_or("", |option| option.text.as_str());
                        violations.push(disallowed(
                            token,
                            format!("`set_option {name}` is not allowed in added text"),
                        ));
                    }
                }
                Kind::Word if self.declares_theorem(token) => {
                    match next.filter(|name| name.kind == Kind::Word) {
                        Some(name) => {
                            let short = last_component(&name.text);
                            if self.names.contains(short) {
                                violations.push(Violation::at(
                                    ViolationKind::NameCollision,
                                    name,
                                    format!(
                                        "helper theorem `{}` reuses the name `{short}`, which already occurs in the spec; choose a fresh name",
                                        name.text
                                    ),
                                ));
                            }
                        }
                        None => violations.push(disallowed(
                            token,
                            format!("`{}` must be followed by a name", token.text),
                        )),
                    }
                }
                Kind::Symbol => {
                    if let Some(command) = hash_command(cand, index, gap.end) {
                        violations.push(disallowed(
                            token,
                            format!("`#{command}` commands are not allowed in added text"),
                        ));
                    }
                }
                Kind::Str if token.glued && index > gap.start => {
                    let previous = &cand[index - 1];
                    if previous.kind == Kind::Word && previous.text.ends_with('!') {
                        violations.push(disallowed(
                            previous,
                            format!(
                                "interpolated strings (`{}\"...\"`) are not allowed in added text",
                                previous.text
                            ),
                        ));
                    }
                }
                _ => {}
            }
        }
    }

    /// Divides the candidate among the targets, so that every position Lean
    /// reports belongs to exactly one of them.
    ///
    /// A target's region ends where its last proof ends; everything from there
    /// up to the end of the next target's proof (frozen text, helper theorems,
    /// statement, proof) is the next target's region. The text after a proof
    /// holds the helper theorems of whatever comes next, and since a proof
    /// cannot contain a command, the proof ends where the first helper starts.
    fn spans(&self, cand: &[Token], matches: &[Range<usize>]) -> Vec<TargetSpan> {
        let last_chunk = self.chunks.len() - 1;
        let position = |index: usize| cand.get(index).map(|token| (token.line, token.col));

        let mut spans = Vec::with_capacity(self.targets.len());
        let mut start = Some((1, 0));
        for (target, info) in self.targets.iter().enumerate() {
            let Some(region_start) = start else { break };
            let end = if target + 1 == self.targets.len() {
                None
            } else {
                let after = info.last_header + 1;
                let proof_end = if after < last_chunk && self.chunks[after].is_empty() {
                    // The next frozen text follows without anything in between.
                    self.first_command(
                        cand,
                        matches[info.last_header].end,
                        matches[after + 1].start,
                    )
                } else {
                    matches[after].start
                };
                position(proof_end)
            };
            spans.push(TargetSpan {
                target,
                start: region_start,
                end,
            });
            start = end;
        }
        spans
    }

    /// Whether `token` is a keyword that starts a helper theorem.
    fn declares_theorem(&self, token: &Token) -> bool {
        token.is_word("theorem") || (self.lemma_is_keyword && token.is_word("lemma"))
    }

    /// Index of the first helper declaration in `tokens[from..to]`, or `to` if
    /// the range holds no command. The only commands the guard lets a candidate
    /// add are theorems and `example`s.
    fn first_command(&self, tokens: &[Token], from: usize, to: usize) -> usize {
        (from..to)
            .find(|&i| self.declares_theorem(&tokens[i]) || tokens[i].is_word("example"))
            .map_or(to, |keyword| {
                declaration_start(tokens, keyword, from).unwrap_or(keyword)
            })
    }

    /// Prepares the `#print axioms` audit for every named target, or `None` if
    /// the spec has no declaration that can be audited by name.
    pub fn axiom_audit(&self) -> Option<AxiomAudit> {
        let names: Vec<String> = self
            .targets
            .iter()
            .filter(|info| info.target.kind != "example")
            .filter_map(|info| info.target.name.clone())
            .collect();
        (!names.is_empty()).then_some(AxiomAudit { names })
    }
}

/// If the token at `index` is the `#` of a command such as `#eval`, `#exit` or
/// `#guard_msgs`, returns the command's name. Array (`#[1, 2]`) and vector
/// (`#v[1, 2]`) literals are not commands.
fn hash_command(tokens: &[Token], index: usize, end: usize) -> Option<&str> {
    let hash = &tokens[index];
    if hash.kind != Kind::Symbol || !hash.text.ends_with('#') || index + 1 >= end {
        return None;
    }
    let command = &tokens[index + 1];
    if command.kind != Kind::Word || !command.glued {
        return None;
    }
    let vector_literal = command.text == "v"
        && tokens
            .get(index + 2)
            .is_some_and(|bracket| bracket.glued && bracket.is_symbol("["));
    (!vector_literal).then_some(command.text.as_str())
}

fn is_opening_bracket(text: &str) -> bool {
    matches!(text, "(" | "[" | "{" | "⟨" | "⦃")
}

fn is_closing_bracket(text: &str) -> bool {
    matches!(text, ")" | "]" | "}" | "⟩" | "⦄")
}

fn is_declaration_keyword(token: &Token, lemma_is_keyword: bool) -> bool {
    token.kind == Kind::Word
        && (DECL_KEYWORDS.contains(&token.text.as_str())
            || (lemma_is_keyword && token.text == "lemma"))
}

/// Where the declaration introduced by the keyword at `keyword` really starts:
/// in front of its modifiers (`private`), attribute blocks (`@[simp]`) and
/// scoping prefixes (`open Foo in`, `set_option maxRecDepth 2000 in`).
///
/// Text inserted at that point cannot come between the declaration and anything
/// that shapes its meaning. `None` means no such point was found, because a
/// `... in` prefix could not be traced back to its command; nothing may be
/// inserted in front of that declaration then. The search stops at `floor`.
fn declaration_start(tokens: &[Token], keyword: usize, floor: usize) -> Option<usize> {
    let mut start = keyword;
    while start > floor {
        let previous = &tokens[start - 1];
        if previous.kind == Kind::Word && DECL_MODIFIERS.contains(&previous.text.as_str()) {
            start -= 1;
        } else if let Some(open) = attribute_block_start(tokens, start - 1, floor) {
            start = open;
        } else if previous.is_word("in") {
            start = scoping_command_start(tokens, start - 1, floor)?;
        } else {
            break;
        }
    }
    Some(start)
}

/// Given the index of the `in` that ends a prefix such as `open Foo in`, finds
/// the keyword that starts the prefix.
fn scoping_command_start(tokens: &[Token], in_index: usize, floor: usize) -> Option<usize> {
    (floor..in_index)
        .rev()
        // Do not run back into the previous command.
        .take_while(|&i| !(tokens[i].is_word("in") || tokens[i].is_symbol(":=")))
        .find(|&i| {
            tokens[i].kind == Kind::Word && SCOPING_COMMANDS.contains(&tokens[i].text.as_str())
        })
}

/// If `close` is the `]` of an attribute block `@[...]`, returns the index of its `@`.
fn attribute_block_start(tokens: &[Token], close: usize, floor: usize) -> Option<usize> {
    if !tokens[close].is_symbol("]") {
        return None;
    }
    let mut depth = 0usize;
    let mut index = close;
    loop {
        if tokens[index].is_symbol("]") {
            depth += 1;
        } else if tokens[index].is_symbol("[") {
            depth -= 1;
            if depth == 0 {
                break;
            }
        }
        if index == floor {
            return None;
        }
        index -= 1;
    }
    let at = index.checked_sub(1).filter(|&at| at >= floor)?;
    (tokens[at].is_symbol("@") && tokens[index].glued).then_some(at)
}

/// The fully qualified name a declaration keyword introduces, if any.
fn declaration_name(tokens: &[Token], keyword: usize) -> Option<String> {
    if tokens[keyword].text == "example" {
        return None;
    }
    let name = tokens.get(keyword + 1).filter(|t| t.kind == Kind::Word)?;
    // `instance : Foo` is anonymous; `instance foo : Foo` is not. A keyword in
    // name position (`instance priority ...`) never happens for the other kinds.
    if let Some(absolute) = name.text.strip_prefix("_root_.") {
        return Some(absolute.to_string());
    }
    let mut path = namespace_at(tokens, keyword);
    path.push(name.text.clone());
    Some(path.join("."))
}

/// The namespace that is open at token `index`.
fn namespace_at(tokens: &[Token], index: usize) -> Vec<String> {
    // `Some(n)` is a namespace that pushed `n` components, `None` a section or
    // `mutual` block; all three are closed by `end`.
    let mut scopes: Vec<Option<usize>> = Vec::new();
    let mut path: Vec<String> = Vec::new();
    let mut i = 0;
    while i < index {
        let token = &tokens[i];
        if token.kind == Kind::Word {
            match token.text.as_str() {
                "namespace" => {
                    if let Some(name) = tokens.get(i + 1).filter(|t| t.kind == Kind::Word) {
                        let parts = name_parts(&name.text);
                        scopes.push(Some(parts.len()));
                        path.extend(parts.into_iter().map(str::to_string));
                        i += 1;
                    }
                }
                "section" | "mutual" => scopes.push(None),
                "end" => {
                    if let Some(Some(count)) = scopes.pop() {
                        path.truncate(path.len().saturating_sub(count));
                    }
                }
                _ => {}
            }
        }
        i += 1;
    }
    path
}

// ---------------------------------------------------------------------------
// Axiom audit
// ---------------------------------------------------------------------------

static AXIOM_REPORT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r"(?m)^'(?P<name>[^'\n]+)' (?:depends on axioms: \[(?P<axioms>[^\]]*)\]|does not depend on any axioms)",
    )
    .expect("valid regex")
});

/// Semantic check run after Lean accepts a candidate: every target theorem must
/// exist and depend on nothing but [`STANDARD_AXIOMS`].
///
/// This catches what no textual rule can see reliably, for example a proof whose
/// errors were swallowed (`sorryAx`), an axiom smuggled in through a macro, or a
/// target that silently disappeared from the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AxiomAudit {
    names: Vec<String>,
}

/// A theorem that depends on axioms outside [`STANDARD_AXIOMS`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TaintedTheorem {
    pub name: String,
    pub axioms: Vec<String>,
}

/// Outcome of an [`AxiomAudit`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
pub enum AuditOutcome {
    /// All targets exist and use only these (standard) axioms.
    Clean(Vec<String>),
    /// Some targets depend on non-standard axioms.
    Tainted(Vec<TaintedTheorem>),
    /// Lean did not report on every target, e.g. because one does not exist.
    Inconclusive(String),
}

impl AxiomAudit {
    /// Fully qualified names of the audited declarations.
    pub fn names(&self) -> &[String] {
        &self.names
    }

    /// Lean commands to append to a candidate before compiling it once more.
    pub fn commands(&self) -> String {
        let mut commands = String::from("\n\n");
        for name in &self.names {
            commands.push_str("#print axioms _root_.");
            commands.push_str(name);
            commands.push('\n');
        }
        commands
    }

    /// Interprets Lean's stdout for a candidate followed by [`Self::commands`].
    pub fn evaluate(&self, lean_stdout: &str) -> AuditOutcome {
        // Our commands come last in the file, so their reports come last too;
        // earlier reports belong to `#print axioms` lines of the spec itself.
        let reports: Vec<_> = AXIOM_REPORT.captures_iter(lean_stdout).collect();
        let Some(skip) = reports.len().checked_sub(self.names.len()) else {
            return AuditOutcome::Inconclusive(format!(
                "Lean reported on {} of {} target declarations; the others do not exist under their original names",
                reports.len(),
                self.names.len()
            ));
        };

        let mut used = Vec::new();
        let mut tainted = Vec::new();
        for (expected, report) in self.names.iter().zip(&reports[skip..]) {
            let reported = &report["name"];
            // Private declarations are printed with a mangled prefix.
            if reported != expected && !reported.ends_with(&format!(".{expected}")) {
                return AuditOutcome::Inconclusive(format!(
                    "expected an axiom report for `{expected}` but Lean reported on `{reported}`"
                ));
            }
            let axioms: Vec<String> = report
                .name("axioms")
                .map(|list| list.as_str())
                .unwrap_or_default()
                .split(',')
                .map(|axiom| axiom.trim().to_string())
                .filter(|axiom| !axiom.is_empty())
                .collect();
            let (standard, foreign): (Vec<String>, Vec<String>) = axioms
                .into_iter()
                .partition(|axiom| STANDARD_AXIOMS.contains(&axiom.as_str()));
            for axiom in standard {
                if !used.contains(&axiom) {
                    used.push(axiom);
                }
            }
            if !foreign.is_empty() {
                tainted.push(TaintedTheorem {
                    name: expected.clone(),
                    axioms: foreign,
                });
            }
        }

        if tainted.is_empty() {
            AuditOutcome::Clean(used)
        } else {
            AuditOutcome::Tainted(tainted)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SPEC: &str = "\
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

/-- Appending the empty list changes nothing. -/
theorem myAppend_nil {α : Type} (xs : List α) : myAppend xs [] = xs := by
  sorry
";

    const PROOF: &str = "\
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

theorem myAppend_nil {α : Type} (xs : List α) : myAppend xs [] = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myAppend, ih]
";

    fn texts(source: &str) -> Vec<String> {
        lex(source).into_iter().map(|t| t.text).collect()
    }

    fn kinds_of(report: &GuardReport) -> Vec<ViolationKind> {
        report.violations.iter().map(|v| v.kind).collect()
    }

    fn spec() -> Spec {
        Spec::parse(SPEC).expect("spec parses")
    }

    // -- forbidden tokens ---------------------------------------------------

    #[test]
    fn blocks_every_forbidden_token() {
        for token in FORBIDDEN_TOKENS.iter().chain(EXTRA_FORBIDDEN_TOKENS) {
            let source = format!("theorem t : True := by\n  {token}\n");
            let found = scan(&source);
            assert_eq!(found.len(), 1, "{token} should be rejected");
            assert_eq!(found[0].kind, ViolationKind::ForbiddenToken);
            assert_eq!((found[0].line, found[0].column), (Some(2), Some(2)));
            assert!(found[0].message.contains(token));
        }
    }

    #[test]
    fn blocks_sorry_and_axiom_in_any_position() {
        assert_eq!(scan("theorem t : 1 = 1 := sorry").len(), 1);
        assert_eq!(scan("theorem t : 1 = 1 := by exact (sorry)").len(), 1);
        assert_eq!(scan("theorem t : p ∧ q := ⟨sorry, sorry⟩").len(), 2);
        assert_eq!(
            scan("axiom cheat : False\ntheorem t : False := cheat").len(),
            1
        );
        assert_eq!(
            scan("theorem t : False := by\n  first | rfl | admit").len(),
            1
        );
        assert_eq!(scan("@[implemented_by fast] def slow := 1").len(), 1);
        assert_eq!(scan("@[extern \"c_fn\"] def f : Nat := 0").len(), 1);
        assert_eq!(scan("unsafe def f : Nat := 0").len(), 1);
    }

    #[test]
    fn ignores_comments() {
        let source = "\
-- sorry, this lemma took a while; no axiom needed
/- an earlier draft used `sorry` and `admit`
   /- nested: unsafe extern -/
   still inside the outer comment: axiom -/
/-- Doc comment mentioning sorry. -/
theorem t : True := trivial -- sorry
/-! module doc: implemented_by -/
";
        assert_eq!(scan(source), vec![]);
    }

    #[test]
    fn strips_comments_but_keeps_positions() {
        let source = "\
def f := 1 -- one
/- block
   /- nested -/ still -/ def g := \"-- not a comment\"
/-- doc -/ theorem t : f = 1 := rfl
";
        let stripped = strip_comments(source);
        let lines: Vec<&str> = stripped.lines().collect();
        assert_eq!(lines[0], format!("def f := 1{}", " ".repeat(7)));
        assert_eq!(lines[1], " ".repeat(8));
        // The block comment ran from line 2 to the middle of line 3.
        assert_eq!(
            lines[2],
            format!("{}def g := \"-- not a comment\"", " ".repeat(25))
        );
        assert_eq!(
            lines[3],
            format!("{}theorem t : f = 1 := rfl", " ".repeat(11))
        );
        // Same shape: every token is still where it was.
        assert_eq!(stripped.lines().count(), source.lines().count());
        assert_eq!(stripped.chars().count(), source.chars().count());
        assert_eq!(texts(&stripped), texts(source));
        assert_eq!(strip_comments("no comments here"), "no comments here");
        assert_eq!(strip_comments("x /- unterminated"), "x                ");
    }

    #[test]
    fn ignores_string_and_char_literals() {
        let source = r##"
def apology : String := "sorry, the axiom is unsafe"
def escaped : String := "quote \" sorry"
def raw : String := r#"sorry "quoted" admit"#
def quote : Char := '"'
def after : String := "sorry"
"##;
        assert_eq!(scan(source), vec![]);
    }

    #[test]
    fn matches_whole_words_only() {
        let fine = "\
theorem sorry_lemma : True := trivial
theorem not_sorry : True := trivial
theorem mysorry' : True := trivial
theorem axiomatic (unsafeCount externs : Nat) : True := trivial
";
        assert_eq!(scan(fine), vec![]);

        for cheat in [
            "exact Foo.sorry",
            "exact sorry.elim",
            "exact «sorry»",
            "exact (fun (_ : Nat) (h : False) => h) 1sorry",
            "exact Lean.Elab.admit",
            "exact `sorry",
            "exact\tsorry",
        ] {
            assert_eq!(scan(cheat).len(), 1, "{cheat:?} should be rejected");
        }
    }

    // -- lexer ---------------------------------------------------------------

    #[test]
    fn comments_do_not_start_inside_operators() {
        // In Lean `//-x` is the subtype bar followed by a negation, not a comment.
        assert_eq!(
            texts("{x : Int //-x = 0} sorry"),
            ["{", "x", ":", "Int", "//-", "x", "=", "0", "}", "sorry"]
        );
        // `<--x` is `<-` followed by `-x`.
        assert_eq!(texts("a <--sorry"), ["a", "<--", "sorry"]);
        // After a word, a bracket or whitespace a comment does start.
        assert_eq!(texts("foo--bar\nbaz"), ["foo", "baz"]);
        assert_eq!(texts("f (x)-- c\ny"), ["f", "(", "x", ")", "y"]);
        assert_eq!(texts("a /- b -/ c"), ["a", "c"]);
        assert_eq!(texts("a/- b -/c"), ["a", "c"]);
    }

    #[test]
    fn block_comments_nest_like_lean() {
        assert_eq!(texts("a /- x /- y -/ z -/ b"), ["a", "b"]);
        assert_eq!(texts("a /-/ still a comment -/ b"), ["a", "b"]);
        assert_eq!(texts("a /- unterminated sorry"), ["a"]);
    }

    #[test]
    fn lexes_literals_like_lean() {
        assert_eq!(texts(r#"'"' "a" 'x' '\n' '\x41' '\''"#).len(), 6);
        assert_eq!(
            texts(r###"r"a\" b r#"c "d" e"# r##"f"#g"## h"###),
            [
                r#"r"a\""#,
                "b",
                r##"r#"c "d" e"#"##,
                r###"r##"f"#g"##"###,
                "h"
            ]
        );
        // `x'` is an identifier, not the start of a character literal.
        assert_eq!(texts("x' y'' h'"), ["x'", "y''", "h'"]);
        // Escaped identifiers are opaque, even when they contain comment markers.
        assert_eq!(
            texts("def «--» := 1 sorry"),
            ["def", "«--»", ":=", "1", "sorry"]
        );
        assert_eq!(
            texts("Foo.«bar baz».qux x.1"),
            ["Foo.«bar baz».qux", "x", ".", "1"]
        );
    }

    #[test]
    fn numbers_never_swallow_identifiers() {
        assert_eq!(texts("1sorry"), ["1", "sorry"]);
        assert_eq!(texts("0xFFsorry"), ["0xFF", "sorry"]);
        assert_eq!(
            texts("1_000 2.5 1e10 3.25e-2 0b101 1..2"),
            ["1_000", "2.5", "1e10", "3.25e-2", "0b101", "1", "..", "2"]
        );
        assert_eq!(texts("2 ^ 32"), ["2", "^", "32"]);
    }

    #[test]
    fn tracks_positions_like_lean() {
        let tokens = lex("theorem t :\n    α → sorry");
        let sorry = tokens.last().unwrap();
        // Lean counts columns in code points, starting at 0.
        assert_eq!((sorry.line, sorry.col), (2, 8));
    }

    // -- frozen spec ---------------------------------------------------------

    #[test]
    fn accepts_a_faithful_proof() {
        let spec = spec();
        let report = spec.check(PROOF);
        assert!(report.passed(), "{:?}", report.violations);
        assert_eq!(report.spans.len(), 1);
        // With a single target, the whole file is its region.
        assert_eq!(report.target_at(7, 2), Some(0));
        assert_eq!(report.target_at(2, 0), Some(0));
    }

    #[test]
    fn describes_targets() {
        let spec = spec();
        assert_eq!(
            spec.targets(),
            vec![Target {
                kind: "theorem".into(),
                name: Some("myAppend_nil".into()),
                line: 6,
                holes: 1
            }]
        );
        assert_eq!(spec.hole_count(), 1);
    }

    #[test]
    fn tolerates_reformatting_and_comments() {
        let candidate = "\
-- definition copied from the spec
def myAppend {α : Type} :
    List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

theorem myAppend_nil {α : Type} (xs : List α) :
    myAppend xs [] = xs := by   /- proof follows -/
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [myAppend, ih]
";
        assert!(spec().check(candidate).passed());
    }

    #[test]
    fn accepts_helper_theorems_before_the_target() {
        let candidate = "\
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

@[simp] theorem myAppend_cons_nil {α : Type} (x : α) (xs : List α) :
    myAppend (x :: xs) [] = x :: myAppend xs [] := rfl

private theorem helper_two : True := trivial

theorem myAppend_nil {α : Type} (xs : List α) : myAppend xs [] = xs := by
  induction xs with
  | nil => rfl
  | cons x xs ih => simp [ih]
";
        let report = spec().check(candidate);
        assert!(report.passed(), "{:?}", report.violations);
        // Helpers count as part of the target they serve.
        assert_eq!(report.target_at(5, 0), Some(0));
    }

    #[test]
    fn catches_a_weakened_statement() {
        // An extra hypothesis makes the theorem trivially true.
        let candidate = PROOF.replace(
            "(xs : List α) : myAppend xs [] = xs",
            "(xs : List α) (h : xs = []) : myAppend xs [] = xs",
        );
        let report = spec().check(&candidate);
        assert_eq!(kinds_of(&report), [ViolationKind::AlteredSpec]);
        let violation = &report.violations[0];
        assert_eq!(violation.line, Some(5));
        assert!(
            violation.message.contains("expected `:`"),
            "{}",
            violation.message
        );
        assert!(
            violation
                .expected
                .as_deref()
                .unwrap()
                .contains("theorem myAppend_nil")
        );
    }

    #[test]
    fn catches_a_changed_conclusion() {
        let candidate = PROOF.replace(
            "myAppend xs [] = xs := by",
            "myAppend xs [] = myAppend xs [] := by",
        );
        assert_eq!(
            kinds_of(&spec().check(&candidate)),
            [ViolationKind::AlteredSpec]
        );
    }

    #[test]
    fn catches_a_removed_theorem() {
        let candidate = "\
def myAppend {α : Type} : List α → List α → List α
  | [], ys => ys
  | x :: xs, ys => x :: myAppend xs ys

theorem something_else : True := trivial
";
        let report = spec().check(candidate);
        assert_eq!(kinds_of(&report), [ViolationKind::AlteredSpec]);
        assert!(report.violations[0].message.contains("missing"));
        assert!(report.spans.is_empty());
    }

    #[test]
    fn catches_a_changed_definition() {
        // "Fixing" the function under test is the classic way to make a false theorem true.
        let candidate = PROOF.replace("| [], ys => ys", "| [], _ => []");
        let report = spec().check(&candidate);
        assert_eq!(kinds_of(&report), [ViolationKind::AlteredSpec]);
        assert_eq!(report.violations[0].line, Some(2));
    }

    #[test]
    fn catches_text_around_the_spec() {
        // The spec opens and closes with text that is not a declaration, so
        // nothing may come before or after it.
        let closed = "namespace A\ntheorem t : True := by sorry\nend A\n";
        let spec = Spec::parse(closed).unwrap();
        assert!(
            spec.check("namespace A\ntheorem t : True := by trivial\nend A\n")
                .passed()
        );

        let report = spec.check(
            "theorem early : True := trivial\nnamespace A\ntheorem t : True := by trivial\nend A\n",
        );
        assert_eq!(kinds_of(&report), [ViolationKind::AlteredSpec]);
        assert!(report.violations[0].message.contains("must start with"));

        let report = spec.check(
            "namespace A\ntheorem t : True := by trivial\nend A\ntheorem late : True := trivial\n",
        );
        assert_eq!(kinds_of(&report), [ViolationKind::AlteredSpec]);
        assert!(report.violations[0].message.contains("after the end"));
    }

    #[test]
    fn accepts_helpers_in_front_of_any_declaration() {
        let source = "\
def f (n : Nat) : Nat := n + 1

def g (n : Nat) : Nat := f (f n)

theorem g_eq (n : Nat) : g n = n + 2 := by sorry
";
        let spec = Spec::parse(source).unwrap();

        // Lemmas next to the definitions they are about, as a human would write.
        let candidate = "\
theorem zero_eq : (0 : Nat) = 0 := rfl

def f (n : Nat) : Nat := n + 1

@[simp] theorem f_eq (n : Nat) : f n = n + 1 := rfl

def g (n : Nat) : Nat := f (f n)

theorem g_eq (n : Nat) : g n = n + 2 := by
  simp [g]
";
        let report = spec.check(candidate);
        assert!(report.passed(), "{:?}", report.violations);

        // Still only theorems, wherever they are put ...
        let with_def = candidate.replace("theorem zero_eq : (0 : Nat) = 0 := rfl", "def zero := 0");
        assert_eq!(
            kinds_of(&spec.check(&with_def)),
            [ViolationKind::DisallowedAddition]
        );
        let with_import = format!("import Mathlib\n{candidate}");
        assert_eq!(
            kinds_of(&spec.check(&with_import)),
            [ViolationKind::DisallowedAddition]
        );

        // ... and the original declarations keep their order.
        let swapped = "\
def g (n : Nat) : Nat := f (f n)

def f (n : Nat) : Nat := n + 1

theorem g_eq (n : Nat) : g n = n + 2 := by
  simp [g, f]
";
        assert!(kinds_of(&spec.check(swapped)).contains(&ViolationKind::AlteredSpec));
    }

    #[test]
    fn keeps_scoping_prefixes_with_their_declaration() {
        // `open Foo in` decides how the statement after it resolves names, so a
        // helper must not be able to come between the two.
        let source = "\
def f := 1

open Nat in
set_option maxRecDepth 2000 in
theorem t : f = 1 := by sorry
";
        let spec = Spec::parse(source).unwrap();
        assert_eq!(spec.targets()[0].name.as_deref(), Some("t"));

        let before_prefix = "\
def f := 1

theorem aux : True := trivial

open Nat in
set_option maxRecDepth 2000 in
theorem t : f = 1 := by rfl
";
        assert!(spec.check(before_prefix).passed());

        // Here the `open Nat in` would apply to `aux` instead of `t`.
        let stolen = "\
def f := 1

open Nat in
theorem aux : True := trivial

set_option maxRecDepth 2000 in
theorem t : f = 1 := by rfl
";
        let report = spec.check(stolen);
        assert!(
            kinds_of(&report).contains(&ViolationKind::AlteredSpec),
            "{:?}",
            report.violations
        );

        // An `in` that cannot be traced to a known prefix command leaves no
        // insertion point in front of the declaration at all.
        let odd =
            Spec::parse("def f := 1\nfrobnicate x in\ntheorem t : f = 1 := by sorry\n").unwrap();
        assert!(
            odd.check("def f := 1\nfrobnicate x in\ntheorem t : f = 1 := by rfl\n")
                .passed()
        );
        let squeezed = "def f := 1\nfrobnicate x in\ntheorem aux : True := trivial\ntheorem t : f = 1 := by rfl\n";
        assert!(!odd.check(squeezed).passed());
        let ahead = "def f := 1\ntheorem aux : True := trivial\nfrobnicate x in\ntheorem t : f = 1 := by rfl\n";
        assert!(!odd.check(ahead).passed());
    }

    #[test]
    fn keywords_inside_brackets_do_not_split_the_spec() {
        let source = "\
def zero : Nat := 0

@[reducible, instance] def inst : Inhabited Nat := ⟨zero⟩

theorem t : zero = 0 := by sorry
";
        let spec = Spec::parse(source).unwrap();
        let candidate = "\
def zero : Nat := 0

theorem aux : zero = 0 := rfl

@[reducible, instance] def inst : Inhabited Nat := ⟨zero⟩

theorem t : zero = 0 := by
  exact aux
";
        assert!(spec.check(candidate).passed());
        // The attribute block stays attached to its definition.
        let detached = candidate.replace(
            "theorem aux : zero = 0 := rfl\n\n@[reducible, instance] def",
            "@[reducible, instance] theorem aux : zero = 0 := rfl\n\ndef",
        );
        assert!(!spec.check(&detached).passed());
    }

    #[test]
    fn rejects_commands_that_change_meaning() {
        let cheats = [
            // Redefine `≤` so the frozen statement becomes trivial.
            "instance : LE Nat := ⟨fun _ _ => True⟩",
            // Smuggle `False` in as a hypothesis of every later theorem.
            "variable (cheat : False)\ninclude cheat",
            "local notation a \" = \" b => True",
            "macro_rules | `($a = $b) => `(True)",
            "open Classical in",
            "namespace Hack",
            "def myAppend' := 1",
            "abbrev α := Empty",
            "attribute [simp] myAppend",
            "set_option debug.skipKernelTC true in",
            "import Lean",
            "universe u",
        ];
        for cheat in cheats {
            let candidate = PROOF.replace(
                "theorem myAppend_nil",
                &format!("{cheat}\ntheorem myAppend_nil"),
            );
            let report = spec().check(&candidate);
            assert!(
                kinds_of(&report).contains(&ViolationKind::DisallowedAddition),
                "{cheat:?} should be rejected, got {:?}",
                report.violations
            );
        }
    }

    #[test]
    fn rejects_hash_commands_but_not_array_literals() {
        for cheat in [
            "#exit",
            "#eval IO.println \"hi\"",
            "/-- error: x -/\n#guard_msgs in",
        ] {
            let candidate = PROOF.replace(
                "theorem myAppend_nil",
                &format!("{cheat}\ntheorem myAppend_nil"),
            );
            assert_eq!(
                kinds_of(&spec().check(&candidate)),
                [ViolationKind::DisallowedAddition],
                "{cheat:?}"
            );
        }
        let fine = PROOF.replace(
            "| nil => rfl",
            "| nil => have _ := #[1, 2].size; have _ := #v[1, 2]; rfl",
        );
        assert!(spec().check(&fine).passed());
    }

    #[test]
    fn allows_harmless_options_only() {
        let fine = PROOF.replace(
            "theorem myAppend_nil",
            "set_option maxRecDepth 4000 in\ntheorem myAppend_nil",
        );
        assert!(spec().check(&fine).passed());
        let lint = PROOF.replace(
            "theorem myAppend_nil",
            "set_option linter.unusedVariables false in\ntheorem myAppend_nil",
        );
        assert!(spec().check(&lint).passed());
        let bad = PROOF.replace(
            "theorem myAppend_nil",
            "set_option autoImplicit true in\ntheorem myAppend_nil",
        );
        assert_eq!(
            kinds_of(&spec().check(&bad)),
            [ViolationKind::DisallowedAddition]
        );
    }

    #[test]
    fn rejects_interpolated_strings_in_added_text() {
        let candidate = PROOF.replace("| nil => rfl", "| nil => have _ := s!\"{1}\"; rfl");
        assert_eq!(
            kinds_of(&spec().check(&candidate)),
            [ViolationKind::DisallowedAddition]
        );
    }

    #[test]
    fn rejects_helper_names_that_shadow_the_spec() {
        // A helper named like something the statement mentions could capture it.
        let candidate = PROOF.replace(
            "theorem myAppend_nil",
            "theorem Hack.myAppend : True := trivial\n\ntheorem myAppend_nil",
        );
        let report = spec().check(&candidate);
        assert_eq!(kinds_of(&report), [ViolationKind::NameCollision]);
        assert!(report.violations[0].message.contains("fresh name"));
    }

    #[test]
    fn reports_forbidden_tokens_alongside_structure() {
        let candidate = PROOF.replace("| nil => rfl", "| nil => sorry");
        let report = spec().check(&candidate);
        assert_eq!(kinds_of(&report), [ViolationKind::ForbiddenToken]);
        assert_eq!(report.violations[0].line, Some(7));
        // The structure itself is intact, so spans are still available.
        assert_eq!(report.spans.len(), 1);
    }

    #[test]
    fn handles_several_targets_and_holes() {
        let source = "\
def f (n : Nat) : Nat := n + 1

theorem first (n : Nat) : f n > n := by sorry

def g (n : Nat) : Nat := f (f n)

theorem second (n : Nat) : g n > n ∧ g n = n + 2 := by
  constructor
  · sorry
  · sorry
";
        let spec = Spec::parse(source).unwrap();
        let targets = spec.targets();
        assert_eq!(targets.len(), 2);
        assert_eq!(
            (targets[0].name.as_deref(), targets[0].holes),
            (Some("first"), 1)
        );
        assert_eq!(
            (targets[1].name.as_deref(), targets[1].holes),
            (Some("second"), 2)
        );
        assert_eq!(spec.hole_count(), 3);

        let candidate = "\
def f (n : Nat) : Nat := n + 1

theorem first (n : Nat) : f n > n := by unfold f; omega

def g (n : Nat) : Nat := f (f n)

theorem g_unfold (n : Nat) : g n = n + 2 := by unfold g f; omega

theorem second (n : Nat) : g n > n ∧ g n = n + 2 := by
  constructor
  · rw [g_unfold]; omega
  · exact g_unfold n
";
        let report = spec.check(candidate);
        assert!(report.passed(), "{:?}", report.violations);
        assert_eq!(report.target_at(3, 40), Some(0));
        assert_eq!(report.target_at(7, 0), Some(1));
        assert_eq!(report.target_at(12, 4), Some(1));
        // Frozen text between two targets counts towards the later one.
        assert_eq!(report.target_at(5, 0), Some(1));

        // Dropping the second definition is noticed even though both theorems are present.
        let without_g = candidate.replace("def g (n : Nat) : Nat := f (f n)\n", "");
        assert_eq!(
            kinds_of(&spec.check(&without_g)),
            [ViolationKind::AlteredSpec]
        );
    }

    #[test]
    fn attributes_positions_between_adjacent_theorems() {
        // No frozen text separates the two theorems, so the gap between their
        // statements holds the first proof *and* the second one's helpers.
        let spec = Spec::parse(
            "theorem first (n : Nat) : n + 0 = n := by sorry\n\ntheorem second (n : Nat) : 0 + n = n := by sorry\n",
        )
        .unwrap();
        let candidate = "\
theorem first (n : Nat) : n + 0 = n := by
  simp
  omega

@[simp] private theorem aux (n : Nat) : 0 + n = n := by
  omega

theorem second (n : Nat) : 0 + n = n := by
  exact aux n
";
        let report = spec.check(candidate);
        assert!(report.passed(), "{:?}", report.violations);
        assert_eq!(report.spans.len(), 2);
        // Statement and proof of the first theorem.
        assert_eq!(report.target_at(1, 0), Some(0));
        assert_eq!(report.target_at(3, 2), Some(0));
        // The helper, attribute included, already belongs to the second one.
        assert_eq!(report.target_at(5, 0), Some(1));
        assert_eq!(report.target_at(6, 2), Some(1));
        assert_eq!(report.target_at(9, 2), Some(1));

        // Without helpers the first proof simply ends at the next statement.
        let plain = "theorem first (n : Nat) : n + 0 = n := by\n  simp\n\ntheorem second (n : Nat) : 0 + n = n := by\n  omega\n";
        let report = spec.check(plain);
        assert_eq!(report.target_at(2, 2), Some(0));
        assert_eq!(report.target_at(4, 0), Some(1));
        assert_eq!(report.target_at(5, 2), Some(1));
    }

    #[test]
    fn lemma_is_a_keyword_only_with_imports() {
        // In core Lean `lemma` is an ordinary identifier.
        let core = Spec::parse("theorem t (p : Prop) (h : p) : p := by sorry\n").unwrap();
        assert!(
            core.check(
                "theorem t (p : Prop) (h : p) : p := by\n  have lemma : p := h\n  exact lemma\n"
            )
            .passed()
        );

        // With Mathlib it declares a theorem, and helper names must be fresh.
        let mathlib =
            Spec::parse("import Mathlib\ntheorem t (p : Prop) (h : p) : p := by sorry\n").unwrap();
        let shadowing = "import Mathlib\nlemma h : True := trivial\ntheorem t (p : Prop) (h : p) : p := by\n  exact h\n";
        assert_eq!(
            kinds_of(&mathlib.check(shadowing)),
            [ViolationKind::NameCollision]
        );
        let fresh = shadowing.replace("lemma h ", "lemma aux ");
        assert!(mathlib.check(&fresh).passed());
    }

    #[test]
    fn keeps_attributes_and_modifiers_with_their_declaration() {
        let source = "def f := 1\n\n@[simp] private theorem t : f = 1 := by sorry\n";
        let spec = Spec::parse(source).unwrap();
        // The helper goes in front of the attribute, not between it and `theorem`.
        let good = "def f := 1\n\ntheorem aux : True := trivial\n\n@[simp] private theorem t : f = 1 := by rfl\n";
        assert!(spec.check(good).passed());
        let bad = "def f := 1\n\n@[simp] theorem aux : True := trivial\n\nprivate theorem t : f = 1 := by rfl\n";
        assert!(!spec.check(bad).passed());
    }

    #[test]
    fn rejects_specs_with_other_forbidden_tokens() {
        let error = Spec::parse("axiom em : True\ntheorem t : True := by sorry").unwrap_err();
        let SpecError::ForbiddenTokens(found) = &error;
        assert_eq!(found.len(), 1);
        assert!(error.to_string().contains("axiom"));
    }

    #[test]
    fn a_spec_without_holes_must_be_reproduced_exactly() {
        let spec = Spec::parse("theorem t : True := trivial\n").unwrap();
        assert!(spec.targets().is_empty());
        assert!(spec.check("theorem t : True := trivial -- same\n").passed());
        assert!(!spec.check("theorem t : True := by trivial\n").passed());
    }

    // -- axiom audit ---------------------------------------------------------

    #[test]
    fn audit_uses_fully_qualified_names() {
        let source = "\
namespace Outer.Inner
section
theorem a : True := by sorry
end
end Outer.Inner
namespace Other
private theorem b : True := by sorry
theorem _root_.c : True := by sorry
example : True := by sorry
";
        let audit = Spec::parse(source).unwrap().axiom_audit().unwrap();
        assert_eq!(audit.names(), ["Outer.Inner.a", "Other.b", "c"]);
        assert_eq!(
            audit.commands(),
            "\n\n#print axioms _root_.Outer.Inner.a\n#print axioms _root_.Other.b\n#print axioms _root_.c\n"
        );
    }

    #[test]
    fn audit_accepts_standard_axioms() {
        let audit = Spec::parse("theorem a : True := by sorry\ntheorem b : True := by sorry")
            .unwrap()
            .axiom_audit()
            .unwrap();
        let stdout = "\
'earlier' depends on axioms: [sorryAx]
'a' depends on axioms: [propext, Classical.choice, Quot.sound]
'b' does not depend on any axioms
";
        assert_eq!(
            audit.evaluate(stdout),
            AuditOutcome::Clean(vec![
                "propext".into(),
                "Classical.choice".into(),
                "Quot.sound".into()
            ])
        );
    }

    #[test]
    fn audit_flags_foreign_axioms() {
        let audit =
            Spec::parse("theorem a : True := by sorry\nprivate theorem b : True := by sorry")
                .unwrap()
                .axiom_audit()
                .unwrap();
        let stdout = "\
'a' depends on axioms: [propext,
 a._native.native_decide.ax_1_1]
'_private._stdin.0.b' depends on axioms: [sorryAx]
";
        assert_eq!(
            audit.evaluate(stdout),
            AuditOutcome::Tainted(vec![
                TaintedTheorem {
                    name: "a".into(),
                    axioms: vec!["a._native.native_decide.ax_1_1".into()]
                },
                TaintedTheorem {
                    name: "b".into(),
                    axioms: vec!["sorryAx".into()]
                },
            ])
        );
    }

    #[test]
    fn audit_is_inconclusive_when_a_target_is_missing() {
        let audit = Spec::parse("theorem a : True := by sorry\ntheorem b : True := by sorry")
            .unwrap()
            .axiom_audit()
            .unwrap();
        let missing =
            "'a' does not depend on any axioms\nfile.lean:9:14: error: Unknown constant `b`\n";
        assert!(matches!(
            audit.evaluate(missing),
            AuditOutcome::Inconclusive(_)
        ));
        let wrong = "'a' does not depend on any axioms\n'zzz' does not depend on any axioms\n";
        assert!(matches!(
            audit.evaluate(wrong),
            AuditOutcome::Inconclusive(_)
        ));
    }

    #[test]
    fn nothing_to_audit_without_named_targets() {
        assert!(
            Spec::parse("example : True := by sorry")
                .unwrap()
                .axiom_audit()
                .is_none()
        );
    }
}
