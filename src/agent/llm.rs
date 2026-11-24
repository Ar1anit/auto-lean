//! Language-model backends.
//!
//! Everything above this module talks to a model through [`LlmProvider`], a single
//! "system prompt + conversation in, text out" call. Four implementations exist:
//!
//! * [`AnthropicProvider`] for the Claude Messages API (`ANTHROPIC_API_KEY`).
//! * [`OpenAiProvider`] for the OpenAI Chat Completions API and anything that
//!   speaks it, such as Ollama or vLLM (`OPENAI_API_KEY`, `OPENAI_BASE_URL`).
//! * [`FoundryProvider`] for models deployed in Microsoft Foundry, which serves
//!   Claude through the Messages API and the rest of its catalog through an
//!   OpenAI-compatible one (`FOUNDRY_RESOURCE_ENDPOINT`, `FOUNDRY_API_KEY`).
//! * [`MockProvider`], which replays scripted answers for deterministic tests
//!   and offline demos.

use std::collections::VecDeque;
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result};
use async_trait::async_trait;
use reqwest::header::{HeaderMap, RETRY_AFTER};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use thiserror::Error;

/// Who wrote a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// One turn of a conversation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    pub content: String,
}

impl Message {
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
        }
    }
}

/// A chat-completion backend.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Sends `system` and the conversation so far, returning the model's reply.
    async fn complete(&self, system: &str, messages: &[Message]) -> Result<String>;
}

/// Failures of a model request.
#[derive(Debug, Error)]
pub enum LlmError {
    #[error("{var} is not set; export it to use the {provider} provider")]
    MissingApiKey {
        provider: &'static str,
        var: &'static str,
    },

    #[error("{provider} API error{}{}: {message}{}",
        optional(" ", .status, ""),
        optional(" (", .kind, ")"),
        optional(" [request ", .request_id, "]"))]
    Api {
        provider: &'static str,
        /// HTTP status, or `None` for an error delivered inside a response stream.
        status: Option<u16>,
        kind: Option<String>,
        message: String,
        request_id: Option<String>,
    },

    #[error("could not reach the {provider} API at {url}: {source}")]
    Transport {
        provider: &'static str,
        url: String,
        #[source]
        source: reqwest::Error,
    },

    #[error("the model declined the request{}", optional(" (category: ", .category, ")"))]
    Refused { category: Option<String> },

    #[error("the reply was cut off at the output-token limit before the model finished")]
    Truncated,

    #[error("the {provider} API returned a reply without any text")]
    Empty { provider: &'static str },

    #[error("unexpected response from the {provider} API: {detail}")]
    Malformed {
        provider: &'static str,
        detail: String,
    },

    /// A provider cannot be set up from the given settings.
    #[error("{0}")]
    Configuration(String),
}

/// Renders `prefix value suffix`, or nothing when the value is absent.
fn optional<T: std::fmt::Display>(prefix: &str, value: &Option<T>, suffix: &str) -> String {
    value
        .as_ref()
        .map(|value| format!("{prefix}{value}{suffix}"))
        .unwrap_or_default()
}

impl LlmError {
    /// Whether sending the same request again can reasonably succeed.
    fn is_transient(&self) -> bool {
        match self {
            LlmError::Api { status, kind, .. } => {
                matches!(status, Some(408 | 409 | 429) | Some(500..))
                    || matches!(kind.as_deref(), Some("overloaded_error" | "api_error"))
            }
            LlmError::Transport { .. } => true,
            _ => false,
        }
    }
}

/// How many times a transient failure is retried.
const MAX_RETRIES: u32 = 2;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);

fn backoff(attempt: u32, retry_after: Option<Duration>) -> Duration {
    retry_after
        .unwrap_or_else(|| Duration::from_millis(500 * 2u64.pow(attempt.min(6))))
        .min(Duration::from_secs(60))
}

fn retry_after(headers: &HeaderMap) -> Option<Duration> {
    let seconds: f64 = headers
        .get(RETRY_AFTER)?
        .to_str()
        .ok()?
        .trim()
        .parse()
        .ok()?;
    (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds))
}

/// Servers may veto or request a retry explicitly.
fn should_retry_header(headers: &HeaderMap) -> Option<bool> {
    match headers.get("x-should-retry")?.to_str().ok()? {
        "true" => Some(true),
        "false" => Some(false),
        _ => None,
    }
}

fn trim_base_url(url: &str) -> String {
    url.trim().trim_end_matches('/').to_string()
}

fn env_var(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

/// How a request proves who is asking. Deliberately not `Debug`: it holds secrets.
#[derive(Clone)]
enum Credential {
    /// No authentication, for local servers.
    None,
    /// `Authorization: Bearer <token>`.
    Bearer(String),
    /// A key in a dedicated header such as `x-api-key` or `api-key`.
    Header(&'static str, String),
}

impl Credential {
    fn apply(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match self {
            Credential::None => request,
            Credential::Bearer(token) => request.bearer_auth(token),
            Credential::Header(name, key) => request.header(*name, key),
        }
    }
}

// ---------------------------------------------------------------------------
// Anthropic
// ---------------------------------------------------------------------------

/// Default Claude model.
pub const DEFAULT_ANTHROPIC_MODEL: &str = "claude-opus-5-5";

const ANTHROPIC: &str = "Anthropic";
const ANTHROPIC_BASE_URL: &str = "https://api.anthropic.com";
const ANTHROPIC_VERSION: &str = "2026-10-21";
/// Beta that lets the API re-run a request another model's safety classifier
/// declined, instead of returning the refusal.
const FALLBACK_BETA: &str = "server-side-fallback-2026-07-01";
/// Proof search is reasoning-heavy, so ask for more than the default depth.
const ANTHROPIC_EFFORT: &str = "high";
/// Thinking counts towards the limit; streaming makes a generous one safe.
const ANTHROPIC_MAX_TOKENS: u32 = 64_000;
const LEGACY_MAX_TOKENS: u32 = 8_192;
/// Longest silence tolerated on an open response stream.
const STREAM_IDLE_TIMEOUT: Duration = Duration::from_secs(600);

/// Which optional request fields a Claude model accepts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ModelProfile {
    /// `thinking: {type: "adaptive"}` and `output_config.effort`.
    adaptive_thinking: bool,
    /// The server-side `fallbacks` parameter.
    fallbacks: bool,
}

/// Parameters differ between model generations and an unsupported one is a hard
/// 400, so they are only sent to model families known to accept them. Unknown
/// models get a bare request and run with their own defaults.
///
/// The family is looked for anywhere in the name, because on platforms with
/// named deployments the model ID is often only part of it (`prod-claude-opus-5-5`).
fn model_profile(model: &str) -> ModelProfile {
    const ADAPTIVE: &[&str] = &[
        "claude-fable-5",
        "claude-mythos-5",
        "claude-opus-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-5",
        "claude-sonnet-4-6",
        "claude-haiku-5",
    ];
    const FALLBACKS: &[&str] = &[
        "claude-fable-5-2",
        "claude-fable-5-1",
        "claude-opus-5-5",
        "claude-opus-5",
        "claude-sonnet-5-5",
    ];
    ModelProfile {
        adaptive_thinking: ADAPTIVE.iter().any(|family| model.contains(family)),
        fallbacks: FALLBACKS.contains(&model),
    }
}

/// Claude via the Anthropic Messages API.
pub struct AnthropicProvider {
    http: reqwest::Client,
    credential: Credential,
    model: String,
    base_url: String,
    /// Who is on the other end, for error messages.
    platform: &'static str,
    /// `max_tokens` for models whose family is not recognised.
    unknown_model_max_tokens: u32,
    /// Cleared if the API rejects the fallback beta, e.g. for an organisation
    /// that does not have it enabled.
    fallbacks: AtomicBool,
}

// Written by hand so the API key can never end up in logs or panic messages.
impl std::fmt::Debug for AnthropicProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .read_timeout(STREAM_IDLE_TIMEOUT)
            .build()
            .context("failed to initialise the HTTP client")?;
        let model = model.into();
        Ok(Self {
            fallbacks: AtomicBool::new(model_profile(&model).fallbacks),
            http,
            credential: Credential::Header("x-api-key", api_key.into()),
            model,
            base_url: ANTHROPIC_BASE_URL.to_string(),
            platform: ANTHROPIC,
            unknown_model_max_tokens: LEGACY_MAX_TOKENS,
        })
    }

    /// A Claude deployment in Microsoft Foundry: the same Messages API under the
    /// resource's `/anthropic` path, with Foundry's own credentials. `model` is
    /// the deployment name. Server-side fallbacks do not exist there.
    fn for_foundry(endpoint: &str, credential: Credential, deployment: String) -> Result<Self> {
        let mut provider =
            Self::new(String::new(), deployment)?.with_base_url(format!("{endpoint}/anthropic"));
        provider.credential = credential;
        provider.platform = FOUNDRY;
        // Every Claude model Foundry offers can produce at least this much, so
        // a deployment with an unrecognisable name still gets room to think.
        provider.unknown_model_max_tokens = ANTHROPIC_MAX_TOKENS;
        Ok(provider)
    }

    /// Reads `ANTHROPIC_API_KEY` and, optionally, `ANTHROPIC_BASE_URL`.
    pub fn from_env(model: Option<String>) -> Result<Self> {
        let api_key = env_var("ANTHROPIC_API_KEY").ok_or(LlmError::MissingApiKey {
            provider: ANTHROPIC,
            var: "ANTHROPIC_API_KEY",
        })?;
        let provider = Self::new(
            api_key,
            model.unwrap_or_else(|| DEFAULT_ANTHROPIC_MODEL.to_string()),
        )?;
        Ok(match env_var("ANTHROPIC_BASE_URL") {
            Some(url) => provider.with_base_url(url),
            None => provider,
        })
    }

    /// Points the provider at a proxy or gateway. Server-side fallbacks are a
    /// first-party feature, so they are only requested from the official API.
    pub fn with_base_url(mut self, url: impl AsRef<str>) -> Self {
        self.base_url = trim_base_url(url.as_ref());
        if self.base_url != ANTHROPIC_BASE_URL {
            self.fallbacks.store(false, Ordering::Relaxed);
        }
        self
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    fn request_body(&self, system: &str, messages: &[Message], fallbacks: bool) -> Value {
        let profile = model_profile(&self.model);
        let mut body = json!({
            "model": self.model,
            "max_tokens": if profile.adaptive_thinking { ANTHROPIC_MAX_TOKENS } else { self.unknown_model_max_tokens },
            "stream": true,
            "messages": messages,
        });
        if !system.is_empty() {
            body["system"] = json!(system);
        }
        if profile.adaptive_thinking {
            body["thinking"] = json!({ "type": "adaptive" });
            body["output_config"] = json!({ "effort": ANTHROPIC_EFFORT });
        }
        if fallbacks {
            body["fallbacks"] = json!("default");
        }
        body
    }

    async fn send(
        &self,
        system: &str,
        messages: &[Message],
    ) -> Result<String, (LlmError, Option<Duration>, Option<bool>)> {
        let url = format!("{}/v1/messages", self.base_url);
        let fallbacks = self.fallbacks.load(Ordering::Relaxed);
        let mut request = self
            .credential
            .apply(self.http.post(&url))
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&self.request_body(system, messages, fallbacks));
        if fallbacks {
            request = request.header("anthropic-beta", FALLBACK_BETA);
        }

        let transport = |source| {
            let error = LlmError::Transport {
                provider: self.platform,
                url: url.clone(),
                source,
            };
            (error, None, None)
        };
        let mut response = request.send().await.map_err(transport)?;

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let headers = response.headers().clone();
            let body = response.text().await.unwrap_or_default();
            let error = api_error(self.platform, status, &headers, &body);
            let spend_cap = body.contains("enforced_spend_limit_reached");
            let retry = if spend_cap {
                Some(false)
            } else {
                should_retry_header(&headers)
            };
            return Err((error, retry_after(&headers), retry));
        }

        let mut decoder = SseDecoder::default();
        let mut stream = AnthropicStream::default();
        while let Some(chunk) = response.chunk().await.map_err(transport)? {
            for event in decoder.push(&chunk) {
                stream.apply(&event);
            }
        }
        if let Some(event) = decoder.finish() {
            stream.apply(&event);
        }
        stream
            .finish(self.platform)
            .map_err(|error| (error, None, None))
    }
}

#[async_trait]
impl LlmProvider for AnthropicProvider {
    async fn complete(&self, system: &str, messages: &[Message]) -> Result<String> {
        let mut attempt = 0;
        loop {
            let (error, wait, retry) = match self.send(system, messages).await {
                Ok(text) => return Ok(text),
                Err(failure) => failure,
            };

            // An account without the fallback beta gets a 400 naming the header
            // or the parameter. Drop the opt-in for this session and go on.
            if let LlmError::Api {
                status: Some(400),
                message,
                ..
            } = &error
            {
                let about_fallbacks =
                    message.contains("anthropic-beta") || message.contains("fallback");
                if about_fallbacks && self.fallbacks.swap(false, Ordering::Relaxed) {
                    continue;
                }
            }

            if attempt >= MAX_RETRIES || !retry.unwrap_or_else(|| error.is_transient()) {
                return Err(error.into());
            }
            tokio::time::sleep(backoff(attempt, wait)).await;
            attempt += 1;
        }
    }
}

/// Builds an [`LlmError::Api`] from an error response. All the APIs use the body
/// shape `{"error": {"type" or "code": ..., "message": ...}}`; the request ID is
/// in the body (Anthropic) or in a header (`request-id`, or Azure's
/// `apim-request-id`).
fn api_error(provider: &'static str, status: u16, headers: &HeaderMap, body: &str) -> LlmError {
    let parsed: Option<Value> = serde_json::from_str(body).ok();
    let field = |path: &[&str]| -> Option<String> {
        let mut value = parsed.as_ref()?;
        for key in path {
            value = value.get(key)?;
        }
        value.as_str().map(str::to_string)
    };
    let message = field(&["error", "message"])
        .or_else(|| field(&["error"]))
        .or_else(|| field(&["message"]))
        .unwrap_or_else(|| {
            let text = body.trim();
            if text.is_empty() {
                "no error details returned".to_string()
            } else {
                text.chars().take(500).collect()
            }
        });
    LlmError::Api {
        provider,
        status: Some(status),
        kind: field(&["error", "type"]).or_else(|| field(&["error", "code"])),
        message,
        request_id: field(&["request_id"]).or_else(|| {
            ["request-id", "apim-request-id", "x-request-id"]
                .iter()
                .find_map(|name| headers.get(*name)?.to_str().ok().map(str::to_string))
        }),
    }
}

/// Incremental decoder for a `text/event-stream` body. Chunks may split lines
/// and even UTF-8 sequences, so bytes are buffered until a full line arrives.
#[derive(Debug, Default)]
struct SseDecoder {
    pending: Vec<u8>,
    data: Vec<String>,
}

impl SseDecoder {
    /// Feeds bytes, returning the `data` payload of every event completed by them.
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.pending.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(newline) = self.pending.iter().position(|&byte| byte == b'\n') {
            let line: Vec<u8> = self.pending.drain(..=newline).collect();
            let line = String::from_utf8_lossy(&line);
            let line = line.trim_end_matches(['\n', '\r']);
            if line.is_empty() {
                events.extend(self.take_event());
            } else if let Some(value) = line.strip_prefix("data:") {
                self.data
                    .push(value.strip_prefix(' ').unwrap_or(value).to_string());
            }
        }
        events
    }

    /// Flushes an event that was not terminated by a blank line.
    fn finish(&mut self) -> Option<String> {
        if !self.pending.is_empty() {
            let line = String::from_utf8_lossy(&std::mem::take(&mut self.pending)).into_owned();
            if let Some(value) = line.trim_end_matches(['\n', '\r']).strip_prefix("data:") {
                self.data
                    .push(value.strip_prefix(' ').unwrap_or(value).to_string());
            }
        }
        self.take_event()
    }

    fn take_event(&mut self) -> Option<String> {
        (!self.data.is_empty()).then(|| std::mem::take(&mut self.data).join("\n"))
    }
}

/// Accumulates a streamed Messages API response.
#[derive(Debug, Default)]
struct AnthropicStream {
    text: String,
    stop_reason: Option<String>,
    refusal_category: Option<String>,
    error: Option<(Option<String>, String)>,
}

impl AnthropicStream {
    fn apply(&mut self, data: &str) {
        // Unknown or unparseable events are skipped: new event types may be
        // added to the stream at any time.
        let Ok(event) = serde_json::from_str::<Value>(data) else {
            return;
        };
        match event["type"].as_str() {
            Some("content_block_start") => {
                let block = &event["content_block"];
                if block["type"] == "text" {
                    self.text
                        .push_str(block["text"].as_str().unwrap_or_default());
                }
            }
            Some("content_block_delta") => {
                let delta = &event["delta"];
                if delta["type"] == "text_delta" {
                    self.text
                        .push_str(delta["text"].as_str().unwrap_or_default());
                }
            }
            Some("message_delta") => {
                let delta = &event["delta"];
                if let Some(reason) = delta["stop_reason"].as_str() {
                    self.stop_reason = Some(reason.to_string());
                }
                // Informational only; it may be absent even on a refusal.
                for details in [&delta["stop_details"], &event["stop_details"]] {
                    if let Some(category) = details["category"].as_str() {
                        self.refusal_category = Some(category.to_string());
                    }
                }
            }
            Some("error") => {
                let error = &event["error"];
                self.error = Some((
                    error["type"].as_str().map(str::to_string),
                    error["message"]
                        .as_str()
                        .unwrap_or("stream error")
                        .to_string(),
                ));
            }
            _ => {}
        }
    }

    fn finish(self, provider: &'static str) -> Result<String, LlmError> {
        if let Some((kind, message)) = self.error {
            return Err(LlmError::Api {
                provider,
                status: None,
                kind,
                message,
                request_id: None,
            });
        }
        match self.stop_reason.as_deref() {
            // A refusal can arrive after partial output; that output is not an answer.
            Some("refusal") => Err(LlmError::Refused {
                category: self.refusal_category,
            }),
            Some("max_tokens") => Err(LlmError::Truncated),
            None => Err(LlmError::Malformed {
                provider,
                detail: "the response stream ended before the message was complete".to_string(),
            }),
            Some(_) if self.text.trim().is_empty() => Err(LlmError::Empty { provider }),
            Some(_) => Ok(self.text),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAI-compatible
// ---------------------------------------------------------------------------

/// Default OpenAI model.
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-6.2-astra";

const OPENAI: &str = "OpenAI";
const OPENAI_BASE_URL: &str = "https://api.openai.com/v1";
const OPENAI_TIMEOUT: Duration = Duration::from_secs(900);

/// Any server implementing the OpenAI Chat Completions API: OpenAI itself, or a
/// local Ollama / vLLM / llama.cpp endpoint selected with `OPENAI_BASE_URL`.
pub struct OpenAiProvider {
    http: reqwest::Client,
    credential: Credential,
    model: String,
    base_url: String,
    /// Who is on the other end, for error messages.
    platform: &'static str,
}

impl std::fmt::Debug for OpenAiProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiProvider")
            .field("model", &self.model)
            .field("base_url", &self.base_url)
            .finish_non_exhaustive()
    }
}

impl OpenAiProvider {
    /// `api_key` may be `None` for local servers that do not authenticate.
    pub fn new(
        api_key: Option<String>,
        model: impl Into<String>,
        base_url: impl AsRef<str>,
    ) -> Result<Self> {
        let http = reqwest::Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .timeout(OPENAI_TIMEOUT)
            .build()
            .context("failed to initialise the HTTP client")?;
        Ok(Self {
            http,
            credential: api_key.map_or(Credential::None, Credential::Bearer),
            model: model.into(),
            base_url: trim_base_url(base_url.as_ref()),
            platform: OPENAI,
        })
    }

    /// A deployment in Microsoft Foundry, called through the resource's
    /// OpenAI-compatible `/openai/v1` route. `model` is the deployment name.
    fn for_foundry(endpoint: &str, credential: Credential, deployment: String) -> Result<Self> {
        let mut provider = Self::new(None, deployment, format!("{endpoint}/openai/v1"))?;
        provider.credential = credential;
        provider.platform = FOUNDRY;
        Ok(provider)
    }

    /// Reads `OPENAI_API_KEY` and `OPENAI_BASE_URL`. The key is only required
    /// for the official endpoint.
    pub fn from_env(model: Option<String>) -> Result<Self> {
        let base_url = env_var("OPENAI_BASE_URL").unwrap_or_else(|| OPENAI_BASE_URL.to_string());
        let api_key = env_var("OPENAI_API_KEY");
        if api_key.is_none() && trim_base_url(&base_url) == OPENAI_BASE_URL {
            return Err(LlmError::MissingApiKey {
                provider: OPENAI,
                var: "OPENAI_API_KEY",
            }
            .into());
        }
        Self::new(
            api_key,
            model.unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string()),
            base_url,
        )
    }

    pub fn model(&self) -> &str {
        &self.model
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Only `model` and `messages` are sent: sampling and token-limit parameters
    /// are spelled differently, or rejected outright, across the servers and
    /// model generations this provider has to work with.
    fn request_body(&self, system: &str, messages: &[Message]) -> Value {
        let mut conversation = Vec::with_capacity(messages.len() + 1);
        if !system.is_empty() {
            conversation.push(json!({ "role": "system", "content": system }));
        }
        conversation.extend(messages.iter().map(|message| json!(message)));
        json!({ "model": self.model, "messages": conversation })
    }

    async fn send(
        &self,
        system: &str,
        messages: &[Message],
    ) -> Result<String, (LlmError, Option<Duration>)> {
        let url = format!("{}/chat/completions", self.base_url);
        let request = self
            .credential
            .apply(self.http.post(&url))
            .json(&self.request_body(system, messages));

        let transport = |source| {
            let error = LlmError::Transport {
                provider: self.platform,
                url: url.clone(),
                source,
            };
            (error, None)
        };
        let response = request.send().await.map_err(transport)?;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.text().await.map_err(transport)?;
        if !status.is_success() {
            let error = api_error(self.platform, status.as_u16(), &headers, &body);
            return Err((error, retry_after(&headers)));
        }
        parse_chat_completion(self.platform, &body).map_err(|error| (error, None))
    }
}

#[async_trait]
impl LlmProvider for OpenAiProvider {
    async fn complete(&self, system: &str, messages: &[Message]) -> Result<String> {
        let mut attempt = 0;
        loop {
            let (error, wait) = match self.send(system, messages).await {
                Ok(text) => return Ok(text),
                Err(failure) => failure,
            };
            if attempt >= MAX_RETRIES || !error.is_transient() {
                return Err(error.into());
            }
            tokio::time::sleep(backoff(attempt, wait)).await;
            attempt += 1;
        }
    }
}

fn parse_chat_completion(provider: &'static str, body: &str) -> Result<String, LlmError> {
    let malformed = |detail: String| LlmError::Malformed { provider, detail };
    let response: Value =
        serde_json::from_str(body).map_err(|error| malformed(error.to_string()))?;
    let choice = response["choices"]
        .get(0)
        .ok_or_else(|| malformed("the response contains no choices".to_string()))?;
    let message = &choice["message"];

    if message["refusal"]
        .as_str()
        .is_some_and(|refusal| !refusal.is_empty())
        || choice["finish_reason"] == "content_filter"
    {
        return Err(LlmError::Refused { category: None });
    }
    if choice["finish_reason"] == "length" {
        return Err(LlmError::Truncated);
    }

    // Normally a string; some servers send a list of typed parts instead.
    let text = match &message["content"] {
        Value::String(text) => text.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|part| part["text"].as_str())
            .collect::<Vec<_>>()
            .concat(),
        _ => String::new(),
    };
    if text.trim().is_empty() {
        return Err(LlmError::Empty { provider });
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// Microsoft Foundry
// ---------------------------------------------------------------------------

const FOUNDRY: &str = "Microsoft Foundry";

/// Environment variable that forces which of Foundry's APIs a deployment is
/// called through, for deployments whose name does not give it away.
pub const FOUNDRY_API_ENV: &str = "AUTO_LEAN_FOUNDRY_API";

/// The two inference APIs a Foundry resource exposes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FoundryApi {
    /// The Anthropic Messages API at `/anthropic/v1/messages`. Claude models
    /// are served here and nowhere else.
    Anthropic,
    /// The OpenAI-compatible Chat Completions API at `/openai/v1/chat/completions`,
    /// for the rest of the catalog.
    OpenAi,
}

impl FoundryApi {
    /// Picks the API from a deployment name. A deployment is named after its
    /// model unless someone chose otherwise, so `claude` in the name is the
    /// signal; [`FOUNDRY_API_ENV`] overrides it.
    pub fn for_deployment(deployment: &str) -> Self {
        if deployment.to_ascii_lowercase().contains("claude") {
            FoundryApi::Anthropic
        } else {
            FoundryApi::OpenAi
        }
    }

    /// A short description for status output.
    pub fn label(self) -> &'static str {
        match self {
            FoundryApi::Anthropic => "Anthropic Messages API",
            FoundryApi::OpenAi => "OpenAI-compatible API",
        }
    }
}

impl std::str::FromStr for FoundryApi {
    type Err = LlmError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().as_str() {
            "anthropic" | "claude" | "messages" => Ok(FoundryApi::Anthropic),
            "openai" | "chat" | "chat-completions" => Ok(FoundryApi::OpenAi),
            other => Err(LlmError::Configuration(format!(
                "{FOUNDRY_API_ENV} must be `anthropic` or `openai`, not `{other}`"
            ))),
        }
    }
}

/// How to authenticate against a Foundry resource.
#[derive(Clone)]
pub enum FoundryCredential {
    /// The resource's API key, sent in the `api-key` header.
    ApiKey(String),
    /// A Microsoft Entra ID access token for the scope
    /// `https://ai.azure.com/.default`, sent as a bearer token. Required for
    /// the Claude Mythos models.
    EntraToken(String),
}

enum FoundryBackend {
    Anthropic(AnthropicProvider),
    OpenAi(OpenAiProvider),
}

/// Models deployed in a Microsoft Foundry resource.
///
/// Foundry serves its catalog through two wire protocols: Claude deployments
/// answer the Anthropic Messages API, everything else an OpenAI-compatible Chat
/// Completions API. This provider picks the right one for the deployment and
/// otherwise behaves exactly like [`AnthropicProvider`] or [`OpenAiProvider`].
pub struct FoundryProvider {
    api: FoundryApi,
    endpoint: String,
    deployment: String,
    backend: FoundryBackend,
}

impl std::fmt::Debug for FoundryProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FoundryProvider")
            .field("api", &self.api)
            .field("endpoint", &self.endpoint)
            .field("deployment", &self.deployment)
            .finish_non_exhaustive()
    }
}

impl FoundryProvider {
    /// `resource` is the resource endpoint
    /// (`https://<resource>.services.ai.azure.com`) or just the resource name.
    /// `api` is detected from the deployment name when `None`.
    pub fn new(
        resource: &str,
        credential: FoundryCredential,
        deployment: impl Into<String>,
        api: Option<FoundryApi>,
    ) -> Result<Self> {
        let endpoint = foundry_endpoint(resource)?;
        let deployment = deployment.into();
        let api = api.unwrap_or_else(|| FoundryApi::for_deployment(&deployment));
        let credential = match credential {
            FoundryCredential::ApiKey(key) => Credential::Header("api-key", key),
            FoundryCredential::EntraToken(token) => Credential::Bearer(token),
        };
        let backend = match api {
            FoundryApi::Anthropic => FoundryBackend::Anthropic(AnthropicProvider::for_foundry(
                &endpoint,
                credential,
                deployment.clone(),
            )?),
            FoundryApi::OpenAi => FoundryBackend::OpenAi(OpenAiProvider::for_foundry(
                &endpoint,
                credential,
                deployment.clone(),
            )?),
        };
        Ok(Self {
            api,
            endpoint,
            deployment,
            backend,
        })
    }

    /// Reads the resource from `FOUNDRY_RESOURCE_ENDPOINT` (or `FOUNDRY_RESOURCE`)
    /// and the credential from `FOUNDRY_AUTH_TOKEN` or `FOUNDRY_API_KEY`. The
    /// variables of Anthropic's Foundry SDKs (`ANTHROPIC_FOUNDRY_RESOURCE`,
    /// `ANTHROPIC_FOUNDRY_BASE_URL`, `ANTHROPIC_FOUNDRY_API_KEY`) are accepted
    /// as well. There is no default deployment: `deployment` is required.
    pub fn from_env(deployment: Option<String>) -> Result<Self> {
        Self::from_settings(deployment, env_var)
    }

    /// [`Self::from_env`] with the environment passed in, so that it can be
    /// tested without touching the process environment.
    fn from_settings(
        deployment: Option<String>,
        setting: impl Fn(&str) -> Option<String>,
    ) -> Result<Self> {
        let first = |names: &[&str]| names.iter().find_map(|name| setting(name));
        let configuration = |message: &str| LlmError::Configuration(message.to_string());

        let deployment = deployment.ok_or_else(|| {
            configuration(
                "the Microsoft Foundry provider has no default model; \
                 pass the name of your deployment with --model",
            )
        })?;
        let resource = first(&[
            "FOUNDRY_RESOURCE_ENDPOINT",
            "FOUNDRY_RESOURCE",
            "ANTHROPIC_FOUNDRY_BASE_URL",
            "ANTHROPIC_FOUNDRY_RESOURCE",
        ])
        .ok_or_else(|| {
            configuration(
                "FOUNDRY_RESOURCE_ENDPOINT is not set; export the endpoint of your Microsoft \
                 Foundry resource (https://<resource>.services.ai.azure.com) to use the \
                 Microsoft Foundry provider",
            )
        })?;
        // A token is short-lived, so one that is set was set on purpose.
        let credential = match first(&["FOUNDRY_AUTH_TOKEN"]) {
            Some(token) => FoundryCredential::EntraToken(token),
            None => first(&["FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_API_KEY"])
                .map(FoundryCredential::ApiKey)
                .ok_or_else(|| {
                    configuration(
                        "FOUNDRY_API_KEY is not set; export it, or a Microsoft Entra ID token \
                         in FOUNDRY_AUTH_TOKEN, to use the Microsoft Foundry provider",
                    )
                })?,
        };
        let api = setting(FOUNDRY_API_ENV)
            .map(|value| value.parse::<FoundryApi>())
            .transpose()?;
        Self::new(&resource, credential, deployment, api)
    }

    /// The API the deployment is called through.
    pub fn api(&self) -> FoundryApi {
        self.api
    }

    /// The resource endpoint, without any API path.
    pub fn endpoint(&self) -> &str {
        &self.endpoint
    }

    /// The deployment name, sent as `model`.
    pub fn deployment(&self) -> &str {
        &self.deployment
    }
}

#[async_trait]
impl LlmProvider for FoundryProvider {
    async fn complete(&self, system: &str, messages: &[Message]) -> Result<String> {
        match &self.backend {
            FoundryBackend::Anthropic(provider) => provider.complete(system, messages).await,
            FoundryBackend::OpenAi(provider) => provider.complete(system, messages).await,
        }
    }
}

/// Reduces whatever the user has for a Foundry resource to its endpoint root.
///
/// Accepts a bare resource name, the resource endpoint, a project endpoint
/// (`.../api/projects/<project>`), or a full URL of either API copied from the
/// portal or the docs (`.../anthropic/v1/messages`, `.../openai/v1`). A path
/// prefix in front of those, as a gateway might add, is kept.
fn foundry_endpoint(resource: &str) -> Result<String, LlmError> {
    /// Path segments of the two APIs, longest first.
    const API_PATHS: &[&str] = &[
        "/v1/messages",
        "/chat/completions",
        "/v1",
        "/anthropic",
        "/openai",
    ];

    let resource = resource.trim();
    let invalid = || {
        LlmError::Configuration(format!(
            "`{resource}` is neither a Microsoft Foundry resource name nor an endpoint such as \
             https://<resource>.services.ai.azure.com"
        ))
    };

    let Some((scheme, rest)) = resource
        .split_once("://")
        .filter(|(scheme, _)| matches!(*scheme, "https" | "http"))
    else {
        let is_name = !resource.is_empty()
            && resource
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-');
        // A host pasted without its scheme, such as `res.services.ai.azure.com`.
        let is_host = resource.contains('.')
            && !resource.contains("://")
            && !resource.chars().any(char::is_whitespace);
        return if is_name {
            Ok(format!("https://{resource}.services.ai.azure.com"))
        } else if is_host {
            foundry_endpoint(&format!("https://{resource}"))
        } else {
            Err(invalid())
        };
    };

    let rest = rest.split(['?', '#']).next().unwrap_or(rest);
    let (host, path) = rest.split_once('/').unwrap_or((rest, ""));
    if host.is_empty() {
        return Err(invalid());
    }

    let mut path = format!("/{path}");
    loop {
        let trimmed = path.trim_end_matches('/');
        let shorter = API_PATHS
            .iter()
            .find_map(|api_path| trimmed.strip_suffix(api_path))
            .unwrap_or(trimmed);
        if shorter.len() == path.len() {
            break;
        }
        path = shorter.to_string();
    }
    // The Messages API only exists on the resource, not on a project.
    if let Some(project) = path.find("/api/projects/") {
        path.truncate(project);
    }
    Ok(format!("{scheme}://{host}{path}"))
}

// ---------------------------------------------------------------------------
// Mock
// ---------------------------------------------------------------------------

/// Separator between replies in a [`MockProvider`] script file.
pub const MOCK_SCRIPT_SEPARATOR: &str = "=== next reply ===";

/// One call received by a [`MockProvider`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordedCall {
    pub system: String,
    pub messages: Vec<Message>,
}

/// Replays a fixed list of replies, one per call, and records what it was asked.
///
/// Used for deterministic tests and for running the whole pipeline offline.
#[derive(Debug, Default)]
pub struct MockProvider {
    replies: Mutex<VecDeque<String>>,
    calls: Mutex<Vec<RecordedCall>>,
}

impl MockProvider {
    pub fn new<I, S>(replies: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            replies: Mutex::new(replies.into_iter().map(Into::into).collect()),
            calls: Mutex::new(Vec::new()),
        }
    }

    /// Parses a script: replies separated by lines reading
    /// [`MOCK_SCRIPT_SEPARATOR`].
    pub fn from_script(script: &str) -> Self {
        let mut replies = vec![String::new()];
        for line in script.lines() {
            if line.trim() == MOCK_SCRIPT_SEPARATOR {
                replies.push(String::new());
            } else if let Some(reply) = replies.last_mut() {
                reply.push_str(line);
                reply.push('\n');
            }
        }
        Self::new(replies.into_iter().filter(|reply| !reply.trim().is_empty()))
    }

    pub fn from_script_file(path: &Path) -> Result<Self> {
        let script = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read mock script {}", path.display()))?;
        Ok(Self::from_script(&script))
    }

    /// Every request received so far, in order.
    pub fn calls(&self) -> Vec<RecordedCall> {
        self.calls.lock().expect("mock lock").clone()
    }

    /// Number of scripted replies not yet used.
    pub fn remaining(&self) -> usize {
        self.replies.lock().expect("mock lock").len()
    }
}

#[async_trait]
impl LlmProvider for MockProvider {
    async fn complete(&self, system: &str, messages: &[Message]) -> Result<String> {
        self.calls.lock().expect("mock lock").push(RecordedCall {
            system: system.to_string(),
            messages: messages.to_vec(),
        });
        self.replies
            .lock()
            .expect("mock lock")
            .pop_front()
            .context("the mock provider has no scripted replies left")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A minimal one-request-per-connection HTTP server that answers with canned
    /// responses and records the raw requests it saw.
    struct FakeServer {
        url: String,
        requests: std::sync::Arc<Mutex<Vec<String>>>,
    }

    impl FakeServer {
        async fn start(responses: Vec<String>) -> Self {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let url = format!("http://{}", listener.local_addr().unwrap());
            let requests = std::sync::Arc::new(Mutex::new(Vec::new()));
            let seen = requests.clone();
            tokio::spawn(async move {
                for response in responses {
                    let Ok((mut socket, _)) = listener.accept().await else {
                        return;
                    };
                    let mut raw = Vec::new();
                    let mut buffer = [0u8; 4096];
                    loop {
                        let read = socket.read(&mut buffer).await.unwrap_or(0);
                        if read == 0 {
                            break;
                        }
                        raw.extend_from_slice(&buffer[..read]);
                        if request_complete(&raw) {
                            break;
                        }
                    }
                    seen.lock()
                        .unwrap()
                        .push(String::from_utf8_lossy(&raw).into_owned());
                    let _ = socket.write_all(response.as_bytes()).await;
                    let _ = socket.shutdown().await;
                }
            });
            Self { url, requests }
        }

        fn requests(&self) -> Vec<String> {
            self.requests.lock().unwrap().clone()
        }

        fn bodies(&self) -> Vec<Value> {
            self.requests()
                .iter()
                .map(|raw| serde_json::from_str(raw.split_once("\r\n\r\n").unwrap().1).unwrap())
                .collect()
        }
    }

    fn request_complete(raw: &[u8]) -> bool {
        let text = String::from_utf8_lossy(raw);
        let Some((head, body)) = text.split_once("\r\n\r\n") else {
            return false;
        };
        let length = head
            .lines()
            .find_map(|line| {
                line.to_ascii_lowercase()
                    .strip_prefix("content-length:")
                    .map(|v| v.trim().parse::<usize>().unwrap_or(0))
            })
            .unwrap_or(0);
        body.len() >= length
    }

    fn http(status: &str, content_type: &str, extra_headers: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\ncontent-type: {content_type}\r\ncontent-length: {}\r\nconnection: close\r\n{extra_headers}\r\n{body}",
            body.len()
        )
    }

    fn sse(events: &[Value]) -> String {
        let body: String = events
            .iter()
            .map(|event| {
                format!(
                    "event: {}\ndata: {event}\n\n",
                    event["type"].as_str().unwrap()
                )
            })
            .collect();
        http("200 OK", "text/event-stream", "", &body)
    }

    fn text_stream(text_parts: &[&str], stop_reason: &str) -> String {
        let mut events = vec![
            json!({"type": "message_start", "message": {"id": "msg_1", "model": "claude-opus-5-5"}}),
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hidden"}}),
            json!({"type": "content_block_stop", "index": 0}),
            json!({"type": "ping"}),
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}),
        ];
        for part in text_parts {
            events.push(json!({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": part}}));
        }
        events.push(json!({"type": "content_block_stop", "index": 1}));
        events.push(json!({"type": "message_delta", "delta": {"stop_reason": stop_reason}, "usage": {"output_tokens": 12}}));
        events.push(json!({"type": "message_stop"}));
        sse(&events)
    }

    fn conversation() -> Vec<Message> {
        vec![
            Message::user("prove it"),
            Message::assistant("```lean\nsorry\n```"),
            Message::user("no cheating"),
        ]
    }

    fn header<'a>(request: &'a str, name: &str) -> Option<&'a str> {
        request.lines().find_map(|line| {
            let (key, value) = line.split_once(':')?;
            key.eq_ignore_ascii_case(name).then(|| value.trim())
        })
    }

    // -- Anthropic ----------------------------------------------------------

    #[tokio::test]
    async fn anthropic_sends_the_documented_request_and_reads_the_stream() {
        let server = FakeServer::start(vec![text_stream(
            &["theorem t", " : True := ", "trivial ∎"],
            "end_turn",
        )])
        .await;
        let provider = AnthropicProvider::new("sk-test", DEFAULT_ANTHROPIC_MODEL)
            .unwrap()
            .with_base_url(&server.url);

        let reply = provider
            .complete("be rigorous", &conversation())
            .await
            .unwrap();
        assert_eq!(reply, "theorem t : True := trivial ∎");

        let request = &server.requests()[0];
        assert!(
            request.starts_with("POST /v1/messages HTTP/1.1"),
            "{request}"
        );
        assert_eq!(header(request, "x-api-key"), Some("sk-test"));
        assert_eq!(header(request, "anthropic-version"), Some("2026-10-21"));
        // Custom endpoints do not get the first-party fallback beta.
        assert_eq!(header(request, "anthropic-beta"), None);

        let body = &server.bodies()[0];
        assert_eq!(body["model"], "claude-opus-5-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "be rigorous");
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
        assert_eq!(body["output_config"], json!({"effort": "high"}));
        assert_eq!(body["max_tokens"], 64_000);
        assert_eq!(
            body["messages"],
            json!([
                {"role": "user", "content": "prove it"},
                {"role": "assistant", "content": "```lean\nsorry\n```"},
                {"role": "user", "content": "no cheating"},
            ])
        );
        // Sampling parameters are rejected by the current API version.
        for removed in ["temperature", "top_p", "top_k", "fallbacks"] {
            assert!(body.get(removed).is_none(), "{removed} must not be sent");
        }
    }

    #[test]
    fn anthropic_requests_fallbacks_only_where_supported() {
        let official = AnthropicProvider::new("k", "claude-opus-5-5").unwrap();
        assert!(official.fallbacks.load(Ordering::Relaxed));
        let body = official.request_body("", &conversation(), true);
        assert_eq!(body["fallbacks"], "default");
        assert!(
            body.get("system").is_none(),
            "an empty system prompt is omitted"
        );

        // Claude Haiku 5.5 rejects the parameter but supports adaptive thinking.
        let haiku = AnthropicProvider::new("k", "claude-haiku-5-5").unwrap();
        assert!(!haiku.fallbacks.load(Ordering::Relaxed));
        assert_eq!(
            haiku.request_body("s", &conversation(), false)["thinking"]["type"],
            "adaptive"
        );

        // Unknown models get a bare request with conservative limits.
        let unknown = AnthropicProvider::new("k", "some-future-model").unwrap();
        let body = unknown.request_body("s", &conversation(), false);
        assert!(body.get("thinking").is_none() && body.get("output_config").is_none());
        assert_eq!(body["max_tokens"], 8_192);
        assert!(!unknown.fallbacks.load(Ordering::Relaxed));

        assert_eq!(
            model_profile("claude-sonnet-5-5"),
            ModelProfile {
                adaptive_thinking: true,
                fallbacks: true
            }
        );
        assert_eq!(
            model_profile("claude-sonnet-5"),
            ModelProfile {
                adaptive_thinking: true,
                fallbacks: false
            }
        );
        assert_eq!(
            model_profile("claude-haiku-4-5"),
            ModelProfile {
                adaptive_thinking: false,
                fallbacks: false
            }
        );
    }

    #[tokio::test]
    async fn anthropic_drops_the_fallback_beta_when_the_api_rejects_it() {
        let rejection = json!({"type": "error", "error": {"type": "invalid_request_error",
            "message": "Unexpected value(s) `server-side-fallback-2026-07-01` for the `anthropic-beta` header."}});
        let server = FakeServer::start(vec![
            http(
                "400 Bad Request",
                "application/json",
                "",
                &rejection.to_string(),
            ),
            text_stream(&["ok"], "end_turn"),
        ])
        .await;
        let mut provider = AnthropicProvider::new("k", "claude-opus-5-5").unwrap();
        provider.base_url = server.url.clone();

        assert_eq!(provider.complete("s", &conversation()).await.unwrap(), "ok");
        let requests = server.requests();
        assert_eq!(header(&requests[0], "anthropic-beta"), Some(FALLBACK_BETA));
        assert_eq!(server.bodies()[0]["fallbacks"], "default");
        assert_eq!(header(&requests[1], "anthropic-beta"), None);
        assert!(server.bodies()[1].get("fallbacks").is_none());
        assert!(!provider.fallbacks.load(Ordering::Relaxed));
    }

    #[tokio::test]
    async fn anthropic_retries_transient_failures() {
        let overloaded = json!({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}});
        let server = FakeServer::start(vec![
            http(
                "529 Overloaded",
                "application/json",
                "retry-after: 0\r\n",
                &overloaded.to_string(),
            ),
            text_stream(&["second time lucky"], "end_turn"),
        ])
        .await;
        let provider = AnthropicProvider::new("k", "claude-opus-5-5")
            .unwrap()
            .with_base_url(&server.url);
        assert_eq!(
            provider.complete("s", &conversation()).await.unwrap(),
            "second time lucky"
        );
        assert_eq!(server.requests().len(), 2);
    }

    #[tokio::test]
    async fn anthropic_reports_api_errors_without_retrying() {
        let unauthorized = json!({"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"},
            "request_id": "req_123"});
        let server = FakeServer::start(vec![http(
            "401 Unauthorized",
            "application/json",
            "",
            &unauthorized.to_string(),
        )])
        .await;
        let provider = AnthropicProvider::new("bad", "claude-opus-5-5")
            .unwrap()
            .with_base_url(&server.url);

        let error = provider.complete("s", &conversation()).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "Anthropic API error 401 (authentication_error): invalid x-api-key [request req_123]"
        );
        assert_eq!(server.requests().len(), 1);
    }

    #[tokio::test]
    async fn anthropic_surfaces_refusals_and_truncation() {
        let refusal = sse(&[
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "partial"}}),
            json!({"type": "message_delta", "delta": {"stop_reason": "refusal", "stop_details": {"type": "refusal", "category": "cyber"}}}),
        ]);
        let server =
            FakeServer::start(vec![refusal, text_stream(&["half a pro"], "max_tokens")]).await;
        let provider = AnthropicProvider::new("k", "claude-opus-5-5")
            .unwrap()
            .with_base_url(&server.url);

        let error = provider.complete("s", &conversation()).await.unwrap_err();
        assert!(matches!(
            error.downcast_ref::<LlmError>(),
            Some(LlmError::Refused { category: Some(category) }) if category == "cyber"
        ));
        let error = provider.complete("s", &conversation()).await.unwrap_err();
        assert!(matches!(
            error.downcast_ref::<LlmError>(),
            Some(LlmError::Truncated)
        ));
    }

    #[test]
    fn sse_decoder_survives_arbitrary_chunk_boundaries() {
        let stream = "event: a\ndata: {\"n\": \"é∎\"}\n\nevent: b\r\ndata: first\r\ndata: second\r\n\r\ndata: tail";
        // Split at every byte offset, including inside multi-byte characters.
        for split in 0..stream.len() {
            let (left, right) = stream.as_bytes().split_at(split);
            let mut decoder = SseDecoder::default();
            let mut events = decoder.push(left);
            events.extend(decoder.push(right));
            events.extend(decoder.finish());
            assert_eq!(
                events,
                ["{\"n\": \"é∎\"}", "first\nsecond", "tail"],
                "split at {split}"
            );
        }
    }

    #[test]
    fn stream_errors_and_incomplete_streams_are_errors() {
        let mut stream = AnthropicStream::default();
        stream.apply(r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}"#);
        stream.apply(
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        );
        let error = stream.finish(ANTHROPIC).unwrap_err();
        assert!(error.is_transient());
        assert_eq!(
            error.to_string(),
            "Anthropic API error (overloaded_error): Overloaded"
        );

        let mut cut_short = AnthropicStream::default();
        cut_short
            .apply(r#"{"type":"content_block_delta","delta":{"type":"text_delta","text":"x"}}"#);
        assert!(matches!(
            cut_short.finish(ANTHROPIC),
            Err(LlmError::Malformed { .. })
        ));

        let mut empty = AnthropicStream::default();
        empty.apply("not json");
        empty.apply(r#"{"type":"message_delta","delta":{"stop_reason":"end_turn"}}"#);
        assert!(matches!(
            empty.finish(ANTHROPIC),
            Err(LlmError::Empty { .. })
        ));
    }

    #[test]
    fn missing_api_keys_are_explained() {
        let error = LlmError::MissingApiKey {
            provider: ANTHROPIC,
            var: "ANTHROPIC_API_KEY",
        };
        assert_eq!(
            error.to_string(),
            "ANTHROPIC_API_KEY is not set; export it to use the Anthropic provider"
        );
    }

    // -- OpenAI-compatible --------------------------------------------------

    #[tokio::test]
    async fn openai_sends_a_minimal_chat_completion_request() {
        let completion = json!({"choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": "```lean\ntheorem t : True := trivial\n```"}}]});
        let server = FakeServer::start(vec![http(
            "200 OK",
            "application/json",
            "",
            &completion.to_string(),
        )])
        .await;
        let provider = OpenAiProvider::new(
            Some("sk-openai".into()),
            "gpt-test",
            format!("{}/v1/", server.url),
        )
        .unwrap();

        let reply = provider
            .complete("be rigorous", &conversation())
            .await
            .unwrap();
        assert_eq!(reply, "```lean\ntheorem t : True := trivial\n```");

        let request = &server.requests()[0];
        assert!(
            request.starts_with("POST /v1/chat/completions HTTP/1.1"),
            "{request}"
        );
        assert_eq!(header(request, "authorization"), Some("Bearer sk-openai"));
        assert_eq!(
            server.bodies()[0],
            json!({
                "model": "gpt-test",
                "messages": [
                    {"role": "system", "content": "be rigorous"},
                    {"role": "user", "content": "prove it"},
                    {"role": "assistant", "content": "```lean\nsorry\n```"},
                    {"role": "user", "content": "no cheating"},
                ]
            })
        );
    }

    #[tokio::test]
    async fn openai_compatible_servers_need_no_key() {
        let completion = json!({"choices": [{"message": {"content": [{"type": "text", "text": "local "}, {"type": "text", "text": "model"}]}}]});
        let server = FakeServer::start(vec![http(
            "200 OK",
            "application/json",
            "",
            &completion.to_string(),
        )])
        .await;
        let provider = OpenAiProvider::new(None, "qwen2.5-coder", &server.url).unwrap();

        assert_eq!(
            provider.complete("", &conversation()).await.unwrap(),
            "local model"
        );
        assert_eq!(header(&server.requests()[0], "authorization"), None);
        assert_eq!(
            server.bodies()[0]["messages"][0]["role"],
            "user",
            "no empty system message"
        );
    }

    #[tokio::test]
    async fn openai_retries_rate_limits_and_reports_errors() {
        let limited = json!({"error": {"message": "Rate limit reached", "type": "rate_limit_error", "code": "rate_limit_exceeded"}});
        let missing = json!({"error": {"message": "The model `nope` does not exist", "type": "invalid_request_error", "code": "model_not_found"}});
        let ok = json!({"choices": [{"finish_reason": "stop", "message": {"content": "fine"}}]});
        let server = FakeServer::start(vec![
            http(
                "429 Too Many Requests",
                "application/json",
                "retry-after: 0\r\n",
                &limited.to_string(),
            ),
            http("200 OK", "application/json", "", &ok.to_string()),
            http(
                "404 Not Found",
                "application/json",
                "",
                &missing.to_string(),
            ),
        ])
        .await;
        let provider = OpenAiProvider::new(Some("k".into()), "nope", &server.url).unwrap();

        assert_eq!(
            provider.complete("s", &conversation()).await.unwrap(),
            "fine"
        );
        let error = provider.complete("s", &conversation()).await.unwrap_err();
        assert_eq!(
            error.to_string(),
            "OpenAI API error 404 (invalid_request_error): The model `nope` does not exist"
        );
        assert_eq!(server.requests().len(), 3);
    }

    #[test]
    fn openai_flags_unusable_completions() {
        let parse = |value: Value| parse_chat_completion(OPENAI, &value.to_string());
        assert!(matches!(
            parse(
                json!({"choices": [{"finish_reason": "length", "message": {"content": "theorem t"}}]})
            ),
            Err(LlmError::Truncated)
        ));
        assert!(matches!(
            parse(
                json!({"choices": [{"finish_reason": "stop", "message": {"content": null, "refusal": "I can't help"}}]})
            ),
            Err(LlmError::Refused { .. })
        ));
        assert!(matches!(
            parse(json!({"choices": [{"finish_reason": "stop", "message": {"content": "  "}}]})),
            Err(LlmError::Empty { .. })
        ));
        assert!(matches!(
            parse(json!({"choices": []})),
            Err(LlmError::Malformed { .. })
        ));
        assert!(matches!(
            parse_chat_completion(OPENAI, "<html>"),
            Err(LlmError::Malformed { .. })
        ));
    }

    // -- Microsoft Foundry ------------------------------------------------------

    fn settings<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<String> + 'a {
        move |name| {
            pairs
                .iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        }
    }

    fn foundry(
        server: &FakeServer,
        credential: FoundryCredential,
        deployment: &str,
    ) -> FoundryProvider {
        FoundryProvider::new(&server.url, credential, deployment, None).unwrap()
    }

    #[tokio::test]
    async fn foundry_calls_claude_deployments_through_the_messages_api() {
        let server = FakeServer::start(vec![text_stream(
            &["theorem t : True := ", "trivial"],
            "end_turn",
        )])
        .await;
        let provider = foundry(
            &server,
            FoundryCredential::ApiKey("azure-key".into()),
            "claude-opus-5-5",
        );
        assert_eq!(provider.api(), FoundryApi::Anthropic);
        assert_eq!(provider.endpoint(), server.url);
        assert_eq!(provider.deployment(), "claude-opus-5-5");

        let reply = provider
            .complete("be rigorous", &conversation())
            .await
            .unwrap();
        assert_eq!(reply, "theorem t : True := trivial");

        let request = &server.requests()[0];
        assert!(
            request.starts_with("POST /anthropic/v1/messages HTTP/1.1"),
            "{request}"
        );
        assert_eq!(header(request, "api-key"), Some("azure-key"));
        assert_eq!(header(request, "x-api-key"), None);
        assert_eq!(header(request, "authorization"), None);
        assert_eq!(header(request, "anthropic-version"), Some("2026-10-21"));
        // Server-side fallbacks do not exist on Foundry.
        assert_eq!(header(request, "anthropic-beta"), None);

        let body = &server.bodies()[0];
        assert_eq!(body["model"], "claude-opus-5-5");
        assert_eq!(body["stream"], true);
        assert_eq!(body["system"], "be rigorous");
        assert_eq!(body["thinking"], json!({"type": "adaptive"}));
        assert_eq!(body["output_config"], json!({"effort": "high"}));
        assert!(body.get("fallbacks").is_none());
    }

    #[tokio::test]
    async fn foundry_calls_other_deployments_through_chat_completions() {
        let completion = json!({"choices": [{"finish_reason": "stop", "message": {"content": "```lean\n-- ok\n```"}}]});
        let server = FakeServer::start(vec![http(
            "200 OK",
            "application/json",
            "",
            &completion.to_string(),
        )])
        .await;
        let provider = foundry(
            &server,
            FoundryCredential::ApiKey("azure-key".into()),
            "gpt-6.2-astra",
        );
        assert_eq!(provider.api(), FoundryApi::OpenAi);

        let reply = provider
            .complete("be rigorous", &conversation())
            .await
            .unwrap();
        assert_eq!(reply, "```lean\n-- ok\n```");

        let request = &server.requests()[0];
        assert!(
            request.starts_with("POST /openai/v1/chat/completions HTTP/1.1"),
            "{request}"
        );
        assert_eq!(header(request, "api-key"), Some("azure-key"));
        assert_eq!(header(request, "authorization"), None);

        let body = &server.bodies()[0];
        assert_eq!(body["model"], "gpt-6.2-astra");
        assert_eq!(
            body["messages"][0],
            json!({"role": "system", "content": "be rigorous"})
        );
        assert_eq!(
            body.as_object().unwrap().len(),
            2,
            "only `model` and `messages` are sent"
        );
    }

    #[tokio::test]
    async fn foundry_sends_entra_tokens_as_bearer_on_both_apis() {
        let completion =
            json!({"choices": [{"finish_reason": "stop", "message": {"content": "fine"}}]});
        let server = FakeServer::start(vec![
            text_stream(&["fine"], "end_turn"),
            http("200 OK", "application/json", "", &completion.to_string()),
        ])
        .await;
        for deployment in ["claude-mythos-5-2", "my-deployment"] {
            let token = FoundryCredential::EntraToken("entra-token".into());
            let provider = foundry(&server, token, deployment);
            assert_eq!(
                provider.complete("s", &conversation()).await.unwrap(),
                "fine"
            );
        }

        let requests = server.requests();
        assert!(
            requests[0].starts_with("POST /anthropic/v1/messages"),
            "{}",
            requests[0]
        );
        assert!(
            requests[1].starts_with("POST /openai/v1/chat/completions"),
            "{}",
            requests[1]
        );
        for request in &requests {
            assert_eq!(header(request, "authorization"), Some("Bearer entra-token"));
            assert_eq!(header(request, "api-key"), None);
            assert_eq!(header(request, "x-api-key"), None);
        }
    }

    #[tokio::test]
    async fn foundry_errors_name_the_platform_and_the_request() {
        let missing = json!({"error": {"code": "DeploymentNotFound", "message": "The API deployment for this resource does not exist."}});
        let unauthorized = json!({"type": "error", "error": {"type": "authentication_error", "message": "Invalid API key"}});
        let server = FakeServer::start(vec![
            http(
                "404 Not Found",
                "application/json",
                "apim-request-id: abc-123\r\n",
                &missing.to_string(),
            ),
            http(
                "401 Unauthorized",
                "application/json",
                "request-id: req_9\r\n",
                &unauthorized.to_string(),
            ),
        ])
        .await;

        let key = || FoundryCredential::ApiKey("wrong".into());
        let error = foundry(&server, key(), "typo")
            .complete("s", &conversation())
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Microsoft Foundry API error 404 (DeploymentNotFound): The API deployment for this resource does not exist. [request abc-123]"
        );
        let error = foundry(&server, key(), "claude-opus-5-5")
            .complete("s", &conversation())
            .await
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Microsoft Foundry API error 401 (authentication_error): Invalid API key [request req_9]"
        );
    }

    #[test]
    fn foundry_picks_the_api_from_the_deployment_name() {
        for claude in [
            "claude-opus-5-5",
            "claude-haiku-5-5",
            "prod-Claude-Sonnet",
            "CLAUDE",
        ] {
            assert_eq!(
                FoundryApi::for_deployment(claude),
                FoundryApi::Anthropic,
                "{claude}"
            );
        }
        for other in ["gpt-6.2-astra", "model-router", "my-deployment", ""] {
            assert_eq!(
                FoundryApi::for_deployment(other),
                FoundryApi::OpenAi,
                "{other}"
            );
        }
        assert_eq!(
            "anthropic".parse::<FoundryApi>().unwrap(),
            FoundryApi::Anthropic
        );
        assert_eq!(
            " OpenAI ".parse::<FoundryApi>().unwrap(),
            FoundryApi::OpenAi
        );
        let error = "gemini".parse::<FoundryApi>().unwrap_err();
        assert_eq!(
            error.to_string(),
            "AUTO_LEAN_FOUNDRY_API must be `anthropic` or `openai`, not `gemini`"
        );
    }

    #[test]
    fn foundry_accepts_names_and_any_documented_url() {
        let root = "https://my-res.services.ai.azure.com";
        for input in [
            "my-res",
            "my-res.services.ai.azure.com",
            "https://my-res.services.ai.azure.com",
            "https://my-res.services.ai.azure.com/",
            "https://my-res.services.ai.azure.com/anthropic/",
            "https://my-res.services.ai.azure.com/anthropic/v1/messages",
            "https://my-res.services.ai.azure.com/openai/v1/",
            "https://my-res.services.ai.azure.com/openai/v1/chat/completions",
            "https://my-res.services.ai.azure.com/api/projects/my-project",
            "https://my-res.services.ai.azure.com/api/projects/my-project/openai/v1",
            "  https://my-res.services.ai.azure.com/openai/v1?api-version=preview ",
        ] {
            assert_eq!(foundry_endpoint(input).unwrap(), root, "{input:?}");
        }

        // Other hosts, ports and gateway prefixes are kept as they are.
        assert_eq!(
            foundry_endpoint("https://my-res.openai.azure.com/openai/v1/").unwrap(),
            "https://my-res.openai.azure.com"
        );
        assert_eq!(
            foundry_endpoint("https://gateway.example.com/foundry/anthropic").unwrap(),
            "https://gateway.example.com/foundry"
        );
        assert_eq!(
            foundry_endpoint("http://127.0.0.1:8080").unwrap(),
            "http://127.0.0.1:8080"
        );
        // A host that merely looks like an API path is not mangled.
        assert_eq!(
            foundry_endpoint("http://openai:8080/openai/v1").unwrap(),
            "http://openai:8080"
        );

        for invalid in [
            "",
            "my res",
            "ftp://my-res.services.ai.azure.com",
            "https://",
            "https:///openai",
        ] {
            let error = foundry_endpoint(invalid).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("neither a Microsoft Foundry resource name nor an endpoint"),
                "{invalid:?}: {error}"
            );
        }
    }

    #[test]
    fn foundry_reads_its_settings_in_a_fixed_order() {
        let build = |deployment: Option<&str>, pairs: &[(&str, &str)]| {
            FoundryProvider::from_settings(deployment.map(str::to_string), settings(pairs))
        };

        // The names Microsoft's documentation uses.
        let provider = build(
            Some("claude-opus-5-5"),
            &[
                (
                    "FOUNDRY_RESOURCE_ENDPOINT",
                    "https://res.services.ai.azure.com",
                ),
                ("FOUNDRY_API_KEY", "k"),
            ],
        )
        .unwrap();
        assert_eq!(provider.endpoint(), "https://res.services.ai.azure.com");
        assert_eq!(provider.api(), FoundryApi::Anthropic);

        // The variables of Anthropic's Foundry SDKs work as well.
        let provider = build(
            Some("claude-opus-5-5"),
            &[
                ("ANTHROPIC_FOUNDRY_RESOURCE", "res"),
                ("ANTHROPIC_FOUNDRY_API_KEY", "k"),
            ],
        )
        .unwrap();
        assert_eq!(provider.endpoint(), "https://res.services.ai.azure.com");
        let provider = build(
            Some("gpt-6.2-astra"),
            &[
                (
                    "ANTHROPIC_FOUNDRY_BASE_URL",
                    "https://res.services.ai.azure.com/anthropic/",
                ),
                ("FOUNDRY_AUTH_TOKEN", "t"),
            ],
        )
        .unwrap();
        assert_eq!(provider.endpoint(), "https://res.services.ai.azure.com");
        assert_eq!(provider.api(), FoundryApi::OpenAi);

        // The explicit override beats the deployment name.
        let provider = build(
            Some("prod-reasoner"),
            &[
                ("FOUNDRY_RESOURCE", "res"),
                ("FOUNDRY_API_KEY", "k"),
                (FOUNDRY_API_ENV, "anthropic"),
            ],
        )
        .unwrap();
        assert_eq!(provider.api(), FoundryApi::Anthropic);

        // Every missing piece is named.
        let complaint = |deployment: Option<&str>, pairs: &[(&str, &str)]| {
            build(deployment, pairs).unwrap_err().to_string()
        };
        assert!(complaint(None, &[]).contains("--model"));
        assert!(complaint(Some("d"), &[]).contains("FOUNDRY_RESOURCE_ENDPOINT is not set"));
        assert!(
            complaint(Some("d"), &[("FOUNDRY_RESOURCE", "res")])
                .contains("FOUNDRY_API_KEY is not set")
        );
        let bad_api = [
            ("FOUNDRY_RESOURCE", "res"),
            ("FOUNDRY_API_KEY", "k"),
            (FOUNDRY_API_ENV, "x"),
        ];
        assert!(complaint(Some("d"), &bad_api).contains("must be `anthropic` or `openai`"));

        // A key for Anthropic's own API is never offered to a Foundry resource.
        let wrong_key = [
            ("FOUNDRY_RESOURCE", "res"),
            ("ANTHROPIC_API_KEY", "sk-ant"),
            ("OPENAI_API_KEY", "sk"),
        ];
        assert!(
            complaint(Some("claude-opus-5-5"), &wrong_key).contains("FOUNDRY_API_KEY is not set")
        );
    }

    #[test]
    fn foundry_tunes_requests_for_claude_deployments_by_name() {
        let body_for = |deployment: &str| {
            let key = FoundryCredential::ApiKey("k".into());
            let provider = FoundryProvider::new("res", key, deployment, None).unwrap();
            let FoundryBackend::Anthropic(backend) = &provider.backend else {
                panic!("{deployment} should use the Messages API");
            };
            assert!(!backend.fallbacks.load(Ordering::Relaxed));
            backend.request_body("s", &conversation(), false)
        };

        // The model ID anywhere in the deployment name is enough.
        let named = body_for("eu-claude-sonnet-5-5-prod");
        assert_eq!(named["thinking"], json!({"type": "adaptive"}));
        assert_eq!(named["max_tokens"], 64_000);

        // An opaque name still gets room for a model that thinks by default.
        let opaque = body_for("claude-prod");
        assert!(opaque.get("thinking").is_none() && opaque.get("output_config").is_none());
        assert_eq!(opaque["max_tokens"], 64_000);
    }

    // -- Mock -----------------------------------------------------------------

    #[tokio::test]
    async fn mock_replays_replies_in_order_and_records_calls() {
        let mock = MockProvider::new(["first", "second"]);
        assert_eq!(
            mock.complete("sys", &[Message::user("a")]).await.unwrap(),
            "first"
        );
        assert_eq!(mock.remaining(), 1);
        assert_eq!(
            mock.complete("sys", &conversation()).await.unwrap(),
            "second"
        );

        let error = mock.complete("sys", &[]).await.unwrap_err();
        assert!(error.to_string().contains("no scripted replies left"));

        let calls = mock.calls();
        assert_eq!(calls.len(), 3);
        assert_eq!(
            calls[0],
            RecordedCall {
                system: "sys".into(),
                messages: vec![Message::user("a")]
            }
        );
        assert_eq!(calls[1].messages.len(), 3);
    }

    #[test]
    fn mock_scripts_are_split_on_the_separator_line() {
        let script = "\
```lean
theorem a : True := by sorry
```
=== next reply ===

```lean
theorem a : True := trivial
```
NOTE: done
  === next reply ===
";
        let mock = MockProvider::from_script(script);
        assert_eq!(mock.remaining(), 2);
        let replies: Vec<String> = mock.replies.lock().unwrap().iter().cloned().collect();
        assert_eq!(replies[0], "```lean\ntheorem a : True := by sorry\n```\n");
        assert!(replies[1].ends_with("NOTE: done\n"));
    }
}
