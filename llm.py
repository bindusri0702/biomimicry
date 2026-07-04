"""Thin LiteLLM wrapper.

`LLM.complete` is the only method the stage nodes use. It always calls the
configured live model through LiteLLM (an API key is required — there is no
offline stub), restricts the output to a Pydantic schema, and returns the
validated instance.

Output is restricted per provider (see `complete`): native response-schema where the
provider supports it (Gemini/OpenAI), and a plain `json_object` request otherwise
(Groq/NIM — tool-call-based output was tried and abandoned as unreliable). The reply is
validated through the schema regardless.

For tests, inject a fake client exposing the same `complete` signature in place of a
stage's module-level `_llm` rather than relying on any built-in stub.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from typing import Any

from pydantic import BaseModel

from . import config

# litellm/pydantic-core emit a benign serializer UserWarning when a provider's response
# (e.g. a tool-call message with content=None, or NVIDIA NIM's 5-field message) doesn't
# match litellm's ModelResponse schema. pydantic-core attributes it to litellm's call
# frame, so a module="pydantic.main" filter misses it — filter by message text instead.
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

# litellm registers an atexit cleanup (litellm/llms/custom_httpx/async_client_cleanup.py)
# that calls the deprecated asyncio.get_event_loop() with no running loop, raising a
# DeprecationWarning on Python 3.12+. It's harmless and internal to litellm, and still
# present in the latest release (1.90.1) — so there's no version to upgrade to; suppress
# just that one message.
warnings.filterwarnings(
    "ignore", message="There is no current event loop", category=DeprecationWarning)


# --- rate-limit header capture -----------------------------------------------
# litellm drops provider response headers, so Groq's x-ratelimit-* counters never reach
# the ModelResponse. We recover them for free (no extra request) via an httpx response
# event hook on a client we hand to litellm.completion(client=...). Only Groq returns these
# headers, and this client is passed only for groq/ models (it breaks the Gemini handler),
# so a single shared holder is safe given the pipeline calls the LLM sequentially.
_RATELIMIT_HEADERS: dict[str, str] = {}
_http_client = None


def _capture_hook(response) -> None:
    for k, v in response.headers.items():
        kl = k.lower()
        if "ratelimit" in kl or kl == "retry-after":
            _RATELIMIT_HEADERS[kl] = v


def _get_capture_client():
    """A litellm HTTPHandler whose httpx client records rate-limit headers per response."""
    global _http_client
    if _http_client is None:
        from litellm.llms.custom_httpx.http_handler import HTTPHandler
        _http_client = HTTPHandler(timeout=config.LLM_TIMEOUT)
        _http_client.client.event_hooks = {"response": [_capture_hook]}
    return _http_client


class MissingAPIKeyError(RuntimeError):
    """Raised when no LLM provider key is configured (no offline fallback exists)."""


class LLM:
    def __init__(self, model: str | None = None):
        # An explicit model pins every task to it (used by tests). None => route by task tier.
        self.model = model
        # Latest per-call usage snapshot + lowest remaining-requests seen this run (set by
        # _record_usage; available programmatically even when config.LOG_USAGE is off).
        self.last_usage: dict | None = None
        self.min_remaining_requests: int | None = None

    # -- public ---------------------------------------------------------------
    def complete(
        self,
        *,
        task: str,
        system: str,
        user: str,
        schema: type[BaseModel],
        temperature: float = 0.4,
        ctx: dict | None = None,
    ) -> BaseModel:
        """Restrict the model's output to `schema` and return the validated instance.

        The model is chosen by task complexity (see `_models_for`): the SUPER tier
        for reasoning-heavy tasks, the NANO tier otherwise (both default to Mistral small).
        Each model is retried with exponential backoff (litellm), then we fall back down the
        chain (other tier, then the Gemini fallbacks) on persistent overload/503.

        Output restriction is chosen per model: native response-schema when the provider
        supports it (Gemini/OpenAI), else a plain json_object request (Groq/NIM). The reply is
        parsed and validated through `schema` in all cases; a validation/parse failure is
        non-transient and raises. `ctx` is accepted for call-site symmetry.
        """
        if not config.HAS_LLM_KEY:
            raise MissingAPIKeyError(
                "No LLM API key found. Set MISTRAL_API_KEY (default provider) or one of "
                "GROQ_API_KEY / NVIDIA_NIM_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY / "
                "OPENAI_API_KEY / ANTHROPIC_API_KEY (e.g. in biomimicry/.env). This "
                "pipeline has no offline mode."
            )

        import litellm
        # NVIDIA NIM does not advertise response_format support; drop unsupported params per
        # provider instead of erroring (we still rely on _parse_json + the prompt for JSON).
        litellm.drop_params = True
        litellm.enable_json_schema_validation = True

        models = self._models_for(task)
        for i, mdl in enumerate(models):
            messages = [
                {"role": "system", "content": _prep_system(mdl, system)},
                {"role": "user", "content": user},
            ]
            try:
                kwargs = dict(
                    model=mdl,
                    messages=messages,
                    temperature=temperature,
                    num_retries=config.LLM_NUM_RETRIES,
                    timeout=config.LLM_TIMEOUT,
                )
                kwargs.update(_output_mode(litellm, mdl, schema))
                # Groq returns x-ratelimit-* headers; capture them via our hooked client.
                # (Passing this client to the Gemini handler breaks it, so groq-only.)
                if mdl.startswith("groq/"):
                    _RATELIMIT_HEADERS.clear()
                    kwargs["client"] = _get_capture_client()
                resp = litellm.completion(**kwargs)
                self._record_usage(resp, mdl, task)
                return schema.model_validate(
                    _parse_json(resp.choices[0].message.content or ""))
            except Exception as e:
                # Non-transient (auth, bad request, JSON/validation error) or no models
                # left -> raise.
                if not _is_transient(e) or i == len(models) - 1:
                    raise
                print(f"[llm] {mdl} unavailable ({getattr(e, 'status_code', '?')}); "
                      f"falling back to {models[i + 1]}", file=sys.stderr)

    # -- routing --------------------------------------------------------------
    def _models_for(self, task: str) -> list[str]:
        """Ordered model chain for a task: primary tier -> other tier -> configured fallbacks.

        An explicit `self.model` or `config.MODEL_OVERRIDE` pins all tasks to one model
        (followed by the fallbacks)."""
        pin = self.model or config.MODEL_OVERRIDE
        if pin:
            return [pin, *config.LLM_FALLBACKS]
        if task in config.COMPLEX_TASKS:
            tiers = [config.MODEL_SUPER, config.MODEL_NANO]
        else:
            tiers = [config.MODEL_NANO, config.MODEL_SUPER]
        return [*tiers, *config.LLM_FALLBACKS]

    # -- usage / quota --------------------------------------------------------
    def _record_usage(self, resp, model: str, task: str) -> None:
        """Snapshot token usage + Groq rate-limit headers after a successful call.

        Stored on the instance and, when config.LOG_USAGE is on, logged as one concise
        line to stderr. Never raises — usage telemetry must not break a completion."""
        headers = dict(_RATELIMIT_HEADERS) if model.startswith("groq/") else {}
        snap = _usage_snapshot(resp, model, headers)
        self.last_usage = snap
        rr = snap.get("remaining_requests")
        if rr is not None and (self.min_remaining_requests is None
                               or rr < self.min_remaining_requests):
            self.min_remaining_requests = rr
        if config.LOG_USAGE:
            print(_fmt_usage(snap, task), file=sys.stderr)


# --- structured-output mode selection ----------------------------------------
def _output_mode(litellm, model: str, schema: type[BaseModel]) -> dict:
    """litellm.completion kwargs restricting output to `schema` for this provider:
    native response-schema where supported (Gemini/OpenAI), else json_object (Groq/NIM);
    the reply is validated through `schema` either way.

    Forced tool-calling is deliberately NOT used: Groq's llama-3.3-70b returns tool calls
    as literal text and rejects them with a 400 tool_use_failed, so json_object (which it
    handles reliably) is the fallback for providers without native response-schema.
    """
    try:
        if litellm.supports_response_schema(model):
            return {"response_format": schema}
    except Exception:  # capability probe must never break a completion
        pass
    return {"response_format": {"type": "json_object"}}


def _prep_system(model: str, system: str) -> str:
    """Nemotron models are reasoning models: prepend `detailed thinking off` so they emit no
    <think> traces, and demand bare JSON since NVIDIA NIM may ignore response_format. No-op
    for non-Nemotron providers (e.g. the Gemini fallbacks)."""
    if "nvidia_nim" in model or "nemotron" in model.lower():
        return ("detailed thinking off\n\n"
                "Return ONLY a single valid JSON object — no markdown fences, no prose.\n\n"
                + system)
    return system


# --- usage snapshot / formatting ---------------------------------------------
def _usage_snapshot(resp, model: str, headers: dict) -> dict:
    """Token usage (all providers) + Groq rate-limit headers (if present). Never raises."""
    snap = {"model": model, "prompt_tokens": None, "completion_tokens": None,
            "total_tokens": None, "remaining_requests": None, "limit_requests": None,
            "reset_requests": None, "remaining_tokens": None, "limit_tokens": None}
    try:
        u = getattr(resp, "usage", None)
        if u is not None:
            snap["prompt_tokens"] = getattr(u, "prompt_tokens", None)
            snap["completion_tokens"] = getattr(u, "completion_tokens", None)
            snap["total_tokens"] = getattr(u, "total_tokens", None)
    except Exception:
        pass

    def _h(suffix):  # match regardless of any provider prefix on the header name
        for k, v in headers.items():
            if k.endswith(suffix):
                return v
        return None

    try:
        rr = _h("x-ratelimit-remaining-requests")
        snap["remaining_requests"] = int(rr) if rr is not None else None
        lr = _h("x-ratelimit-limit-requests")
        snap["limit_requests"] = int(lr) if lr is not None else None
        snap["reset_requests"] = _h("x-ratelimit-reset-requests")
        rt = _h("x-ratelimit-remaining-tokens")
        snap["remaining_tokens"] = int(rt) if rt is not None else None
        lt = _h("x-ratelimit-limit-tokens")
        snap["limit_tokens"] = int(lt) if lt is not None else None
    except (TypeError, ValueError):
        pass  # non-numeric header -> leave as-is / None
    return snap


def _fmt_usage(s: dict, task: str) -> str:
    line = f"[llm] {s['model']} (task={task}) used {s.get('total_tokens')} tok"
    if s.get("remaining_requests") is not None:
        line += f" | req {s['remaining_requests']}/{s.get('limit_requests')}"
        if s.get("reset_requests"):
            line += f" (reset {s['reset_requests']})"
        if s.get("remaining_tokens") is not None:
            line += f" | tok {s['remaining_tokens']}/{s.get('limit_tokens')}"
    else:
        line += " | req n/a (no rate-limit headers)"
    return line


# --- transient-error detection (retry/fallback) ------------------------------
_TRANSIENT_CODES = {408, 409, 429, 500, 502, 503, 504}


def _is_transient(e: Exception) -> bool:
    """Overload / rate-limit / connectivity errors worth a fallback, by HTTP status
    (robust across litellm exception classes, incl. the raised VertexAIError)."""
    if getattr(e, "status_code", None) in _TRANSIENT_CODES:
        return True
    try:
        import litellm
        return isinstance(e, (litellm.exceptions.Timeout,
                              litellm.exceptions.APIConnectionError,
                              litellm.exceptions.ServiceUnavailableError,
                              litellm.exceptions.RateLimitError,
                              litellm.exceptions.InternalServerError))
    except Exception:
        return False


# --- JSON parsing -------------------------------------------------------------
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _parse_json(text: str) -> Any:
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fall back to the first balanced { } or [ ] span.
        for opener, closer in (("{", "}"), ("[", "]")):
            i, j = text.find(opener), text.rfind(closer)
            if 0 <= i < j:
                try:
                    return json.loads(text[i : j + 1])
                except json.JSONDecodeError:
                    continue
        raise ValueError(f"LLM did not return valid JSON:\n{text[:500]}")
