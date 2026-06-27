"""Thin LiteLLM wrapper.

`LLM.complete_json` is the only method the stage nodes use. It always calls the
configured live model through LiteLLM (an API key is required — there is no
offline stub) and parses the JSON response robustly.

For tests, inject a fake client exposing the same `complete_json` signature in
place of a stage's module-level `_llm` rather than relying on any built-in stub.
"""
from __future__ import annotations

import json
import re
import sys
import warnings
from typing import Any

from . import config

# litellm serializes its own ModelResponse/Message pydantic models internally;
# under pydantic v2 this emits a benign UserWarning. It does not affect the parsed
# output (we only read message.content), so suppress the noise.
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic.main")


class MissingAPIKeyError(RuntimeError):
    """Raised when no LLM provider key is configured (no offline fallback exists)."""


class LLM:
    def __init__(self, model: str | None = None):
        self.model = model or config.MODEL

    # -- public ---------------------------------------------------------------
    def complete_json(
        self,
        *,
        task: str,
        system: str,
        user: str,
        temperature: float = 0.4,
        ctx: dict | None = None,
    ) -> Any:
        """Return parsed JSON (dict or list) from the live model.

        Retries the chosen model with exponential backoff (litellm), then
        auto-falls back to lighter models on persistent overload/503. `task`/`ctx`
        are accepted for call-site symmetry and logging but are not otherwise used.
        """
        if not config.HAS_LLM_KEY:
            raise MissingAPIKeyError(
                "No LLM API key found. Set one of GEMINI_API_KEY / GOOGLE_API_KEY / "
                "OPENAI_API_KEY / ANTHROPIC_API_KEY (e.g. in biomimicry/.env). "
                "This pipeline has no offline mode."
            )

        import litellm

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        models = [self.model, *config.LLM_FALLBACKS]
        for i, mdl in enumerate(models):
            try:
                resp = litellm.completion(
                    model=mdl,
                    messages=messages,
                    temperature=temperature,
                    response_format={"type": "json_object"},
                    num_retries=config.LLM_NUM_RETRIES,
                    timeout=config.LLM_TIMEOUT,
                )
                return _parse_json(resp.choices[0].message.content or "")
            except Exception as e:
                # Non-transient (auth, bad request, JSON parse) or no models left -> raise.
                if not _is_transient(e) or i == len(models) - 1:
                    raise
                print(f"[llm] {mdl} unavailable ({getattr(e, 'status_code', '?')}); "
                      f"falling back to {models[i + 1]}", file=sys.stderr)


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
