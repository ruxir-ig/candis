"""Minimal, dependency-free OpenAI-compatible LLM client.

Why hand-rolled: the ranking path must stay stdlib-only, and the offline LLM
work (judge / re-rank / feature extraction) should not pull a heavy SDK into
the base environment. This speaks the OpenAI /v1/chat/completions protocol,
which is also spoken by Groq, Together, OpenRouter, Gemini's OpenAI-compat
layer, vLLM, LM Studio, and Ollama (`/v1/...` alias). So one client covers
every backend we might use.

Configure via env (no code changes to switch backend):
    LLM_BASE_URL   e.g. https://integrate.api.nvidia.com/v1
    LLM_API_KEY    provider API key
    LLM_MODEL      e.g. meta/llama-3.3-70b-instruct
    LLM_TIMEOUT_S  per-call timeout                       (default: 180)

There is deliberately no implicit local-model default. The LLM judge is a high
leverage offline precompute step, so callers must choose the backend explicitly.
"""
from __future__ import annotations

import json
import os
import urllib.request

DEFAULT_BASE = os.environ.get("LLM_BASE_URL")
DEFAULT_KEY = os.environ.get("LLM_API_KEY")
DEFAULT_MODEL = os.environ.get("LLM_MODEL")
DEFAULT_TIMEOUT = int(os.environ.get("LLM_TIMEOUT_S", "180"))


class LLMError(RuntimeError):
    pass


def chat(
    messages: list[dict],
    *,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE,
    api_key: str = DEFAULT_KEY,
    timeout: int = DEFAULT_TIMEOUT,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    response_format_json: bool = False,
) -> str:
    """Send a chat completion; return the assistant content string.

    response_format_json=True asks the backend to force valid JSON output
    (supported by OpenAI, Groq, Together, Gemini-compat). Falls through
    silently for backends that ignore it.
    """
    if not base_url or not api_key or not model:
        raise LLMError(
            "LLM_BASE_URL, LLM_API_KEY, and LLM_MODEL must be set for offline "
            "LLM precompute. Ranking itself does not need these env vars."
        )
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format_json:
        body["response_format"] = {"type": "json_object"}

    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:500]}") from None
    except Exception as e:
        raise LLMError(f"{type(e).__name__}: {e}") from None

    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError):
        raise LLMError(f"unexpected response: {json.dumps(data)[:400]}")


def config_summary() -> str:
    """One-line description of the active backend, for logging."""
    key = DEFAULT_KEY
    masked = key[:3] + "***" if key else "<unset>"
    return f"model={DEFAULT_MODEL} base={DEFAULT_BASE} key={masked}"
