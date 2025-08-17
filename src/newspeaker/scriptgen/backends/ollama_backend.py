from __future__ import annotations

import json
from typing import List, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

from .base import (
    ChatBackend,
    GenerationParams,
    GenerationResult,
    GenerationUsage,
    BackendError,
    MessageDict,
    normalize_messages,
)


class OllamaBackend(ChatBackend):
    """
    Chat backend for a local Ollama server.

    Defaults
    - base_url: "http://127.0.0.1:11434"
    - endpoint: "/api/chat"
    - streaming is disabled; we read the complete response at once.

    Notes
    - No API key required.
    - Seed control is supported by Ollama via options.seed.
    - Temperature and top_p are supported via options.
    - max_tokens is not strictly enforced by Ollama; we pass it as options.num_predict
      when provided.
    """

    name: str = "ollama"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:11434",
        default_model: str = "llama3",
        timeout: float = 6000.0,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout

    # -----------------------------
    # Public interface
    # -----------------------------
    def generate(
        self,
        messages: List[MessageDict],
        params: GenerationParams,
        strict: bool = False,
    ) -> GenerationResult:
        try:
            msgs = normalize_messages(messages)
            if not msgs:
                raise BackendError("empty messages")

            model = (params.model or self.default_model).strip()
            if not model:
                raise BackendError("missing model name")

            # Build payload for Ollama /api/chat
            # https://github.com/ollama/ollama/blob/main/docs/api.md#chat
            payload = {
                "model": model,
                "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                "stream": False,
                "options": {},
            }

            # Map generic params to Ollama options
            options = payload["options"]
            if params.temperature is not None:
                options["temperature"] = float(params.temperature)
            if params.top_p is not None:
                options["top_p"] = float(params.top_p)
            if params.seed is not None:
                # reproducibility where supported
                options["seed"] = int(params.seed)
            if params.max_tokens is not None and params.max_tokens > 0:
                # Ollama calls this num_predict (max completion tokens)
                options["num_predict"] = int(params.max_tokens)

            resp_obj = self._post_json("/api/chat", payload, timeout=self.timeout)

            # Expected response shape (non-streaming):
            # {
            #   "model": "llama3",
            #   "message": {"role": "assistant", "content": "..."},
            #   "done": true,
            #   "eval_count": 123, "prompt_eval_count": 456, ...
            # }
            text = ""
            model_used = resp_obj.get("model") or model
            message = resp_obj.get("message") or {}
            if isinstance(message, dict):
                text = (message.get("content") or "").strip()

            # Usage (best-effort; Ollama returns eval_count/prompt_eval_count)
            usage = None
            try:
                prompt_tokens = resp_obj.get("prompt_eval_count")
                completion_tokens = resp_obj.get("eval_count")
                total_tokens = None
                if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
                    total_tokens = prompt_tokens + completion_tokens
                usage = GenerationUsage(
                    prompt_tokens=prompt_tokens if isinstance(prompt_tokens, int) else None,
                    completion_tokens=completion_tokens if isinstance(completion_tokens, int) else None,
                    total_tokens=total_tokens,
                )
            except Exception:
                usage = None  # keep it optional

            if not text:
                # Return a clear error for the caller; they can fall back to template
                return GenerationResult(
                    text="",
                    model_used=model_used,
                    usage=usage,
                    error="ollama_empty_response",
                )

            return GenerationResult(
                text=text,
                model_used=model_used,
                usage=usage,
                error=None,
            )

        except Exception as e:
            if strict:
                raise BackendError(str(e)) from e
            return GenerationResult(text="", model_used=params.model or self.default_model, usage=None, error=str(e))

    def supports_seed(self) -> bool:
        return True

    # -----------------------------
    # Internals
    # -----------------------------
    def _post_json(self, path: str, payload: dict, timeout: float) -> dict:
        url = urljoin(self.base_url + "/", path.lstrip("/"))
        data = json.dumps(payload).encode("utf-8")
        req = Request(url=url, data=data, method="POST", headers={"Content-Type": "application/json"})
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except HTTPError as he:
            # Read body for diagnostics if present
            try:
                body = he.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise BackendError(f"ollama_http_error {he.code}: {body}") from he
        except URLError as ue:
            raise BackendError(f"ollama_url_error: {ue.reason}") from ue
        except json.JSONDecodeError as je:
            raise BackendError(f"ollama_bad_json: {je}") from je
