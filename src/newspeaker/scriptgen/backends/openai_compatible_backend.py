from __future__ import annotations

import json
from typing import List, Optional, Dict, Any
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


class OpenAICompatibleBackend(ChatBackend):
    """
    Chat backend for any OpenAI compatible server.
    Examples
    - LM Studio local server at http://127.0.0.1:1234/v1
    - llama.cpp server in OpenAI mode
    - Other gateways that mimic /v1/chat/completions

    Notes
    - API key is optional. Many local servers do not require it.
    - Seed is passed if provided. Some servers honor it, some ignore it.
    - Usage is returned when the server provides it.
    """

    name: str = "openai_compat"

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:1234/v1",
        api_key: Optional[str] = None,
        default_model: str = "qwen2.5-7b-instruct",
        timeout: float = 60.0,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.default_model = default_model
        self.timeout = timeout
        self.extra_headers = extra_headers or {}

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

            body: Dict[str, Any] = {
                "model": model,
                "messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                "temperature": float(params.temperature) if params.temperature is not None else 0.2,
                "max_tokens": int(params.max_tokens) if params.max_tokens is not None else 700,
            }
            if params.top_p is not None:
                body["top_p"] = float(params.top_p)
            if params.seed is not None:
                # Many compatible servers accept this
                body["seed"] = int(params.seed)
            if params.stop:
                body["stop"] = list(params.stop)

            resp = self._post_json("/chat/completions", body, timeout=self.timeout)

            # Expected response shape:
            # {
            #   "id": "...",
            #   "object": "chat.completion",
            #   "model": "xxx",
            #   "choices": [
            #       {"index": 0, "message": {"role": "assistant", "content": "..."}, "finish_reason": "stop"}
            #   ],
            #   "usage": {"prompt_tokens": 123, "completion_tokens": 456, "total_tokens": 579}
            # }
            model_used = resp.get("model") or model
            text = ""
            choices = resp.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0].get("message") if isinstance(choices[0], dict) else None
                if isinstance(msg, dict):
                    text = (msg.get("content") or "").strip()

            usage_block = resp.get("usage") if isinstance(resp, dict) else None
            usage = None
            if isinstance(usage_block, dict):
                pt = usage_block.get("prompt_tokens")
                ct = usage_block.get("completion_tokens")
                tt = usage_block.get("total_tokens")
                usage = GenerationUsage(
                    prompt_tokens=pt if isinstance(pt, int) else None,
                    completion_tokens=ct if isinstance(ct, int) else None,
                    total_tokens=tt if isinstance(tt, int) else None,
                )

            if not text:
                return GenerationResult(
                    text="",
                    model_used=model_used,
                    usage=usage,
                    error="openai_compat_empty_response",
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
            return GenerationResult(
                text="",
                model_used=params.model or self.default_model,
                usage=None,
                error=f"openai_compat_error: {e}",
            )

    def supports_seed(self) -> bool:
        return True

    # -----------------------------
    # Internals
    # -----------------------------
    def _post_json(self, path: str, payload: dict, timeout: float) -> dict:
        # Base can end with /v1 or without it. Always build cleanly.
        endpoint = "chat/completions" if path.strip("/") == "chat/completions" else path.strip("/")
        url = urljoin(self.base_url + "/", endpoint)
        data = json.dumps(payload).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)

        req = Request(url=url, data=data, method="POST", headers=headers)
        try:
            with urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw)
        except HTTPError as he:
            try:
                body = he.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
            raise BackendError(f"openai_compat_http_error {he.code}: {body}") from he
        except URLError as ue:
            raise BackendError(f"openai_compat_url_error: {ue.reason}") from ue
        except json.JSONDecodeError as je:
            raise BackendError(f"openai_compat_bad_json: {je}") from je
