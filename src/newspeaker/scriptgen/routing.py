from __future__ import annotations

import os
import socket
from typing import Optional, Tuple
from urllib.parse import urlparse

from .backends.base import ChatBackend, BackendError, GenerationParams
from .backends.template_backend import TemplateBackend


# Defaults for local servers
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_LMSTUDIO_URL = "http://127.0.0.1:1234/v1"

ALLOWED_BACKENDS = {"template", "ollama", "openai_compat", "openai", "auto"}


def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _host_port_from_url(base_url: str) -> Tuple[Optional[str], Optional[int]]:
    try:
        u = urlparse(base_url)
        host = u.hostname
        port = u.port
        return host, port
    except Exception:
        return None, None


def _lazy_import_ollama_backend():
    from .backends.ollama_backend import OllamaBackend  # type: ignore
    return OllamaBackend


def _lazy_import_openai_compat_backend():
    from .backends.openai_compatible_backend import OpenAICompatibleBackend  # type: ignore
    return OpenAICompatibleBackend


def _lazy_import_openai_backend():
    from .backends.openai_backend import OpenAIBackend  # type: ignore
    return OpenAIBackend


def detect_local_endpoints(
    explicit_base_url: Optional[str] = None,
) -> dict:
    """
    Probe common local endpoints. Very fast check, no HTTP request.
    Returns a small dict you can log for debugging if needed.
    """
    out = {
        "base_url_given": explicit_base_url or "",
        "lmstudio_open": False,
        "ollama_open": False,
        "lmstudio_url": DEFAULT_LMSTUDIO_URL,
        "ollama_url": DEFAULT_OLLAMA_URL,
    }

    # If a base_url is provided, detect its port too
    if explicit_base_url:
        h, p = _host_port_from_url(explicit_base_url)
        if h and p and _is_port_open(h, p):
            out["lmstudio_open"] = True  # treat any openai-compatible base as available
            out["lmstudio_url"] = explicit_base_url

    # Probe defaults only if no explicit base_url
    if not explicit_base_url:
        # LM Studio default
        h, p = _host_port_from_url(DEFAULT_LMSTUDIO_URL)
        if h and p:
            out["lmstudio_open"] = _is_port_open(h, p)

    # Ollama default port
    h, p = _host_port_from_url(DEFAULT_OLLAMA_URL)
    if h and p:
        out["ollama_open"] = _is_port_open(h, p)

    return out


def pick_backend_name(
    backend: Optional[str],
    base_url: Optional[str],
) -> str:
    """
    Decide which backend to use when user passes --backend or env, or asks for auto.
    Priority in auto mode:
      1) If base_url is provided and reachable, use openai_compat
      2) LM Studio default port if open, use openai_compat
      3) Ollama default port if open, use ollama
      4) Else template
    """
    chosen = (backend or os.getenv("NEWSPEAKER_BACKEND") or "auto").lower()
    if chosen not in ALLOWED_BACKENDS:
        chosen = "auto"

    if chosen != "auto":
        return chosen

    probe = detect_local_endpoints(base_url)
    if base_url:
        # If caller gave a base_url and it is open, prefer openai_compat
        h, p = _host_port_from_url(base_url)
        if h and p and _is_port_open(h, p):
            return "openai_compat"

    if probe["lmstudio_open"]:
        return "openai_compat"
    if probe["ollama_open"]:
        return "ollama"
    return "template"


def build_backend(
    *,
    backend: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> ChatBackend:
    """
    Construct a ChatBackend instance based on parameters and environment.

    Parameters
    - backend: template, ollama, openai_compat, openai, or auto
    - model: model name for the chosen backend
    - base_url: for openai_compat servers like LM Studio or llama.cpp server
    - api_key: optional key for openai_compat if your gateway requires it

    Env vars
    - NEWSPEAKER_BACKEND selects default backend when backend is None
    - NEWSPEAKER_BASE_URL default base_url for openai_compat
    - OPENAI_API_KEY used only by the OpenAI backend
    """
    chosen = pick_backend_name(backend, base_url or os.getenv("NEWSPEAKER_BASE_URL"))

    # Always have a working fallback in hand
    fallback = TemplateBackend()

    try:
        if chosen == "template":
            return fallback

        if chosen == "ollama":
            OllamaBackend = _lazy_import_ollama_backend()
            # model can still be None, the backend will validate later
            return OllamaBackend(
                base_url=DEFAULT_OLLAMA_URL,
                default_model=model or "llama3",
            )

        if chosen == "openai_compat":
            OpenAICompatibleBackend = _lazy_import_openai_compat_backend()
            url = base_url or os.getenv("NEWSPEAKER_BASE_URL") or DEFAULT_LMSTUDIO_URL
            return OpenAICompatibleBackend(
                base_url=url,
                api_key=api_key,  # can be None for local servers
                default_model=model or "qwen2.5-7b-instruct",
            )

        if chosen == "openai":
            OpenAIBackend = _lazy_import_openai_backend()
            key = api_key or os.getenv("OPENAI_API_KEY")
            # Let the backend raise clearly if the key is missing and strict mode is used,
            # but for normal runs it will return an error in GenerationResult if needed.
            return OpenAIBackend(
                api_key=key,
                default_model=model or "gpt-4o-mini",
            )

    except Exception as e:
        # If anything goes wrong, return template to keep the pipeline running
        # The caller can log e and record an error field per item if needed.
        return fallback

    # Defensive fallback
    return fallback
