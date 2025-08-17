from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, TypedDict, Union
import abc


# -----------------------------
# Types shared by all backends
# -----------------------------

Role = Literal["system", "user", "assistant"]


class MessageDict(TypedDict):
    """
    Wire format used across the project.
    Matches the common OpenAI style messages.
    """
    role: Role
    content: str


@dataclass
class GenerationParams:
    """
    Knobs that control a single text generation call.
    Only include options that are widely supported.
    Backend specific settings will live inside each backend.
    """
    model: str
    temperature: float = 0.2
    max_tokens: int = 700
    top_p: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[List[str]] = None
    # You can extend later with presence_penalty and frequency_penalty if needed


@dataclass
class GenerationUsage:
    """
    Optional token usage info when a backend provides it.
    """
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None


@dataclass
class GenerationResult:
    """
    Unified result for any backend.
    If error is not None, text may be empty and the caller should fall back.
    """
    text: str
    model_used: str
    usage: Optional[GenerationUsage] = None
    error: Optional[str] = None


class BackendError(Exception):
    """Base class for backend related errors."""


# ----------------------------------------
# Abstract interface for chat text models
# ----------------------------------------

class ChatBackend(abc.ABC):
    """
    Minimal interface that every backend must implement.

    Contract
    1. Input messages are a list of dicts with role and content.
    2. The backend returns a GenerationResult with plain text.
    3. On failure, set error in the result. Do not raise by default.
       Only raise if the caller explicitly requested strict mode.
    4. The method should be side effect free with respect to global state.
    """

    name: str = "base"

    def __init__(self) -> None:
        # Subclasses can put reusable clients here
        pass

    @abc.abstractmethod
    def generate(
        self,
        messages: List[MessageDict],
        params: GenerationParams,
        strict: bool = False,
    ) -> GenerationResult:
        """
        Produce a text completion for the given chat messages.

        strict False means catch exceptions and return them in the error field.
        strict True means raise BackendError on transport or API failure.
        """
        raise NotImplementedError

    def supports_seed(self) -> bool:
        """Return True if the backend honors the seed parameter."""
        return False

    def close(self) -> None:
        """Optional cleanup for network clients or file handles."""
        return


# ----------------------------------------
# Helpers that all backends can reuse
# ----------------------------------------

def normalize_messages(messages: List[Union[MessageDict, Dict]]) -> List[MessageDict]:
    """
    Validate and coerce messages into the canonical list of MessageDict.
    Drops entries with empty content and trims whitespace.
    Raises BackendError for invalid roles.
    """
    out: List[MessageDict] = []
    for i, m in enumerate(messages):
        role = m.get("role")  # type: ignore
        content = m.get("content")  # type: ignore

        if role not in ("system", "user", "assistant"):
            raise BackendError(f"invalid role at index {i}: {role}")
        if not isinstance(content, str):
            raise BackendError(f"content must be str at index {i}")
        content = content.strip()
        if not content:
            # skip empty lines to avoid confusing some backends
            continue
        out.append(MessageDict(role=role, content=content))
    return out
