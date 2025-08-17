from __future__ import annotations

import re
from typing import List, Optional
from urllib.parse import urlparse

from .backends.base import MessageDict, Role


# -----------------------------
# Default system instruction
# -----------------------------
DEFAULT_SYSTEM = (
    "Eres guionista de pódcast en español de España. "
    "Escribe guiones claros y naturales para ser locutados. "
    "No inventes hechos. Usa únicamente lo que viene en 'summary' y, si está, en 'text'. "
    "Mantén nombres y cifras tal cual aparecen. "
    "No incluyas URLs en el cuerpo del guion. "
    "Incluye una única cita de la fuente cerca del final, usando el nombre del medio y el dominio. "
    "Longitud objetivo entre ciento cincuenta y doscientas veinticinco palabras. "
    "Frases cortas, sin emojis ni markdown."
)


# -----------------------------
# Helpers (pure, no I/O)
# -----------------------------
def _strip_www(domain: str) -> str:
    return domain[4:] if domain.startswith("www.") else domain


def outlet_from_title(title: str) -> str:
    """
    Extract a likely outlet name from common title separators.
    Examples:
      'Algo - El País' -> 'El País'
      'Algo | El Mundo' -> 'El Mundo'
    Fallback: return the whole title stripped.
    """
    parts = re.split(r"[-|–—]\s*", title or "")
    if len(parts) >= 2:
        outlet = parts[-1].strip()
        outlet = re.split(r"[:|•]", outlet)[0].strip()
        return outlet or (title or "").strip()
    return (title or "").strip()


def domain_from_url(url: str) -> str:
    try:
        netloc = urlparse(url or "").netloc.lower()
        return _strip_www(netloc)
    except Exception:
        return ""


def trim_text_for_context(text: str, max_words: int = 1200) -> str:
    """
    Keep prompts lean. If article text is long, keep the first ~max_words.
    LLMs handle summaries better when inputs are concise.
    """
    if not text:
        return ""
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[:max_words]).strip()


# -----------------------------
# Public builder
# -----------------------------
def build_messages_for_item(
    *,
    style: str,
    title: str,
    resolved_url: str,
    summary: str,
    text: Optional[str],
    query: str,
    intro_template: Optional[str],
    outro_template: Optional[str],
    system_text: Optional[str] = None,
    extra_user_instructions: Optional[str] = None,
) -> List[MessageDict]:
    """
    Build a minimal, strict chat prompt for a single item.

    Returns a list of messages:
      - one 'system' message with global rules
      - one 'user' message with item data and concrete instructions

    Notes
    - We do not add an 'assistant' example by default to keep prompts short.
    - The spoken script must not include URLs; we still pass the resolved_url
      only to derive the domain and keep it in metadata.
    """
    system_msg: MessageDict = {
        "role": "system",
        "content": (system_text or DEFAULT_SYSTEM).strip(),
    }

    outlet = outlet_from_title(title) if title else "la fuente original"
    domain = domain_from_url(resolved_url) or "el sitio del medio"
    cleaned_text = trim_text_for_context(text or "")

    # One compact, explicit user message. Clear, testable instructions.
    user_lines = [
        f"Estilo: {style}",
        f"Query del episodio: {query}",
        "Plantillas:",
        f"- Intro: {intro_template or 'Hola, aquí tienes las noticias clave sobre {query}.'}",
        f"- Outro: {outro_template or 'Gracias por escuchar. Hasta la próxima.'}",
        "",
        "Instrucciones de salida:",
        "1) Devuelve solo el guion final en texto plano.",
        "2) Este guion corresponde a un único ítem del episodio.",
        "3) Empieza si procede con una línea breve que sitúe el tema.",
        "4) Explica de forma concisa a partir de 'summary'. Si hay 'text', úsalo solo para reforzar, sin agregar hechos nuevos.",
        "5) Incluye una línea de 'por qué importa'.",
        f"6) Cita cerca del final exactamente así: 'Fuente. {outlet}, según información publicada en {domain}.'",
        "7) No incluyas URLs en el cuerpo. Frases cortas. Sin emojis ni markdown.",
    ]
    if extra_user_instructions:
        user_lines.append("")
        user_lines.append("Instrucciones extra:")
        user_lines.append(extra_user_instructions.strip())

    user_lines += [
        "",
        "Datos del ítem:",
        f"title: {title}",
        f"resolved_url: {resolved_url}",
        f"outlet: {outlet}",
        f"domain: {domain}",
        f"summary: {summary}",
        f"text: {cleaned_text if cleaned_text else '(sin texto, usa solo el summary)'}",
    ]

    user_msg: MessageDict = {
        "role": "user",
        "content": "\n".join(user_lines).strip(),
    }

    return [system_msg, user_msg]
