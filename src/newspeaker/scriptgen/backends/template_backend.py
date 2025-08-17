from __future__ import annotations

import re
from typing import List, Optional

from .base import (
    ChatBackend,
    GenerationParams,
    GenerationResult,
    BackendError,
    MessageDict,
    normalize_messages,
)


class TemplateBackend(ChatBackend):
    """
    Deterministic offline generator.
    Expands a short summary into a spoken script in Spanish.
    No network access and no API keys.

    Input expectations
    - Messages follow the OpenAI style with roles.
    - We read the latest user message and try to extract:
        style: educativo | conversacional | humoristico
        summary: the short summary text
        outlet: outlet or source name
        domain: source domain without protocol
      If these fields are not present, we fall back to simple heuristics.

    Output
    - A single, plain text script ready for TTS, 150 to 200 words.
    - One citation line near the end if outlet or domain are known.
    """

    name: str = "template"

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
            user_text = self._last_user_content(msgs)

            # Extract fields from the user payload if present
            style = _extract_field(user_text, ["style", "estilo"]) or "educativo"
            style = style.strip().lower()
            if style not in ("educativo", "conversacional", "humoristico"):
                style = "educativo"

            summary = _extract_multiline(user_text, ["summary", "resumen"])
            if not summary:
                # Second attempt: grab text after 'summary:' anywhere in the payload
                m = re.search(r"(?is)\bsummary\s*:\s*(.+?)(?:\n|$)", user_text)
                if m:
                    summary = m.group(1).strip()

            outlet = _extract_field(user_text, ["outlet"])
            domain = _extract_field(user_text, ["domain", "dominio"])

            # Build deterministic script
            script = _build_script(summary, outlet, domain, style)

            return GenerationResult(
                text=script,
                model_used=params.model or "template-v1",
                usage=None,
                error=None,
            )
        except Exception as e:
            if strict:
                raise BackendError(str(e)) from e
            return GenerationResult(
                text="",
                model_used=params.model or "template-v1",
                usage=None,
                error=f"template_backend_failed: {e}",
            )

    def supports_seed(self) -> bool:
        # Deterministic by construction
        return False

    # -----------------------------
    # Internals
    # -----------------------------
    @staticmethod
    def _last_user_content(messages: List[MessageDict]) -> str:
        # Take the last user message. If none, fall back to last message.
        for m in reversed(messages):
            if m["role"] == "user":
                return m["content"]
        return messages[-1]["content"] if messages else ""


# ---------- helpers used only in this file ----------

def _extract_field(payload: str, keys: list[str]) -> Optional[str]:
    """
    Extract single line fields like 'style: educativo' or 'domain: elpais.com'.
    Case insensitive. Returns the first match.
    """
    for k in keys:
        m = re.search(rf"(?im)^\s*[-*•]?\s*{re.escape(k)}\s*:\s*(.+?)\s*$", payload)
        if m:
            return m.group(1).strip()
    return None


def _extract_multiline(payload: str, keys: list[str]) -> Optional[str]:
    """
    Extract a field that may span multiple sentences on the same line.
    In our prompts, 'summary:' is usually a single line, so this is simple.
    """
    return _extract_field(payload, keys)


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s for s in sentences if s]
    return " ".join(sentences[:max_sentences]).strip()


def _normalize_small_numbers_to_words(text: str) -> str:
    """
    Convert small integers to Spanish words for better TTS.
    Keep years as digits. Handle 0..29 and round tens up to 90.
    """
    units = {
        0: "cero", 1: "uno", 2: "dos", 3: "tres", 4: "cuatro", 5: "cinco",
        6: "seis", 7: "siete", 8: "ocho", 9: "nueve", 10: "diez",
        11: "once", 12: "doce", 13: "trece", 14: "catorce", 15: "quince",
        16: "dieciséis", 17: "diecisiete", 18: "dieciocho", 19: "diecinueve",
        20: "veinte", 21: "veintiuno", 22: "veintidós", 23: "veintitrés",
        24: "veinticuatro", 25: "veinticinco", 26: "veintiséis",
        27: "veintisiete", 28: "veintiocho", 29: "veintinueve",
    }
    tens = {30: "treinta", 40: "cuarenta", 50: "cincuenta",
            60: "sesenta", 70: "setenta", 80: "ochenta", 90: "noventa"}

    def repl(m):
        s = m.group(0)
        if re.fullmatch(r"\d{4}", s):
            return s
        try:
            n = int(s)
        except Exception:
            return s
        if 0 <= n <= 29:
            return units[n]
        if n in tens:
            return tens[n]
        return s

    return re.sub(r"\b\d+\b", repl, text)


def _build_script(summary: str, outlet: Optional[str], domain: Optional[str], style: str) -> str:
    """
    Build a didactic, spoken script from the summary.
    Rules
    - Spanish es ES
    - No invented facts. Only rephrase the summary
    - One citation line near the end using outlet and domain if provided
    - No URLs in the body
    - Short sentences. 150 to 200 words target
    - Style options tweak delivery in a minimal way
    """
    outlet = (outlet or "la fuente original").strip()
    domain = (domain or "el sitio del medio").strip()

    lead = "Titular del día. " + summary.strip()

    explain = (
        "En pocas palabras, este es el punto central. "
        "Para entenderlo mejor, piensa en las consecuencias directas y en quién se ve afectado. "
        "La clave es quedarse con la idea principal sin perderse en los detalles. "
        "Si solo recuerdas una cosa, que sea esta. "
    )

    why = (
        "¿Por qué importa. "
        "Porque influye en decisiones públicas, en empresas o en la vida diaria de muchas personas. "
        "También ayuda a interpretar otros datos y a anticipar lo que puede venir después. "
    )

    source_line = f"Fuente. {outlet}, según información publicada en {domain}. "
    close = "Con esto cerramos este tema."

    base = " ".join([lead, explain, why, source_line, close])
    base = re.sub(r"\s+", " ", base).strip()

    if style == "conversacional":
        # Alternate short lines between Voz A and Voz B
        sentences = re.split(r"(?<=[.!?])\s+", base)
        a, b = [], []
        for i, s in enumerate(sentences):
            (a if i % 2 == 0 else b).append(s)
        base = "Voz A. " + " ".join(a).strip() + " Voz B. " + " ".join(b).strip()
    elif style == "humoristico":
        base = base + " Nota aparte. Un poco de ironía ayuda a recordar la idea, pero sin perder el respeto por los datos."

    # Length control to hit 150 to 200 words
    words = base.split()
    target_min, target_max = 150, 200
    if len(words) < target_min:
        base += " Para cerrar. Quédate con la idea principal y el impacto directo."
        words = base.split()
    elif len(words) > target_max:
        sentences = re.split(r"(?<=[.!?])\s+", base)
        while len(" ".join(sentences).split()) > target_max and len(sentences) > 3:
            sentences.pop(-2)
        base = " ".join(sentences)

    # Tidy and normalize small numbers for TTS
    base = re.sub(r"\s+", " ", base).strip()
    base = _normalize_small_numbers_to_words(base)
    return base
