from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, List

# Optional progress bar
try:
    from tqdm import tqdm  # type: ignore
except Exception:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

from .backends.base import GenerationParams
from .backends.template_backend import TemplateBackend
from .routing import build_backend
from .prompt import build_messages_for_item, outlet_from_title, domain_from_url

WORDS_PER_MINUTE = 150


# -----------------------------
# Public API
# -----------------------------

def generate_script_for_item(
    item: Dict,
    *,
    style: str = "educativo",
    intro_template: Optional[str] = None,
    outro_template: Optional[str] = None,
    # Model and backend selection
    backend: Optional[str] = None,          # "template", "ollama", "openai_compat", or "auto"
    model: str = "llama3",                  # model name for the chosen backend
    base_url: Optional[str] = None,         # for openai_compat, e.g., "http://127.0.0.1:1234/v1"
    api_key: Optional[str] = None,          # optional for openai_compat
    # Generation knobs
    temperature: float = 0.2,
    max_tokens: int = 700,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> Dict:
    """
    Turn one summarized item into a narratable script block.

    The returned dict matches the output JSONL schema:
      title, resolved_url, summary, script, style, script_model,
      duration_sec_est, words, segment_index, episode_id, citations, errors.
    """
    title = item.get("title") or ""
    resolved_url = item.get("resolved_url") or ""
    summary = (item.get("summary") or "").strip()
    query = item.get("query") or ""
    text_opt = (item.get("text") or "").strip()

    # Skip logic: we still emit a line with an error so downstream steps can trace the item
    if not resolved_url or not summary:
        return _empty_result(item, style, model, reason="missing resolved_url or summary")

    # Build messages for any backend
    messages = build_messages_for_item(
        style=style,
        title=title,
        resolved_url=resolved_url,
        summary=summary,
        text=text_opt if text_opt else None,
        query=query,
        intro_template=intro_template,
        outro_template=outro_template,
    )

    # Choose a backend and run generation
    errors: Optional[str] = None
    script_block = ""
    picked = build_backend(backend=backend, model=model, base_url=base_url, api_key=api_key)
    params = GenerationParams(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
        stop=stop,
    )
    result = picked.generate(messages, params, strict=False)

    if result.error or not result.text:
        # Fall back to deterministic template backend
        fallback = TemplateBackend()
        fb = fallback.generate(messages, params, strict=False)
        script_block = fb.text or ""
        # Combine error messages for traceability
        errors = (result.error or "llm_empty") if not fb.error else f"{result.error or 'llm_empty'} | {fb.error}"
        script_model = fb.model_used
    else:
        script_block = result.text
        script_model = result.model_used

    # Compose with episode intro, transition, outro flags
    parts = []
    if item.get("_is_first") and intro_template:
        parts.append(_fill_template(intro_template, query=query))
    parts.append(script_block.strip())
    if item.get("_add_transition"):
        parts.append("Vamos con la siguiente noticia.")
    if item.get("_is_last") and outro_template:
        parts.append(_fill_template(outro_template, query=query))

    script = " ".join(p.strip() for p in parts if p and p.strip())

    # Final cleanup for audio: numbers, URLs, whitespace
    script = _remove_urls(script)
    script = _normalize_small_numbers_to_words(script)
    script = re.sub(r"\s+", " ", script).strip()

    # Stats and metadata
    word_count = len(script.split())
    duration_sec_est = _estimate_duration_seconds(word_count, WORDS_PER_MINUTE)

    citations = [{"title": title or outlet_from_title(title) or domain_from_url(resolved_url), "resolved_url": resolved_url}]
    if not text_opt:
        # non fatal warning that we relied only on the summary
        warn = "no full text available, script based only on summary"
        errors = warn if not errors else f"{errors} | {warn}"

    return {
        "title": title,
        "resolved_url": resolved_url,
        "summary": summary,
        "script": script,
        "style": style,
        "script_model": script_model,
        "duration_sec_est": duration_sec_est,
        "words": word_count,
        "segment_index": int(item.get("_segment_index", 0)),
        "episode_id": item.get("_episode_id", ""),
        "citations": citations,
        "errors": errors,
    }


def generate_episode(
    input_path: str,
    output_path: str,
    *,
    style: str = "educativo",
    intro_template: str = "Hola, aquí tienes las noticias clave sobre {query}.",
    outro_template: str = "Gracias por escuchar. Hasta la próxima.",
    max_items: int = 10,
    episode_id: Optional[str] = None,
    # Backend selection
    backend: Optional[str] = None,
    model: str = "llama3",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    # Generation knobs
    temperature: float = 0.2,
    max_tokens: int = 700,
    top_p: Optional[float] = None,
    seed: Optional[int] = None,
    stop: Optional[List[str]] = None,
) -> None:
    """
    Read a summarized JSONL and write a scripted JSONL.

    Each input line produces one output line with the required fields.
    Items missing resolved_url or summary are recorded with an error.
    """
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    items = list(_iter_jsonl(in_path))
    if max_items and max_items > 0:
        items = items[:max_items]

    total = len(items)
    for idx, it in enumerate(tqdm(items, desc="scriptgen")):
        it = dict(it)  # shallow copy
        it["_segment_index"] = idx
        it["_episode_id"] = episode_id or out_path.stem
        it["_is_first"] = idx == 0
        it["_is_last"] = idx == total - 1
        it["_add_transition"] = total > 1 and idx < total - 1

        out_obj = generate_script_for_item(
            it,
            style=style,
            intro_template=intro_template,
            outro_template=outro_template,
            backend=backend,
            model=model,
            base_url=base_url,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            seed=seed,
            stop=stop,
        )

        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out_obj, ensure_ascii=False) + "\n")


# -----------------------------
# Internals and helpers
# -----------------------------

def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _fill_template(template: str, *, query: str) -> str:
    # Minimal helper for {query} replacement
    try:
        return template.format(query=query)
    except Exception:
        return template


def _estimate_duration_seconds(word_count: int, wpm: int = WORDS_PER_MINUTE) -> int:
    minutes = word_count / max(1, wpm)
    return max(1, int(round(minutes * 60)))


def _remove_urls(text: str) -> str:
    # Remove any http/https or www patterns from the spoken script
    return re.sub(r"(https?://\S+|www\.\S+)", "", text).strip()


def _normalize_small_numbers_to_words(text: str) -> str:
    """
    Convert small integers to Spanish words for better TTS.
    Keep four-digit years as digits. Handle 0..29 and tens 30..90.
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


def _empty_result(item: Dict, style: str, model: str, reason: str) -> Dict:
    title = item.get("title") or ""
    resolved_url = item.get("resolved_url") or ""
    summary = (item.get("summary") or "").strip()
    return {
        "title": title,
        "resolved_url": resolved_url,
        "summary": summary,
        "script": "",
        "style": style,
        "script_model": model,
        "duration_sec_est": 0,
        "words": 0,
        "segment_index": int(item.get("_segment_index", 0)),
        "episode_id": item.get("_episode_id", ""),
        "citations": [],
        "errors": reason,
    }
