from __future__ import annotations
from typing import Iterator, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from .io_utils import read_jsonl, write_jsonl
from .scrape import extract_article_text
from .clean import clean_text
from .summarizers import textrank_summary

def _process_item(item: Dict[str, Any],
                  max_sentences: int,
                  model_name: str) -> Dict[str, Any]:
    rec = dict(item)  
    url = rec.get("resolved_url") or rec.get("link")
    if not url:
        rec["scrape_error"] = "missing_url"
        return rec

    text: Optional[str] = extract_article_text(url)
    if not text:
        rec["scrape_error"] = "scrape_failed"
        return rec

    rec["text"] = text

    if len(text.split()) < 80:
        rec["summary"] = text if len(text) < 800 else text[:800]
        rec["summary_model"] = "pass_through"
        return rec

    summary = textrank_summary(text, max_sentences=max_sentences)
    rec["summary"] = clean_text(summary)
    rec["summary_model"] = model_name
    return rec

def run(input_jsonl: str,
        output_jsonl: str,
        max_workers: int = 8,
        max_sentences: int = 5,
        model_name: str = "sumy_textrank") -> None:
    items = list(read_jsonl(input_jsonl))
    results: list[Dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [
            ex.submit(_process_item, it, max_sentences, model_name)
            for it in items
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="summarize"):
            results.append(fut.result())

    idx_by_url = {r.get("resolved_url") or r.get("link"): r for r in results}
    ordered = []
    for it in items:
        key = it.get("resolved_url") or it.get("link")
        ordered.append(idx_by_url.get(key, it))

    write_jsonl(output_jsonl, ordered)
