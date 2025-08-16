from __future__ import annotations
from typing import Optional
from .clean import clean_text

def _newspaper_extract(url: str) -> Optional[str]:
    try:
        from newspaper import Article
        art = Article(url)
        art.download()
        art.parse()
        text = f"{art.title}\n\n{art.text}".strip() if art.title else art.text
        return clean_text(text)
    except Exception:
        return None

def _trafilatura_extract(url: str) -> Optional[str]:
    try:
        import trafilatura
        downloaded = trafilatura.fetch_url(url, no_ssl=True)
        if not downloaded:
            return None
        text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
        return clean_text(text) if text else None
    except Exception:
        return None

def extract_article_text(url: str) -> Optional[str]:
    text = _newspaper_extract(url)
    if text:
        return text
    return _trafilatura_extract(url)
