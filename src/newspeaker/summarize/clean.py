import re

_whitespace_re = re.compile(r"\s+")

def clean_text(text: str) -> str:
    t = text or ""
    # Remove boilerplate artifacts that often appear
    t = t.replace("\u00a0", " ")
    t = t.replace("\u200b", "")
    # Collapse whitespace
    t = _whitespace_re.sub(" ", t).strip()
    return t
