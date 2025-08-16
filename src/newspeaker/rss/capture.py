import json, pathlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
import feedparser
from urllib.parse import quote_plus

def build_google_news_rss(query, hl="es", gl="ES", ceid="ES:es"):
    q = quote_plus(query.strip())
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"

def _entry_dt(e):
    for k in ("published", "updated", "created"):
        v = e.get(k)
        if v:
            try:
                return parsedate_to_datetime(v)
            except Exception:
                pass
    return None

def _is_recent(dt, days):
    if not dt:
        return False
    return datetime.now(timezone.utc) - dt <= timedelta(days=days)

def capture_google_news(query, recent_days, max_articles, out_dir):
    url = build_google_news_rss(query)
    d = feedparser.parse(url)
    items = []
    for e in d.entries:
        dt = _entry_dt(e)
        if not _is_recent(dt, recent_days):
            continue
        link = (e.get("link") or "").strip()
        if not link:
            continue
        items.append({
            "title": e.get("title"),
            "link": link,
            "published_at": dt.isoformat() if dt else None,
            "source_feed": url,
            "query": query,
        })
    
    seen, clean = set(), []
    for it in items:
        if it["link"].lower() in seen:
            continue
        seen.add(it["link"].lower())
        clean.append(it)
    clean = clean[:max_articles]

    out = pathlib.Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = out / f"rss_gnews_{stamp}.jsonl"
    with open(path, "w", encoding="utf-8") as w:
        for it in clean:
            w.write(json.dumps(it, ensure_ascii=False) + "\n")
    return len(clean), str(path)
