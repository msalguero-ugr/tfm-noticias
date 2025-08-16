import json, pathlib, asyncio, re, os, time
from urllib.parse import urlparse, parse_qs, urljoin
from playwright.async_api import async_playwright

G_HL = os.getenv("GOOGLE_NEWS_HL", "es")
G_GL = os.getenv("GOOGLE_NEWS_GL", "ES")
G_CEID = os.getenv("GOOGLE_NEWS_CEID", "ES:es")

def _try_decode_direct(gn_link: str) -> str | None:
    qs = parse_qs(urlparse(gn_link).query)
    return qs.get("url", [None])[0]

def _maybe_add_locale(link: str) -> str:
    u = urlparse(link)
    if u.netloc.endswith("news.google.com") and "/rss/articles/" in u.path:
        sep = "&" if u.query else ""
        return f"{link}{sep}&hl={G_HL}&gl={G_GL}&ceid={G_CEID}"
    return link

def _looks_google(u: str) -> bool:
    host = urlparse(u).netloc
    return any(host.endswith(d) for d in (
        "news.google.com", "consent.google.com", "google.com"
    ))

def _extract_meta_refresh(html: str, base: str) -> str | None:
    m = re.search(r'http-equiv=["\']?refresh["\']?\s+content=["\'][^"\']*url=([^"\']+)', html, re.I)
    if m:
        return urljoin(base, m.group(1).strip())
    return None

async def _safe_get_attr(page, selector: str, attr: str) -> str | None:
    try:
        loc = page.locator(selector).first
        # do not wait; just attempt to read if present
        return await loc.get_attribute(attr)
    except Exception:
        return None

async def _bypass_consent(context) -> None:
    # preload consent cookies so G services skip the interstitial
    now = int(time.time())
    future = now + 3600 * 24 * 365
    cookies = []
    for domain in [".google.com", ".consent.google.com", ".news.google.com"]:
        cookies.append({"name": "CONSENT", "value": "YES+", "domain": domain, "path": "/", "expires": future})
        # Some regions use SOCS/ANID. Not strictly needed, but harmless:
        cookies.append({"name": "SOCS", "value": "CAI", "domain": domain, "path": "/", "expires": future})
    await context.add_cookies(cookies)

async def _resolve_one(context, url: str) -> str | None:
    # direct param first
    direct = _try_decode_direct(url)
    if direct:
        return direct

    url = _maybe_add_locale(url)

    # request-level hop first (faster than opening a page)
    try:
        resp = await context.request.get(url, max_redirects=10, timeout=20000, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept-Language": "es-ES,es;q=0.9",
        })
        if resp:
            final_url = resp.url
            if final_url and not _looks_google(final_url):
                return final_url
            html = await resp.text()
            meta_url = _extract_meta_refresh(html, base=final_url or url)
            if meta_url and not _looks_google(meta_url):
                return meta_url
    except Exception:
        pass

    # page-level navigation with consent bypass and retry
    async def _page_attempt(u: str) -> str | None:
        page = await context.new_page()
        try:
            await page.goto(u, wait_until="domcontentloaded", timeout=25000)
            # try canonical/og
            canonical = await _safe_get_attr(page, "head link[rel=canonical]", "href")
            if canonical and not _looks_google(canonical):
                return canonical
            og = await _safe_get_attr(page, "head meta[property='og:url']", "content")
            if og and not _looks_google(og):
                return og
            # meta refresh in head
            meta = await _safe_get_attr(page, "head meta[http-equiv='refresh']", "content")
            if meta and "url=" in meta.lower():
                target = meta.split("url=", 1)[1].strip()
                if target:
                    target = urljoin(page.url, target)
                    if not _looks_google(target):
                        return target
            # settle and take page.url
            try:
                await page.wait_for_load_state("networkidle", timeout=15000)
            except Exception:
                pass
            fin = page.url
            if fin and not _looks_google(fin):
                return fin
            return None
        finally:
            await page.close()

    # attempt 1
    final = await _page_attempt(url)
    if final and not _looks_google(final):
        return final

    # if stuck on consent or a google host, set cookies and try once more
    await _bypass_consent(context)
    final = await _page_attempt(url)
    if final and not _looks_google(final):
        return final

    return None

async def _resolver(urls: list[str], concurrency: int = 5) -> list[str | None]:
    sem = asyncio.Semaphore(concurrency)
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0",
            locale="es-ES",
            extra_http_headers={"Accept-Language": "es-ES,es;q=0.9"},
        )
        async def worker(u):
            async with sem:
                return await _resolve_one(context, u)
        results = await asyncio.gather(*(worker(u) for u in urls))
        await context.close()
        await browser.close()
        return results

def resolve_links_from_jsonl(input_path: str, output_path: str | None = None, concurrency: int = 5) -> str:
    in_path = pathlib.Path(input_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    items = [json.loads(line) for line in in_path.open("r", encoding="utf-8") if line.strip()]
    urls = [it.get("link", "") for it in items]

    resolved = asyncio.run(_resolver(urls, concurrency=concurrency))

    # output path handling: accept directory or filename
    if output_path:
        out_path = pathlib.Path(output_path)
        if out_path.exists() and out_path.is_dir():
            out_path = out_path / (in_path.stem + "_resolved.jsonl")
        elif out_path.suffix == "":
            out_path = out_path.with_name(out_path.name + "_resolved.jsonl")
    else:
        out_path = in_path.with_name(in_path.stem + "_resolved.jsonl")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for it, ru in zip(items, resolved):
            it["resolved_url"] = ru
            w.write(json.dumps(it, ensure_ascii=False) + "\n")
    return str(out_path)
