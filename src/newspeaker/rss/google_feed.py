from urllib.parse import quote_plus
import os

def build_google_news_rss(query: str, language: str, country: str) -> str:
    hl = os.getenv("GOOGLE_NEWS_HL", language)
    gl = os.getenv("GOOGLE_NEWS_GL", country)
    ceid = os.getenv("GOOGLE_NEWS_CEID", f"{country}:{language}")
    q = quote_plus(query.strip())
    return f"https://news.google.com/rss/search?q={q}&hl={hl}&gl={gl}&ceid={ceid}"
