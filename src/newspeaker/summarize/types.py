from typing import TypedDict, Optional

class NewsItem(TypedDict, total=False):
    title: str
    link: str
    published_at: str
    source_feed: str
    query: str
    resolved_url: str
    text: str
    summary: str
    summary_model: str
    scrape_error: Optional[str]
