from .rss.capture import capture_google_news
from .rss.resolve_links import resolve_links_from_jsonl

if __name__ == "__main__":
    #capture_google_news("tecnología espacial", 3, 5, "data/staging")
    resolve_links_from_jsonl("data/staging/rss_gnews_20250816_124405.jsonl")
