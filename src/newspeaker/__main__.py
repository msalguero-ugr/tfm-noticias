from .rss.capture import capture_google_news
from .rss.resolve_links import resolve_links_from_jsonl
from .summarize.pipeline import run as run_summarize

if __name__ == "__main__":
    #capture_google_news("tecnología espacial", 3, 5, "data/staging")
    #resolve_links_from_jsonl("data/staging/rss_gnews_20250816_124405.jsonl")
    run_summarize(
        input_jsonl="data/staging/rss_gnews_20250816_124405_resolved.jsonl",
        output_jsonl="data/staging/rss_gnews_20250816_124405_summarized.jsonl",
        max_workers=8,
        max_sentences=4,
        model_name="sumy_textrank",
    )