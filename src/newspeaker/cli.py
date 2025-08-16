import typer
from .rss.capture import capture_google_news
from .rss.resolve_links import resolve_links_from_jsonl
from .summarize.pipeline import run as run_summarize


app = typer.Typer(help="TFM News pipeline")

@app.command()
def capture(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    recent_days: int = typer.Option(3, "--recent-days"),
    max_articles: int = typer.Option(100, "--max-articles"),
    out: str = typer.Option("data/staging", "--out"),
):
    count, path = capture_google_news(query, recent_days, max_articles, out)
    typer.echo(f"Saved {count} items to {path}")

@app.command()
def resolve(
    input_path: str = typer.Option(..., "--in", "-i", help="Input JSONL from capture step"),
    output_path: str = typer.Option(None, "--out", "-o", help="Output JSONL with resolved_url"),
    concurrency: int = typer.Option(5, "--concurrency", "-c", help="Parallel pages to open"),
):
    out = resolve_links_from_jsonl(input_path, output_path, concurrency)
    typer.echo(f"Resolved links written to {out}")

@app.command("summarize")
def summarize(
    input_jsonl: str = typer.Option(..., "--input", "-i", help="Path to input JSONL with resolved_url"),
    output_jsonl: str = typer.Option(..., "--output", "-o", help="Path to write JSONL with text and summary"),
    max_workers: int = typer.Option(8, help="Parallel workers for scraping"),
    max_sentences: int = typer.Option(5, help="Max sentences in the summary"),
):
    """
    Scrape article text from resolved_url and produce concise summaries.
    """
    run_summarize(
        input_jsonl=input_jsonl,
        output_jsonl=output_jsonl,
        max_workers=max_workers,
        max_sentences=max_sentences,
        model_name="sumy_textrank",
    )

if __name__ == "__main__":
    app()
