import typer
from .rss.capture import capture_google_news
from .rss.resolve_links import resolve_links_from_jsonl

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

if __name__ == "__main__":
    app()
