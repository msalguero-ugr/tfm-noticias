import typer
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from .rss.capture import capture_google_news
from .rss.resolve_links import resolve_links_from_jsonl
from .summarize.pipeline import run as run_summarize
from .scriptgen.generate import generate_episode

app = typer.Typer(help="TFM News pipeline")

@app.command("capture")
def capture(
    query: str = typer.Option(..., "--query", "-q", help="Search query"),
    recent_days: int = typer.Option(3, "--recent-days"),
    max_articles: int = typer.Option(100, "--max-articles"),
    out: str = typer.Option("data/staging", "--out"),
):
    count, path = capture_google_news(query, recent_days, max_articles, out)
    typer.echo(f"Saved {count} items to {path}")

@app.command("resolve_links")
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

@app.command("write_script")
def scriptgen_run(
    # IO
    in_path: Path = typer.Option(..., "--in", help="Ruta al JSONL de resúmenes"),
    out_path: Optional[Path] = typer.Option(
        None,
        "--out",
        help="Ruta de salida. Por defecto usa el nombre de entrada con sufijo _scripted.jsonl",
    ),
    # Script style
    style: str = typer.Option(
        "educativo",
        "--style",
        help="educativo, conversacional o humoristico",
        case_sensitive=False,
    ),
    max_items: int = typer.Option(10, "--max-items", help="Máximo de ítems a procesar"),
    intro_template: str = typer.Option(
        "Hola, aquí tienes las noticias clave sobre {query}.",
        "--intro-template",
        help="Plantilla de intro con {query}",
    ),
    outro_template: str = typer.Option(
        "Gracias por escuchar. Hasta la próxima.",
        "--outro-template",
        help="Plantilla de cierre con {query}",
    ),
    episode_id: Optional[str] = typer.Option(
        None,
        "--episode-id",
        help="Identificador del episodio. Si no se indica se usa el nombre del archivo de salida",
    ),
    # Backend selection
    backend: Optional[str] = typer.Option(
        None,
        "--backend",
        help="template, ollama, openai_compat o auto. Si no se indica se usa auto",
        case_sensitive=False,
    ),
    model: str = typer.Option(
        "llama3",
        "--model",
        help="Nombre del modelo para el backend elegido. Ejemplo: llama3, qwen2.5-7b-instruct",
    ),
    base_url: Optional[str] = typer.Option(
        None,
        "--base-url",
        help="URL base para un servidor OpenAI compatible. Ejemplo: http://127.0.0.1:1234/v1",
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        help="Clave opcional para servidores OpenAI compatibles que la requieran",
        show_default=False,
    ),
    # Generation knobs
    temperature: float = typer.Option(0.2, "--temperature", help="Creatividad del modelo"),
    max_tokens: int = typer.Option(700, "--max-tokens", help="Límite de tokens de salida"),
    top_p: Optional[float] = typer.Option(None, "--top-p", help="Nucleus sampling"),
    seed: Optional[int] = typer.Option(None, "--seed", help="Semilla para reproducibilidad", show_default=False),
    stop: Optional[List[str]] = typer.Option(
        None,
        "--stop",
        help="Secuencias de corte. Puedes repetir la opción varias veces",
        show_default=False,
    ),
):
    """
    Lee el JSONL de summarize y escribe un JSONL con:
    title, resolved_url, summary, script, style, script_model, duration_sec_est,
    words, segment_index, episode_id, citations, errors.
    Selección de backend flexible: template, ollama o servidores OpenAI compatibles.
    """
    style = style.lower().strip()
    if style not in {"educativo", "conversacional", "humoristico"}:
        raise typer.BadParameter("style debe ser: educativo, conversacional o humoristico")

    if out_path is None:
        stem = in_path.stem
        out_path = in_path.parent / f"{stem}_scripted.jsonl"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Evitar mezclar ejecuciones en el mismo archivo
    if out_path.exists():
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        out_path = out_path.with_name(out_path.stem + f".{ts}" + out_path.suffix)

    generate_episode(
        str(in_path),
        str(out_path),
        style=style,
        intro_template=intro_template,
        outro_template=outro_template,
        max_items=max_items,
        episode_id=episode_id,
        # backend selection
        backend=(backend.lower() if backend else None),
        model=model,
        base_url=base_url,
        api_key=api_key,
        # generation knobs
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        seed=seed,
        stop=list(stop) if isinstance(stop, tuple) else stop,  # Typer puede entregar tuple
    )

    typer.echo(f"Listo. Archivo escrito en {out_path}")

if __name__ == "__main__":
    app()
