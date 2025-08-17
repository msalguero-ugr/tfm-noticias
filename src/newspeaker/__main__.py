from .rss.capture import capture_google_news
from .rss.resolve_links import resolve_links_from_jsonl
from .summarize.pipeline import run as run_summarize
from .scriptgen.generate import generate_episode

if __name__ == "__main__":
    #capture_google_news("tecnología espacial", 3, 5, "data/staging")
    #resolve_links_from_jsonl("data/staging/rss_gnews_20250816_124405.jsonl")
    #run_summarize(
    #    input_jsonl="data/staging/rss_gnews_20250816_124405_resolved.jsonl",
    #    output_jsonl="data/staging/rss_gnews_20250816_124405_summarized.jsonl",
    #    max_workers=8,
    #    max_sentences=4,
    #    model_name="sumy_textrank",
    #)
    generate_episode(
        "data/staging/rss_gnews_20250816_124405_summarized.jsonl",
        "data/staging/rss_gnews_20250816_124405_scripted_ollama.jsonl",
        style="educativo",
        intro_template="Hola, aquí tienes las noticias clave sobre {query}.",
        outro_template="Gracias por escuchar. Hasta la próxima.",
        max_items=10,
        episode_id=None,
        backend="ollama",           # use local Ollama
        model="gemma3:270m",        # the model you pulled
        temperature=0.2,
        max_tokens=700,
        seed=42,    
    )