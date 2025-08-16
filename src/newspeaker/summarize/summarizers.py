from __future__ import annotations
from typing import List

DEFAULT_MAX_SENTENCES = 5

def textrank_summary(text: str, max_sentences: int = DEFAULT_MAX_SENTENCES, language: str = "english") -> str:
    """
    Extractive summary using sumy's TextRank.
    Keeps sentence order stable for readability.
    """
    if not text:
        return ""
    # sumy imports are local to avoid import cost if unused
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.text_rank import TextRankSummarizer

    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = TextRankSummarizer()
    
    sentences = summarizer(parser.document, max_sentences)
    summary_sentences: List[str] = [str(s) for s in sentences]
    # Preserve original order
    # sumy returns in importance order, so we reorder by their index in the original text
    original = [str(s) for s in parser.document.sentences]
    idx_map = {s: i for i, s in enumerate(original)}
    summary_sentences.sort(key=lambda s: idx_map.get(s, 10**9))
    return " ".join(summary_sentences).strip()
