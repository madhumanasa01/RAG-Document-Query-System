"""Answer generation: grounded LLM answer when an API key is set, extractive fallback otherwise."""

import re

import numpy as np

from app.chunker import split_sentences
from app.config import settings
from app.store import SearchHit

_HEADING = re.compile(r"^\s*#+ [^\n]*\n?", re.MULTILINE)

SYSTEM_PROMPT = (
    "You answer questions strictly from the provided document excerpts. "
    "Cite excerpts inline as [1], [2], ... If the excerpts do not contain the answer, "
    "say you could not find it in the documents. Do not use outside knowledge."
)


def build_context(hits: list[SearchHit]) -> str:
    return "\n\n".join(
        f"[{i}] ({h.chunk.filename}, page {h.chunk.page})\n{h.chunk.text}" for i, h in enumerate(hits, start=1)
    )


def generate(question: str, hits: list[SearchHit], embedder) -> tuple[str, str]:
    """Returns (answer, mode)."""
    if not hits:
        return "I could not find anything relevant to that question in the uploaded documents.", "none"
    if settings.openai_api_key:
        return _llm_answer(question, hits), "llm"
    return _extractive_answer(question, hits, embedder), "extractive"


def _llm_answer(question: str, hits: list[SearchHit]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=settings.openai_api_key)
    response = client.chat.completions.create(
        model=settings.openai_model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Excerpts:\n{build_context(hits)}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content.strip()


def _extractive_answer(question: str, hits: list[SearchHit], embedder, max_sentences: int = 3) -> str:
    """Pick the sentences from retrieved chunks that are closest to the question."""
    candidates: list[tuple[str, int]] = []
    seen: set[str] = set()
    for i, hit in enumerate(hits, start=1):
        # Markdown headings help retrieval but read badly inside an answer.
        body = _HEADING.sub("", hit.chunk.text)
        for sentence in split_sentences(body):
            # Overlapping chunks repeat sentences; keep the first occurrence.
            if len(sentence) > 25 and sentence not in seen:
                seen.add(sentence)
                candidates.append((sentence, i))
    if not candidates:
        return hits[0].chunk.text
    vectors = embedder.encode([question] + [s for s, _ in candidates])
    scores = vectors[1:] @ vectors[0]
    top = np.argsort(-scores)[:max_sentences]
    # Keep only sentences reasonably close to the best match.
    best = sorted(j for j in top if scores[j] >= 0.85 * scores[top[0]])
    return " ".join(f"{_clean(candidates[j][0])} [{candidates[j][1]}]" for j in best)


def _clean(sentence: str) -> str:
    return re.sub(r"\s+", " ", sentence).strip()
