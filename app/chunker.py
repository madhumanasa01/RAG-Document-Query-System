"""Sentence-aware chunking with character overlap."""

import re
from dataclasses import dataclass

from app.parsers import Page

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_HEADING_LINE = re.compile(r"^ ?#+ [^\n]*$", re.MULTILINE)


@dataclass
class Chunk:
    text: str
    page: int
    ocr: bool


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"[ \t]+", " ", text)
    # Isolate markdown headings so they never merge with the next sentence.
    text = _HEADING_LINE.sub(lambda m: f"\n\n{m.group(0).strip()}\n\n", text)
    return [s.strip() for s in _SENTENCE_END.split(text) if s and s.strip()]


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than chunk size")
    chunks: list[Chunk] = []
    for page in pages:
        current = ""
        for sentence in split_sentences(page.text):
            # Hard-split sentences longer than a whole chunk.
            while len(sentence) > size:
                if current:
                    chunks.append(Chunk(current, page.number, page.ocr))
                    current = ""
                chunks.append(Chunk(sentence[:size], page.number, page.ocr))
                sentence = sentence[size - overlap:]
            if current and len(current) + 1 + len(sentence) > size:
                chunks.append(Chunk(current, page.number, page.ocr))
                current = current[-overlap:].lstrip() if overlap else ""
            # Headings stay on their own line so answers can drop them later.
            after_heading = current.rsplit("\n", 1)[-1].startswith("#")
            sep = "\n" if sentence.startswith("#") or after_heading else " "
            current = f"{current}{sep}{sentence}".strip()
        if current:
            chunks.append(Chunk(current, page.number, page.ocr))
    return chunks
