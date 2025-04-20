import pytest

from app.chunker import chunk_pages, split_sentences
from app.parsers import Page


def test_split_sentences():
    assert split_sentences("One. Two!  Three?\n\nFour") == ["One.", "Two!", "Three?", "Four"]


def test_chunks_respect_size_and_keep_page_numbers():
    text = " ".join(f"Sentence number {i} is here." for i in range(200))
    chunks = chunk_pages([Page(3, text)], size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert all(c.page == 3 for c in chunks)


def test_chunks_overlap():
    text = " ".join(f"Fact {i} matters." for i in range(100))
    chunks = chunk_pages([Page(1, text)], size=120, overlap=30)
    for a, b in zip(chunks, chunks[1:]):
        assert a.text[-15:] in b.text


def test_long_sentence_is_hard_split():
    chunks = chunk_pages([Page(1, "x" * 1000)], size=300, overlap=50)
    assert all(len(c.text) <= 300 for c in chunks)
    assert len(chunks) >= 4


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        chunk_pages([Page(1, "text")], size=100, overlap=100)
