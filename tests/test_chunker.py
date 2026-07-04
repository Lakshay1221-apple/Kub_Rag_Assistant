from __future__ import annotations

import pytest

from app.ingestion.chucking.splitter import chunk_text


def test_chunk_text_normal_case() -> None:
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."

    chunks = chunk_text(text, chunk_size=30)

    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_empty_input() -> None:
    assert chunk_text("") == []


def test_chunk_text_invalid_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_text("text", chunk_size=0)