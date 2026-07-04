from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.services.retrieval import embeddings


class _FakeGeminiModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] * 2 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] * 2


class _FakeFallbackModel:
    def encode(self, texts: list[str]):
        rows = [[float(len(text))] * 2 for text in texts]

        class _EncodedRows:
            def __init__(self, values: list[list[float]]):
                self._values = values

            def tolist(self) -> list[list[float]]:
                return self._values

            def __getitem__(self, index: int):
                return _EncodedRow(self._values[index])

        class _EncodedRow:
            def __init__(self, values: list[float]):
                self._values = values

            def tolist(self) -> list[float]:
                return self._values

        return _EncodedRows(rows)


@pytest.fixture(autouse=True)
def reset_embedding_state() -> None:
    embeddings._active_model = None
    embeddings._model_type = None


def test_embedding_initializes_with_gemini(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "_probe_gemini", lambda: _FakeGeminiModel())
    monkeypatch.setattr(embeddings, "_load_fallback", lambda: _FakeFallbackModel())

    embeddings._init()

    assert embeddings._model_type == "gemini"
    assert embeddings.get_embedding_dim() == 3072
    assert embeddings.embed_texts(["alpha"]) == [[5.0, 5.0]]


def test_embedding_fallback_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "_probe_gemini", lambda: None)
    monkeypatch.setattr(embeddings, "_load_fallback", lambda: _FakeFallbackModel())

    embeddings._init()

    assert embeddings._model_type == "fallback"
    assert embeddings.get_embedding_dim() == 768
    assert embeddings.embed_texts(["beta"]) == [[4.0, 4.0]]
    assert embeddings.embed_query("beta") == [4.0, 4.0]


def test_embed_texts_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "_probe_gemini", lambda: _FakeGeminiModel())
    monkeypatch.setattr(embeddings, "_load_fallback", lambda: _FakeFallbackModel())

    assert embeddings.embed_texts([]) == []