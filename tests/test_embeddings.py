from __future__ import annotations

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


class _QuotaGeminiModel:
    def __init__(self) -> None:
        self.calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        raise RuntimeError("429 RESOURCE_EXHAUSTED")

    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("429 RESOURCE_EXHAUSTED")


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
    fallback_vectors = embeddings.embed_texts(["beta"])
    assert len(fallback_vectors) == 1
    assert len(fallback_vectors[0]) == 768
    assert fallback_vectors[0][:2] == [4.0, 4.0]
    query_vector = embeddings.embed_query("beta")
    assert len(query_vector) == 768
    assert query_vector[:2] == [4.0, 4.0]


def test_embed_texts_empty_input(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(embeddings, "_probe_gemini", lambda: _FakeGeminiModel())
    monkeypatch.setattr(embeddings, "_load_fallback", lambda: _FakeFallbackModel())

    assert embeddings.embed_texts([]) == []


def test_embedding_falls_back_on_quota_error(monkeypatch: pytest.MonkeyPatch) -> None:
    quota_model = _QuotaGeminiModel()

    monkeypatch.setattr(embeddings, "_probe_gemini", lambda: quota_model)
    monkeypatch.setattr(embeddings, "_load_fallback", lambda: _FakeFallbackModel())

    embeddings._init()

    vectors = embeddings.embed_texts(["alpha"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 3072
    assert vectors[0][:2] == [5.0, 5.0]
    assert embeddings._model_type == "fallback"
    assert embeddings.get_embedding_dim() == 3072