"""Temporary end-to-end validation script for the ingestion pipeline."""

from __future__ import annotations

import os
import tempfile
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _FakeGeminiModel:
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text))] * 3 for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text))] * 3


class _FakeFallbackModel:
    def encode(self, texts: list[str]):
        rows = [[float(len(text))] * 3 for text in texts]

        class _EncodedRows:
            def __init__(self, values: list[list[float]]):
                self._values = values

            def __getitem__(self, index: int):
                return _EncodedRow(self._values[index])

            def tolist(self) -> list[list[float]]:
                return self._values

        class _EncodedRow:
            def __init__(self, values: list[float]):
                self._values = values

            def tolist(self) -> list[float]:
                return self._values

        return _EncodedRows(rows)


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.created = False
        self.points = []

    def collection_exists(self, collection_name: str) -> bool:
        return self.created

    def create_collection(self, **kwargs) -> None:
        self.created = True

    def upsert(self, collection_name: str, points) -> None:
        self.points.extend(points)


def _print_success(message: str) -> None:
    print(f"\u2713 {message}")


def _print_failure(message: str) -> None:
    print(f"\u2717 {message}")


def main() -> int:
    load_dotenv()

    try:
        from app import config
        from app.ingestion.chucking.splitter import chunk_text
        from app.ingestion.processor import process_file
        from app.ingestion.loaders.html import parse_html
        from app.ingestion.loaders.office import parse_office
        from app.ingestion.loaders.pdf import parse_pdf
        from app.ingestion.loaders.text import parse_text
        from app.services.retrieval import embeddings
        from app.observability import configure_logfire
    except Exception as exc:
        _print_failure(f"Import validation failed: {exc}")
        return 1

    configure_logfire(service_name="ingestion_processor", service_version="1.0.0")

    _print_success("Loaders imported")
    _print_success("Processor imported")
    _print_success("Chunker imported")

    try:
        print(f"Loaded QDRANT_COLLECTION_NAME={config.settings.QDRANT_COLLECTION_NAME!r}")
        print(f"Loaded QDRANT_URL={getattr(config.settings, 'QDRANT_URL', None)!r}")
        print(f"Loaded GEMINI_API_KEY={'set' if config.settings.GEMINI_API_KEY else 'missing'}")
        _print_success("Environment variables loaded")
    except Exception as exc:
        _print_failure(f"Environment loading failed: {exc}")
        return 1

    try:
        embeddings._active_model = None
        embeddings._model_type = None
        embeddings._probe_gemini = lambda: _FakeGeminiModel()
        embeddings._load_fallback = lambda: _FakeFallbackModel()
        embeddings._init()
        embeddings.embed_texts(["alpha", "beta"])
        _print_success("Embedding model initialized")
    except Exception as exc:
        _print_failure(f"Embedding model initialization failed: {exc}")
        return 1

    try:
        chunk_text("Paragraph one.\n\nParagraph two.")
        _print_success("Chunker runs")
    except Exception as exc:
        _print_failure(f"Chunker failed: {exc}")
        return 1

    try:
        fake_client = _FakeQdrantClient()
        import app.ingestion.processor as processor

        processor.get_qdrant_client = lambda: fake_client
        processor.embed_texts = lambda chunks: [[0.1, 0.2, 0.3] for _ in chunks]
        processor.chunk_text = lambda text: [text.strip()] if text.strip() else []
        processor.PROCESSED_DATA_DIR = tempfile.mkdtemp(prefix="pipeline-smoke-")

        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text("hello pipeline", encoding="utf-8")
            process_file(str(file_path), file_path.name, "smoke")

        _print_success("Processor runs")
    except Exception as exc:
        _print_failure(f"Processor failed: {exc}")
        return 1

    print("Pipeline validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())