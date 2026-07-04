from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import app.ingestion.processor as processor


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.upserts = []
        self.created_collections = []
        self.collection_state = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_state

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)
        self.collection_state = True

    def upsert(self, collection_name: str, points) -> None:
        self.upserts.append((collection_name, points))


@pytest.fixture(autouse=True)
def reset_processor_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_client = _FakeQdrantClient()
    monkeypatch.setattr(processor, "qdrant_client", fake_client)
    monkeypatch.setattr(processor, "PROCESSED_DATA_DIR", str(tmp_path / "processed"))
    monkeypatch.setattr(processor, "save_processed_locally", lambda data, source_type, filename: str(tmp_path / f"{filename}.json"))
    monkeypatch.setattr(processor, "embed_texts", lambda chunks: [[0.1, 0.2] for _ in chunks])
    monkeypatch.setattr(processor, "chunk_text", lambda text: [text.strip()] if text.strip() else [])
    monkeypatch.setattr(processor, "parse_text", lambda file_path: "processor text")
    monkeypatch.setattr(processor, "parse_html", lambda file_path: "processor html")
    monkeypatch.setattr(processor, "parse_pdf", lambda file_path: "processor pdf")
    monkeypatch.setattr(processor, "parse_office", lambda file_path: "processor office")


def test_process_file_normal_case(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content", encoding="utf-8")

    processor.process_file(str(file_path), file_path.name, "true")

    assert processor.qdrant_client.upserts
    collection_name, points = processor.qdrant_client.upserts[0]
    assert collection_name == processor.settings.QDRANT_COLLECTION_NAME
    assert len(points) == 1
    assert points[0].payload["filename"] == "sample.txt"


def test_process_file_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.exe"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError):
        processor.process_file(str(file_path), file_path.name, "true")


def test_process_file_no_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(processor, "chunk_text", lambda text: [])

    with pytest.raises(ValueError):
        processor.process_file(str(file_path), file_path.name, "true")


def test_run_universal_ingestion_creates_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    (base_dir / "sample.txt").write_text("content", encoding="utf-8")

    calls = []

    def fake_process_directory(dir_path: str, source_type: str) -> None:
        calls.append((dir_path, source_type))

    monkeypatch.setattr(processor, "process_directory", fake_process_directory)
    monkeypatch.setattr(processor, "get_embedding_dim", lambda: 2)
    processor.qdrant_client.collection_state = False

    processor.run_universal_ingestion(str(base_dir), explicit_source_type=None, wipe=False)

    assert processor.qdrant_client.created_collections
    assert calls == [(str(base_dir), "general")]