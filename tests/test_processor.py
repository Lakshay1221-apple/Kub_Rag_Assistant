from __future__ import annotations

from pathlib import Path

import pytest

import app.ingestion.processor as processor


class _FakeQdrantClient:
    def __init__(self) -> None:
        self.upserts = []
        self.created_collections = []
        self.deleted_collections = []
        self.collection_state = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.collection_state

    def create_collection(self, **kwargs) -> None:
        self.created_collections.append(kwargs)
        self.collection_state = True

    def delete_collection(self, collection_name: str) -> None:
        self.deleted_collections.append(collection_name)
        self.collection_state = False

    def upsert(self, collection_name: str, points) -> None:
        self.upserts.append((collection_name, points))


@pytest.fixture(autouse=True)
def reset_processor_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake_client = _FakeQdrantClient()
    monkeypatch.setattr(processor, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(processor, "PROCESSED_DATA_DIR", str(tmp_path / "processed"))
    monkeypatch.setattr(processor, "embed_texts", lambda chunks: [[0.1, 0.2] for _ in chunks])
    monkeypatch.setattr(processor, "chunk_text", lambda text: [text.strip()] if text.strip() else [])
    monkeypatch.setattr(processor, "parse_text", lambda file_path: "processor text")
    monkeypatch.setattr(processor, "parse_html", lambda file_path: "processor html")
    monkeypatch.setattr(processor, "parse_pdf", lambda file_path: "processor pdf")
    monkeypatch.setattr(processor, "parse_office", lambda file_path: "processor office")


def test_process_file_normal_case(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content", encoding="utf-8")

    processor.process_file(str(file_path), file_path.name, "true", qdrant_client=processor.get_qdrant_client())

    client = processor.get_qdrant_client()
    assert client.upserts
    collection_name, points = client.upserts[0]
    assert collection_name == processor.settings.QDRANT_COLLECTION_NAME
    assert len(points) == 1
    assert points[0].payload["filename"] == "sample.txt"


def test_save_processed_locally_strips_extension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(processor, "PROCESSED_DATA_DIR", str(tmp_path))

    output_path = processor.save_processed_locally({"chunks": ["a"]}, "true", "document.pdf")

    assert output_path.endswith("document.json")
    assert Path(output_path).exists()


def test_process_file_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.exe"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError):
        processor.process_file(str(file_path), file_path.name, "true", qdrant_client=processor.get_qdrant_client())


def test_process_file_handles_multi_dot_filename(tmp_path: Path) -> None:
    file_path = tmp_path / "report.v1.pdf"
    file_path.write_text("content", encoding="utf-8")

    processor.process_file(str(file_path), file_path.name, "true", qdrant_client=processor.get_qdrant_client())

    client = processor.get_qdrant_client()
    assert client.upserts


def test_process_file_no_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("content", encoding="utf-8")
    monkeypatch.setattr(processor, "chunk_text", lambda text: [])

    with pytest.raises(ValueError):
        processor.process_file(str(file_path), file_path.name, "true", qdrant_client=processor.get_qdrant_client())


def test_run_universal_ingestion_creates_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()
    (base_dir / "sample.txt").write_text("content", encoding="utf-8")

    calls = []

    def fake_process_directory(dir_path: str, source_type: str, qdrant_client=None) -> None:
        calls.append((dir_path, source_type))

    monkeypatch.setattr(processor, "process_directory", fake_process_directory)
    monkeypatch.setattr(processor, "get_embedding_dim", lambda: 2)

    processor.run_universal_ingestion(str(base_dir), explicit_source_type=None, wipe=False)

    client = processor.get_qdrant_client()
    assert client.created_collections
    assert calls == [(str(base_dir), "general")]


def test_run_universal_ingestion_wipes_existing_collection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base_dir = tmp_path / "data"
    base_dir.mkdir()

    fake_client = processor.get_qdrant_client()
    fake_client.collection_state = True
    monkeypatch.setattr(processor, "get_qdrant_client", lambda: fake_client)
    monkeypatch.setattr(processor, "get_embedding_dim", lambda: 2)
    monkeypatch.setattr(processor, "process_directory", lambda dir_path, source_type, qdrant_client=None: None)

    processor.run_universal_ingestion(str(base_dir), explicit_source_type=None, wipe=True)

    assert fake_client.deleted_collections == [processor.settings.QDRANT_COLLECTION_NAME]


def test_process_directory_continues_on_file_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dir_path = tmp_path / "data"
    dir_path.mkdir()
    file_bad = dir_path / "bad.txt"
    file_bad.write_text("bad file content", encoding="utf-8")
    file_good = dir_path / "good.txt"
    file_good.write_text("good file content", encoding="utf-8")

    processed_files = []

    def mock_process_file(file_path: str, filename: str, source_type: str, qdrant_client=None) -> None:
        processed_files.append(filename)
        if filename == "bad.txt":
            raise ValueError("No chunks created for file bad.txt")

    monkeypatch.setattr(processor, "process_file", mock_process_file)

    processor.process_directory(str(dir_path), "test_source")

    assert "bad.txt" in processed_files
    assert "good.txt" in processed_files