from __future__ import annotations

from pathlib import Path

import pytest

from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.office import parse_office
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.text import parse_text


def test_parse_html_normal_case(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.html"
    file_path.write_text(
        """
        <html>
          <head>
            <meta charset="utf-8" />
            <style>body { color: red; }</style>
            <script>console.log('ignore');</script>
          </head>
          <body>
            <h1>Title</h1>
            <p>First paragraph.</p>
          </body>
        </html>
        """,
        encoding="utf-8",
    )

    text = parse_html(str(file_path))

    assert "Title" in text
    assert "First paragraph." in text
    assert "console.log" not in text


def test_parse_html_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.html"
    file_path.write_text("", encoding="utf-8")

    assert parse_html(str(file_path)) == ""


def test_parse_html_invalid_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_html(str(tmp_path / "missing.html"))


def test_parse_text_normal_case(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    file_path.write_text("Line 1\n\nLine 2", encoding="utf-8")

    assert parse_text(str(file_path)) == "Line 1\nLine 2"


def test_parse_text_empty_file(tmp_path: Path) -> None:
    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    assert parse_text(str(file_path)) == ""


def test_parse_text_invalid_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_text(str(tmp_path / "missing.txt"))


def test_parse_text_unsupported_file_type(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.bin"
    file_path.write_text("content", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_text(str(file_path))


class _FakePdfPage:
    def __init__(self, text: str | None) -> None:
        self._text = text

    def extract_text(self) -> str | None:
        return self._text


class _FakePdfReader:
    def __init__(self, pages: list[_FakePdfPage]) -> None:
        self.pages = pages


def test_parse_pdf_normal_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pdf"
    file_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.loaders.pdf.PdfReader",
        lambda _: _FakePdfReader([
            _FakePdfPage("Page one."),
            _FakePdfPage(""),
            _FakePdfPage("Page two."),
        ]),
    )

    text = parse_pdf(str(file_path))

    assert "Page one." in text
    assert "Page two." in text


def test_parse_pdf_empty_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.pdf"
    file_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.loaders.pdf.PdfReader",
        lambda _: _FakePdfReader([_FakePdfPage(None)]),
    )

    assert parse_pdf(str(file_path)) == ""


def test_parse_pdf_invalid_path() -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf("/tmp/does-not-exist.pdf")


class _FakeParagraph:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeShape:
    def __init__(self, text: str = "") -> None:
        self.text = text


class _FakeSlide:
    def __init__(self, shapes: list[_FakeShape]) -> None:
        self.shapes = shapes


class _FakeDocument:
    def __init__(self, paragraphs: list[_FakeParagraph]) -> None:
        self.paragraphs = paragraphs


class _FakePresentation:
    def __init__(self, slides: list[_FakeSlide]) -> None:
        self.slides = slides


def test_parse_office_docx_normal_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.docx"
    file_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.loaders.office.Document",
        lambda _: _FakeDocument([
            _FakeParagraph("First paragraph."),
            _FakeParagraph(""),
            _FakeParagraph("Second paragraph."),
        ]),
    )

    text = parse_office(str(file_path))

    assert "First paragraph." in text
    assert "Second paragraph." in text


def test_parse_office_pptx_normal_case(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "sample.pptx"
    file_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr(
        "app.ingestion.loaders.office.Presentation",
        lambda _: _FakePresentation([
            _FakeSlide([_FakeShape("Slide title"), _FakeShape("Slide body")]),
            _FakeSlide([]),
        ]),
    )

    text = parse_office(str(file_path))

    assert "Slide title" in text
    assert "Slide body" in text


def test_parse_office_empty_docx(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    file_path = tmp_path / "empty.docx"
    file_path.write_text("placeholder", encoding="utf-8")
    monkeypatch.setattr("app.ingestion.loaders.office.Document", lambda _: _FakeDocument([]))

    assert parse_office(str(file_path)) == ""


def test_parse_office_invalid_path() -> None:
    with pytest.raises(FileNotFoundError):
        parse_office("/tmp/does-not-exist.docx")


def test_parse_office_unsupported_file_type(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.odt"
    file_path.write_text("placeholder", encoding="utf-8")

    with pytest.raises(ValueError):
        parse_office(str(file_path))