"""PDF document loader."""

from __future__ import annotations

from pathlib import Path

import logfire
from loguru import logger
from pypdf import PdfReader

__all__ = ["parse_pdf"]


def _validate_file_path(file_path: str) -> Path:
	"""Validate that the provided path exists and points to a file."""

	path = Path(file_path)
	if not path.exists():
		logger.error("PDF file does not exist: {}", file_path)
		raise FileNotFoundError(f"PDF file does not exist: {file_path}")
	if not path.is_file():
		logger.error("PDF path is not a file: {}", file_path)
		raise IsADirectoryError(f"PDF path is not a file: {file_path}")
	return path


def _normalize_text(text: str) -> str:
	"""Normalize whitespace in extracted text."""

	return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def parse_pdf(file_path: str) -> str:
	"""Parse a PDF document page by page and return cleaned text."""

	logger.info("Starting PDF parse: {}", file_path)
	try:
		path = _validate_file_path(file_path)
		reader = PdfReader(str(path))
		logger.info("PDF page count for {}: {}", file_path, len(reader.pages))

		pages: list[str] = []
		for page_index, page in enumerate(reader.pages, start=1):
			try:
				page_text = page.extract_text() or ""
			except Exception as exc:
				logger.warning(
					"Skipping page {} in {} due to extraction error: {}",
					page_index,
					file_path,
					exc,
				)
				continue

			if not page_text.strip():
				logger.debug("Skipping empty page {} in {}", page_index, file_path)
				continue

			pages.append(_normalize_text(page_text))
			logger.debug("Extracted text from page {} in {}", page_index, file_path)

		clean_text = "\n\n".join(page for page in pages if page.strip())
		logger.info("Completed PDF parse: {} characters extracted", len(clean_text))
		return clean_text
	except Exception as exc:
		logfire.error(f"PDF parse failed: {exc}")
		logger.exception("PDF parse failed for {}", file_path)
		raise
