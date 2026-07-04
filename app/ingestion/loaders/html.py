"""HTML document loader."""

from __future__ import annotations

from pathlib import Path

import logfire
from bs4 import BeautifulSoup
from loguru import logger

__all__ = ["parse_html"]


def _validate_file_path(file_path: str) -> Path:
	"""Validate that the provided path exists and points to a file."""

	path = Path(file_path)
	if not path.exists():
		logger.error("HTML file does not exist: {}", file_path)
		raise FileNotFoundError(f"HTML file does not exist: {file_path}")
	if not path.is_file():
		logger.error("HTML path is not a file: {}", file_path)
		raise IsADirectoryError(f"HTML path is not a file: {file_path}")
	return path


def _normalize_html_text(text: str) -> str:
	"""Normalize whitespace while preserving paragraph structure."""

	lines = (line.strip() for line in text.splitlines())
	chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
	return "\n".join(chunk for chunk in chunks if chunk)


def parse_html(file_path: str) -> str:
	"""Parse an HTML document and return cleaned text.

	The parser removes script, style, metadata, and noscript tags before
	extracting meaningful text content.
	"""

	logger.info("Starting HTML parse: {}", file_path)
	try:
		path = _validate_file_path(file_path)
		with path.open("r", encoding="utf-8") as file_handle:
			content = file_handle.read()

		soup = BeautifulSoup(content, "html.parser")
		for tag in soup(["script", "style", "meta", "noscript"]):
			tag.decompose()

		text = soup.get_text(separator="\n")
		clean_text = _normalize_html_text(text)

		logger.info("Completed HTML parse: {} characters extracted", len(clean_text))
		return clean_text
	except Exception as exc:
		logfire.error(f"HTML parse failed: {exc}")
		logger.exception("HTML parse failed for {}", file_path)
		raise
