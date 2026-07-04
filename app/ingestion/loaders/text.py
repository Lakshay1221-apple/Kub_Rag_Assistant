"""Plain text loader."""

from __future__ import annotations

from pathlib import Path

import logfire
from loguru import logger

__all__ = ["parse_text"]

SUPPORTED_EXTENSIONS = {".txt", ".md", ".csv", ".json"}


def _validate_file_path(file_path: str) -> Path:
	"""Validate that the provided path exists and points to a file."""

	path = Path(file_path)
	if not path.exists():
		logger.error("Text file does not exist: {}", file_path)
		raise FileNotFoundError(f"Text file does not exist: {file_path}")
	if not path.is_file():
		logger.error("Text path is not a file: {}", file_path)
		raise IsADirectoryError(f"Text path is not a file: {file_path}")
	if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
		logger.error("Unsupported text file type: {}", file_path)
		raise ValueError(f"Unsupported text file type: {file_path}")
	return path


def _normalize_text(text: str) -> str:
	"""Normalize text by collapsing excessive whitespace."""

	return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def parse_text(file_path: str) -> str:
	"""Parse plain text-like files and return cleaned text."""

	logger.info("Starting text parse: {}", file_path)
	try:
		path = _validate_file_path(file_path)
		with path.open("r", encoding="utf-8") as file_handle:
			content = file_handle.read()

		clean_text = _normalize_text(content)
		logger.info(
			"Completed text parse: {} characters read, {} characters returned",
			len(content),
			len(clean_text),
		)
		return clean_text
	except Exception as exc:
		logfire.error(f"Text parse failed: {exc}")
		logger.exception("Text parse failed for {}", file_path)
		raise
