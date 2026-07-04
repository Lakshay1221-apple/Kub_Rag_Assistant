from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestion.processor import run_universal_ingestion

run_universal_ingestion(
    base_dir="DATA",
    wipe=True
)