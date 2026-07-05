from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    from app.ingestion.processor import run_universal_ingestion
    from app.observability import configure_logfire

    configure_logfire(service_name="ingestion_processor", service_version="1.0.0")
    run_universal_ingestion(base_dir="DATA", wipe=True)


if __name__ == "__main__":
    main()