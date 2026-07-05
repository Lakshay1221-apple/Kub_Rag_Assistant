"""Application observability helpers."""

from __future__ import annotations

import os
from pathlib import Path

import logfire

LOGFIRE_BASE_URL = "https://logfire-us.pydantic.dev"
_LOGFIRE_AUTH_FILE = Path.home() / ".logfire" / "default.toml"
_CONFIGURED = False


def configure_logfire(service_name: str, service_version: str | None = None) -> None:
    """Configure Logfire for application entrypoints.

    Local development uses the CLI auth cache when available. If credentials are
    missing, Logfire is disabled so imports and tests remain safe.
    """

    global _CONFIGURED

    if _CONFIGURED:
        return

    should_send = bool(os.getenv("LOGFIRE_TOKEN")) or _LOGFIRE_AUTH_FILE.exists()

    try:
        if should_send:
            configure_kwargs: dict[str, object] = {
                "service_name": service_name,
                "advanced": logfire.AdvancedOptions(base_url=LOGFIRE_BASE_URL),
            }
            if service_version:
                configure_kwargs["service_version"] = service_version
            logfire.configure(**configure_kwargs)
        else:
            logfire.configure(send_to_logfire=False)
    except Exception:
        logfire.configure(send_to_logfire=False)

    _CONFIGURED = True


