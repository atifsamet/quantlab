"""Logging helpers. Never log API keys, secrets, or passphrases."""

from __future__ import annotations

import logging
import sys

_SENSITIVE_KEYS = ("api_key", "secret", "passphrase", "password", "token")


class RedactFilter(logging.Filter):
    """Best-effort filter to redact known secret field names from log records."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage().lower()
        for key in _SENSITIVE_KEYS:
            if key in message and ("=" in message or ":" in message):
                record.msg = "[REDACTED — possible secret in log message]"
                record.args = ()
                break
        return True


def setup_logging(level: str = "INFO") -> None:
    """Configure root logging once for the process."""
    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    handler.addFilter(RedactFilter())

    root.setLevel(level.upper())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger."""
    return logging.getLogger(name)
