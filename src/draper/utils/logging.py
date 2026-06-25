"""Structured logging setup for Draper.ai."""

from __future__ import annotations

import logging

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", module: str = "draper") -> logging.Logger:
    """Configure and return a logger with rich formatting.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        module: Logger name / module identifier.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(module)
    logger.setLevel(getattr(logging, level.upper()))

    if not logger.handlers:
        handler = RichHandler(
            show_time=True,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            tracebacks_show_locals=True,
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)

    # Suppress noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    return logger


def get_logger(name: str = "draper") -> logging.Logger:
    """Get an existing logger by name. Call setup_logging() first."""
    return logging.getLogger(name)


# Module-level logger for convenience
log = setup_logging()


if __name__ == "__main__":
    log.info("Draper.ai logging initialized")
    log.debug("Debug message (won't show at INFO level)")
    log.warning("Warning message")
