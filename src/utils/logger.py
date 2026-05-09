"""
src/utils/logger.py
Centralised logging setup for Finnie.

Usage:
    from src.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("something happened")

Logs appear in the terminal where `uv run streamlit run` was launched.
Set LOG_LEVEL=DEBUG in .env for verbose output.
"""

import logging
import os
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module name."""
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured

    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logger.setLevel(getattr(logging, level, logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger
