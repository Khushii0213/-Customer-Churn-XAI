"""
logger.py
---------
Provides a single, reusable logger factory for the entire project.

Why this file exists:
    Every module (EDA, preprocessing, training, explainability, the Streamlit
    app) needs to report progress and errors. Rather than calling
    `print()` everywhere -- which is hard to filter, silence, or redirect
    to a file in production -- we configure one standard logger here and
    import `get_logger()` wherever logging is needed.
"""

import logging
import sys
from pathlib import Path


def get_logger(name: str, log_to_file: bool = True) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Args:
        name: Usually `__name__` of the calling module, so log lines show
            their origin.
        log_to_file: If True, also append logs to `logs/project.log` in
            addition to streaming to the console.

    Returns:
        A configured `logging.Logger` instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        # Logger already configured (avoids duplicate handlers on re-import)
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_to_file:
        log_dir = Path(__file__).resolve().parent.parent.parent / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "project.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.propagate = False
    return logger
