from __future__ import annotations

import logging
import pathlib


_LOGGER_NAME = "die_vfm"


def configure_logging(log_dir: pathlib.Path) -> logging.Logger:
    """Configures and returns the project logger.

    Args:
      log_dir: Directory where the run log file will be written.

    Returns:
      A configured logger instance.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger