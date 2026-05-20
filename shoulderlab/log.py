"""Common ShoulderLab logging helpers."""

from __future__ import annotations

import logging
import sys
from typing import Optional, TextIO


LOGGER_NAME = "ShoulderLab"
LOG_FORMAT = "[ShoulderLab][%(asctime)s][%(levelname)s] %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
RESET = "\033[0m"
CYAN = "\033[36m"
COLORS = {
    logging.DEBUG: "\033[36m",
    logging.INFO: "\033[32m",
    logging.WARNING: "\033[33m",
    logging.ERROR: "\033[31m",
    logging.CRITICAL: "\033[1;31m",
}


class ShoulderLabFormatter(logging.Formatter):
    """Colorize ShoulderLab log metadata when writing to a terminal."""

    def __init__(
        self,
        fmt: str = LOG_FORMAT,
        datefmt: str = LOG_DATE_FORMAT,
        use_color: bool = True,
    ) -> None:
        super().__init__(fmt, datefmt=datefmt)
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        if record.stack_info:
            message = f"{message}\n{self.formatStack(record.stack_info)}"

        timestamp = self.formatTime(record, self.datefmt)
        if not self.use_color:
            return f"[{LOGGER_NAME}][{timestamp}][{record.levelname}] {message}"

        level_color = COLORS.get(record.levelno, "")
        project = f"{CYAN}{LOGGER_NAME}{RESET}"
        time_text = f"{CYAN}{timestamp}{RESET}"
        level = f"{level_color}{record.levelname}{RESET}" if level_color else record.levelname
        return f"[{project}][{time_text}][{level}] {message}"


def configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project logger."""
    stream = sys.stdout
    formatter = _make_formatter(stream)
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(stream)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Return the project logger, configuring it on first use."""
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        configure_logging()
    return logger


def _make_formatter(stream: Optional[TextIO]) -> ShoulderLabFormatter:
    use_color = bool(stream and hasattr(stream, "isatty") and stream.isatty())
    return ShoulderLabFormatter(use_color=use_color)
