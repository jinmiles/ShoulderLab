"""Common ShoulderLab logging helpers."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
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
ANSI_PATTERN = re.compile(r"\x1b\[[0-9;]*m")
_TEE_FILE: Optional[TextIO] = None
_TEE_PATH: Optional[Path] = None
_ORIGINAL_STDOUT: TextIO = sys.stdout
_ORIGINAL_STDERR: TextIO = sys.stderr


class TeeStream:
    """Mirror writes to the terminal and a plain-text log file."""

    def __init__(self, stream: TextIO, log_file: TextIO) -> None:
        self.stream = stream
        self.log_file = log_file

    def write(self, text: str) -> int:
        written = self.stream.write(text)
        self.log_file.write(ANSI_PATTERN.sub("", text))
        return written

    def flush(self) -> None:
        self.stream.flush()
        self.log_file.flush()

    def isatty(self) -> bool:
        return self.stream.isatty()

    @property
    def encoding(self) -> Optional[str]:
        return getattr(self.stream, "encoding", None)

    @property
    def errors(self) -> Optional[str]:
        return getattr(self.stream, "errors", None)


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


def configure_logging(level: int = logging.INFO, log_path: Optional[Path] = None) -> logging.Logger:
    """Configure and return the project logger."""
    if log_path is not None:
        _configure_tee(log_path)
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


def _configure_tee(log_path: Path) -> None:
    global _TEE_FILE, _TEE_PATH

    log_path = log_path.resolve()
    if _TEE_PATH == log_path:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if _TEE_FILE is not None:
        _TEE_FILE.close()
    _TEE_FILE = log_path.open("a", encoding="utf-8", buffering=1)
    _TEE_PATH = log_path
    sys.stdout = TeeStream(_ORIGINAL_STDOUT, _TEE_FILE)  # type: ignore[assignment]
    sys.stderr = TeeStream(_ORIGINAL_STDERR, _TEE_FILE)  # type: ignore[assignment]
