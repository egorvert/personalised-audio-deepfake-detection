from __future__ import annotations

import logging
import re
from typing import Iterable

# Strict UUID format (any version). We don't restrict to v4 because participant
# IDs may come from sources other than the browser's crypto.randomUUID().
UUID_RE = re.compile(
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)

# Pragmatic email matcher — good enough to scrub addresses from log lines.
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def scrub(text: str) -> str:
    return EMAIL_RE.sub("<email>", UUID_RE.sub("<uuid>", text))


class PiiScrubbingFormatter(logging.Formatter):
    # Scrub the rendered message so UUIDs in formatted paths/JSON also get masked.
    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


_DEFAULT_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access", "vdetect")


def install(loggers: Iterable[str] = _DEFAULT_LOGGERS) -> None:
    formatter = PiiScrubbingFormatter(fmt="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in loggers:
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        else:
            for handler in logger.handlers:
                handler.setFormatter(formatter)


# Install on import — any later `from vdetect import logging as _` hardens descendants too.
install()
