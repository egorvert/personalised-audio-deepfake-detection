from __future__ import annotations

import os

import uvicorn

# Side effect: installs the PII-scrubbing formatter on uvicorn loggers.
from vdetect import logging as _vdetect_logging  # noqa: F401


def main() -> None:
    host = os.environ.get("VDETECT_HOST", "127.0.0.1")
    port = int(os.environ.get("VDETECT_PORT", "8000"))
    # Caddy logs every request in production, so uvicorn's access log is off by default.
    access_log = os.environ.get("VDETECT_ACCESS_LOG", "0").lower() in ("1", "true", "yes")
    uvicorn.run("vdetect.api:app", host=host, port=port, access_log=access_log)


if __name__ == "__main__":
    main()
