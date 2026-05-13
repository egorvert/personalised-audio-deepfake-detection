from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv

# Side-effect import: configures the PII-scrubbing formatter on uvicorn + vdetect loggers.
import vdetect.logging  # noqa: F401

# Pick up Supabase local-dev keys from webapp/.env.local if present.
_WEBAPP_ENV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "webapp", ".env.local")
if os.path.exists(_WEBAPP_ENV):
    load_dotenv(_WEBAPP_ENV)
load_dotenv()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(f"vdetect.scripts.{name}")
    logger.setLevel(logging.INFO)
    return logger


def require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        print(f"ERROR: environment variable {name} is required.", file=sys.stderr)
        sys.exit(2)
    return value


def supabase_client():
    # Lazy import so psycopg-only scripts don't pay the supabase-py import cost.
    from supabase import create_client
    return create_client(require_env("SUPABASE_URL"), require_env("SUPABASE_SERVICE_ROLE_KEY"))


def postgres_dsn() -> str:
    dsn = os.environ.get("SUPABASE_DB_URL")
    if dsn:
        return dsn
    # Default DSN for the Supabase local stack.
    return (
        f"postgresql://postgres:postgres@"
        f"{os.environ.get('SUPABASE_DB_HOST', '127.0.0.1')}:"
        f"{os.environ.get('SUPABASE_DB_PORT', '54322')}/postgres"
    )


def pg_connect():
    import psycopg
    return psycopg.connect(postgres_dsn(), autocommit=False)


def enrollments_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "assets", "enrollments")
    os.makedirs(path, exist_ok=True)
    return path


def prototypes_json_path() -> str:
    return os.environ.get("VDETECT_DB_PATH") or os.path.join(enrollments_dir(), "prototypes.json")
