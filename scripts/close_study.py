#!/usr/bin/env python
# Stamp study_lifecycle timestamps. phase1/phase2 set study_closed_at;
# followup sets followup_concluded_at. Refuses to overwrite non-null stamps
# unless --force is passed. Updates only — never inserts.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect  # noqa: E402

log = get_logger("close_study")

_COLUMN_FOR_PHASE = {
    "phase1": "study_closed_at",
    "phase2": "study_closed_at",
    "followup": "followup_concluded_at",
}


def _fetch_row(cur) -> Optional[tuple]:
    cur.execute("SELECT study_closed_at, followup_concluded_at FROM study_lifecycle WHERE id = 1")
    return cur.fetchone()


def _stamp(phase: str, force: bool, dry_run: bool) -> int:
    column = _COLUMN_FOR_PHASE[phase]
    with pg_connect() as conn, conn.cursor() as cur:
        row = _fetch_row(cur)
        if row is None:
            print(
                "ERROR: study_lifecycle singleton row is missing. Re-run migration 0001_schema.sql.",
                file=sys.stderr,
            )
            return 1

        study_closed_at, followup_concluded_at = row
        current = study_closed_at if column == "study_closed_at" else followup_concluded_at
        print(f"{column} (before): {current}")

        if current is not None and not force:
            print(
                f"ERROR: {column} already set to {current}. Pass --force to overwrite.",
                file=sys.stderr,
            )
            return 1
        if current is not None and force:
            print(f"WARNING: overwriting existing {column} ({current}) because --force was passed.")

        if dry_run:
            print(f"DRY-RUN: would UPDATE study_lifecycle SET {column} = now(), updated_at = now()")
            return 0

        cur.execute(
            f"UPDATE study_lifecycle SET {column} = now(), updated_at = now() WHERE id = 1"
        )
        conn.commit()
        row = _fetch_row(cur)
        new_value = row[0] if column == "study_closed_at" else row[1]
        print(f"{column} (after):  {new_value}")
        log.info("stamped %s", column)
        return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Stamp a study_lifecycle timestamp.")
    parser.add_argument("--phase", required=True, choices=sorted(_COLUMN_FOR_PHASE.keys()))
    parser.add_argument("--force", action="store_true", help="Overwrite an existing non-null stamp.")
    parser.add_argument("--dry-run", action="store_true", help="Print intended change without committing.")
    args = parser.parse_args(argv)
    return _stamp(args.phase, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
