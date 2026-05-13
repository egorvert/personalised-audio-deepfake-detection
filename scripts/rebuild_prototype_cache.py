#!/usr/bin/env python
# Rebuild assets/enrollments/prototypes.json from the prototype_embeddings
# table (the authoritative store). Writes via a flock'd .lock file + atomic
# rename so concurrent readers never see a half-written file. Importable as
# rebuild() — used directly by enrollment_worker.py and withdraw_participant.py.

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "scripts"))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect, prototypes_json_path  # noqa: E402

log = get_logger("rebuild_prototype_cache")


def _fetch_embeddings() -> List[Dict[str, Any]]:
    sql = """
        SELECT participant_id,
               embedding,
               source_recording_ids,
               computed_at,
               model_version
          FROM prototype_embeddings
         ORDER BY computed_at ASC
    """
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
        return [
            {
                "participant_id": str(r[0]),
                "embedding": list(r[1]),
                "source_recording_ids": [str(x) for x in (r[2] or [])],
                "computed_at": r[3].isoformat() if r[3] else None,
                "model_version": r[4],
            }
            for r in rows
        ]


def _assemble_db(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not rows:
        return {"schema_version": 1, "embedding_dim": 0, "speakers": {}}

    dim = len(rows[0]["embedding"])
    speakers: Dict[str, Any] = {}
    for r in rows:
        if len(r["embedding"]) != dim:
            raise ValueError(
                f"embedding dimension mismatch for participant (expected {dim}, "
                f"got {len(r['embedding'])})"
            )
        speakers[r["participant_id"]] = {
            "embedding": r["embedding"],
            "num_samples": len(r["source_recording_ids"]),
            "sample_paths": r["source_recording_ids"],
            "created_at": r["computed_at"],
            "normalized": True,
        }
    return {"schema_version": 1, "embedding_dim": dim, "speakers": speakers}


def rebuild(target_path: Optional[str] = None) -> int:
    """Rebuild the on-disk cache from the authoritative table.

    The cross-process advisory lock covers the ENTIRE read-modify-write —
    fetch, assemble, write, replace — so two concurrent rebuilders serialise
    at the DB-read boundary rather than racing and letting the later
    ``os.replace`` clobber a newer snapshot with a stale one.

    Importable from other scripts (enrollment_worker, withdraw_participant).
    """
    target = Path(target_path or prototypes_json_path())
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = target.with_suffix(target.suffix + ".lock")
    tmp = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")

    lock_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        # DB read happens INSIDE the flock so a concurrent writer that just
        # finished committing cannot have its snapshot overwritten by ours.
        rows = _fetch_embeddings()
        payload = _assemble_db(rows)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, target)
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)

    count = len(payload["speakers"])
    log.info("rebuilt %s with %d participants (embedding_dim=%d)",
             target, count, payload["embedding_dim"])
    return count


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Rebuild assets/enrollments/prototypes.json from prototype_embeddings.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Target JSON path (default: $VDETECT_DB_PATH or assets/enrollments/prototypes.json).",
    )
    args = parser.parse_args(argv)

    count = rebuild(args.out)
    print(f"rebuild_prototype_cache: wrote {count} participant(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
