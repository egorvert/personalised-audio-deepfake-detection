#!/usr/bin/env python
# Event-driven retention sweep keyed off the study_lifecycle stamps:
#   study_closed_at + 14d  -> delete Phase 1 audio + deepfakes + prototype embeddings
#   study_closed_at + 180d -> delete Phase 2 sessions + responses + phase2 clips
#   followup_concluded_at + 14d -> delete phase1_emails
# No branch uses raw now() - created_at. No-op while both stamps are NULL.

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect, prototypes_json_path, supabase_client  # noqa: E402
from vdetect.logging import scrub  # noqa: E402

log = get_logger("purge_expired_data")

RECORDINGS_BUCKET = "recordings"
DEEPFAKES_BUCKET = "deepfakes"
PHASE2_BUCKET = "phase2-clips"


def _fetch_lifecycle():
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT
                study_closed_at,
                followup_concluded_at,
                (study_closed_at IS NOT NULL AND now() > study_closed_at + interval '14 days')  AS p1_ready,
                (study_closed_at IS NOT NULL AND now() > study_closed_at + interval '180 days') AS p2_ready,
                (followup_concluded_at IS NOT NULL AND now() > followup_concluded_at + interval '14 days') AS fu_ready
              FROM study_lifecycle WHERE id = 1
            """
        )
        return cur.fetchone()


def _list_storage_keys(sb, bucket: str, prefix: str) -> List[str]:
    """List object keys under `prefix` in `bucket`."""
    try:
        entries = sb.storage.from_(bucket).list(path=prefix)
    except Exception as exc:
        log.error("list(%s/%s) failed: %s", bucket, prefix, scrub(str(exc)))
        return []
    return [f"{prefix}/{e['name']}" for e in entries or []]


def _remove_storage(sb, bucket: str, keys: List[str], dry_run: bool) -> int:
    if not keys:
        return 0
    if dry_run:
        return len(keys)
    try:
        sb.storage.from_(bucket).remove(keys)
    except Exception as exc:
        log.error("remove(%s) failed: %s", bucket, scrub(str(exc)))
        return 0
    return len(keys)


# ---------------------------------------------------------------------------
# Branches
# ---------------------------------------------------------------------------


def _branch_p1(sb, dry_run: bool) -> None:
    """14 days past study_closed_at: delete P1 audio, deepfakes, prototype_embeddings."""
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM participants")
        participant_ids = [r[0] for r in cur.fetchall()]

    recording_blobs = 0
    deepfake_blobs = 0
    for pid in participant_ids:
        prefix = str(pid)
        recording_blobs += _remove_storage(
            sb, RECORDINGS_BUCKET, _list_storage_keys(sb, RECORDINGS_BUCKET, prefix), dry_run,
        )
        deepfake_blobs += _remove_storage(
            sb, DEEPFAKES_BUCKET, _list_storage_keys(sb, DEEPFAKES_BUCKET, prefix), dry_run,
        )

    cache_path = Path(prototypes_json_path())
    if cache_path.exists():
        if dry_run:
            print(f"  [dry-run] would unlink {cache_path}")
        else:
            cache_path.unlink()

    if dry_run:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recordings")
            r = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM deepfakes")
            d = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM prototype_embeddings")
            p = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM participants")
            part = cur.fetchone()[0]
        print(f"  [dry-run] would delete {r} recording(s), {d} deepfake(s), {p} prototype_embedding(s), {part} participant(s)")
    else:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM recordings")
            r = cur.rowcount
            cur.execute("DELETE FROM deepfakes")
            d = cur.rowcount
            cur.execute("DELETE FROM prototype_embeddings")
            p = cur.rowcount
            cur.execute("DELETE FROM participants")
            part = cur.rowcount
            conn.commit()
        print(f"  deleted {r} recording(s), {d} deepfake(s), {p} prototype_embedding(s), {part} participant(s)")
    print(f"  storage: removed {recording_blobs} recording blob(s), {deepfake_blobs} deepfake blob(s)")


def _branch_p2(sb, dry_run: bool) -> None:
    """180 days past study_closed_at: delete Phase 2 sessions + responses + clips."""
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT storage_path FROM phase2_clips")
        keys = [r[0] for r in cur.fetchall()]

    blobs_removed = _remove_storage(sb, PHASE2_BUCKET, keys, dry_run)

    if dry_run:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM responses")
            resp = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM phase2_sessions")
            sess = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM phase2_clips")
            clips = cur.fetchone()[0]
        print(f"  [dry-run] would delete {resp} response(s), {sess} session(s), {clips} clip(s)")
    else:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM responses")
            resp = cur.rowcount
            cur.execute("DELETE FROM phase2_sessions")
            sess = cur.rowcount
            cur.execute("DELETE FROM phase2_clips")
            clips = cur.rowcount
            conn.commit()
        print(f"  deleted {resp} response(s), {sess} session(s), {clips} clip(s)")
    print(f"  storage: removed {blobs_removed} phase2-clip blob(s)")


def _branch_followup(dry_run: bool) -> None:
    """14 days past followup_concluded_at: delete phase1_emails."""
    if dry_run:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM phase1_emails")
            n = cur.fetchone()[0]
        print(f"  [dry-run] would delete {n} phase1_email(s)")
    else:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM phase1_emails")
            n = cur.rowcount
            conn.commit()
        print(f"  deleted {n} phase1_email(s)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _purge(dry_run: bool) -> int:
    row = _fetch_lifecycle()
    if row is None:
        print("study_lifecycle singleton missing — re-run migration 0001_schema.sql.")
        return 1
    study_closed_at, followup_concluded_at, p1_ready, p2_ready, fu_ready = row
    print(f"study_closed_at:        {study_closed_at}")
    print(f"followup_concluded_at:  {followup_concluded_at}")

    if not (p1_ready or p2_ready or fu_ready):
        print("no lifecycle stamps — nothing to do")
        return 0

    sb = supabase_client()

    if p1_ready:
        print("\n-- branch P1 (study_closed_at + 14d) --")
        _branch_p1(sb, dry_run)
    if p2_ready:
        print("\n-- branch P2 (study_closed_at + 180d) --")
        _branch_p2(sb, dry_run)
    if fu_ready:
        print("\n-- branch followup (followup_concluded_at + 14d) --")
        _branch_followup(dry_run)

    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Event-driven retention sweep.")
    parser.add_argument("--dry-run", action="store_true", help="Print counts without deleting.")
    args = parser.parse_args(argv)
    return _purge(args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
