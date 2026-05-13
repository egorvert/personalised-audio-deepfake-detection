#!/usr/bin/env python
# Withdraw a participant. Hard-deletes their recordings, prototype embeddings
# and enrolment jobs (via cascade). Phase 2 clips derived from them are
# soft-retired (active=false) so anonymous responses pointing at those clips
# remain valid. Aborts without --force if the active pool would drop below 20.

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence
from uuid import UUID

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect, supabase_client  # noqa: E402

log = get_logger("withdraw_participant")

RECORDINGS_BUCKET = "recordings"
DEEPFAKES_BUCKET = "deepfakes"

POOL_MIN_ACTIVE = 20


def _storage_prefix_keys(sb, bucket: str, prefix: str) -> List[str]:
    try:
        entries = sb.storage.from_(bucket).list(path=prefix)
    except Exception as exc:
        log.error("list(%s/%s) failed: %s", bucket, prefix, exc)
        return []
    return [f"{prefix}/{e['name']}" for e in entries or []]


def _count_would_remain_active(pid: UUID) -> int:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT COUNT(*) FROM phase2_clips
             WHERE active = true
               AND (source_participant_id IS DISTINCT FROM %s)
            """,
            (str(pid),),
        )
        return cur.fetchone()[0]


def _participant_exists(pid: UUID) -> bool:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM participants WHERE id = %s", (str(pid),))
        return cur.fetchone() is not None


def _withdraw(pid: UUID, force: bool, dry_run: bool) -> int:
    if not _participant_exists(pid):
        print(f"ERROR: participant {pid} not found.", file=sys.stderr)
        return 1

    would_remain = _count_would_remain_active(pid)
    if would_remain < POOL_MIN_ACTIVE:
        print(
            f"\n!!! PHASE2 POOL INTEGRITY WARNING — only {would_remain} active clips would remain !!!",
            file=sys.stderr,
        )
        print(
            "New Phase 2 sessions will 503 until a curator backfills the pool.",
            file=sys.stderr,
        )
        if not force:
            print("Aborting. Pass --force to proceed anyway.", file=sys.stderr)
            return 1

    sb = supabase_client()
    rec_keys = _storage_prefix_keys(sb, RECORDINGS_BUCKET, str(pid))
    df_keys = _storage_prefix_keys(sb, DEEPFAKES_BUCKET, str(pid))

    if dry_run:
        with pg_connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM recordings WHERE participant_id = %s", (str(pid),))
            n_rec = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM deepfakes WHERE source_participant_id = %s", (str(pid),))
            n_df = cur.fetchone()[0]
            cur.execute(
                "SELECT COUNT(*) FROM phase2_clips WHERE active=true AND source_participant_id = %s",
                (str(pid),),
            )
            n_clip = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM prototype_embeddings WHERE participant_id = %s", (str(pid),))
            n_proto = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM enrollment_jobs WHERE participant_id = %s", (str(pid),))
            n_jobs = cur.fetchone()[0]
        print(f"[dry-run] would remove {len(rec_keys)} recording blob(s), {len(df_keys)} deepfake blob(s)")
        print(f"[dry-run] would delete {n_rec} recordings, {n_df} deepfakes, {n_proto} prototype, {n_jobs} job(s)")
        print(f"[dry-run] would soft-retire {n_clip} active phase2_clip(s)")
        print(f"[dry-run] post-action active pool size: {would_remain}")
        return 0

    # Storage: remove per-prefix keys explicitly (deletion is by path list, not prefix).
    if rec_keys:
        sb.storage.from_(RECORDINGS_BUCKET).remove(rec_keys)
    if df_keys:
        sb.storage.from_(DEEPFAKES_BUCKET).remove(df_keys)

    with pg_connect() as conn, conn.cursor() as cur:
        # 1. Soft-retire derived phase2_clips FIRST (before participant cascade nulls source_participant_id).
        cur.execute(
            """
            UPDATE phase2_clips
               SET active = false,
                   retired_at = now(),
                   retired_reason = 'withdrawal'
             WHERE source_participant_id = %s
               AND active = true
            """,
            (str(pid),),
        )
        retired = cur.rowcount

        # 2. Drop the prototype row (cascade would catch it, but be explicit for the cache refresh).
        cur.execute("DELETE FROM prototype_embeddings WHERE participant_id = %s", (str(pid),))

        # 3. Drop the participant row; cascades cover recordings, enrollment_jobs, deepfakes.
        cur.execute("DELETE FROM participants WHERE id = %s", (str(pid),))
        conn.commit()

    # 4. Refresh on-disk cache so vdetect-api stops serving the stale embedding.
    from rebuild_prototype_cache import rebuild
    rebuild()

    print(f"withdrew participant {pid}: retired {retired} phase2_clip(s), removed {len(rec_keys) + len(df_keys)} blob(s)")
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM phase2_clips WHERE active = true")
        active = cur.fetchone()[0]
    if active < POOL_MIN_ACTIVE:
        print(
            f"!!! PHASE2 POOL INTEGRITY BROKEN — only {active} active clips remain !!!",
            file=sys.stderr,
        )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Withdraw a participant from the study.")
    parser.add_argument("--participant", required=True, help="Participant UUID")
    parser.add_argument("--force", action="store_true", help="Proceed even if post-action active pool < 20")
    parser.add_argument("--dry-run", action="store_true", help="Print intent without executing")
    args = parser.parse_args(argv)

    try:
        pid = UUID(args.participant)
    except ValueError:
        print(f"ERROR: invalid UUID: {args.participant}", file=sys.stderr)
        return 2

    return _withdraw(pid, force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
