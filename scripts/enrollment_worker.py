#!/usr/bin/env python
# Durable enrolment queue daemon. Runs under PM2, drains the enrollment_jobs
# table. For each job: claim atomically with SELECT FOR UPDATE SKIP LOCKED,
# hold the MPS lock for load + inference, download the 5 recordings, call
# DetectionService.enroll_speaker in-process, upsert prototype_embeddings,
# rebuild the prototypes.json cache, and mark the job done/failed.

# A startup reaper kicks any 'processing' rows older than 10 minutes back to
# 'queued' so a worker crash never strands a job permanently. SIGTERM finishes
# the in-flight job, releases the lock, and exits cleanly.

from __future__ import annotations

import os
import signal
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import UUID

# Ensure repo root is importable when PM2 launches us from /Users/.../scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect, supabase_client  # noqa: E402
from vdetect.logging import scrub  # noqa: E402
from vdetect.mps_lock import mps_lock  # noqa: E402

log = get_logger("enrollment_worker")

POLL_INTERVAL_S = 2.0
STUCK_JOB_TIMEOUT = "10 minutes"
RECORDINGS_BUCKET = "recordings"


class _ShutdownFlag:
    """SIGTERM-aware flag. Set once, never unset."""

    def __init__(self) -> None:
        self._set = False

    def trip(self, *_: object) -> None:
        if not self._set:
            log.info("shutdown signal received; finishing current job then exiting")
        self._set = True

    def __bool__(self) -> bool:
        return self._set


# ---------------------------------------------------------------------------
# DB ops
# ---------------------------------------------------------------------------


def _reap_stuck_jobs() -> None:
    """Reset any `processing` rows older than STUCK_JOB_TIMEOUT back to queued."""
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            UPDATE enrollment_jobs
               SET status='queued', started_at=NULL
             WHERE status='processing'
               AND started_at < now() - interval '{STUCK_JOB_TIMEOUT}'
            """
        )
        reaped = cur.rowcount
        conn.commit()
    if reaped:
        log.info("startup reaper: reset %d stuck 'processing' row(s)", reaped)


def _claim_job() -> Optional[Tuple[UUID, UUID]]:
    """Atomically claim one queued job. Returns (job_id, participant_id) or None."""
    sql = """
        UPDATE enrollment_jobs
           SET status='processing', attempts=attempts+1, started_at=now()
         WHERE id = (
             SELECT id
               FROM enrollment_jobs
              WHERE status='queued'
              ORDER BY created_at
              LIMIT 1
              FOR UPDATE SKIP LOCKED
         )
        RETURNING id, participant_id
    """
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        conn.commit()
        return (row[0], row[1]) if row else None


def _fetch_recording_rows(participant_id: UUID) -> List[Tuple[UUID, str]]:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, storage_path
              FROM recordings
             WHERE participant_id = %s
             ORDER BY sentence_index
            """,
            (str(participant_id),),
        )
        return [(r[0], r[1]) for r in cur.fetchall()]


def _upsert_prototype(
    participant_id: UUID, embedding: List[float], source_ids: List[UUID]
) -> None:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO prototype_embeddings
                (participant_id, embedding, source_recording_ids, model_version, computed_at)
            VALUES (%s, %s, %s, 'two_stream_v1', now())
            ON CONFLICT (participant_id) DO UPDATE
                SET embedding = EXCLUDED.embedding,
                    source_recording_ids = EXCLUDED.source_recording_ids,
                    model_version = EXCLUDED.model_version,
                    computed_at = EXCLUDED.computed_at
            """,
            (str(participant_id), embedding, [str(x) for x in source_ids]),
        )
        conn.commit()


def _mark_done(job_id: UUID) -> None:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE enrollment_jobs SET status='done', completed_at=now(), error=NULL WHERE id=%s",
            (str(job_id),),
        )
        conn.commit()


def _mark_failed(job_id: UUID, error: str) -> None:
    scrubbed = scrub(error)[:2000]
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE enrollment_jobs SET status='failed', completed_at=now(), error=%s WHERE id=%s",
            (scrubbed, str(job_id)),
        )
        conn.commit()


# ---------------------------------------------------------------------------
# Embedding pipeline
# ---------------------------------------------------------------------------


def _download_recordings(storage_paths: List[str], tmpdir: Path) -> List[Path]:
    """Download each storage object to tmpdir; return local paths in order."""
    sb = supabase_client()
    local_paths: List[Path] = []
    for idx, key in enumerate(storage_paths):
        blob = sb.storage.from_(RECORDINGS_BUCKET).download(key)
        # Supabase returns bytes (older supabase-py) or a Response-like object.
        data = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
        suffix = Path(key).suffix or ".webm"
        dst = tmpdir / f"sample_{idx}{suffix}"
        dst.write_bytes(data)
        local_paths.append(dst)
    return local_paths


def _enroll_and_persist(
    participant_id: UUID,
    recording_rows: List[Tuple[UUID, str]],
) -> None:
    """Load model + enroll under the MPS lock; persist embedding; refresh cache."""
    # Deferred imports keep startup cheap + side-effect-free for test harnesses.
    import torch

    from vdetect.enrollment import compute_prototype
    from vdetect.service import get_device

    weights_env = os.environ.get(
        "VDETECT_ENROLL_WEIGHTS",
        str(_REPO_ROOT / "assets" / "checkpoints" / "two_stream_film.pt"),
    )
    weights = Path(weights_env)
    if not weights.exists():
        raise FileNotFoundError(f"enrollment weights not found: {weights}")

    source_ids = [rid for rid, _ in recording_rows]
    storage_paths = [path for _, path in recording_rows]
    if len(storage_paths) < 3:
        raise ValueError(
            f"participant has only {len(storage_paths)} recording(s); need >= 3"
        )

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        local_paths = _download_recordings(storage_paths, tmpdir)

        # Pre-validate each download by attempting to decode it. Some browsers
        # (notably Chrome on Windows) occasionally produce webm files without
        # a complete EBML header that neither torchcodec nor ffmpeg can recover.
        # Rather than failing the whole enrolment we skip those files and
        # proceed with whatever decodes — provided at least 3 survive, which
        # is enough for a usable speaker prototype.
        from vdetect.data.audio import crop_or_pad, load_audio

        survivor_wavs: List[torch.Tensor] = []
        survivor_source_ids: List[UUID] = []
        skipped: List[Tuple[str, str]] = []
        for path, sid in zip(local_paths, source_ids):
            try:
                wav = load_audio(path)
                wav = crop_or_pad(wav, max_len=64600, train=False)
            except Exception as exc:
                skipped.append((path.name, exc.__class__.__name__))
                continue
            survivor_wavs.append(wav)
            survivor_source_ids.append(sid)

        if skipped:
            log.warning(
                "enrolment skipping %d undecodable file(s): %s",
                len(skipped),
                ", ".join(f"{n} ({why})" for n, why in skipped),
            )

        if len(survivor_wavs) < 3:
            raise ValueError(
                f"only {len(survivor_wavs)} of {len(local_paths)} recording(s) decoded; "
                "need >= 3 for a prototype"
            )

        with mps_lock("enrollment_worker"):
            from vdetect.models.two_stream import TwoStreamDetector

            device = get_device()
            model = TwoStreamDetector(
                wavlm_checkpoint=None,
                aasist_checkpoint=None,
                freeze_wavlm=True,
                freeze_aasist=True,
            ).to(device)
            checkpoint = torch.load(weights, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            model.train(False)

            batch = torch.stack(survivor_wavs).to(device)
            with torch.no_grad():
                embeddings = model.extract_embedding(batch)
            prototype = compute_prototype(embeddings, normalize=True)

    embedding_list = prototype.detach().cpu().tolist()
    _upsert_prototype(participant_id, embedding_list, survivor_source_ids)

    # Refresh prototypes.json so the running API picks up the new prototype.
    from rebuild_prototype_cache import rebuild
    rebuild()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _process_one(job_id: UUID, participant_id: UUID) -> None:
    """Run the pipeline for one claimed job."""
    log.info("processing job (participant id redacted)")
    try:
        recordings = _fetch_recording_rows(participant_id)
        _enroll_and_persist(participant_id, recordings)
        _mark_done(job_id)
        log.info("job completed")
    except Exception as exc:
        tb = traceback.format_exc(limit=4)
        log.error("job failed: %s", scrub(str(exc)))
        _mark_failed(job_id, f"{exc.__class__.__name__}: {exc}\n{tb}")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Durable enrollment queue daemon (PM2-supervised).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Claim and process at most one job, then exit (for smoke tests).",
    )
    args = parser.parse_args(argv)

    shutdown = _ShutdownFlag()
    signal.signal(signal.SIGTERM, shutdown.trip)
    signal.signal(signal.SIGINT, shutdown.trip)

    log.info("enrollment_worker starting (poll=%.1fs, once=%s)", POLL_INTERVAL_S, args.once)
    _reap_stuck_jobs()

    while not shutdown:
        try:
            claimed = _claim_job()
        except Exception as exc:
            log.error("claim transaction failed: %s", scrub(str(exc)))
            time.sleep(POLL_INTERVAL_S)
            continue

        if claimed is None:
            time.sleep(POLL_INTERVAL_S)
            continue

        job_id, participant_id = claimed
        _process_one(job_id, participant_id)

        if args.once:
            break

    log.info("enrollment_worker exiting cleanly")
    return 0


if __name__ == "__main__":
    sys.exit(main())
