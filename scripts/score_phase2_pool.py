#!/usr/bin/env python
# Score every active phase2_clips row with two_stream_film.pt. Loads the model
# once under the MPS lock and iterates. Retired clips (active=false) are
# skipped. Rebuilds prototypes.json first so FiLM sees the latest embeddings.

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect, prototypes_json_path, supabase_client  # noqa: E402
from vdetect.logging import scrub  # noqa: E402
from vdetect.mps_lock import mps_lock  # noqa: E402

log = get_logger("score_phase2_pool")

PHASE2_BUCKET = "phase2-clips"


def _fetch_active_clips():
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, storage_path, is_fake, source_participant_id
              FROM phase2_clips
             WHERE active = true
             ORDER BY created_at
            """
        )
        return cur.fetchall()


def _persist_score(clip_id, score: float, prediction: bool) -> None:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            UPDATE phase2_clips
               SET system_score = %s, system_prediction = %s
             WHERE id = %s
            """,
            (score, prediction, str(clip_id)),
        )
        conn.commit()


def _download(sb, key: str, dst: Path) -> None:
    blob = sb.storage.from_(PHASE2_BUCKET).download(key)
    data = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    dst.write_bytes(data)


def _score_all(threshold: float, weights: Path, db_path: Path) -> int:
    from vdetect.service import DetectionService, ModelType

    # Ensure the on-disk cache reflects authoritative table state.
    from rebuild_prototype_cache import rebuild
    rebuild(str(db_path))

    sb = supabase_client()
    clips = _fetch_active_clips()
    if not clips:
        print("No active phase2_clips rows found; nothing to score.")
        return 0

    service = DetectionService()
    real_scores: List[float] = []
    fake_scores: List[float] = []
    correct = 0

    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        with mps_lock("score_phase2_pool"):
            service.load_model(ModelType.fusion, weights)
            for row in clips:
                clip_id, storage_path, is_fake, source_pid = row
                local = tmpdir / f"{clip_id}{Path(storage_path).suffix or '.wav'}"
                try:
                    _download(sb, storage_path, local)
                    result = service.detect_file(
                        audio_path=local,
                        threshold=threshold,
                        speaker_id=str(source_pid) if source_pid else None,
                        db_path=db_path if source_pid else None,
                    )
                    prediction = bool(result.score >= threshold)
                    _persist_score(clip_id, float(result.score), prediction)
                    if prediction == bool(is_fake):
                        correct += 1
                    (fake_scores if is_fake else real_scores).append(float(result.score))
                except Exception as exc:
                    log.error("scoring failed for clip: %s", scrub(str(exc)))

    total = len(clips)
    mean_real = sum(real_scores) / len(real_scores) if real_scores else float("nan")
    mean_fake = sum(fake_scores) / len(fake_scores) if fake_scores else float("nan")
    print(
        f"score_phase2_pool: {correct}/{total} correct | "
        f"mean real score {mean_real:.3f} | mean fake score {mean_fake:.3f}"
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Score Phase 2 clips with two_stream_film.pt.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold (default 0.5)")
    parser.add_argument(
        "--weights",
        default=os.environ.get(
            "VDETECT_WEIGHTS",
            str(_REPO_ROOT / "assets" / "checkpoints" / "two_stream_film.pt"),
        ),
        help="Path to two_stream_film.pt (default: VDETECT_WEIGHTS env or repo default)",
    )
    parser.add_argument(
        "--db-path",
        default=prototypes_json_path(),
        help="Path to prototypes.json (default: VDETECT_DB_PATH or assets/enrollments/prototypes.json)",
    )
    args = parser.parse_args(argv)
    return _score_all(args.threshold, Path(args.weights), Path(args.db_path))


if __name__ == "__main__":
    sys.exit(main())
