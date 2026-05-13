#!/usr/bin/env python
# Dump study data to CSV (one per table) + embeddings.jsonl from
# prototype_embeddings. Idempotent (overwrites). phase1_emails is emitted as a
# per-day aggregate count only, never joined with participant rows.

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect  # noqa: E402

log = get_logger("export_study_data")

# name -> (csv filename, SQL)
TABLES: List[tuple] = [
    ("participants",        "participants.csv",
     "SELECT id, created_at, consented_at FROM participants ORDER BY created_at"),
    ("recordings",          "recordings.csv",
     "SELECT id, participant_id, sentence_index, storage_path, duration_seconds, created_at "
     "FROM recordings ORDER BY created_at"),
    # Ethics E4: emails MUST stay separate from the anonymised research bundle.
    # Even the row id is a joinable handle, so this export emits an aggregate
    # count per day only — zero per-row granularity. Follow-up invite addresses
    # are pulled via a separate operational path, never co-located with
    # participant/recording CSVs.
    ("phase1_emails",       "phase1_emails.csv",
     "SELECT bucket_day, COUNT(*) AS n FROM phase1_emails "
     "GROUP BY bucket_day ORDER BY bucket_day"),
    ("enrollment_jobs",     "enrollment_jobs.csv",
     "SELECT id, participant_id, status, attempts, error, created_at, started_at, completed_at "
     "FROM enrollment_jobs ORDER BY created_at"),
    ("deepfakes",           "deepfakes.csv",
     "SELECT id, source_participant_id, reference_recording_id, sentence_index, storage_path, created_at "
     "FROM deepfakes ORDER BY created_at"),
    ("phase2_clips",        "phase2_clips.csv",
     "SELECT id, storage_path, is_fake, source_participant_id, sentence_index, "
     "system_score, system_prediction, active, retired_at, retired_reason, created_at "
     "FROM phase2_clips ORDER BY created_at"),
    ("phase2_sessions",     "phase2_sessions.csv",
     "SELECT id, created_at, consented_at, clip_order FROM phase2_sessions ORDER BY created_at"),
    ("responses",           "responses.csv",
     "SELECT id, session_id, clip_id, answer_is_fake, confidence, created_at "
     "FROM responses ORDER BY created_at"),
    ("study_lifecycle",     "study_lifecycle.csv",
     "SELECT id, study_closed_at, followup_concluded_at, updated_at FROM study_lifecycle"),
]


def _dump_csv(cur, sql: str, out_path: Path) -> int:
    cur.execute(sql)
    cols = [d.name for d in cur.description]
    rows = cur.fetchall()
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: _stringify(v) for c, v in zip(cols, row)})
    return len(rows)


def _stringify(value) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        # psycopg returns arrays as Python lists. Serialise as JSON for round-trip.
        return json.dumps([_stringify(v) for v in value])
    return str(value)


def _dump_embeddings(cur, out_path: Path) -> int:
    cur.execute(
        """
        SELECT participant_id, embedding, model_version, source_recording_ids, computed_at
          FROM prototype_embeddings
         ORDER BY computed_at
        """
    )
    rows = cur.fetchall()
    with out_path.open("w", encoding="utf-8") as f:
        for pid, emb, model_version, source_ids, computed_at in rows:
            f.write(json.dumps({
                "participant_id": str(pid),
                "embedding": list(emb),
                "model_version": model_version,
                "source_recording_ids": [str(x) for x in (source_ids or [])],
                "computed_at": computed_at.isoformat() if computed_at else None,
            }) + "\n")
    return len(rows)


def _export(out_dir: Path) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    with pg_connect() as conn, conn.cursor() as cur:
        for name, filename, sql in TABLES:
            count = _dump_csv(cur, sql, out_dir / filename)
            print(f"  {filename}: {count} row(s)")
        emb_count = _dump_embeddings(cur, out_dir / "embeddings.jsonl")
        print(f"  embeddings.jsonl: {emb_count} row(s)")
    print(f"export_study_data: wrote to {out_dir}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Export study data as CSV + JSONL.")
    parser.add_argument("--out", default="exports", help="Output directory (default: exports/)")
    args = parser.parse_args(argv)
    return _export(Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
