#!/usr/bin/env python
# Offline F5-TTS deepfake generation. Per participant: pick a reference
# recording, generate fakes for two OTHER sentences from the 5-sentence prompt
# set, upload the WAVs to the deepfakes bucket, and index each row in the
# deepfakes table. The whole load + iteration loop runs under the MPS lock so
# it must not be run concurrently with live Phase 2 traffic.

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from uuid import UUID

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from _common import get_logger, pg_connect, supabase_client  # noqa: E402
from vdetect.logging import scrub  # noqa: E402
from vdetect.mps_lock import mps_lock  # noqa: E402

log = get_logger("generate_deepfakes")

RECORDINGS_BUCKET = "recordings"
DEEPFAKES_BUCKET = "deepfakes"

# MUST match webapp/lib/constants.ts::PROMPT_SENTENCES byte-for-byte so the
# generated audio lines up with the ``sentence_index`` column.
PROMPT_SENTENCES: Tuple[str, ...] = (
    "The morning sun cast a warm golden light over the quiet village as the birds began to sing",
    "She picked up the heavy box from the table and carried it carefully through the narrow hallway",
    "Please remember to bring your jacket with you today because the weather may change later this evening",
    "The children laughed and played together in the open field just behind the old village school",
    "I would like a fresh cup of coffee with just a small amount of sugar and a dash of milk please",
)


def _sentence_text(index: int) -> str:
    if not 1 <= index <= len(PROMPT_SENTENCES):
        raise ValueError(f"sentence_index out of range: {index}")
    return PROMPT_SENTENCES[index - 1]


def _pick_target_sentences(reference_index: int, per_participant: int) -> List[int]:
    """Deterministic pick: the `per_participant` sentences furthest (lexicographically) from the reference.

    Per-spec simplification: if reference is 1, default is [2, 4]. More
    generally, take candidates (all other indices), sort by descending
    absolute distance from the reference then ascending index to break ties,
    take the first N.
    """
    candidates = [i for i in range(1, len(PROMPT_SENTENCES) + 1) if i != reference_index]
    candidates.sort(key=lambda i: (-abs(i - reference_index), i))
    return sorted(candidates[:per_participant])


def _list_participants() -> List[UUID]:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT id FROM participants ORDER BY created_at")
        return [r[0] for r in cur.fetchall()]


def _fetch_recordings(participant_id: UUID):
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, sentence_index, storage_path
              FROM recordings
             WHERE participant_id = %s
             ORDER BY sentence_index
            """,
            (str(participant_id),),
        )
        return cur.fetchall()


def _insert_deepfake_row(
    source_participant_id: UUID,
    reference_recording_id: UUID,
    sentence_index: int,
    storage_path: str,
) -> UUID:
    with pg_connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO deepfakes
                (source_participant_id, reference_recording_id, sentence_index, storage_path)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (str(source_participant_id), str(reference_recording_id), sentence_index, storage_path),
        )
        new_id = cur.fetchone()[0]
        conn.commit()
        return new_id


def _download(sb, bucket: str, key: str, dst: Path) -> None:
    blob = sb.storage.from_(bucket).download(key)
    data = blob if isinstance(blob, (bytes, bytearray)) else bytes(blob)
    dst.write_bytes(data)


def _upload(sb, bucket: str, key: str, src: Path) -> None:
    sb.storage.from_(bucket).upload(
        key,
        src.read_bytes(),
        {"content-type": "audio/wav", "upsert": "true"},
    )

# F5-TTS bridge
def _load_f5tts():
    """Load F5-TTS exactly once. Returns a callable (ref_path, target_text, out_path) -> None."""
    # f5_tts's Python API surface has evolved; prefer the stable high-level class.
    from f5_tts.api import F5TTS  # type: ignore[import-not-found]

    tts = F5TTS()  # default E2-TTS checkpoint from SWivid/F5-TTS
    log.info("F5-TTS loaded")

    def synth(ref_audio: Path, ref_text: str, target_text: str, out_path: Path) -> None:
        # F5TTS.infer writes a 24 kHz WAV; we resample to 16 kHz mono below.
        # The API returns (wav, sr, _) or writes directly depending on version;
        # we call the documented file-output form.
        tts.infer(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=target_text,
            file_wave=str(out_path),
        )

    return synth


def _resample_to_16k_mono(src: Path, dst: Path) -> None:
    """Force 16 kHz mono WAV so all deepfakes share a uniform sample rate."""
    import soundfile as sf  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    wav, sr = sf.read(str(src), always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        import librosa  # type: ignore[import-not-found]
        wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000)
        sr = 16000
    sf.write(str(dst), wav, sr, subtype="PCM_16")


# Main
def _process_participant(
    pid: UUID,
    per_participant: int,
    reference_sentence: int,
    synth,
    sb,
    tmpdir: Path,
) -> Tuple[int, int]:
    """Return (generated, skipped) counts for one participant."""
    rows = _fetch_recordings(pid)
    if not rows:
        log.warning("participant has no recordings; skipping")
        return 0, 0

    by_idx = {r[1]: (r[0], r[2]) for r in rows}  # sentence_index -> (rec_id, path)
    if reference_sentence not in by_idx:
        log.warning("reference sentence %d missing for participant; skipping", reference_sentence)
        return 0, 0

    ref_rec_id, ref_key = by_idx[reference_sentence]
    ref_text = _sentence_text(reference_sentence)

    ref_local = tmpdir / f"ref_{pid}{Path(ref_key).suffix or '.webm'}"
    _download(sb, RECORDINGS_BUCKET, ref_key, ref_local)

    targets = _pick_target_sentences(reference_sentence, per_participant)
    generated = 0
    skipped = 0
    for sentence_index in targets:
        target_text = _sentence_text(sentence_index)
        raw_out = tmpdir / f"gen_{pid}_{sentence_index}_raw.wav"
        final_out = tmpdir / f"gen_{pid}_{sentence_index}.wav"
        dest_key = f"{pid}/{sentence_index}.wav"
        try:
            synth(ref_local, ref_text, target_text, raw_out)
            _resample_to_16k_mono(raw_out, final_out)
            _upload(sb, DEEPFAKES_BUCKET, dest_key, final_out)
            _insert_deepfake_row(pid, ref_rec_id, sentence_index, dest_key)
            generated += 1
            log.info("generated clip sentence_index=%d", sentence_index)
        except Exception as exc:
            skipped += 1
            tb = traceback.format_exc(limit=3)
            log.error(
                "clip sentence_index=%d failed: %s | %s",
                sentence_index, scrub(str(exc)), scrub(tb),
            )
    return generated, skipped


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generate F5-TTS deepfakes for study participants.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participant", help="UUID of a single participant to process")
    group.add_argument("--all", action="store_true", help="Process every participant in the DB")
    parser.add_argument("--per-participant", type=int, default=2,
                        help="Number of fake clips per participant (default 2)")
    parser.add_argument("--reference-sentence", type=int, default=1,
                        help="Which sentence_index to use as the F5-TTS reference (default 1)")
    args = parser.parse_args(argv)

    if "PYTORCH_ENABLE_MPS_FALLBACK" not in os.environ:
        log.warning("PYTORCH_ENABLE_MPS_FALLBACK not set; F5-TTS may fail on unsupported MPS ops")

    if args.all:
        participants = _list_participants()
    else:
        try:
            participants = [UUID(args.participant)]
        except ValueError:
            print(f"ERROR: invalid UUID: {args.participant}", file=sys.stderr)
            return 2

    if not participants:
        print("No participants to process.")
        return 0

    sb = supabase_client()

    total_generated = 0
    total_skipped = 0
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        with mps_lock("generate_deepfakes"):
            synth = _load_f5tts()
            for pid in participants:
                try:
                    g, s = _process_participant(
                        pid, args.per_participant, args.reference_sentence, synth, sb, tmpdir,
                    )
                    total_generated += g
                    total_skipped += s
                except Exception as exc:
                    total_skipped += args.per_participant
                    log.error("participant failed entirely: %s", scrub(str(exc)))

    print(f"generate_deepfakes: {total_generated} generated, {total_skipped} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
