import { NextResponse } from "next/server";

import { BUCKETS } from "@/lib/constants";
import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { getStudyPid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const MAX_BYTES = 5 * 1024 * 1024; // 5 MB upload cap; matches the Supabase bucket limit.

function pickExtension(mime: string): string {
  if (mime.startsWith("audio/webm")) return "webm";
  if (mime.startsWith("audio/mp4")) return "m4a";
  if (mime.startsWith("audio/ogg")) return "ogg";
  if (mime.startsWith("audio/wav")) return "wav";
  return "webm";
}

export async function POST(request: Request): Promise<Response> {
  const pid = await getStudyPid();
  if (!pid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  let form: FormData;
  try {
    form = await request.formData();
  } catch {
    return NextResponse.json({ error: "invalid_form" }, { status: 400 });
  }

  const file = form.get("file");
  const sentenceIndexRaw = form.get("sentence_index");
  const durationRaw = form.get("duration_seconds");

  if (!(file instanceof Blob)) {
    return NextResponse.json({ error: "missing_file" }, { status: 400 });
  }
  if (file.size === 0) {
    return NextResponse.json({ error: "empty_file" }, { status: 400 });
  }
  if (file.size > MAX_BYTES) {
    return NextResponse.json({ error: "file_too_large" }, { status: 413 });
  }

  const contentType = file.type || "";
  if (!contentType.startsWith("audio/")) {
    return NextResponse.json({ error: "unsupported_media_type" }, { status: 415 });
  }

  const sentenceIndex = Number(sentenceIndexRaw);
  if (!Number.isInteger(sentenceIndex) || sentenceIndex < 1 || sentenceIndex > 5) {
    return NextResponse.json({ error: "bad_sentence_index" }, { status: 400 });
  }

  const durationSeconds = Number(durationRaw);
  if (!Number.isFinite(durationSeconds) || durationSeconds <= 0) {
    return NextResponse.json({ error: "bad_duration" }, { status: 400 });
  }

  const ext = pickExtension(contentType);
  const storagePath = `${pid}/${sentenceIndex}.${ext}`;
  const supabase = getServiceSupabase();

  const { error: uploadErr } = await supabase.storage
    .from(BUCKETS.recordings)
    .upload(storagePath, file, {
      upsert: true,
      contentType,
      cacheControl: "no-store",
    });

  if (uploadErr) {
    logger.error("recording: storage upload failed", uploadErr);
    return NextResponse.json({ error: "upload_failed" }, { status: 500 });
  }

  const { error: upsertErr } = await supabase
    .from("recordings")
    .upsert(
      {
        participant_id: pid,
        sentence_index: sentenceIndex,
        storage_path: storagePath,
        duration_seconds: durationSeconds,
        created_at: new Date().toISOString(),
      },
      { onConflict: "participant_id,sentence_index" },
    );

  if (upsertErr) {
    logger.error("recording: DB upsert failed", upsertErr);
    return NextResponse.json({ error: "db_error" }, { status: 500 });
  }

  return NextResponse.json({ ok: true, sentence_index: sentenceIndex });
}
