// Pure enqueue. This handler only inserts a row into `enrollment_jobs` (or
// returns an existing queued/processing row, so callers can retry idempotently).
// The actual enrolment runs in scripts/enrollment_worker.py off the queue.
// Don't add background tasks (waitUntil, fire-and-forget IIFEs, direct vdetect
// calls) here — they reintroduce the race the durable queue exists to avoid.

import { NextResponse } from "next/server";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { getStudyPid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const REQUIRED_SENTENCES = 5;

export async function POST(): Promise<Response> {
  const pid = await getStudyPid();
  if (!pid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const supabase = getServiceSupabase();

  const { count, error: countErr } = await supabase
    .from("recordings")
    .select("id", { count: "exact", head: true })
    .eq("participant_id", pid);

  if (countErr) {
    logger.error("enroll: recording count failed", countErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  if ((count ?? 0) < REQUIRED_SENTENCES) {
    return NextResponse.json({ error: "not_ready" }, { status: 400 });
  }

  // Idempotency probe: if a queued or processing job already exists for this
  // participant, return its id instead of inserting a duplicate.
  const { data: existing, error: existingErr } = await supabase
    .from("enrollment_jobs")
    .select("id")
    .eq("participant_id", pid)
    .in("status", ["queued", "processing"])
    .limit(1)
    .maybeSingle();

  if (existingErr) {
    logger.error("enroll: existing-job probe failed", existingErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  if (existing?.id) {
    return NextResponse.json({ jobId: existing.id });
  }

  const { data: inserted, error: insertErr } = await supabase
    .from("enrollment_jobs")
    .insert({ participant_id: pid, status: "queued" })
    .select("id")
    .single();

  if (insertErr || !inserted) {
    logger.error("enroll: job insert failed", insertErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  return NextResponse.json({ jobId: inserted.id });
}
