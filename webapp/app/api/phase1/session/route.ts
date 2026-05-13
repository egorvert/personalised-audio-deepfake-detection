import { NextResponse } from "next/server";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { clearStudyPid, getStudyPid } from "@/lib/supabase/session";

export const runtime = "nodejs";

type EnrollmentStatus =
  | "not_started"
  | "queued"
  | "processing"
  | "done"
  | "failed";

export async function GET(): Promise<Response> {
  const pid = await getStudyPid();
  if (!pid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const supabase = getServiceSupabase();

  const { data: participant, error: partErr } = await supabase
    .from("participants")
    .select("id")
    .eq("id", pid)
    .maybeSingle();

  if (partErr) {
    logger.error("phase1/session: participant lookup failed", partErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }
  if (!participant) {
    // Stale cookie — participant row no longer exists (withdrew, purged,
    // or cookie forged). Surface as 401 so client can retry consent.
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const { data: recordings, error: recErr } = await supabase
    .from("recordings")
    .select("sentence_index")
    .eq("participant_id", pid)
    .order("sentence_index", { ascending: true });

  if (recErr) {
    logger.error("phase1/session: recordings query failed", recErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  const completedSentences = (recordings ?? []).map(
    (r) => r.sentence_index as number,
  );

  // Most-recent job wins. ORDER BY created_at DESC + LIMIT 1.
  const { data: latestJob, error: jobErr } = await supabase
    .from("enrollment_jobs")
    .select("status")
    .eq("participant_id", pid)
    .order("created_at", { ascending: false })
    .limit(1)
    .maybeSingle();

  if (jobErr) {
    logger.error("phase1/session: job lookup failed", jobErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  const enrollmentStatus: EnrollmentStatus = (latestJob?.status as
    | EnrollmentStatus
    | undefined) ?? "not_started";

  return NextResponse.json({
    pid,
    completedSentences,
    enrollmentStatus,
  });
}

/**
 * Clear the study_pid cookie. Called by the "Close" button on /phase1/done
 * (Spec §5.1.5). Idempotent — safe to call without a cookie.
 */
export async function DELETE(): Promise<Response> {
  await clearStudyPid();
  return NextResponse.json({ ok: true });
}
