import { NextResponse } from "next/server";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { getStudyPid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

type RouteContext = { params: Promise<{ jobId: string }> };

export async function GET(
  _request: Request,
  ctx: RouteContext,
): Promise<Response> {
  const pid = await getStudyPid();
  if (!pid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const { jobId } = await ctx.params;
  if (!UUID_RE.test(jobId)) {
    return NextResponse.json({ error: "bad_job_id" }, { status: 400 });
  }

  const supabase = getServiceSupabase();

  const { data: job, error } = await supabase
    .from("enrollment_jobs")
    .select("status, error, participant_id, attempts")
    .eq("id", jobId)
    .maybeSingle();

  if (error) {
    logger.error("phase1/enrollment: job lookup failed", error);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  // Return 404 (not 403) if the job isn't owned by the current study_pid —
  // we don't want to leak job existence to unrelated sessions.
  if (!job || job.participant_id !== pid) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }

  const payload: {
    status: string;
    attempts: number;
    error?: string;
  } = {
    status: job.status as string,
    attempts: (job.attempts as number) ?? 0,
  };
  if (job.status === "failed" && job.error) {
    payload.error = job.error as string;
  }

  return NextResponse.json(payload);
}
