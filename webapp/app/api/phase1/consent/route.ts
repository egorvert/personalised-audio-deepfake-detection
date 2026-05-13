import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { getStudyPid, setStudyPid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const ConsentSchema = z.object({ agreed: z.literal(true) });

const UUID_RE =
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export async function POST(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const parsed = ConsentSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "agreement_required" }, { status: 400 });
  }

  const supabase = getServiceSupabase();

  // Idempotent rehydrate: if the caller already has a valid study_pid cookie
  // pointing at an existing participant whose enrolment hasn't finished,
  // refresh the cookie and return the existing pid instead of creating a new one.
  const existingPid = await getStudyPid();
  if (existingPid && UUID_RE.test(existingPid)) {
    const { data: participant, error: lookupErr } = await supabase
      .from("participants")
      .select("id")
      .eq("id", existingPid)
      .maybeSingle();

    if (lookupErr) {
      logger.error("consent: participant lookup failed", lookupErr);
      return NextResponse.json({ error: "server_error" }, { status: 500 });
    }

    if (participant) {
      const { data: doneJob, error: jobErr } = await supabase
        .from("enrollment_jobs")
        .select("id")
        .eq("participant_id", existingPid)
        .eq("status", "done")
        .limit(1)
        .maybeSingle();

      if (jobErr) {
        logger.error("consent: enrollment_jobs lookup failed", jobErr);
        return NextResponse.json({ error: "server_error" }, { status: 500 });
      }

      if (!doneJob) {
        // Valid resume — refresh Max-Age.
        await setStudyPid(existingPid);
        return NextResponse.json({ pid: existingPid, resumed: true });
      }
    }
  }

  const pid = randomUUID();
  const consentedAt = new Date().toISOString();
  const { error: insertErr } = await supabase
    .from("participants")
    .insert({ id: pid, consented_at: consentedAt });

  if (insertErr) {
    logger.error("consent: participant insert failed", insertErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  await setStudyPid(pid);
  return NextResponse.json({ pid, resumed: false });
}
