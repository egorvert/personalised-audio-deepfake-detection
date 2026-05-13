import { NextResponse } from "next/server";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { clearStudySid, getStudySid } from "@/lib/supabase/session";

export const runtime = "nodejs";

export async function GET(): Promise<Response> {
  const sid = await getStudySid();
  if (!sid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const supabase = getServiceSupabase();

  const { data: session, error: sessionErr } = await supabase
    .from("phase2_sessions")
    .select("clip_order")
    .eq("id", sid)
    .maybeSingle();

  if (sessionErr) {
    logger.error("phase2/session: session lookup failed", sessionErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }
  if (!session) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const { count, error: countErr } = await supabase
    .from("responses")
    .select("id", { count: "exact", head: true })
    .eq("session_id", sid);

  if (countErr) {
    logger.error("phase2/session: response count failed", countErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  return NextResponse.json({
    sid,
    clip_order: session.clip_order as string[],
    next_index: count ?? 0,
  });
}

/**
 * Clear the study_sid cookie. Called by the "Close" button on /phase2/done
 * (Spec §5.2.4). Idempotent — safe to call without a cookie.
 */
export async function DELETE(): Promise<Response> {
  await clearStudySid();
  return NextResponse.json({ ok: true });
}
