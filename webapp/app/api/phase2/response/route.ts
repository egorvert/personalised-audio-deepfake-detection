import { NextResponse } from "next/server";
import { z } from "zod";

import { logger } from "@/lib/logging";
import { phase2ResponseLimiter } from "@/lib/rate-limit";
import { getServiceSupabase } from "@/lib/supabase/server";
import { getStudySid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const ResponseSchema = z.object({
  clip_id: z.string().uuid(),
  answer_is_fake: z.boolean(),
  confidence: z.number().int().min(1).max(5),
});

// Postgres unique_violation SQLSTATE (per §5.2.3 + Section 11 — duplicate is
// treated as idempotent success).
const PG_UNIQUE_VIOLATION = "23505";

export async function POST(request: Request): Promise<Response> {
  const sid = await getStudySid();
  if (!sid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  if (!phase2ResponseLimiter.take(sid)) {
    return NextResponse.json({ error: "rate_limited" }, { status: 429 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const parsed = ResponseSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "bad_body" }, { status: 400 });
  }
  const { clip_id, answer_is_fake, confidence } = parsed.data;

  const supabase = getServiceSupabase();

  const { data: session, error: sessionErr } = await supabase
    .from("phase2_sessions")
    .select("clip_order")
    .eq("id", sid)
    .maybeSingle();

  if (sessionErr) {
    logger.error("phase2/response: session lookup failed", sessionErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }
  if (!session) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const order = (session.clip_order ?? []) as string[];
  if (!order.includes(clip_id)) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }

  const { error: insertErr } = await supabase.from("responses").insert({
    session_id: sid,
    clip_id,
    answer_is_fake,
    confidence,
  });

  if (insertErr) {
    if ((insertErr as { code?: string }).code === PG_UNIQUE_VIOLATION) {
      return NextResponse.json({ ok: true, duplicate: true });
    }
    logger.error("phase2/response: insert failed", insertErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
