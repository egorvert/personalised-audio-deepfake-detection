import { NextResponse } from "next/server";
import { randomUUID } from "node:crypto";
import { z } from "zod";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";
import { setStudySid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const REQUIRED_POOL_SIZE = 20;

const ConsentSchema = z.object({ agreed: z.literal(true) });

/** Fisher–Yates shuffle seeded via crypto.getRandomValues. */
function shuffle<T>(input: readonly T[]): T[] {
  const arr = input.slice();
  const buf = new Uint32Array(arr.length);
  crypto.getRandomValues(buf);
  for (let i = arr.length - 1; i > 0; i--) {
    const j = buf[i] % (i + 1);
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}

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

  // Only active clips count toward pool readiness. The required size is a
  // floor (>= 20), not strict equality, so curators backfilling extra clips
  // doesn't lock out new participants. We still take exactly 20 per session.
  const { data: clips, error: poolErr } = await supabase
    .from("phase2_clips")
    .select("id")
    .eq("active", true)
    .order("created_at", { ascending: true });

  if (poolErr) {
    logger.error("phase2/consent: pool query failed", poolErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  if (!clips || clips.length < REQUIRED_POOL_SIZE) {
    return NextResponse.json(
      { error: "Study not ready" },
      { status: 503 },
    );
  }

  const clipOrder = shuffle(clips.map((c) => c.id as string)).slice(
    0,
    REQUIRED_POOL_SIZE,
  );
  const sid = randomUUID();

  const { error: insertErr } = await supabase.from("phase2_sessions").insert({
    id: sid,
    consented_at: new Date().toISOString(),
    clip_order: clipOrder,
  });

  if (insertErr) {
    logger.error("phase2/consent: session insert failed", insertErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  await setStudySid(sid);
  return NextResponse.json({ ok: true });
}
