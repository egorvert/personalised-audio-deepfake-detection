import { NextResponse } from "next/server";

import { BUCKETS } from "@/lib/constants";
import { logger } from "@/lib/logging";
import { toPublicSignedUrl } from "@/lib/phase2/signed-url";
import { getServiceSupabase } from "@/lib/supabase/server";
import { getStudySid } from "@/lib/supabase/session";

export const runtime = "nodejs";

const SIGNED_URL_TTL_SECONDS = 1200; // 20 minutes.

type RouteContext = { params: Promise<{ id: string }> };

export async function GET(
  _request: Request,
  ctx: RouteContext,
): Promise<Response> {
  const sid = await getStudySid();
  if (!sid) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }

  const { id: clipId } = await ctx.params;
  const supabase = getServiceSupabase();

  // Defence-in-depth: confirm the requested clip id is in this session's
  // clip_order before handing out a signed URL.
  const { data: session, error: sessionErr } = await supabase
    .from("phase2_sessions")
    .select("clip_order")
    .eq("id", sid)
    .maybeSingle();

  if (sessionErr) {
    logger.error("phase2/clip: session lookup failed", sessionErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }
  if (!session) {
    return NextResponse.json({ error: "unauthorized" }, { status: 401 });
  }
  const order = (session.clip_order ?? []) as string[];
  if (!order.includes(clipId)) {
    return NextResponse.json({ error: "forbidden" }, { status: 403 });
  }

  const { data: clip, error: clipErr } = await supabase
    .from("phase2_clips")
    .select("storage_path")
    .eq("id", clipId)
    .maybeSingle();

  if (clipErr) {
    logger.error("phase2/clip: clip lookup failed", clipErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }
  if (!clip) {
    return NextResponse.json({ error: "not_found" }, { status: 404 });
  }

  const { data: signed, error: signErr } = await supabase.storage
    .from(BUCKETS.phase2Clips)
    .createSignedUrl(clip.storage_path as string, SIGNED_URL_TTL_SECONDS);

  if (signErr || !signed?.signedUrl) {
    logger.error("phase2/clip: signed URL failed", signErr);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  const expiresAt = new Date(Date.now() + SIGNED_URL_TTL_SECONDS * 1000).toISOString();
  return NextResponse.json({ url: toPublicSignedUrl(signed.signedUrl), expires_at: expiresAt });
}
