// IMPORTANT: this endpoint is intentionally decoupled from the participant
// record (ethics commitment). It must NOT read the study_pid cookie or any
// other participant identifier. It writes only to phase1_emails, which stores
// `bucket_day DATE` rather than a timestamp so emails cannot be correlated
// back to a session. Any change here needs ethics review.

import { NextResponse } from "next/server";
import { z } from "zod";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";

export const runtime = "nodejs";

const EmailSchema = z.object({
  email: z.string().trim().min(3).max(254).email(),
});

export async function POST(request: Request): Promise<Response> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "invalid_json" }, { status: 400 });
  }

  const parsed = EmailSchema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: "bad_email" }, { status: 400 });
  }

  const supabase = getServiceSupabase();
  const { error } = await supabase
    .from("phase1_emails")
    .insert({ email: parsed.data.email });

  if (error) {
    logger.error("email: insert failed", error);
    return NextResponse.json({ error: "server_error" }, { status: 500 });
  }

  return NextResponse.json({ ok: true });
}
