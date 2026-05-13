import "server-only";

import { logger } from "@/lib/logging";
import { getServiceSupabase } from "@/lib/supabase/server";

type ResponseRow = {
  clip_id: string;
  answer_is_fake: boolean;
  confidence: number;
};

type ClipMeta = {
  id: string;
  is_fake: boolean;
  system_prediction: boolean | null;
};

export type BreakdownRow = {
  clip_number: number;
  ground_truth: "real" | "fake";
  participant_answer: "real" | "fake";
  participant_confidence: number;
  system_answer: "real" | "fake" | "unknown";
};

export type ResultsPayload = {
  participant_score: number;
  system_score: number;
  total: number;
  breakdown: BreakdownRow[];
};

export type ComputeResultsOutcome =
  | { kind: "ok"; payload: ResultsPayload }
  | { kind: "unauthorized" }
  | { kind: "server_error" };

export async function computePhase2Results(
  sid: string,
): Promise<ComputeResultsOutcome> {
  const supabase = getServiceSupabase();

  const { data: session, error: sessionErr } = await supabase
    .from("phase2_sessions")
    .select("clip_order")
    .eq("id", sid)
    .maybeSingle();

  if (sessionErr) {
    logger.error("phase2/results: session lookup failed", sessionErr);
    return { kind: "server_error" };
  }
  if (!session) {
    return { kind: "unauthorized" };
  }
  const clipOrder = (session.clip_order ?? []) as string[];

  const { data: responses, error: respErr } = await supabase
    .from("responses")
    .select("clip_id, answer_is_fake, confidence")
    .eq("session_id", sid);

  if (respErr) {
    logger.error("phase2/results: responses query failed", respErr);
    return { kind: "server_error" };
  }

  const { data: clips, error: clipsErr } = await supabase
    .from("phase2_clips")
    .select("id, is_fake, system_prediction")
    .in("id", clipOrder);

  if (clipsErr) {
    logger.error("phase2/results: clips query failed", clipsErr);
    return { kind: "server_error" };
  }

  const respByClip = new Map<string, ResponseRow>();
  for (const r of (responses ?? []) as ResponseRow[]) respByClip.set(r.clip_id, r);
  const clipById = new Map<string, ClipMeta>();
  for (const c of (clips ?? []) as ClipMeta[]) clipById.set(c.id, c);

  const breakdown: BreakdownRow[] = [];
  let participantCorrect = 0;
  let systemCorrect = 0;
  let answered = 0;

  clipOrder.forEach((clipId, idx) => {
    const resp = respByClip.get(clipId);
    const clip = clipById.get(clipId);
    if (!resp || !clip) return;
    answered += 1;

    const groundTruth: "real" | "fake" = clip.is_fake ? "fake" : "real";
    const participantAnswer: "real" | "fake" = resp.answer_is_fake
      ? "fake"
      : "real";
    const systemAnswer: "real" | "fake" | "unknown" =
      clip.system_prediction === null
        ? "unknown"
        : clip.system_prediction
          ? "fake"
          : "real";

    if (participantAnswer === groundTruth) participantCorrect += 1;
    if (systemAnswer === groundTruth) systemCorrect += 1;

    breakdown.push({
      clip_number: idx + 1,
      ground_truth: groundTruth,
      participant_answer: participantAnswer,
      participant_confidence: resp.confidence,
      system_answer: systemAnswer,
    });
  });

  return {
    kind: "ok",
    payload: {
      participant_score: participantCorrect,
      system_score: systemCorrect,
      total: answered,
      breakdown,
    },
  };
}
