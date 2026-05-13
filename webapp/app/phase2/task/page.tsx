"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { AudioPlayer } from "@/components/audio-player";
import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";

const TOTAL_CLIPS = 20;
const CONFIDENCE_LEVELS = [1, 2, 3, 4, 5] as const;
const REFRESH_BUFFER_MS = 2 * 60 * 1000; // refresh URL if it expires in <2 min

interface SessionPayload {
  sid: string;
  clip_order: string[];
  next_index: number;
}

interface SignedUrlPayload {
  url: string;
  expires_at: string;
}

async function fetchSession(): Promise<SessionPayload | null> {
  const res = await fetch("/api/phase2/session", { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as SessionPayload;
}

async function fetchSignedUrl(clipId: string): Promise<SignedUrlPayload | null> {
  const res = await fetch(`/api/phase2/clip/${clipId}`, { cache: "no-store" });
  if (!res.ok) return null;
  return (await res.json()) as SignedUrlPayload;
}

async function refreshSignedUrl(clipId: string): Promise<SignedUrlPayload | null> {
  const res = await fetch(`/api/phase2/refresh-url/${clipId}`, {
    method: "POST",
  });
  if (!res.ok) return null;
  return (await res.json()) as SignedUrlPayload;
}

export default function Phase2TaskPage() {
  const router = useRouter();

  const [session, setSession] = React.useState<SessionPayload | null>(null);
  const [clipIdx, setClipIdx] = React.useState(0);
  const [signedUrl, setSignedUrl] = React.useState<string | null>(null);
  const [expiresAt, setExpiresAt] = React.useState<string | null>(null);

  const [answerIsFake, setAnswerIsFake] = React.useState<boolean | null>(null);
  const [confidence, setConfidence] = React.useState<number | null>(null);
  const [submitting, setSubmitting] = React.useState(false);
  const [loadError, setLoadError] = React.useState<string | null>(null);

  // 1. Rehydrate session on mount
  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      const payload = await fetchSession();
      if (cancelled) return;
      if (!payload) {
        router.replace("/phase2/consent");
        return;
      }
      if (payload.next_index >= TOTAL_CLIPS) {
        router.replace("/phase2/done");
        return;
      }
      setSession(payload);
      setClipIdx(payload.next_index);
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  // 2. Whenever clipIdx changes, fetch a fresh signed URL
  const currentClipId = session?.clip_order[clipIdx] ?? null;
  React.useEffect(() => {
    if (!currentClipId) return;
    let cancelled = false;
    setLoadError(null);
    setSignedUrl(null);
    setExpiresAt(null);
    setAnswerIsFake(null);
    setConfidence(null);
    (async () => {
      const payload = await fetchSignedUrl(currentClipId);
      if (cancelled) return;
      if (!payload) {
        setLoadError("Could not load the audio clip. Please refresh.");
        return;
      }
      setSignedUrl(payload.url);
      setExpiresAt(payload.expires_at);
    })();
    return () => {
      cancelled = true;
    };
  }, [currentClipId]);

  // 3. Proactive URL refresh when within 2 min of expiry
  React.useEffect(() => {
    if (!currentClipId || !expiresAt) return;
    const msUntilExpiry = new Date(expiresAt).getTime() - Date.now();
    const msUntilRefresh = msUntilExpiry - REFRESH_BUFFER_MS;
    if (msUntilRefresh <= 0) return;
    const timer = setTimeout(async () => {
      const fresh = await refreshSignedUrl(currentClipId);
      if (fresh) {
        setSignedUrl(fresh.url);
        setExpiresAt(fresh.expires_at);
      }
    }, msUntilRefresh);
    return () => clearTimeout(timer);
  }, [currentClipId, expiresAt]);

  // 4. beforeunload warning mid-task
  React.useEffect(() => {
    const handler = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  }, []);

  const replaceUrl = React.useCallback(async () => {
    if (!currentClipId) return null;
    const fresh = await refreshSignedUrl(currentClipId);
    if (!fresh) return null;
    setExpiresAt(fresh.expires_at);
    return fresh.url;
  }, [currentClipId]);

  const handleNext = async () => {
    if (!session || !currentClipId) return;
    if (answerIsFake === null || confidence === null) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/phase2/response", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          clip_id: currentClipId,
          answer_is_fake: answerIsFake,
          confidence,
        }),
      });
      if (res.status === 429) {
        toast.error("Slow down a moment — try again.");
        return;
      }
      if (!res.ok) {
        toast.error("Could not save your answer — please try again.");
        return;
      }
    } catch {
      toast.error("Could not save your answer — please try again.");
      return;
    } finally {
      setSubmitting(false);
    }

    if (clipIdx === TOTAL_CLIPS - 1) {
      router.replace("/phase2/done");
      return;
    }
    setClipIdx((i) => i + 1);
  };

  if (!session) {
    return (
      <main className="mx-auto flex min-h-svh w-full max-w-2xl items-center justify-center px-6 py-12">
        <p
          role="status"
          aria-live="polite"
          className="text-sm text-muted-foreground"
        >
          Loading your session…
        </p>
      </main>
    );
  }

  const clipNumber = clipIdx + 1;
  const progressValue = (clipNumber / TOTAL_CLIPS) * 100;
  const nextDisabled =
    answerIsFake === null || confidence === null || submitting || !signedUrl;

  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col gap-8 px-6 py-12">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 2 — Listening test
        </p>
        <div className="flex items-center justify-between gap-4">
          <h1 className="text-lg font-medium tracking-tight">
            Clip {clipNumber} of {TOTAL_CLIPS}
          </h1>
        </div>
        <Progress
          value={progressValue}
          aria-label={`Progress: clip ${clipNumber} of ${TOTAL_CLIPS}`}
        />
      </header>

      {loadError && (
        <Alert variant="destructive">
          <AlertTitle>Could not load clip</AlertTitle>
          <AlertDescription>{loadError}</AlertDescription>
        </Alert>
      )}

      <section aria-label="Audio clip" className="flex justify-center">
        {signedUrl ? (
          <AudioPlayer
            src={signedUrl}
            onReplaceUrl={replaceUrl}
            className="w-full"
            aria-label={`Audio clip ${clipNumber} of ${TOTAL_CLIPS}`}
          />
        ) : (
          <div
            role="status"
            aria-live="polite"
            className="flex h-14 items-center justify-center text-sm text-muted-foreground"
          >
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            <span className="ml-2">Loading audio…</span>
          </div>
        )}
      </section>

      <fieldset className="space-y-3">
        <legend className="text-sm font-medium">Is this clip real or fake?</legend>
        <div className="grid grid-cols-2 gap-3">
          <Button
            type="button"
            size="lg"
            variant={answerIsFake === false ? "default" : "outline"}
            aria-pressed={answerIsFake === false}
            onClick={() => setAnswerIsFake(false)}
          >
            Real
          </Button>
          <Button
            type="button"
            size="lg"
            variant={answerIsFake === true ? "default" : "outline"}
            aria-pressed={answerIsFake === true}
            onClick={() => setAnswerIsFake(true)}
          >
            Fake
          </Button>
        </div>
      </fieldset>

      <fieldset className="space-y-3">
        <legend className="text-sm font-medium">How confident are you?</legend>
        <div
          role="radiogroup"
          aria-label="Confidence level from 1 (not confident) to 5 (very confident)"
          className="grid grid-cols-5 gap-2"
        >
          {CONFIDENCE_LEVELS.map((level) => (
            <Button
              key={level}
              type="button"
              variant={confidence === level ? "default" : "outline"}
              role="radio"
              aria-checked={confidence === level}
              onClick={() => setConfidence(level)}
              className={cn("tabular-nums")}
            >
              {level}
            </Button>
          ))}
        </div>
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>1 — Not confident</span>
          <span>5 — Very confident</span>
        </div>
      </fieldset>

      <div className="pt-2">
        <Button onClick={handleNext} disabled={nextDisabled} size="lg">
          {submitting ? (
            <>
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              Saving…
            </>
          ) : clipIdx === TOTAL_CLIPS - 1 ? (
            "Finish"
          ) : (
            "Next"
          )}
        </Button>
      </div>
    </main>
  );
}
