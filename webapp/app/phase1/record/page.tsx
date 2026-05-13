"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Recorder } from "@/components/recorder";
import { useRecorder } from "@/hooks/use-recorder";
import { PROMPT_SENTENCES } from "@/lib/constants";

interface SessionSnapshot {
  pid: string;
  completedSentences: number[];
  enrollmentStatus:
    | "not_started"
    | "queued"
    | "processing"
    | "done"
    | "failed";
}

async function recoverJobId(): Promise<string | null> {
  // /api/phase1/enroll is idempotent: if a queued/processing job exists for
  // this participant it returns the existing jobId without inserting a duplicate.
  try {
    const res = await fetch("/api/phase1/enroll", { method: "POST" });
    if (!res.ok) return null;
    const data = (await res.json()) as { jobId?: string };
    return data.jobId ?? null;
  } catch {
    return null;
  }
}

async function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function uploadRecording(
  blob: Blob,
  sentenceIndex: number,
  durationSec: number,
): Promise<void> {
  const delays = [1000, 2000, 4000];
  let lastErr: unknown = null;

  for (let attempt = 0; attempt <= delays.length; attempt++) {
    try {
      const form = new FormData();
      form.append("file", blob, `sentence-${sentenceIndex}.webm`);
      form.append("sentence_index", String(sentenceIndex));
      form.append("duration_seconds", String(durationSec));
      const res = await fetch("/api/phase1/recording", {
        method: "POST",
        body: form,
      });
      if (res.ok) return;
      // 4xx responses are not retried
      if (res.status >= 400 && res.status < 500) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.error ?? `upload failed (${res.status})`);
      }
      lastErr = new Error(`upload failed (${res.status})`);
    } catch (err) {
      lastErr = err;
    }
    if (attempt < delays.length) {
      await sleep(delays[attempt]);
    }
  }
  throw lastErr ?? new Error("upload failed");
}

export default function Phase1RecordPage() {
  const router = useRouter();
  const recorder = useRecorder();
  const TOTAL = PROMPT_SENTENCES.length;

  const [hydrated, setHydrated] = React.useState(false);
  const [completed, setCompleted] = React.useState<boolean[]>(() =>
    PROMPT_SENTENCES.map(() => false),
  );
  const [currentIdx, setCurrentIdx] = React.useState(0);
  const [rehydrateError, setRehydrateError] = React.useState<string | null>(
    null,
  );

  // Batch mode: recordings are held in memory until the user clicks Finish
  // on the last sentence. Nothing is uploaded until then.
  const blobsRef = React.useRef<(Blob | null)[]>(
    PROMPT_SENTENCES.map(() => null),
  );
  const durationsRef = React.useRef<number[]>(
    PROMPT_SENTENCES.map(() => 0),
  );
  const [uploadProgress, setUploadProgress] = React.useState<{
    current: number;
    total: number;
  } | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch("/api/phase1/session", {
          method: "GET",
          cache: "no-store",
        });
        if (res.status === 401) {
          router.replace("/phase1/consent");
          return;
        }
        if (!res.ok) {
          setRehydrateError("Could not load your session. Please refresh.");
          setHydrated(true);
          return;
        }
        const data = (await res.json()) as SessionSnapshot;
        if (cancelled) return;

        if (data.enrollmentStatus === "done") {
          router.replace("/phase1/done");
          return;
        }
        if (
          data.enrollmentStatus === "queued" ||
          data.enrollmentStatus === "processing" ||
          data.enrollmentStatus === "failed"
        ) {
          const jobId = await recoverJobId();
          if (cancelled) return;
          router.replace(
            jobId
              ? `/phase1/enrolling?jobId=${encodeURIComponent(jobId)}`
              : "/phase1/enrolling",
          );
          return;
        }

        // Batch mode: recordings live in browser memory, so we always
        // start from sentence 1 when no enrollment has been enqueued.
        // Any partially-uploaded rows from a prior tab-close will be
        // overwritten via UPSERT when the participant finishes this time.
        setCurrentIdx(0);
        setCompleted(PROMPT_SENTENCES.map(() => false));
        setHydrated(true);
      } catch {
        setRehydrateError("Could not load your session. Please refresh.");
        setHydrated(true);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [router]);

  const handleNext = async () => {
    if (!recorder.blob) return;

    // Save this sentence's blob + duration locally
    blobsRef.current[currentIdx] = recorder.blob;
    durationsRef.current[currentIdx] = recorder.durationSec;

    const isLast = currentIdx === TOTAL - 1;

    if (!isLast) {
      setCompleted((prev) => {
        const next = prev.slice();
        next[currentIdx] = true;
        return next;
      });
      recorder.reset();
      setCurrentIdx((i) => i + 1);
      return;
    }

    // Last sentence — Finish. Batch-upload all 5 recordings, then enroll.
    recorder.setUploading();
    setCompleted((prev) => {
      const next = prev.slice();
      next[currentIdx] = true;
      return next;
    });

    const allBlobs = blobsRef.current;
    const allDurations = durationsRef.current;

    try {
      for (let i = 0; i < TOTAL; i++) {
        const blob = allBlobs[i];
        if (!blob) {
          // Shouldn't happen in normal flow - defensive.
          throw new Error(`Missing recording for sentence ${i + 1}`);
        }
        setUploadProgress({ current: i + 1, total: TOTAL });
        await uploadRecording(blob, i + 1, allDurations[i]);
      }
    } catch {
      toast.error("Upload failed — please try again");
      setUploadProgress(null);
      // Roll the local "completed" flag off the last sentence so the
      // Finish button re-enables and the user can retry.
      setCompleted((prev) => {
        const next = prev.slice();
        next[currentIdx] = false;
        return next;
      });
      recorder.setReview();
      return;
    }

    // All uploaded — enqueue enrollment
    try {
      const res = await fetch("/api/phase1/enroll", { method: "POST" });
      if (!res.ok) {
        toast.error("Enrolment failed — please contact the researcher");
        setUploadProgress(null);
        recorder.setReview();
        return;
      }
      const data = (await res.json()) as { jobId: string };
      router.replace(
        `/phase1/enrolling?jobId=${encodeURIComponent(data.jobId)}`,
      );
    } catch {
      toast.error("Enrolment failed — please contact the researcher");
      setUploadProgress(null);
      recorder.setReview();
    }
  };

  const sentence = PROMPT_SENTENCES[currentIdx];
  const sentenceNumber = currentIdx + 1;

  if (recorder.state === "error" && recorder.errorReason === "permission_denied") {
    return (
      <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col justify-center gap-6 px-6 py-12">
        <Alert variant="destructive">
          <AlertTitle>Microphone access required</AlertTitle>
          <AlertDescription>
            This study needs microphone access to record your voice. Please
            grant permission in your browser settings and refresh this page.
          </AlertDescription>
        </Alert>
      </main>
    );
  }

  if (!hydrated) {
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

  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col gap-10 px-6 py-12">
      <header className="space-y-1">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 1 — Recording
        </p>
        <h1 className="text-2xl font-medium tracking-tight">
          Read each sentence aloud
        </h1>
      </header>

      {rehydrateError && (
        <Alert variant="destructive">
          <AlertDescription>{rehydrateError}</AlertDescription>
        </Alert>
      )}

      <Recorder
        sentence={sentence}
        sentenceNumber={sentenceNumber}
        totalSentences={TOTAL}
        completed={completed}
        state={recorder.state}
        errorReason={recorder.errorReason}
        elapsedMs={recorder.elapsedMs}
        blob={recorder.blob}
        uploadProgress={uploadProgress}
        onStart={() => {
          void recorder.start();
        }}
        onStop={recorder.stop}
        onRetry={recorder.reset}
        onNext={() => {
          void handleNext();
        }}
      />
    </main>
  );
}
