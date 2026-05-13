"use client";

import * as React from "react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { Loader2 } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";

interface EnrollmentStatus {
  status: "queued" | "processing" | "done" | "failed";
  attempts: number;
  error?: string;
}

const POLL_INTERVAL_MS = 2000;
const SLOW_THRESHOLD_MS = 60_000;

function EnrollingInner() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const jobId = searchParams.get("jobId");

  const [snapshot, setSnapshot] = React.useState<EnrollmentStatus | null>(null);
  const [fatal, setFatal] = React.useState<"not_found" | "no_job" | null>(
    null,
  );
  const [slow, setSlow] = React.useState(false);

  React.useEffect(() => {
    if (!jobId) setFatal("no_job");
  }, [jobId]);

  React.useEffect(() => {
    const timer = setTimeout(() => setSlow(true), SLOW_THRESHOLD_MS);
    return () => clearTimeout(timer);
  }, []);

  React.useEffect(() => {
    if (!jobId || fatal) return;
    let cancelled = false;

    const poll = async () => {
      try {
        const res = await fetch(
          `/api/phase1/enrollment/${encodeURIComponent(jobId)}`,
          { cache: "no-store" },
        );
        if (cancelled) return;

        if (res.status === 401) {
          router.replace("/phase1/consent");
          return;
        }
        if (res.status === 404) {
          setFatal("not_found");
          return;
        }
        if (!res.ok) return;

        const data = (await res.json()) as EnrollmentStatus;
        setSnapshot(data);

        if (data.status === "done") {
          router.replace("/phase1/done");
        }
      } catch {
        // network blip — keep polling
      }
    };

    void poll();
    const interval = setInterval(poll, POLL_INTERVAL_MS);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, [jobId, fatal, router]);

  if (fatal === "no_job") {
    return (
      <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col items-start justify-center gap-6 px-6 py-12">
        <Alert>
          <AlertTitle>No enrolment in progress</AlertTitle>
          <AlertDescription>
            Start from the recording page to begin voice enrolment.
          </AlertDescription>
        </Alert>
        <Button asChild>
          <Link href="/phase1/record">Go to recording</Link>
        </Button>
      </main>
    );
  }

  if (fatal === "not_found") {
    return (
      <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col items-start justify-center gap-6 px-6 py-12">
        <Alert variant="destructive">
          <AlertTitle>Session not found</AlertTitle>
          <AlertDescription>
            We couldn&apos;t find this enrolment job. Your session may have
            expired, or this link belongs to a different participant.
          </AlertDescription>
        </Alert>
        <Button asChild variant="outline">
          <Link href="/">Return home</Link>
        </Button>
      </main>
    );
  }

  if (snapshot?.status === "failed") {
    return (
      <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col items-start justify-center gap-6 px-6 py-12">
        <header className="space-y-3">
          <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
            Phase 1 — Enrolment
          </p>
          <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">
            Something went wrong
          </h1>
        </header>
        <Alert variant="destructive">
          <AlertTitle>Processing failed</AlertTitle>
          <AlertDescription>
            Please contact the researcher at{" "}
            <a
              className="underline underline-offset-4"
              href="mailto:e.vert@se22.qmul.ac.uk"
            >
              e.vert@se22.qmul.ac.uk
            </a>{" "}
            and quote job id{" "}
            <code className="rounded bg-muted px-1.5 py-0.5 font-mono text-xs">
              {jobId}
            </code>
            .
          </AlertDescription>
        </Alert>
        <Button asChild variant="outline">
          <Link href="/">Return home</Link>
        </Button>
      </main>
    );
  }

  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col items-start justify-center gap-8 px-6 py-12">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 1 — Enrolment
        </p>
        <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">
          Processing your voice…
        </h1>
      </header>

      <div
        role="status"
        aria-live="polite"
        className="flex items-center gap-3 text-sm text-muted-foreground"
      >
        <Loader2 className="size-5 animate-spin" aria-hidden="true" />
        <span>
          {snapshot?.status === "processing"
            ? "Building your personalised voice profile."
            : "Waiting for a processing slot."}
        </span>
      </div>

      <p className="max-w-xl text-sm leading-relaxed text-muted-foreground">
        The system is analysing your 5 recordings to create a numerical profile
        of your voice. This usually takes less than a minute. Please keep this
        tab open — you&apos;ll be moved on automatically when it&apos;s done.
      </p>

      {snapshot && snapshot.attempts > 0 && (
        <p className="text-xs text-muted-foreground">
          Attempt {snapshot.attempts}
        </p>
      )}

      {slow && (
        <p className="text-xs text-muted-foreground">
          This is taking longer than usual — please keep this tab open.
        </p>
      )}
    </main>
  );
}

function EnrollingFallback() {
  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl items-center justify-center px-6 py-12">
      <div className="flex items-center gap-3 text-sm text-muted-foreground">
        <Loader2 className="size-4 animate-spin" aria-hidden="true" />
        <span>Loading enrolment status…</span>
      </div>
    </main>
  );
}

export default function Phase1EnrollingPage() {
  return (
    <React.Suspense fallback={<EnrollingFallback />}>
      <EnrollingInner />
    </React.Suspense>
  );
}
