import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function Phase2LandingPage() {
  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col justify-center gap-10 px-6 py-12">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 2 — Listening test
        </p>
        <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">
          Can you tell real voices from fake ones?
        </h1>
      </header>

      <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          This study compares how well people can spot computer-generated
          voices against an automated deepfake detector. You&apos;ll listen to
          20 short clips of people reading sentences aloud — some are real
          recordings of volunteer participants, others are synthetic copies
          produced by a voice cloning system. For each clip you decide whether
          it&apos;s real or fake and rate how confident you are.
        </p>
        <p>
          This part of the study is fully anonymous — no personal information
          is collected, and your responses cannot be linked back to you.
          Responses are used only to compare human accuracy to the detection
          system.
        </p>
        <p>
          Takes about 10–15 minutes. Headphones or good speakers are
          recommended.
        </p>
      </div>

      <div>
        <Button asChild size="lg" className="w-full sm:w-64">
          <Link href="/phase2/consent">Start</Link>
        </Button>
      </div>
    </main>
  );
}
