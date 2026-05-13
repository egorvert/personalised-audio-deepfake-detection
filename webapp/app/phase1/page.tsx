import Link from "next/link";

import { Button } from "@/components/ui/button";

export default function Phase1LandingPage() {
  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col justify-center gap-10 px-6 py-12">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 1 — Voice enrolment
        </p>
        <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">
          Help test a personalised deepfake detector
        </h1>
      </header>

      <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          This study tests whether tailoring a deepfake detector to an
          individual&apos;s voice helps it spot fakes of that specific person.
          You&apos;ll record yourself reading five short sentences, which we
          use to build a numerical profile of your voice and to generate
          synthetic copies of your voice on a local machine. The personalised
          detector is then measured against a generic one on both real and
          fake audio.
        </p>
        <p>
          Your data is stored under an anonymous identifier — never linked to
          your name. All voice generation happens locally; nothing is sent to
          external services. Recordings and generated audio are deleted within
          14 days of the study ending.
        </p>
        <p>
          Your recordings stay in your browser while you read through the
          sentences and are only sent to the server when you click Finish on
          the last one. You can withdraw at any point before that by simply
          closing the page — nothing will have been submitted.
        </p>
        <p>
          Takes about 10–15 minutes. You&apos;ll need a microphone and a quiet
          room.
        </p>
      </div>

      <div>
        <Button asChild size="lg" className="w-full sm:w-64">
          <Link href="/phase1/consent">Start</Link>
        </Button>
      </div>
    </main>
  );
}
