"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";
import { ArrowLeft, Download } from "lucide-react";
import { toast } from "sonner";

import {
  Alert,
  AlertDescription,
  AlertTitle,
} from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ConsentChecklist } from "@/components/consent-checklist";
import { Separator } from "@/components/ui/separator";
import { PHASE2_CONSENT } from "@/lib/constants";

export default function Phase2ConsentPage() {
  const router = useRouter();
  const [poolNotReady, setPoolNotReady] = React.useState(false);

  const handleAgree = async () => {
    setPoolNotReady(false);
    try {
      const res = await fetch("/api/phase2/consent", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ agreed: true }),
      });
      if (res.status === 503) {
        setPoolNotReady(true);
        return;
      }
      if (!res.ok) {
        toast.error("Something went wrong — please refresh");
        return;
      }
      router.push("/phase2/task");
    } catch {
      toast.error("Something went wrong — please refresh");
    }
  };

  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col gap-8 px-6 py-12">
      <div>
        <Button asChild variant="ghost" size="sm" className="-ml-3">
          <Link href="/phase2">
            <ArrowLeft className="size-4" aria-hidden="true" />
            Back
          </Link>
        </Button>
      </div>

      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 2 — Consent
        </p>
        <h1 className="text-2xl font-medium tracking-tight sm:text-3xl">
          Please review and agree before continuing
        </h1>
      </header>

      {poolNotReady && (
        <Alert>
          <AlertTitle>Study is currently being set up</AlertTitle>
          <AlertDescription>
            Please check back later — the listening clips aren&apos;t ready
            yet.
          </AlertDescription>
        </Alert>
      )}

      <section
        aria-label="Study documents"
        className="flex flex-col gap-3 sm:flex-row"
      >
        <Button asChild variant="outline" size="sm">
          <a href="/pdfs/pis-phase2.pdf" target="_blank" rel="noopener noreferrer">
            <Download className="size-4" aria-hidden="true" />
            Participant Info Sheet (PDF)
          </a>
        </Button>
        <Button asChild variant="outline" size="sm">
          <a href="/pdfs/consent-phase2.pdf" target="_blank" rel="noopener noreferrer">
            <Download className="size-4" aria-hidden="true" />
            Consent Form (PDF)
          </a>
        </Button>
      </section>

      <Separator />

      <p className="text-sm text-muted-foreground">
        Please read each statement and tick the box to confirm before
        continuing.
      </p>

      <ConsentChecklist
        statements={PHASE2_CONSENT}
        onAgree={handleAgree}
        declineHref="/phase2"
      />
    </main>
  );
}
