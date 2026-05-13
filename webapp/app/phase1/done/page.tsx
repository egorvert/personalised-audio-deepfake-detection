"use client";

import * as React from "react";
import { useRouter } from "next/navigation";
import { Loader2 } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

const EMAIL_REGEX = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export default function Phase1DonePage() {
  const router = useRouter();
  const [email, setEmail] = React.useState("");
  const [submitting, setSubmitting] = React.useState(false);
  const [submitted, setSubmitted] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const onSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setError(null);
    if (!EMAIL_REGEX.test(email)) {
      setError("Please enter a valid email address.");
      return;
    }
    setSubmitting(true);
    try {
      const res = await fetch("/api/phase1/email", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email }),
      });
      if (!res.ok) {
        setError("Something went wrong — please try again.");
        return;
      }
      setSubmitted(true);
    } catch {
      setError("Something went wrong — please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  const onClose = async () => {
    try {
      await fetch("/api/phase1/session", { method: "DELETE" });
    } catch {
      // best effort — we still navigate away
    }
    router.push("/phase1");
  };

  return (
    <main className="mx-auto flex min-h-svh w-full max-w-2xl flex-col justify-center gap-10 px-6 py-12">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 1 — Complete
        </p>
        <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">
          Thank you for taking part!
        </h1>
      </header>

      <div className="space-y-4 text-sm leading-relaxed text-muted-foreground">
        <p>
          Your recordings and any generated audio derived from them will be
          deleted within 14 days of the study ending. Your data is stored
          under an anonymous identifier and will not be linked to your name
          in any publications or outputs from this study.
        </p>
        <p>
          If you have any questions or concerns, please contact the researcher
          at{" "}
          <a className="underline underline-offset-4" href="mailto:e.vert@se22.qmul.ac.uk">
            e.vert@se22.qmul.ac.uk
          </a>
          .
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Optional — follow-up invite</CardTitle>
          <CardDescription>
            Would you like to be invited to the separate listening test? Your
            email is stored completely separately from your voice recordings.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {submitted ? (
            <p
              role="status"
              aria-live="polite"
              className="text-sm text-muted-foreground"
            >
              Thanks — your email has been recorded separately from your voice
              data.
            </p>
          ) : (
            <form onSubmit={onSubmit} className="space-y-3" noValidate>
              <div className="space-y-2">
                <Label htmlFor="email">Email address</Label>
                <Input
                  id="email"
                  type="email"
                  autoComplete="email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  disabled={submitting}
                  aria-invalid={error ? true : undefined}
                  aria-describedby={error ? "email-error" : undefined}
                />
                {error && (
                  <p id="email-error" className="text-sm text-destructive">
                    {error}
                  </p>
                )}
              </div>
              <Button type="submit" disabled={submitting || !email}>
                {submitting ? (
                  <>
                    <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                    Submitting…
                  </>
                ) : (
                  "Submit email"
                )}
              </Button>
            </form>
          )}
        </CardContent>
      </Card>

      <div>
        <Button variant="outline" onClick={onClose}>
          Close and return home
        </Button>
      </div>
    </main>
  );
}
