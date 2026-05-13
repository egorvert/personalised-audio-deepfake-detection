import { redirect } from "next/navigation";
import { Check, Minus, X } from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import { computePhase2Results } from "@/lib/phase2/results";
import { getStudySid } from "@/lib/supabase/session";

import { CloseButton } from "./close-button";

type AnswerVerdict = "correct" | "incorrect" | "unknown";

function verdictFor(
  answer: "real" | "fake" | "unknown",
  truth: "real" | "fake",
): AnswerVerdict {
  if (answer === "unknown") return "unknown";
  return answer === truth ? "correct" : "incorrect";
}

function VerdictIcon({ verdict }: { verdict: AnswerVerdict }) {
  if (verdict === "unknown") {
    return <Minus className="size-4" aria-hidden="true" />;
  }
  return verdict === "correct" ? (
    <Check className="size-4" aria-hidden="true" />
  ) : (
    <X className="size-4" aria-hidden="true" />
  );
}

function verdictClass(verdict: AnswerVerdict) {
  if (verdict === "correct") return "text-foreground";
  if (verdict === "incorrect") return "text-destructive";
  return "text-muted-foreground";
}

function verdictSrLabel(verdict: AnswerVerdict) {
  if (verdict === "correct") return "(correct)";
  if (verdict === "incorrect") return "(incorrect)";
  return "(not scored)";
}

export default async function Phase2DonePage() {
  const sid = await getStudySid();
  if (!sid) redirect("/phase2");

  const outcome = await computePhase2Results(sid);
  if (outcome.kind !== "ok") redirect("/phase2");
  const { participant_score, system_score, total, breakdown } = outcome.payload;

  return (
    <main className="mx-auto flex min-h-svh w-full max-w-3xl flex-col gap-10 px-6 py-12">
      <header className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Phase 2 — Complete
        </p>
        <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">
          Thanks for taking part
        </h1>
      </header>

      <section
        aria-label="Results summary"
        className="grid gap-4 sm:grid-cols-2"
      >
        <div className="rounded-lg border p-6">
          <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
            You
          </p>
          <p className="mt-2 text-4xl font-medium tabular-nums">
            {participant_score}
            <span className="text-xl text-muted-foreground"> / {total}</span>
          </p>
        </div>
        <div className="rounded-lg border p-6">
          <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
            Detection system
          </p>
          <p className="mt-2 text-4xl font-medium tabular-nums">
            {system_score}
            <span className="text-xl text-muted-foreground"> / {total}</span>
          </p>
        </div>
      </section>

      <section aria-label="Per-clip breakdown" className="space-y-3">
        <h2 className="text-sm font-medium">Clip-by-clip breakdown</h2>
        <div className="overflow-hidden rounded-lg border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="w-12">#</TableHead>
                <TableHead>Ground truth</TableHead>
                <TableHead>Your answer</TableHead>
                <TableHead className="w-28">Confidence</TableHead>
                <TableHead>System answer</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {breakdown.map((row) => {
                const youVerdict = verdictFor(
                  row.participant_answer,
                  row.ground_truth,
                );
                const sysVerdict = verdictFor(
                  row.system_answer,
                  row.ground_truth,
                );
                return (
                  <TableRow key={row.clip_number}>
                    <TableCell className="tabular-nums">
                      {row.clip_number}
                    </TableCell>
                    <TableCell className="capitalize">
                      {row.ground_truth}
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 capitalize",
                          verdictClass(youVerdict),
                        )}
                      >
                        <VerdictIcon verdict={youVerdict} />
                        {row.participant_answer}
                        <span className="sr-only">
                          {verdictSrLabel(youVerdict)}
                        </span>
                      </span>
                    </TableCell>
                    <TableCell className="tabular-nums text-muted-foreground">
                      {row.participant_confidence}/5
                    </TableCell>
                    <TableCell>
                      <span
                        className={cn(
                          "inline-flex items-center gap-1.5 capitalize",
                          verdictClass(sysVerdict),
                        )}
                      >
                        <VerdictIcon verdict={sysVerdict} />
                        {row.system_answer === "unknown"
                          ? "Not scored"
                          : row.system_answer}
                        <span className="sr-only">
                          {verdictSrLabel(sysVerdict)}
                        </span>
                      </span>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </section>

      <CloseButton />
    </main>
  );
}
