"use client";

import Link from "next/link";
import { Loader2 } from "lucide-react";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";

export interface ConsentChecklistProps {
  statements: readonly string[];
  onAgree: () => Promise<void>;
  declineHref: string;
  agreeLabel?: string;
  declineLabel?: string;
}

export function ConsentChecklist({
  statements,
  onAgree,
  declineHref,
  agreeLabel = "Agree and continue",
  declineLabel = "Decline and return home",
}: ConsentChecklistProps) {
  const reactId = React.useId();
  const liveRegionId = `${reactId}-live`;
  const agreeAllId = `${reactId}-agree-all`;

  const [checked, setChecked] = React.useState<boolean[]>(() =>
    statements.map(() => false),
  );
  const [submitting, setSubmitting] = React.useState(false);

  React.useEffect(() => {
    setChecked(statements.map(() => false));
  }, [statements]);

  const agreedCount = checked.filter(Boolean).length;
  const total = statements.length;
  const allAgreed = total > 0 && agreedCount === total;
  const buttonDisabled = !allAgreed || submitting;

  const toggle = (idx: number, value: boolean) => {
    setChecked((prev) => {
      const next = prev.slice();
      next[idx] = value;
      return next;
    });
  };

  const toggleAll = (value: boolean) => {
    setChecked(statements.map(() => value));
  };

  const handleAgree = async () => {
    if (buttonDisabled) return;
    setSubmitting(true);
    try {
      await onAgree();
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      <ul className="space-y-4" role="list">
        {statements.map((statement, idx) => {
          const id = `${reactId}-stmt-${idx}`;
          return (
            <li key={id} className="flex items-start gap-3">
              <Checkbox
                id={id}
                checked={checked[idx] ?? false}
                onCheckedChange={(value) => toggle(idx, value === true)}
                aria-required="true"
                disabled={submitting}
                className="mt-0.5"
              />
              <Label
                htmlFor={id}
                className="text-sm leading-relaxed font-normal"
              >
                {statement}
              </Label>
            </li>
          );
        })}
      </ul>

      <div className="flex items-start gap-3 border-t pt-4">
        <Checkbox
          id={agreeAllId}
          checked={allAgreed}
          onCheckedChange={(value) => toggleAll(value === true)}
          disabled={submitting}
          className="mt-0.5"
        />
        <Label
          htmlFor={agreeAllId}
          className="text-sm leading-relaxed font-medium"
        >
          I agree to all of the above
        </Label>
      </div>

      <div
        id={liveRegionId}
        role="status"
        aria-live="polite"
        className="text-xs text-muted-foreground"
      >
        {agreedCount} of {total} statements agreed
      </div>

      <div className="flex items-center justify-between gap-3">
        <Button
          type="button"
          onClick={handleAgree}
          disabled={buttonDisabled}
          aria-disabled={buttonDisabled}
          aria-describedby={liveRegionId}
        >
          {submitting ? (
            <>
              <Loader2 className="size-4 animate-spin" aria-hidden="true" />
              <span>Submitting…</span>
            </>
          ) : (
            agreeLabel
          )}
        </Button>
        <Button asChild variant="ghost" size="sm">
          <Link href={declineHref}>{declineLabel}</Link>
        </Button>
      </div>
    </div>
  );
}
