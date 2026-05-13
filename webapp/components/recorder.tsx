"use client";

import * as React from "react";
import { Check, Loader2, Mic, Pause, Play, RotateCcw } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type {
  RecorderState,
  RecorderErrorReason,
} from "@/hooks/use-recorder";

export interface RecorderProps {
  sentence: string;
  sentenceNumber: number;
  totalSentences: number;
  completed: boolean[];
  state: RecorderState;
  errorReason: RecorderErrorReason | null;
  elapsedMs: number;
  blob: Blob | null;
  onStart: () => void;
  onStop: () => void;
  onRetry: () => void;
  onNext: () => void;
  uploadProgress?: { current: number; total: number } | null;
}

function formatElapsed(ms: number): string {
  const totalSeconds = Math.floor(ms / 1000);
  const mm = String(Math.floor(totalSeconds / 60)).padStart(2, "0");
  const ss = String(totalSeconds % 60).padStart(2, "0");
  return `${mm}:${ss}`;
}

function ProgressPills({
  total,
  current,
  completed,
}: {
  total: number;
  current: number;
  completed: boolean[];
}) {
  return (
    <ol
      aria-label="Recording progress"
      className="flex w-full items-center gap-2"
    >
      {Array.from({ length: total }).map((_, idx) => {
        const isCurrent = idx + 1 === current;
        const isDone = completed[idx];
        return (
          <li
            key={idx}
            aria-current={isCurrent ? "step" : undefined}
            aria-label={
              isDone
                ? `Sentence ${idx + 1}, recorded`
                : isCurrent
                  ? `Sentence ${idx + 1}, current`
                  : `Sentence ${idx + 1}, upcoming`
            }
            className={cn(
              "flex h-8 flex-1 items-center justify-center gap-1.5 rounded-full border text-xs font-medium transition-colors",
              isDone
                ? "border-transparent bg-primary text-primary-foreground"
                : isCurrent
                  ? "border-foreground bg-muted text-foreground"
                  : "border-border bg-transparent text-muted-foreground",
            )}
          >
            {isDone ? (
              <>
                <Check className="size-3" aria-hidden="true" />
                <span>{idx + 1}</span>
              </>
            ) : (
              <span>{idx + 1}</span>
            )}
          </li>
        );
      })}
    </ol>
  );
}

export function Recorder({
  sentence,
  sentenceNumber,
  totalSentences,
  completed,
  state,
  errorReason,
  elapsedMs,
  blob,
  onStart,
  onStop,
  onRetry,
  onNext,
  uploadProgress,
}: RecorderProps) {
  const uploadLabel = uploadProgress
    ? `Uploading ${uploadProgress.current} of ${uploadProgress.total}…`
    : "Uploading…";
  const audioUrl = React.useMemo(
    () => (blob ? URL.createObjectURL(blob) : null),
    [blob],
  );

  React.useEffect(() => {
    return () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    };
  }, [audioUrl]);

  const audioRef = React.useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = React.useState(false);

  React.useEffect(() => {
    setIsPlaying(false);
  }, [audioUrl]);

  const togglePlay = () => {
    const el = audioRef.current;
    if (!el) return;
    if (el.paused) {
      el.play().catch(() => setIsPlaying(false));
    } else {
      el.pause();
    }
  };

  const tooShort = state === "review" && errorReason === "too_short";

  return (
    <div className="flex w-full flex-col gap-8">
      <ProgressPills
        total={totalSentences}
        current={sentenceNumber}
        completed={completed}
      />

      <div className="space-y-3">
        <p className="text-xs font-medium tracking-widest text-muted-foreground uppercase">
          Sentence {sentenceNumber} of {totalSentences}
        </p>
        <blockquote className="border-l-2 border-foreground/30 pl-4 text-lg leading-relaxed sm:text-xl">
          {sentence}
        </blockquote>
      </div>

      <div
        role="status"
        aria-live="polite"
        className="min-h-[1.25rem] text-xs text-muted-foreground"
      >
        {state === "idle" && "Ready when you are."}
        {state === "recording" &&
          `Recording… ${formatElapsed(elapsedMs)} (auto-stops at 00:30)`}
        {state === "review" && !tooShort && "Listen back, then continue."}
        {state === "review" &&
          tooShort &&
          "Please speak the full sentence — that clip was too short."}
        {state === "uploading" && uploadLabel}
      </div>

      <div className="flex flex-wrap items-center gap-3">
        {state === "idle" && (
          <Button size="lg" onClick={onStart}>
            <Mic className="size-4" aria-hidden="true" />
            Start recording
          </Button>
        )}

        {state === "recording" && (
          <>
            <Button size="lg" variant="destructive" onClick={onStop}>
              <Pause className="size-4" aria-hidden="true" />
              Stop
            </Button>
            <span className="font-mono text-sm tabular-nums text-muted-foreground">
              {formatElapsed(elapsedMs)}
            </span>
          </>
        )}

        {state === "review" && audioUrl && (
          <>
            <audio
              ref={audioRef}
              src={audioUrl}
              onPlay={() => setIsPlaying(true)}
              onPause={() => setIsPlaying(false)}
              onEnded={() => setIsPlaying(false)}
              className="sr-only"
            />
            <Button
              size="lg"
              variant="outline"
              onClick={togglePlay}
              aria-pressed={isPlaying}
            >
              {isPlaying ? (
                <Pause className="size-4" aria-hidden="true" />
              ) : (
                <Play className="size-4" aria-hidden="true" />
              )}
              {isPlaying ? "Pause" : "Play"}
            </Button>
            <Button size="lg" variant="outline" onClick={onRetry}>
              <RotateCcw className="size-4" aria-hidden="true" />
              Re-record
            </Button>
            <Button size="lg" onClick={onNext} disabled={tooShort}>
              {sentenceNumber === totalSentences ? "Finish" : "Next"}
            </Button>
          </>
        )}

        {state === "uploading" && (
          <Button size="lg" disabled>
            <Loader2 className="size-4 animate-spin" aria-hidden="true" />
            {uploadLabel}
          </Button>
        )}

        {state === "error" && (
          <Button size="lg" variant="outline" onClick={onRetry}>
            <RotateCcw className="size-4" aria-hidden="true" />
            Try again
          </Button>
        )}
      </div>

      {state === "error" && errorReason && (
        <Alert variant="destructive">
          <AlertTitle>Something went wrong</AlertTitle>
          <AlertDescription>
            {errorReason === "permission_denied" &&
              "Microphone access is required. Please grant permission and refresh the page."}
            {errorReason === "unsupported_browser" &&
              "Your browser does not support audio recording. Please try Chrome, Firefox, or Safari."}
            {errorReason === "recorder_error" &&
              "The recorder failed. Please try again."}
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}
