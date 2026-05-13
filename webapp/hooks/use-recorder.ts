"use client";

import * as React from "react";

import { MAX_RECORDING_MS, MIN_RECORDING_MS } from "@/lib/constants";

export type RecorderState =
  | "idle"
  | "recording"
  | "review"
  | "uploading"
  | "error";

export type RecorderErrorReason =
  | "permission_denied"
  | "unsupported_browser"
  | "too_short"
  | "recorder_error";

export interface UseRecorderReturn {
  state: RecorderState;
  errorReason: RecorderErrorReason | null;
  blob: Blob | null;
  mimeType: string | null;
  durationSec: number;
  elapsedMs: number;
  start: () => Promise<void>;
  stop: () => void;
  reset: () => void;
  setUploading: () => void;
  setReview: () => void;
}

const MIMETYPE_CANDIDATES = [
  "audio/webm;codecs=opus",
  "audio/webm",
  "audio/mp4",
] as const;

function pickMimeType(): string | null {
  if (typeof MediaRecorder === "undefined") return null;
  for (const candidate of MIMETYPE_CANDIDATES) {
    if (MediaRecorder.isTypeSupported(candidate)) return candidate;
  }
  return null;
}

export function useRecorder(): UseRecorderReturn {
  const [state, setState] = React.useState<RecorderState>("idle");
  const [errorReason, setErrorReason] =
    React.useState<RecorderErrorReason | null>(null);
  const [blob, setBlob] = React.useState<Blob | null>(null);
  const [mimeType, setMimeType] = React.useState<string | null>(null);
  const [durationSec, setDurationSec] = React.useState(0);
  const [elapsedMs, setElapsedMs] = React.useState(0);

  const recorderRef = React.useRef<MediaRecorder | null>(null);
  const streamRef = React.useRef<MediaStream | null>(null);
  const chunksRef = React.useRef<Blob[]>([]);
  const startedAtRef = React.useRef<number>(0);
  const tickTimerRef = React.useRef<ReturnType<typeof setInterval> | null>(
    null,
  );
  const autoStopTimerRef = React.useRef<ReturnType<typeof setTimeout> | null>(
    null,
  );

  const clearTimers = React.useCallback(() => {
    if (tickTimerRef.current) {
      clearInterval(tickTimerRef.current);
      tickTimerRef.current = null;
    }
    if (autoStopTimerRef.current) {
      clearTimeout(autoStopTimerRef.current);
      autoStopTimerRef.current = null;
    }
  }, []);

  const releaseStream = React.useCallback(() => {
    if (streamRef.current) {
      for (const track of streamRef.current.getTracks()) track.stop();
      streamRef.current = null;
    }
  }, []);

  const reset = React.useCallback(() => {
    clearTimers();
    releaseStream();
    recorderRef.current = null;
    chunksRef.current = [];
    setState("idle");
    setErrorReason(null);
    setBlob(null);
    setMimeType(null);
    setDurationSec(0);
    setElapsedMs(0);
  }, [clearTimers, releaseStream]);

  const stop = React.useCallback(() => {
    const rec = recorderRef.current;
    if (!rec) return;
    if (rec.state === "recording" || rec.state === "paused") {
      try {
        rec.stop();
      } catch {
        // swallow: already stopped
      }
    }
  }, []);

  const start = React.useCallback(async () => {
    setErrorReason(null);
    setBlob(null);
    setDurationSec(0);
    setElapsedMs(0);
    chunksRef.current = [];

    const chosenMime = pickMimeType();
    if (!chosenMime) {
      setState("error");
      setErrorReason("unsupported_browser");
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      setState("error");
      setErrorReason("permission_denied");
      return;
    }
    streamRef.current = stream;
    setMimeType(chosenMime);

    let rec: MediaRecorder;
    try {
      rec = new MediaRecorder(stream, { mimeType: chosenMime });
    } catch {
      releaseStream();
      setState("error");
      setErrorReason("unsupported_browser");
      return;
    }
    recorderRef.current = rec;

    rec.ondataavailable = (ev) => {
      if (ev.data && ev.data.size > 0) chunksRef.current.push(ev.data);
    };

    rec.onstart = () => {
      startedAtRef.current = performance.now();
      setState("recording");
      tickTimerRef.current = setInterval(() => {
        setElapsedMs(performance.now() - startedAtRef.current);
      }, 100);
      autoStopTimerRef.current = setTimeout(() => {
        if (recorderRef.current && recorderRef.current.state === "recording") {
          try {
            recorderRef.current.stop();
          } catch {
            // swallow
          }
        }
      }, MAX_RECORDING_MS);
    };

    rec.onerror = () => {
      clearTimers();
      releaseStream();
      setState("error");
      setErrorReason("recorder_error");
    };

    rec.onstop = () => {
      const stoppedAt = performance.now();
      const elapsed = stoppedAt - startedAtRef.current;
      clearTimers();
      releaseStream();

      const chunks = chunksRef.current;
      const finalBlob = new Blob(chunks, { type: chosenMime });
      chunksRef.current = [];
      setBlob(finalBlob);
      const seconds = Math.max(0, elapsed / 1000);
      setDurationSec(Number(seconds.toFixed(3)));
      setElapsedMs(elapsed);

      if (elapsed < MIN_RECORDING_MS) {
        setErrorReason("too_short");
        setState("review");
        return;
      }
      setState("review");
    };

    // No timeslice: MediaRecorder buffers the entire recording internally
    // and fires a single dataavailable on stop with a complete EBML header.
    // Earlier versions used rec.start(1000) for incremental chunks, but on
    // Chrome/Windows that occasionally produced files where the header
    // chunk was lost — the resulting blob then started mid-stream and was
    // unrecoverable by ffmpeg/torchcodec.
    try {
      rec.start();
    } catch {
      clearTimers();
      releaseStream();
      setState("error");
      setErrorReason("recorder_error");
    }
  }, [clearTimers, releaseStream]);

  const setUploading = React.useCallback(() => {
    setState("uploading");
  }, []);

  const setReview = React.useCallback(() => {
    setState("review");
  }, []);

  React.useEffect(() => {
    return () => {
      clearTimers();
      if (recorderRef.current && recorderRef.current.state !== "inactive") {
        try {
          recorderRef.current.stop();
        } catch {
          // swallow
        }
      }
      releaseStream();
    };
  }, [clearTimers, releaseStream]);

  return {
    state,
    errorReason,
    blob,
    mimeType,
    durationSec,
    elapsedMs,
    start,
    stop,
    reset,
    setUploading,
    setReview,
  };
}
