import 'server-only';

// Slim FastAPI client. Only exposes /health for deployment health checks —
// enrolment is done in-process by scripts/enrollment_worker.py, not over HTTP.
// Don't add an enrollSpeaker() helper here; that would reintroduce the race
// the durable queue exists to avoid.

const VDETECT_BASE_URL = process.env.VDETECT_BASE_URL ?? 'http://127.0.0.1:8000';

export type HealthResponse = {
  status: string;
  model_loaded: boolean;
  version?: string;
  model_type?: string | null;
  device?: string | null;
};

export async function health(
  init?: { signal?: AbortSignal; timeoutMs?: number },
): Promise<HealthResponse> {
  const controller = new AbortController();
  const timeoutMs = init?.timeoutMs ?? 5_000;
  const timer = setTimeout(() => controller.abort(), timeoutMs);

  const combinedSignal = init?.signal
    ? anySignal([init.signal, controller.signal])
    : controller.signal;

  try {
    const res = await fetch(`${VDETECT_BASE_URL}/health`, {
      method: 'GET',
      cache: 'no-store',
      signal: combinedSignal,
    });
    if (!res.ok) {
      throw new Error(`vdetect /health returned HTTP ${res.status}`);
    }
    return (await res.json()) as HealthResponse;
  } finally {
    clearTimeout(timer);
  }
}

function anySignal(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  for (const s of signals) {
    if (s.aborted) {
      controller.abort(s.reason);
      break;
    }
    s.addEventListener('abort', () => controller.abort(s.reason), { once: true });
  }
  return controller.signal;
}
