import 'server-only';

// In-memory token-bucket rate limiter keyed by arbitrary string.
// At study scale we don't need Redis; bucket state resetting on process
// restart is fine. Periodically sweep stale buckets to bound memory.

type Bucket = {
  tokens: number;
  updatedAt: number;
};

type Limiter = {
  take: (key: string) => boolean;
};

const DEFAULT_EVICT_MS = 10 * 60 * 1000;

export function createRateLimiter(opts?: {
  refillPerSecond?: number;
  burst?: number;
  evictAfterMs?: number;
}): Limiter {
  const refillPerSecond = opts?.refillPerSecond ?? 1;
  const capacity = opts?.burst ?? 1;
  const evictAfterMs = opts?.evictAfterMs ?? DEFAULT_EVICT_MS;
  const buckets = new Map<string, Bucket>();

  function evictStale(now: number): void {
    for (const [k, b] of buckets) {
      if (now - b.updatedAt > evictAfterMs) buckets.delete(k);
    }
  }

  return {
    take(key: string): boolean {
      const now = Date.now();
      // Cheap periodic sweep — roughly every 256 calls.
      if (buckets.size > 0 && (now & 0xff) === 0) evictStale(now);

      const existing = buckets.get(key);
      if (!existing) {
        buckets.set(key, { tokens: capacity - 1, updatedAt: now });
        return true;
      }

      const elapsedSec = (now - existing.updatedAt) / 1000;
      const refilled = Math.min(capacity, existing.tokens + elapsedSec * refillPerSecond);

      if (refilled < 1) {
        existing.tokens = refilled;
        existing.updatedAt = now;
        return false;
      }

      existing.tokens = refilled - 1;
      existing.updatedAt = now;
      return true;
    },
  };
}

// Used by /api/phase2/response: 1 req/s per study_sid.
export const phase2ResponseLimiter = createRateLimiter({
  refillPerSecond: 1,
  burst: 1,
});
