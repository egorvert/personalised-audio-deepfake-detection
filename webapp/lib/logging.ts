// Wraps console.* with UUID + email scrubbers so participant identifiers
// never end up in server logs. Mirrors vdetect/logging.py so both sides of
// the wire scrub consistently.
export const UUID_REGEX =
  /\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b/gi;

export const EMAIL_REGEX =
  /\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b/gi;

function scrubString(value: string): string {
  return value.replace(UUID_REGEX, "<uuid>").replace(EMAIL_REGEX, "<email>");
}

function scrub(value: unknown): unknown {
  if (typeof value === "string") return scrubString(value);
  if (value instanceof Error) {
    const cloned = new Error(scrubString(value.message));
    cloned.name = value.name;
    if (value.stack) cloned.stack = scrubString(value.stack);
    return cloned;
  }
  if (Array.isArray(value)) return value.map(scrub);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value)) out[k] = scrub(v);
    return out;
  }
  return value;
}

function scrubArgs(args: unknown[]): unknown[] {
  return args.map(scrub);
}

export const logger = {
  log: (...args: unknown[]) => console.log(...scrubArgs(args)),
  info: (...args: unknown[]) => console.info(...scrubArgs(args)),
  warn: (...args: unknown[]) => console.warn(...scrubArgs(args)),
  error: (...args: unknown[]) => console.error(...scrubArgs(args)),
};
