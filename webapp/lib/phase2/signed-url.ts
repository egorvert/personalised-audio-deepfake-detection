// Rewrites a Supabase storage signed URL so its host is the public site
// origin instead of the internal `127.0.0.1:54321` that the participant's
// browser cannot reach. Caddy on :4000 has a `/storage/*` handler that
// forwards to the local Supabase, so we just swap scheme+host.
export function toPublicSignedUrl(internalUrl: string): string {
  const publicHost = process.env.NEXT_PUBLIC_SITE_URL?.replace(/\/$/, "");
  if (!publicHost) return internalUrl;
  return internalUrl.replace(/^https?:\/\/[^/]+/, publicHost);
}
