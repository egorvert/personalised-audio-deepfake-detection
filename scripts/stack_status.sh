#!/usr/bin/env bash
# One-shot post-boot health probe. Prints the state of every component in the
# study-webapp stack and exits 0 if everything is healthy, 1 otherwise. Useful
# after a cold boot before sharing the Funnel URL.

set -uo pipefail

SUPABASE_ANON_KEY="<anon>"
FAIL=0

pass() { printf "  \033[32mOK\033[0m  %s\n" "$1"; }
fail() { printf "  \033[31mFAIL\033[0m %s — %s\n" "$1" "$2"; FAIL=1; }
warn() { printf "  \033[33mWARN\033[0m %s — %s\n" "$1" "$2"; }

echo "=== study-webapp stack status — $(date '+%Y-%m-%d %H:%M:%S') ==="

# 1. Docker daemon
if docker info >/dev/null 2>&1; then
	pass "Docker daemon"
else
	fail "Docker daemon" "docker info failed"
fi

# 2. Supabase — REST (PostgREST)
REST_URL="http://127.0.0.1:54321/rest/v1/?apikey=${SUPABASE_ANON_KEY}"
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$REST_URL" || echo 000)
if [ "$code" = "200" ]; then
	pass "Supabase REST :54321"
else
	fail "Supabase REST :54321" "http=${code}"
fi

# 3. Supabase — Postgres TCP
if nc -z -G 3 127.0.0.1 54322 2>/dev/null; then
	pass "Supabase Postgres :54322"
else
	fail "Supabase Postgres :54322" "TCP closed"
fi

# 4. Supabase — Studio (non-essential)
if curl -fs --max-time 3 http://127.0.0.1:54323 >/dev/null 2>&1; then
	pass "Supabase Studio :54323"
else
	warn "Supabase Studio :54323" "not reachable (non-essential)"
fi

# 5. vdetect-api (FastAPI) — /health and model_loaded
VDETECT_HEALTH=$(curl -fs --max-time 3 http://127.0.0.1:8000/health 2>/dev/null || echo '')
if [ -n "$VDETECT_HEALTH" ]; then
	if echo "$VDETECT_HEALTH" | grep -q '"model_loaded":[[:space:]]*true'; then
		pass "vdetect-api :8000 (/health, model_loaded=true)"
	else
		fail "vdetect-api :8000" "/health returned but model_loaded != true"
	fi
else
	fail "vdetect-api :8000" "/health unreachable"
fi

# 6. Next.js :3000
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:3000 || echo 000)
if [ "$code" = "200" ] || [ "$code" = "307" ] || [ "$code" = "308" ]; then
	pass "Next.js :3000 (http=${code})"
else
	fail "Next.js :3000" "http=${code}"
fi

# 7. Caddy :4000 (routes both apps through the Funnel)
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:4000 || echo 000)
if [ "$code" = "200" ] || [ "$code" = "307" ] || [ "$code" = "308" ]; then
	pass "Caddy :4000 (http=${code})"
else
	fail "Caddy :4000" "http=${code}"
fi

# 8. Tailscale Funnel
if command -v tailscale >/dev/null 2>&1; then
	if tailscale funnel status 2>/dev/null | grep -q 'http'; then
		pass "Tailscale Funnel"
		echo "     $(tailscale funnel status 2>/dev/null | grep -E 'https?://' | head -1 | xargs)"
	else
		fail "Tailscale Funnel" "tailscale funnel status shows no active mapping"
	fi
else
	fail "Tailscale Funnel" "tailscale CLI not found"
fi

echo ""
if [ $FAIL -eq 0 ]; then
	echo "All checks passed."
	exit 0
else
	echo "One or more checks FAILED — inspect output above."
	exit 1
fi
