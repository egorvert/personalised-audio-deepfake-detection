#!/usr/bin/env bash
# Boot wrapper for the study-webapp stack. Waits for the Docker daemon, starts
# the Supabase local stack, health-gates each service, exits non-zero with a
# clear log line on any failure. Invoked by com.vdetect.supabase.plist (a
# LaunchAgent with KeepAlive on non-successful exit) so launchd retries on
# failure. <anon> and the repo path are substituted at deploy time.

set -euo pipefail

REPO="/Users/egorvert/Developer/personalised-audio-deepfake-detection"
WEBAPP="${REPO}/webapp"
LOG="/Users/egorvert/Library/Logs/vdetect-boot.log"
SUPABASE_ANON_KEY="<anon>"

exec >> "$LOG" 2>&1
echo ""
echo "=== boot_stack.sh $(date '+%Y-%m-%d %H:%M:%S') ==="

die() {
	echo "FATAL: $*"
	exit 1
}

# Docker Desktop takes 20-40s to open the socket after login; wait up to ~2 min.
echo "[1/4] Waiting for Docker daemon..."
for i in $(seq 1 60); do
	if docker info >/dev/null 2>&1; then
		echo "      docker ready (after ${i} check(s))"
		break
	fi
	echo "      docker not ready (${i}/60), sleeping 2s"
	sleep 2
done
docker info >/dev/null 2>&1 || die "docker never came up within 120s"

# `supabase start` is idempotent: a no-op + status print if already up.
echo "[2/4] Starting Supabase stack..."
cd "$WEBAPP" || die "could not cd to $WEBAPP"
if ! npx --yes supabase start; then
	die "supabase start failed (exit=$?) — check ~/.supabase/ logs"
fi

# Health-gate each Supabase service. Studio is non-essential -> warn only.
echo "[3/4] Health-gating Supabase services..."

check_http() {
	local label="$1" url="$2" retries=30
	for j in $(seq 1 $retries); do
		code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$url" || echo 000)
		if [ "$code" = "200" ]; then
			echo "      ${label} OK (http ${code} after ${j} check(s))"
			return 0
		fi
		echo "      ${label} not ready (${j}/${retries}, http=${code}), sleeping 2s"
		sleep 2
	done
	die "${label} never returned 200 within $((retries*2))s"
}

check_tcp() {
	local label="$1" host="$2" port="$3" retries=30
	for j in $(seq 1 $retries); do
		if nc -z -G 3 "$host" "$port" 2>/dev/null; then
			echo "      ${label} OK (tcp ${host}:${port} after ${j} check(s))"
			return 0
		fi
		echo "      ${label} not ready (${j}/${retries}), sleeping 2s"
		sleep 2
	done
	die "${label} tcp ${host}:${port} never opened within $((retries*2))s"
}

check_http "rest (54321)"   "http://127.0.0.1:54321/rest/v1/?apikey=${SUPABASE_ANON_KEY}"
check_tcp  "postgres (54322)" "127.0.0.1" "54322"
# Studio is non-essential for the participant flow; warn-only.
if ! curl -fs --max-time 3 "http://127.0.0.1:54323" >/dev/null 2>&1; then
	echo "      WARN: studio (54323) not reachable — non-fatal, continuing"
fi

# PM2's own launchd agent brings up nextjs / vdetect-api / enrollment-worker
# once this exits 0. We don't start PM2 here so the failure domains stay isolated.
echo "[4/4] boot_stack.sh completed successfully."
exit 0
