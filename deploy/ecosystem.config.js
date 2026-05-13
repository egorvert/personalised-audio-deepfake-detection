// PM2 ecosystem for the study webapp.
//
// Three apps:
//   nextjs            — Next.js production server on :3000
//   vdetect-api       — FastAPI (uvicorn) on :8000, access log off (Caddy logs already)
//   enrollment-worker — drains the enrollment_jobs queue
//
// SUPABASE_SERVICE_ROLE_KEY and NEXT_PUBLIC_SITE_URL placeholders below must be
// substituted from `supabase status` + `tailscale funnel status` before
// `pm2 start`. CORS is dev-only; FastAPI binds to 127.0.0.1 and is only
// reachable via Caddy's server-side proxy.

const REPO = '/Users/egorvert/Developer/personalised-audio-deepfake-detection';

module.exports = {
	apps: [
		{
			name: 'nextjs',
			cwd: `${REPO}/webapp`,
			script: 'npm',
			args: 'start',
			// Force Node 22 via brew PATH — the system /usr/local/bin/node may be a different version.
			env: {
				NODE_ENV: 'production',
				PORT: '3000',
				PATH: '/opt/homebrew/opt/node@22/bin:/opt/homebrew/bin:/usr/bin:/bin',
				// Disable Next.js telemetry so no participant flow makes outbound third-party requests.
				NEXT_TELEMETRY_DISABLED: '1',
				NEXT_PUBLIC_SITE_URL: '<TAILSCALE_FUNNEL_URL>',
				SUPABASE_URL: 'http://127.0.0.1:54321',
				SUPABASE_SERVICE_ROLE_KEY: '<FROM_SUPABASE_STATUS>',
				VDETECT_BASE_URL: 'http://127.0.0.1:8000',
			},
			max_restarts: 10,
			autorestart: true,
		},
		{
			name: 'vdetect-api',
			cwd: REPO,
			// Launched via scripts/run_vdetect_api.sh; see that file for the
			// reason we need a shell wrapper around `python -m vdetect`.
			interpreter: 'none',
			script: 'scripts/run_vdetect_api.sh',
			env: {
				VDETECT_HOST: '127.0.0.1',
				VDETECT_PORT: '8000',
				VDETECT_ACCESS_LOG: '0',
				VDETECT_MODEL_TYPE: 'fusion',
				VDETECT_WEIGHTS: 'assets/checkpoints/two_stream_film.pt',
				VDETECT_ENROLL_WEIGHTS: 'assets/checkpoints/two_stream_film.pt',
				VDETECT_DB_PATH: 'assets/enrollments/prototypes.json',
				VDETECT_CORS_ORIGINS: 'http://localhost:3000',
				VDETECT_MPS_LOCK: '/tmp/vdetect-mps.lock',
				PYTORCH_ENABLE_MPS_FALLBACK: '1',
			},
			max_restarts: 10,
			autorestart: true,
		},
		{
			name: 'enrollment-worker',
			cwd: REPO,
			interpreter: `${REPO}/.venv/bin/python`,
			script: 'scripts/enrollment_worker.py',
			env: {
				SUPABASE_URL: 'http://127.0.0.1:54321',
				SUPABASE_SERVICE_ROLE_KEY: '<FROM_SUPABASE_STATUS>',
				VDETECT_MODEL_TYPE: 'fusion',
				VDETECT_WEIGHTS: 'assets/checkpoints/two_stream_film.pt',
				VDETECT_ENROLL_WEIGHTS: 'assets/checkpoints/two_stream_film.pt',
				VDETECT_DB_PATH: 'assets/enrollments/prototypes.json',
				VDETECT_MPS_LOCK: '/tmp/vdetect-mps.lock',
				PYTORCH_ENABLE_MPS_FALLBACK: '1',
			},
			max_restarts: 10,
			autorestart: true,
		},
	],
};
