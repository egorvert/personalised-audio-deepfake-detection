-- Authoritative study schema. RLS is enabled in 0003_rls.sql; the service
-- role (only the Next.js server) bypasses RLS and is the only writer.

CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- Phase 1 participants. Enrolment status lives on enrollment_jobs, and the
-- prototype vector lives on prototype_embeddings — they used to be columns
-- here.
CREATE TABLE participants (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    consented_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE recordings (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id UUID NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    sentence_index SMALLINT NOT NULL CHECK (sentence_index BETWEEN 1 AND 5),
    storage_path TEXT NOT NULL,
    duration_seconds NUMERIC(6,3),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (participant_id, sentence_index)
);
CREATE INDEX idx_recordings_participant ON recordings(participant_id);

-- Intentionally decoupled from participants. No timestamp column — only a
-- day-bucketed DATE — so an email row cannot be correlated to a participant
-- record via sub-second timing.
CREATE TABLE phase1_emails (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT NOT NULL,
    bucket_day DATE NOT NULL DEFAULT current_date
);

-- Durable enrolment queue. POST /api/phase1/enroll enqueues a row;
-- scripts/enrollment_worker.py drains them.
CREATE TABLE enrollment_jobs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    participant_id UUID NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    status TEXT NOT NULL CHECK (status IN ('queued','processing','done','failed')) DEFAULT 'queued',
    attempts int NOT NULL DEFAULT 0,
    error text,
    created_at timestamptz DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz
);
CREATE INDEX ON enrollment_jobs (status, created_at) WHERE status IN ('queued','processing');
CREATE INDEX idx_enrollment_jobs_participant ON enrollment_jobs(participant_id);

-- Authoritative prototype store. prototypes.json on disk is a derived cache
-- only, regenerated from this table by rebuild_prototype_cache.py.
CREATE TABLE prototype_embeddings (
    participant_id UUID PRIMARY KEY REFERENCES participants(id) ON DELETE CASCADE,
    embedding DOUBLE PRECISION[] NOT NULL,
    model_version TEXT NOT NULL DEFAULT 'two_stream_v1',
    source_recording_ids UUID[] NOT NULL,
    computed_at TIMESTAMPTZ DEFAULT now()
);

-- Deepfakes (one row per generated clip)
CREATE TABLE deepfakes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_participant_id UUID NOT NULL REFERENCES participants(id) ON DELETE CASCADE,
    reference_recording_id UUID NOT NULL REFERENCES recordings(id) ON DELETE CASCADE,
    sentence_index SMALLINT NOT NULL CHECK (sentence_index BETWEEN 1 AND 5),
    storage_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_deepfakes_participant ON deepfakes(source_participant_id);

-- Phase 2 clip pool. `active` + retired_* columns let withdrawal soft-retire
-- a participant's derived clips without breaking anonymous responses.
CREATE TABLE phase2_clips (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    storage_path TEXT NOT NULL,
    is_fake BOOLEAN NOT NULL,
    source_participant_id UUID REFERENCES participants(id) ON DELETE SET NULL,
    sentence_index SMALLINT NOT NULL CHECK (sentence_index BETWEEN 1 AND 5),
    system_score NUMERIC(8,6),
    system_prediction BOOLEAN,
    active BOOLEAN NOT NULL DEFAULT true,
    retired_at TIMESTAMPTZ,
    retired_reason TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_phase2_clips_active ON phase2_clips(active) WHERE active;

CREATE TABLE phase2_sessions (
    id UUID PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    consented_at TIMESTAMPTZ NOT NULL,
    clip_order UUID[] NOT NULL
);

-- We intentionally don't cascade-delete responses when a phase2_clip is
-- removed. Clips should be retired via soft-delete (active=false); the FK is
-- ON DELETE RESTRICT so a stray hard-delete surfaces as an error rather than
-- silently dropping respondents' answers.
CREATE TABLE responses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id UUID NOT NULL REFERENCES phase2_sessions(id) ON DELETE CASCADE,
    clip_id UUID NOT NULL REFERENCES phase2_clips(id) ON DELETE RESTRICT,
    answer_is_fake BOOLEAN NOT NULL,
    confidence SMALLINT NOT NULL CHECK (confidence BETWEEN 1 AND 5),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, clip_id)
);
CREATE INDEX idx_responses_session ON responses(session_id);

-- Lifecycle stamps: single row, id=1. All retention gates read this table.
-- Seeded at migration time so close_study.py only needs to UPDATE.
CREATE TABLE study_lifecycle (
    id INT PRIMARY KEY CHECK (id = 1),
    study_closed_at TIMESTAMPTZ,
    followup_concluded_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
INSERT INTO study_lifecycle (id, study_closed_at, followup_concluded_at, updated_at)
VALUES (1, NULL, NULL, now())
ON CONFLICT (id) DO NOTHING;
