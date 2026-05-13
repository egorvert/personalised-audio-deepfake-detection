-- Align enrollment_jobs.created_at with every other created_at in this schema
-- (which are all NOT NULL). The DEFAULT now() stays, so existing INSERTs still
-- work without supplying a value.

ALTER TABLE enrollment_jobs
    ALTER COLUMN created_at SET NOT NULL;
