-- webapp/supabase/migrations/0003_rls.sql
-- Deny-by-default RLS. The Next.js server uses the service role, which
-- bypasses RLS; this migration is defence-in-depth so that any request
-- arriving with the anon or authenticated key is denied.
--
-- Deliberate: NO `CREATE POLICY` statements targeting anon or authenticated.
-- Deliberate: NO `GRANT` statements to anon or authenticated.
--
-- Storage buckets: inherit `public=false` from 0002_buckets.sql — no extra
-- RLS on `storage.objects` is needed for our use case because signed URLs
-- are the only browser path to audio.

ALTER TABLE participants          ENABLE ROW LEVEL SECURITY;
ALTER TABLE recordings            ENABLE ROW LEVEL SECURITY;
ALTER TABLE phase1_emails         ENABLE ROW LEVEL SECURITY;
ALTER TABLE enrollment_jobs       ENABLE ROW LEVEL SECURITY;
ALTER TABLE prototype_embeddings  ENABLE ROW LEVEL SECURITY;
ALTER TABLE deepfakes             ENABLE ROW LEVEL SECURITY;
ALTER TABLE phase2_clips          ENABLE ROW LEVEL SECURITY;
ALTER TABLE phase2_sessions       ENABLE ROW LEVEL SECURITY;
ALTER TABLE responses             ENABLE ROW LEVEL SECURITY;
ALTER TABLE study_lifecycle       ENABLE ROW LEVEL SECURITY;
