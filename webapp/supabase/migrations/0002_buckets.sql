-- webapp/supabase/migrations/0002_buckets.sql
-- All three buckets PRIVATE (public=false). Access only via signed URLs
-- issued by the Next.js server. 5 MB per-object cap; MIME allowlist enforced
-- by the Supabase Storage API on upload (NOT in SQL — hence this comment).

INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES
  ('recordings',    'recordings',    false, 5242880,  ARRAY['audio/webm','audio/mp4','audio/ogg','audio/wav']),
  ('deepfakes',     'deepfakes',     false, 5242880,  ARRAY['audio/wav','audio/mpeg','audio/webm']),
  ('phase2-clips',  'phase2-clips',  false, 5242880,  ARRAY['audio/wav','audio/mpeg','audio/webm','audio/mp4'])
ON CONFLICT (id) DO NOTHING;
