// Ethics-approved participant-facing strings. VERBATIM from docs/study-webapp-spec.md §§2.2–2.4.
// Do NOT paraphrase, reformat, or translate. Any change requires ethics review.

export const PROMPT_SENTENCES: readonly string[] = [
  "The morning sun cast a warm golden light over the quiet village as the birds began to sing",
  "She picked up the heavy box from the table and carried it carefully through the narrow hallway",
  "Please remember to bring your jacket with you today because the weather may change later this evening",
  "The children laughed and played together in the open field just behind the old village school",
  "I would like a fresh cup of coffee with just a small amount of sugar and a dash of milk please",
] as const;

export const PHASE1_CONSENT: readonly string[] = [
  "I confirm that I have read the Participant Information Sheet dated 26.02.2026 version 0.2 for the above study; or it has been read to me. I have had the opportunity to consider the information, ask questions and have had these answered satisfactorily.",
  "I understand that my participation is voluntary and that I am free to stop taking part in the study at any time during participation without giving any reason and without my rights being affected.",
  "I understand that my data will be accessed by the research team.",
  "I understand that my data will be securely stored in a locally running database instance and in accordance with the data protection guidelines of the Queen Mary University of London for 14 days after the study ends in fully anonymized form.",
  "I understand that I can access the information I have provided and request destruction of that information at any time within 14 days after my participation. I understand that after two weeks have passed, I will not be able to request withdrawal of the information I have provided.",
  "I agree to my voice being audio recorded as part of the enrollment process",
  "I agree to have my voice recordings used to generate synthetic versions of my voice for the sole purpose of testing the deepfake detection system and that generated voice clips will not contain any offensive or derogatory content.",
  "I agree to my anonymised versions of my audio recordings and generated synthetic audio being used in a separate follow-up listening test where participants judge whether audio clips are real or fake.",
  "I understand that the researcher will not identify me in any publications and other study outputs using personal information obtained from this study.",
  "I agree to take part in the above study.",
] as const;

export const PHASE2_CONSENT: readonly string[] = [
  "I confirm that I have read the Participant Information Sheet dated 26.02.2026 version 0.1 for the above study; or it has been read to me. I have had the opportunity to consider the information, ask questions and have had these answered satisfactorily.",
  "I understand that my participation is voluntary and that I am free to stop taking part in the study at any time during participation without giving any reason and without my rights being affected.",
  "I understand that this study is fully anonymous and no personal information is collected.",
  "I understand that my responses will be securely stored in a locally running database instance and in accordance with the data protection guidelines of the Queen Mary University of London for up to 6 months after the study ends in fully anonymised form and will only be accessible to the research team.",
  "I understand that once I submit my responses, they cannot be individually identified or withdrawn because the study is fully anonymous.",
  "I understand that the research team will not identify me in any publications and other study outputs.",
  "I agree to take part in the above study.",
] as const;

export const BUCKETS = {
  recordings: "recordings",
  deepfakes: "deepfakes",
  phase2Clips: "phase2-clips",
} as const;

export const COOKIE_NAMES = {
  studyPid: "study_pid",
  studySid: "study_sid",
} as const;

export const MAX_RECORDING_MS = 30_000;
export const MIN_RECORDING_MS = 1_000;
