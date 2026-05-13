import "server-only";

import { cookies } from "next/headers";

import { COOKIE_NAMES } from "@/lib/constants";

// 24h — long enough that a participant stuck on /phase1/enrolling behind a
// contended MPS queue won't outlive the cookie and start 401ing the poll.
// Phase 2 takes 10–15 min so the wider window is harmless there.
const COOKIE_OPTIONS = {
  httpOnly: true,
  secure: true,
  sameSite: "strict" as const,
  path: "/",
  maxAge: 86_400,
};

async function setCookie(name: string, value: string) {
  const store = await cookies();
  store.set({ name, value, ...COOKIE_OPTIONS });
}

async function getCookie(name: string): Promise<string | null> {
  const store = await cookies();
  return store.get(name)?.value ?? null;
}

async function clearCookie(name: string) {
  const store = await cookies();
  store.set({ name, value: "", ...COOKIE_OPTIONS, maxAge: 0 });
}

export async function setStudyPid(pid: string) {
  await setCookie(COOKIE_NAMES.studyPid, pid);
}

export async function getStudyPid(): Promise<string | null> {
  return getCookie(COOKIE_NAMES.studyPid);
}

export async function clearStudyPid() {
  await clearCookie(COOKIE_NAMES.studyPid);
}

export async function setStudySid(sid: string) {
  await setCookie(COOKIE_NAMES.studySid, sid);
}

export async function getStudySid(): Promise<string | null> {
  return getCookie(COOKIE_NAMES.studySid);
}

export async function clearStudySid() {
  await clearCookie(COOKIE_NAMES.studySid);
}
