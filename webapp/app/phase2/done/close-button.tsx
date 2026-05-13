"use client";

import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";

export function CloseButton() {
  const router = useRouter();
  const close = async () => {
    try {
      await fetch("/api/phase2/session", { method: "DELETE" });
    } catch {
      // best effort
    }
    router.push("/phase2");
  };
  return (
    <Button variant="outline" onClick={close}>
      Close
    </Button>
  );
}
