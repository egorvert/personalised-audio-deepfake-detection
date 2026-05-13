import Link from "next/link"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"

export default function Page() {
  return (
    <main className="mx-auto flex min-h-svh max-w-4xl flex-col gap-12 px-6 py-12">
      <header className="flex flex-col gap-3">
        <p className="text-xs tracking-wide text-muted-foreground uppercase">
          Research Study
        </p>
        <h1 className="font-heading text-2xl font-medium text-balance sm:text-3xl">
          Personalised Audio Deepfake Detection
        </h1>
        <p className="max-w-2xl text-sm text-muted-foreground">
          Choose a phase to take part. Each phase takes around 10–15 minutes
          and can be completed independently.
        </p>
      </header>

      <section className="grid gap-4 sm:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Phase 1 — Voice Enrolment</CardTitle>
            <CardDescription>
              Record five short sentences so we can build a personalised voice
              profile for testing the detection system.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Approx. 10–15 minutes. Microphone required.
          </CardContent>
          <CardFooter>
            <Button asChild>
              <Link href="/phase1">Start Phase 1</Link>
            </Button>
          </CardFooter>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Phase 2 — Listening Test</CardTitle>
            <CardDescription>
              Listen to 20 short audio clips and judge whether each one is a
              real or synthesised voice. Fully anonymous.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-sm text-muted-foreground">
            Approx. 10–15 minutes. Headphones recommended.
          </CardContent>
          <CardFooter>
            <Button asChild>
              <Link href="/phase2">Start Phase 2</Link>
            </Button>
          </CardFooter>
        </Card>
      </section>
    </main>
  )
}
