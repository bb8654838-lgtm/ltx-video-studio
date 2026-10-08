import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/** Liveness for the Vercel deployment itself. */
export async function GET() {
  return NextResponse.json({ ok: true, ts: Date.now() });
}
