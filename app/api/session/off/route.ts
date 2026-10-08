import { NextResponse } from "next/server";
import * as kag from "@/app/lib/kaggle";

export const runtime = "nodejs";

/** OFF — tell the worker to exit so GPU quota is released immediately. */
export async function POST() {
  if (!kag.configured()) {
    return NextResponse.json({ error: "Kaggle not configured" }, { status: 503 });
  }
  const res = await kag.stop();
  return NextResponse.json({
    ...res,
    note:
      "Worker asked to exit. Even if this is unreachable, the session self-terminates at sessionTimeoutSeconds.",
  });
}
