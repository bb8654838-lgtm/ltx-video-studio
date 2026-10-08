import { NextResponse } from "next/server";
import * as kag from "@/app/lib/kaggle";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * GET  -> current session state + quota (the UI polls this)
 * POST -> ON  (start the Kaggle GPU session)
 */
export async function GET() {
  if (!kag.configured()) {
    return NextResponse.json(
      { error: "Kaggle not configured", need: ["KAGGLE_USERNAME", "KAGGLE_KEY"] },
      { status: 503 },
    );
  }
  try {
    const [q, st] = await Promise.all([kag.quota(), kag.status()]);
    return NextResponse.json({
      ...st,
      tunnel: kag.tunnelUrl() || null,
      quota: {
        usedH: +(q.usedS / 3600).toFixed(2),
        reservedH: +(q.reservedS / 3600).toFixed(2),
        totalH: +(q.totalS / 3600).toFixed(1),
        remainingH: +(q.remainingS / 3600).toFixed(2),
        refreshTime: q.refreshTime,
      },
    });
  } catch (e: any) {
    return NextResponse.json({ error: e?.message ?? "kaggle error" }, { status: 502 });
  }
}

export async function POST(req: Request) {
  if (!kag.configured()) {
    return NextResponse.json(
      { error: "Kaggle not configured", need: ["KAGGLE_USERNAME", "KAGGLE_KEY"] },
      { status: 503 },
    );
  }
  let body: { action?: string; probe?: boolean; timeoutSeconds?: number } = {};
  try {
    body = await req.json();
  } catch {
    /* empty body = plain ON */
  }

  // Refuse to start a second session that cannot fit in the remaining quota.
  const q = await kag.quota();
  if (q.remainingS < 300) {
    return NextResponse.json(
      { error: "Not enough weekly GPU quota left", remainingH: q.remainingS / 3600 },
      { status: 429 },
    );
  }

  const res = await kag.start({
    timeoutSeconds: body.timeoutSeconds ?? 5400,
    runProbe: body.probe,
  });
  if (!res.ok) return NextResponse.json(res, { status: 502 });
  return NextResponse.json({
    ok: true,
    started: true,
    timeoutSeconds: body.timeoutSeconds ?? 5400,
    note: "Kaggle queues the session. GPU wait + weight load can take several minutes.",
  });
}
