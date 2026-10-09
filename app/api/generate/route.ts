import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Generate — hands the browser a job to run against the tunnel, never a proxy.
 *
 * Two measured limits shape this. Vercel Hobby caps a function response at
 * 4.5 MB and kills the request at 300 s, and ngrok's free tier cuts any HTTP
 * request at 300 s with ERR_NGROK_3004. A generation takes ~380 s and the mp4 is
 * ~300 KB, so the transfer cannot be held open on either hop.
 *
 * So the job is split: this route returns the job coordinates immediately and
 * the browser talks to the tunnel itself — POST /generate for a job_id, then
 * GET /job/<id> until it reports done and hands back video_b64. Nothing large
 * ever passes through Vercel.
 */
export async function POST(req: Request) {
  const tunnel = process.env.TUNNEL_URL;
  if (!tunnel) {
    return NextResponse.json(
      { error: "TUNNEL_URL not set — worker has no public URL yet" },
      { status: 503 },
    );
  }

  const token = process.env.JOB_TOKEN ?? "";
  let payload: Record<string, unknown> = {};
  try {
    payload = await req.json();
  } catch {
    /* empty */
  }

  // Ask the worker for state first — a 503 here is a much better UX than
  // letting the browser discover a dead tunnel on its own.
  try {
    const h = await fetch(`${tunnel}/healthz`, {
      signal: AbortSignal.timeout(12000),
      headers: { "ngrok-skip-browser-warning": "1" },
    });
    const st = await h.json();
    if (!st.ready) {
      return NextResponse.json(
        {
          error: "Model still loading — weights take ~4 min on a cold start",
          ready: false,
          uptime_s: st.uptime_s,
        },
        { status: 503 },
      );
    }
  } catch {
    return NextResponse.json(
      { error: "Worker unreachable. Is the session ON?", tunnel },
      { status: 503 },
    );
  }

  return NextResponse.json({
    ok: true,
    action: "run-job",
    tunnel,
    submit: { url: `${tunnel}/generate`, method: "POST" },
    poll: (jobId: string) => `${tunnel}/job/${jobId}?token=${encodeURIComponent(token)}`,
    headers: {
      "Content-Type": "application/json",
      "ngrok-skip-browser-warning": "1",
    },
    body: { ...payload, token },
  });
}
