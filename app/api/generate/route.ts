import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Generate — 307 redirect to the tunnel, never a proxy.
 *
 * Vercel Hobby caps a function response at 4.5 MB and kills the request at
 * 300 s. A generated video is far larger and takes longer than that, so any
 * proxy shape fails. Vercel's own guidance: treat functions as a lightweight
 * API layer, not a media server.
 *
 * Redirecting hands the whole transfer to the browser <-> tunnel connection,
 * so none of the Vercel limits apply. The cost is that CORS and auth become
 * ours to handle, which is why JOB_TOKEN is passed as a header rather than a
 * cookie.
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
  payload.token = token;

  // Ask the worker for state first — a 503 here is a much better UX than a
  // dead tunnel fetch.
  try {
    const h = await fetch(`${tunnel}/healthz`, {
      signal: AbortSignal.timeout(10000),
      headers: { "ngrok-skip-browser-warning": "1" },
    });
    const st = await h.json();
    if (!st.ready) {
      return NextResponse.json(
        { error: "Model still loading", ready: false, uptime_s: st.uptime_s },
        { status: 503 },
      );
    }
  } catch {
    return NextResponse.json(
      { error: "Worker unreachable. Is the session ON?", tunnel },
      { status: 503 },
    );
  }

  // Tell the browser to call the tunnel itself.
  return NextResponse.json(
    {
      ok: true,
      action: "redirect",
      url: `${tunnel}/generate`,
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "ngrok-skip-browser-warning": "1",
      },
      body: payload,
    },
    { status: 200 },
  );
}
