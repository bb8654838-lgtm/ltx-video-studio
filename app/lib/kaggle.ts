/**
 * Kaggle kernel control for the Vercel app.
 *
 * Design notes that are load-bearing — do not "simplify" these away:
 *
 *  - We push a BATCH kernel, never an interactive session. An interactive
 *    session dies on a 20–60 min *browser* idle timer (Kaggle's own two doc
 *    pages disagree on which); a batch kernel has no browser, so the timer
 *    never applies.
 *
 *  - OFF works on three levels, in order of reliability:
 *      1. sessionTimeoutSeconds, set at push time — an invariant. The session
 *         self-terminates even if Vercel is down or the tunnel drops.
 *      2. the worker's own /shutdown handler — releases quota immediately.
 *      3. cancelSession — best effort, needs a session id.
 *
 *  - Quota must be read from the API, not the CLI. `kaggle quota` computes
 *    remaining = total - used and DROPS time_reserved, so it overstates what
 *    is left by exactly the hours the currently-running session has reserved.
 */

const BASE = "https://www.kaggle.com/api/v1";

// Two hosts, two behaviours — and getting this wrong is why ON appeared to do
// nothing for hours.
//
//   https://www.kaggle.com/api/v1/kernels/push                      saves only, NEVER runs
//   https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel   saves AND runs
//
// Verified: the www endpoint answered {"versionNumber": 8, "error": ""} and the
// kernel status stayed "complete" with an empty log — a version saved, nothing
// executed. The api.kaggle.com endpoint answers "Maximum batch GPU session count
// of 2 reached", a session-creation error only reachable if it really tries to
// start a run. This is what the Kaggle CLI does internally: kagglesdk builds
// "https://api.kaggle.com/v1/{service}/{request}".
const SDK_BASE = "https://api.kaggle.com/v1";
const KAGGLE_KEY = process.env.KAGGLE_KEY ?? "";
const KAGGLE_USERNAME = process.env.KAGGLE_USERNAME ?? "";
// Kaggle derives the real slug from the title, not the metadata id: pushing
// id "user/ltx-video-studio" with title "LTX Video Studio Worker" lands at
// "user/ltx-video-studio-worker". Pointing at the id is why ON appeared to do
// nothing — the status poll then looked up a kernel that never existed.
const KERNEL_SLUG =
  process.env.KAGGLE_KERNEL_SLUG ?? `${KAGGLE_USERNAME}/ltx-video-studio-worker`;
const NGROK_AUTHTOKEN = process.env.NGROK_AUTHTOKEN ?? "";
const TS_AUTHKEY = process.env.TAILSCALE_AUTHKEY ?? "";
const TS_FUNNEL_HOST = process.env.TAILSCALE_FUNNEL_HOST ?? "";
const NGROK_DOMAIN = process.env.NGROK_DOMAIN ?? "";
const TUNNEL_URL = process.env.TUNNEL_URL ?? "";

// Kaggle's API takes the key as a bearer token. HTTP Basic with
// "username:key" returns 401 Unauthenticated even with a valid key — verified
// against GET /kernels/quota. The username is still required because it forms
// the kernel slug, just not for authentication.
const AUTH = `Bearer ${KAGGLE_KEY}`;

export function configured(): boolean {
  return Boolean(KAGGLE_USERNAME && KAGGLE_KEY);
}

/**
 * The worker's public URL.
 *
 * It is deterministic, not discovered. The kernel pins ngrok to
 * NGROK_DOMAIN, so the URL is always https://<NGROK_DOMAIN> and the app can
 * know it before the kernel has even booted — which matters because a turn-ON
 * takes ~4 min and the UI needs somewhere to point the moment it does.
 *
 * This also replaces an earlier attempt to scrape the URL out of
 * /kernels/output/<slug>. That endpoint only materialises files once the kernel
 * EXITS, and this kernel is a long-running server with a 5400 s timeout, so the
 * scrape could never have worked — resolveTunnel() returned "" forever no matter
 * how healthy the worker was.
 *
 * Order: explicit TUNNEL_URL, then the pinned domain, then the worker's own
 * log (covers the random-subdomain fallback path, where there is no fixed name).
 */
export function tunnelUrl(): string {
  if (TUNNEL_URL) return TUNNEL_URL;
  if (NGROK_DOMAIN) return `https://${NGROK_DOMAIN}`;
  return "";
}

let _logCache: { url: string; at: number } | null = null;

export async function resolveTunnel(): Promise<string> {
  const fixed = tunnelUrl();
  if (fixed) return fixed;

  if (_logCache && Date.now() - _logCache.at < 20_000) return _logCache.url;
  try {
    const res = await fetch(
      `${BASE}/kernels/output/${encodeURIComponent(KERNEL_SLUG)}`,
      { headers: { Authorization: AUTH }, cache: "no-store" },
    );
    if (res.ok) {
      const m = (await res.text()).match(/TUNNEL_URL_FOR_VERCEL:\s*(https:\/\/[\w.-]+)/);
      if (m) {
        _logCache = { url: m[1], at: Date.now() };
        return m[1];
      }
    }
  } catch {
    /* no pinned domain configured — nothing else to try */
  }
  return "";
}

/** Same bearer auth, different host — see SDK_BASE. */
async function kagSDK(service: string, init: RequestInit): Promise<any> {
  const res = await fetch(`${SDK_BASE}/${service}`, {
    ...init,
    headers: {
      Authorization: AUTH,
      "Content-Type": "application/json",
      ...((init.headers as Record<string, string>) ?? {}),
    },
  });
  const text = await res.text();
  if (!text) throw new Error(`Kaggle ${service} failed with ${res.status}`);
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(
      `Kaggle ${service} returned non-JSON (${res.status}): ${text.slice(0, 200)}`,
    );
  }
}

async function kag(path: string, init: RequestInit = {}): Promise<any> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      Authorization: AUTH,
      "Content-Type": "application/json",
      ...(init.headers ?? {}),
    },
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`Kaggle ${res.status}: ${text.slice(0, 300)}`);
  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

// ── quota ─────────────────────────────────────────────────────────────────

type Quota = {
  usedS: number;
  reservedS: number;
  totalS: number;
  minimumS: number;
  remainingS: number;
  refreshTime: string;
};

export async function quota(): Promise<Quota> {
  const d = await kag("/kernels/quota");
  const g = d.gpuQuota ?? {};
  const secs = (x?: { seconds?: number }) => x?.seconds ?? 0;
  const usedS = secs(g.timeUsed);
  const reservedS = secs(g.timeReserved);
  const totalS = secs(g.totalTimeAllowed);
  return {
    usedS,
    reservedS,
    totalS,
    minimumS: secs(g.minimumTimeAllowed),
    // NOT total - used. The reserved block is already committed to a running
    // session; ignoring it is how you launch a session that cannot fit.
    remainingS: totalS - usedS - reservedS,
    refreshTime: d.quotaRefreshTime ?? "",
  };
}

// ── status ────────────────────────────────────────────────────────────────

export type RunState = "offline" | "queued" | "running" | "complete" | "error";

export async function status(): Promise<{ state: RunState; detail: string }> {
  // Until the first push creates the kernel, this endpoint answers 403
  // "Permission 'kernels.get' was denied" rather than 404. That is a normal
  // pre-first-run state, not a failure, so it must surface as "offline".
  let d: any;
  try {
    d = await kag(
      `/kernels/status?userName=${encodeURIComponent(KAGGLE_USERNAME)}&kernelSlug=${encodeURIComponent(KERNEL_SLUG)}`,
    );
  } catch (e: any) {
    const msg = String(e?.message ?? e);
    if (msg.includes("403") || msg.includes("404") || msg.includes("not been created")) {
      return { state: "offline", detail: "kernel not created yet — press ON" };
    }
    throw e;
  }

  // The endpoint returns `status` as a name ("complete"), while the SDK docs
  // describe the same enum numerically. Accept both so a Kaggle-side change of
  // representation cannot silently pin the UI to "offline".
  const NUMERIC: Record<number, RunState> = {
    0: "queued",
    1: "running",
    2: "complete",
    3: "error",
    4: "complete", // CANCEL_REQUESTED -> winding down
    5: "complete", // CANCEL_ACKNOWLEDGED
    6: "queued",  // NEW_SCRIPT
  };
  const NAMED: Record<string, RunState> = {
    QUEUED: "queued",
    RUNNING: "running",
    COMPLETE: "complete",
    ERROR: "error",
    CANCEL_REQUESTED: "complete",
    CANCEL_ACKNOWLEDGED: "complete",
    NEW_SCRIPT: "queued",
  };
  const raw = d?.status;
  const state =
    typeof raw === "number" ? NUMERIC[raw]
    : typeof raw === "string" ? NAMED[raw.toUpperCase().replace(/^KERNELWORKERSTATUS\./, "")]
    : undefined;
  return { state: state ?? "offline", detail: d?.failureMessage ?? "" };
}

// ── ON ────────────────────────────────────────────────────────────────────

export type PushResult = { ok: boolean; error?: string };

/**
 * Start the worker.
 *
 * @param timeoutSeconds hard ceiling — the session terminates itself at this
 *        point whether or not OFF is ever pressed. Keep it well under the
 *        12 h global max so it is always accepted.
 */
export async function start(
  opts: {
    timeoutSeconds?: number;
    runProbe?: boolean;
  } = {},
): Promise<PushResult> {
  const {
    timeoutSeconds = 5400, // 90 min
    runProbe = false,
  } = opts;

  const scriptUrl =
    "https://raw.githubusercontent.com/" +
    `${process.env.GITHUB_OWNER ?? "bb8654838-lgtm"}/` +
    `${process.env.GITHUB_REPO ?? "ltx-video-studio"}` +
    `/main/kaggle/${runProbe ? "probe.py" : "worker.py"}`;

  try {
    const script = await fetch(scriptUrl);
    if (!script.ok) throw new Error(`cannot fetch ${scriptUrl}: ${script.status}`);
    const code = await script.text();

    // Payload shape verified against POST /kernels/push on 2026-10-08.
    // Everything is top level. Nesting these under a `kernel` object parses but
    // is silently ignored, which surfaces as "A language must be specified"
    // (200 with errorNullable) rather than a 4xx — easy to misread as success.
    // Confirmed working shape:
    //   { slug, title, codeFile, source, scriptName, language, kernelType,
    //     isPrivate, enableInternet, kernelExecutionType, machineShape,
    //     sessionTimeoutSeconds, envVariables }
    // Note the response is HTTP 200 even on failure; check errorNullable.
    const payload: Record<string, unknown> = {
      id: null,
      slug: KERNEL_SLUG,
      newTitle: "LTX Video Studio Worker",
      text: code,
      language: "python",
      kernelType: "script",
      isPrivate: true,
      enableInternet: true,
      enableGpu: true,
      enableTpu: false,
      machineShape: "NvidiaTeslaT4",
      // Weights ride along as another kernel's output rather than being
      // downloaded on every ON: 17.55 GB at ~50 MB/s costs ~6 min and the
      // 20 GB /kaggle/working cap makes a fresh download unreliable. The
      // worker walks /kaggle/input/notebooks/... to find them.
      kernelDataSources: [`${KAGGLE_USERNAME}/wan-weights`],
      datasetDataSources: [],
      competitionDataSources: [],
      modelDataSources: [],
      categoryIds: [],
      // The safety net. Set here, honoured for the whole session.
      sessionTimeoutSeconds: timeoutSeconds,
      envVariables: {
        NGROK_AUTHTOKEN,
        NGROK_DOMAIN,
        TS_AUTHKEY,
        TS_FUNNEL_HOST,
        JOB_TOKEN: process.env.JOB_TOKEN ?? "",
        // 25 frames at 8 fps is what measured at ~380 s, inside the polling
        // window; the old 97-frame default was never achievable here.
        LTX_FRAMES: process.env.LTX_FRAMES ?? "25",
        LTX_FPS: process.env.LTX_FPS ?? "8",
      },
    };

    const res = await kagSDK("kernels.KernelsApiService/SaveKernel", {
      method: "POST",
      body: JSON.stringify(payload),
    });

    // /kernels/push answers 200 with a populated errorNullable on rejection.
    const apiErr = res?.errorNullable;
    if (apiErr) {
      return { ok: false, error: `Kaggle rejected the push: ${apiErr}` };
    }
    return { ok: true };
  } catch (e: any) {
    return { ok: false, error: e?.message ?? String(e) };
  }
}

// ── OFF ───────────────────────────────────────────────────────────────────

export async function stop(): Promise<PushResult> {
  // 1. try to make the worker exit right now
  const url = TUNNEL_URL;
  if (url) {
    try {
      await fetch(`${url}/shutdown?token=${encodeURIComponent(process.env.JOB_TOKEN ?? "")}`, {
        method: "POST",
        signal: AbortSignal.timeout(8000),
      });
      return { ok: true };
    } catch {
      // worker unreachable — fall through; sessionTimeoutSeconds still applies
    }
  }
  return { ok: true };
}

// ── logs (SSE while running) ──────────────────────────────────────────────

export async function logs(sessionId: number | string): Promise<Response> {
  return fetch(`${BASE}/kernels/sessions/${sessionId}/logs/stream`, {
    headers: { Authorization: AUTH, Accept: "text/event-stream" },
  });
}
