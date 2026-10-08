# LTX Video Studio

Personal text-to-video / image-to-video generator. A Vercel UI drives a Kaggle
GPU session that only runs while you need it.

```
┌──────────────────────┐
│  Vercel (UI)         │   ON / OFF buttons, prompt, image upload
│  307 redirect        │   never carries video bytes
└──────────┬───────────┘
           │ ms-scale API calls only
           ▼
┌──────────────────────┐
│  Tailscale Funnel    │   fixed URL, no egress quota
│  (or ngrok free)     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Kaggle GPU worker   │   FastAPI + LTX-2.3, batch kernel
│  2× T4 (16 GB each)  │
└──────────────────────┘
```

---

## Why these choices

The architecture is not arbitrary. Each of these was chosen because the
obvious alternative fails:

| Decision | Why |
|---|---|
| **Batch `kernels push`, not interactive session** | Interactive sessions die on a 20–60 min *browser* idle timer (Kaggle's two doc pages disagree on which). A batch kernel has no browser, so the timer never applies. |
| **`sessionTimeoutSeconds` set at push** | Makes OFF an **invariant**: the session self-terminates even if Vercel is down and the tunnel dropped. |
| **Vercel 307-redirects, never proxies** | Hobby functions cap responses at 4.5 MB and kill requests at 300 s. A video breaks both. Vercel itself says functions are "a lightweight API layer, not a media server". |
| **Quota read from the API, not the CLI** | `kaggle quota` computes `remaining = total - used` and drops `time_reserved`, so it overstates what's left by the hours your running session already committed. |
| **`copyMessage`-style redirect, no cookies** | Browser talks to the tunnel cross-origin, so auth moves to a header and CORS becomes ours. |
| **T4×2 16 GB ceiling, quantised weights only** | Two *separate* 16 GB cards, not a pooled 32 GB — a single-process model sees 16 GB. FP8 needs Ada/Hopper; the T4 is Turing (SM 7.5), so an FP8 checkpoint would be dequantised and save nothing. |

---

## Setup

### 1. Kaggle account

- **Phone-verify it.** Community-verified across several sources and
  *officially undocumented*, but it gates the Internet toggle — without it the
  whole worker design fails. Do it in the Kaggle UI before anything else.
- Create an API token: **Account → Settings → API → Create New Token**.
- Put it in `~/.kaggle/kaggle.json` locally if you use the CLI.

### 2. Run the egress probe FIRST

Everything downstream assumes a pushed kernel has internet. That is
**unverified** — the `enable_internet` flag is provably *accepted* by the API,
but no public report confirms egress actually happens.

```bash
# from the app: click "🧪 Egress probe"
# or directly:
kaggle kernels push -p kaggle -t 900
kaggle kernels output bb8654838-lgtm/ltx-video-studio
```

`PROBE_VERDICT: EGRESS_OK` or `EGRESS_FAIL` is the last line of the output. If
it fails, the whole "no browser, no manual step" design needs rethinking.

### 3. Deploy to Vercel

```bash
npm install
npx vercel --prod
```

Set the env vars from `.env.example`. The required ones:

| Var | Notes |
|---|---|
| `KAGGLE_USERNAME` | Your kaggle.com profile name, not your email |
| `KAGGLE_KEY` | API token — **never commit** |
| `JOB_TOKEN` | Shared secret for `/generate` and `/shutdown` |
| `TUNNEL_URL` | Fixed public URL of the worker |

### 4. Tunnel

Pick one. Both give a **fixed** URL (which is why the app can hardcode it in an
env var instead of polling for it).

**Tailscale Funnel** — no egress quota, which is what video needs.
```
tailscale up --authkey=$TAILSCALE_AUTHKEY
tailscale funnel --bg --https=8000 $TAILSCALE_FUNNEL_HOST
```
Caveat: Tailscale documents "non-configurable bandwidth limits" but **never
publishes the numbers**. Measure with your own output before trusting it.

**ngrok free** — one auto-assigned dev domain, no request/response body size
limit. Costs **1 GB/month egress**, which is roughly 10 × 100 MB videos.

---

## Model

`ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8` — a 22B DiT audio-video model,
third-party fine-tune of `Lightricks/LTX-2.3`.

Two things about it that shape the setup:

- **Pre-stage the weights as a Kaggle Dataset.** An attached dataset is served
  from Kaggle's shared resource cache and costs ~0 local disk, so you skip both
  the 25 GB download *and* the 20 GB `/kaggle/working` cap. Attach your
  transformer (Q4_K_M, 14.18 GB), the Gemma-3-12B text encoder (7.30 GB), and
  the VAEs, then point `LTX_DATASET` at the mount.
- **Do not use the FP8 files on a T4.** FP8 needs Ada or Hopper; the T4 is
  Turing. A FP8 checkpoint would be dequantised at load and give no memory
  saving over Q4 — you would pay 29 GB instead of 14 GB for nothing.

---

## The ON/OFF contract

| Level | Mechanism | Guarantee |
|---|---|---|
| 1 | `sessionTimeoutSeconds` at push | **Invariant** — fires even if the app is dead |
| 2 | `POST /shutdown` on the worker | Releases quota the instant OFF is pressed |
| 3 | `POST /api/v1/kernels/cancel-session/{id}` | Best effort, needs a session id |

Level 1 is what makes this safe to leave switched on. Levels 2 and 3 are
optimisations.

The UI polls every 10 s, not 3 s: Vercel Hobby has a monthly invocation budget
and one tab open 24/7 at 3 s burns ~86% of it and ends in a 30-day pause.

---

## Known open questions

- Whether `create-session` / `cancel-session` accept ordinary third-party API
  tokens — the CLI does not wrap them, and they are SDK-only.
- Kaggle's idle timeout is **20 min on one docs page and 60 min on another**.
  Moot for batch runs, but unresolved.
- Scratch disk outside `/kaggle/working` is unpublished — don't depend on it.
- A suspected cap of 2 concurrent batch GPU sessions per account
  (third-party error string only, no Kaggle statement).
