# Vercel env vars — LTX Video Studio

Project: `ltx-video-studio` · URL: https://ltx-video-studio.vercel.app

Add these under **Settings → Environment Variables**. Tick **all three
environments** (Production, Preview, Development) so a preview deploy works
too, then redeploy — env vars are baked in at build time, so an existing
deployment will keep reporting `Kaggle not configured` until you redeploy.

## Required

| Name | Value |
|---|---|
| `KAGGLE_USERNAME` | `msdhoni99770` |
| `KAGGLE_KEY` | `KGAT_f2ae8f3fceaeb1dd6ea7a32d2915d0b0` |
| `GITHUB_OWNER` | `bb8654838-lgtm` |
| `GITHUB_REPO` | `ltx-video-studio` |
| `NGROK_AUTHTOKEN` | `3KPXlIa0V04pwwAIdOG2jYu0Aqx_4mtHZacXb6x4txXQie1LM` |
| `JOB_TOKEN` | `3v_Rl6W_igTMAoeV0H4GwRTn5rIRRj11GXXqrFQT5cM` |

`GITHUB_OWNER` / `GITHUB_REPO` matter because `app/lib/kaggle.ts` fetches
`worker.py` and `probe.py` from raw.githubusercontent.com at push time rather
than storing the kernel source in the function bundle.

## Optional

| Name | Value | Notes |
|---|---|---|
| `KAGGLE_KERNEL_SLUG` | `msdhoni99770/ltx-video-studio-worker` | override the default slug. Kaggle derives the real slug from the **title**, not the metadata id — id `user/ltx-video-studio` + title `LTX Video Studio Worker` lands at `user/ltx-video-studio-worker` |
| `NGROK_DOMAIN` | `rewind-ambition-iodine.ngrok-free.dev` | pinned ngrok dev domain, passed to the kernel |
| `TUNNEL_URL` | `https://rewind-ambition-iodine.ngrok-free.dev` | only a fallback |

`TUNNEL_URL` is not required in the normal path. The worker prints
`TUNNEL_URL_FOR_VERCEL: <url>` into its kernel log and the app reads the URL
from there, which is what makes it survive the hostname changing between
sessions. Set it only if you pin a fixed ngrok dev domain and want a hard
default.

Leave `TAILSCALE_AUTHKEY` and `TAILSCALE_FUNNEL_HOST` unset — ngrok is the
active tunnel and Tailscale stays an unused fallback.

## Verify after redeploying

```bash
curl -s https://ltx-video-studio.vercel.app/api/session
```

`{"error":"Kaggle not configured"}` means the vars did not take. Anything with
quota numbers means they did.

## Rotating anything

`KAGGLE_KEY`, `NGROK_AUTHTOKEN` and `JOB_TOKEN` are all credentials. If any of
them is pasted into a chat, a commit, or a shared screenshot, regenerate it in
the source dashboard and update Vercel — `JOB_TOKEN` can be anything, it only
has to match on both sides.