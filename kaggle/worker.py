#!/usr/bin/env python3
"""
LTX Video Studio — Kaggle GPU worker.

Runs as a BATCH kernel (kaggle kernels push), never an interactive session:
a batch kernel has no browser attached, so the 20-60 min browser-idle timer
that kills interactive sessions never applies. The kernel runs top-to-bottom
until it is told to stop or session_timeout_seconds fires.

Serves a small FastAPI app over a public tunnel so the Vercel UI can drive it:

    GET  /healthz          readiness probe (Vercel polls this after ON)
    POST /generate         text-to-video | image-to-video
    POST /shutdown         graceful stop — releases GPU quota immediately

TUNNEL is read from the environment, so the provider is swappable without
touching this file.
"""
# ── HF env vars MUST be set before huggingface_hub is imported ──────────────
# HF's Xet storage backend HANGS on Kaggle (xet-core#527) and Kaggle's proxy
# rewrites HF URLs stripping the scheme -> MissingSchema + partial downloads.
# These are read at import time, so order is load-bearing.
import os

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "120")
# scratch, NOT /kaggle/working (capped at 20 GB for auto-saved output)
os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf_home")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import argparse
import base64
import json
import subprocess
import sys
import threading
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── config from environment (set by the Vercel app at push time) ────────────
MODEL_ID = os.environ.get("LTX_MODEL_ID", "ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8")
MODEL_DATASET = os.environ.get("LTX_DATASET", "")          # optional pre-staged weights
NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")
TS_AUTHKEY = os.environ.get("TS_AUTHKEY", "")
TS_FUNNEL_HOST = os.environ.get("TS_FUNNEL_HOST", "")       # name.tailnet.ts.net
JOB_TOKEN = os.environ.get("JOB_TOKEN", "")                 # shared secret with Vercel
DEFAULT_RES = os.environ.get("LTX_RESOLUTION", "768x512")
DEFAULT_FRAMES = int(os.environ.get("LTX_FRAMES", "97"))
DEFAULT_FPS = int(os.environ.get("LTX_FPS", "24"))

app = FastAPI(title="LTX Video Studio Worker")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tunnel is public; auth is the job token
    allow_methods=["*"],
    allow_headers=["*"],
)

STATE = {"tunnel": None, "ready": False, "started": time.time(), "jobs": 0}


def log(msg):
    print(f"[worker {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ── model loading ──────────────────────────────────────────────────────────
_pipes = {}


def load_model(kind: str = "t2v"):
    """Load the LTX pipeline once and cache it.

    T4x2 gives two SEPARATE 16 GB cards, not a pooled 32 GB. A single-process
    model sees 16 GB. Quantised weights only — FP8 needs Ada/Hopper and the T4
    is Turing (SM 7.5), so an FP8 checkpoint would be dequantised and give no
    memory saving at all.
    """
    if kind in _pipes:
        return _pipes[kind]
    log(f"loading pipeline: {kind} (device={DEVICE})")
    t0 = time.time()
    try:
        from ltx_pipelines.distilled import DistilledPipeline  # noqa: F401
        _pipes[kind] = _build(kind)
    except Exception as e:                     # fall back to a stub for the probe
        log(f"pipeline import failed ({type(e).__name__}: {e}) — /healthz will report not-ready")
        _pipes[kind] = None
    log(f"pipeline ready in {time.time()-t0:.1f}s")
    return _pipes[kind]


DEVICE = "cuda"


def _build(kind):
    import torch
    from ltx_pipelines.distilled import DistilledPipeline

    src = MODEL_DATASET or MODEL_ID
    log(f"weights source: {src}")
    return DistilledPipeline(
        transformer_path=os.path.join(src, "transformer"),
        text_encoder_path=os.path.join(src, "text_encoder"),
        video_vae_path=os.path.join(src, "video_vae"),
        audio_vae_path=os.path.join(src, "audio_vae"),
        device=DEVICE,
    )


# ── API ────────────────────────────────────────────────────────────────────
@app.get("/healthz")
def healthz():
    """Vercel polls this after pressing ON until it answers."""
    up = int(time.time() - STATE["started"])
    return {
        "ok": True,
        "ready": STATE["ready"],
        "tunnel": STATE["tunnel"],
        "uptime_s": up,
        "jobs": STATE["jobs"],
        "model": MODEL_ID,
    }


class GenReq(BaseModel):
    prompt: str = ""
    image_b64: str | None = None      # image-to-video
    width: int = 768
    height: int = 512
    frames: int = DEFAULT_FRAMES
    fps: int = DEFAULT_FPS
    seed: int = -1
    token: str = ""


@app.post("/generate")
def generate(req: GenReq):
    if JOB_TOKEN and req.token != JOB_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")
    if not STATE["ready"]:
        raise HTTPException(status_code=503, detail="model still loading")
    if not req.prompt and not req.image_b64:
        raise HTTPException(status_code=400, detail="need prompt or image")

    STATE["jobs"] += 1
    kind = "i2v" if req.image_b64 else "t2v"
    log(f"job #{STATE['jobs']} {kind} prompt={req.prompt[:60]!r} {req.width}x{req.height}")

    pipe = load_model(kind)
    if pipe is None:
        raise HTTPException(status_code=503, detail="pipeline unavailable on this GPU")

    out = f"/kaggle/working/out_{uuid.uuid4().hex[:8]}.mp4"
    try:
        import torch
        seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        pipe.generate(
            prompt=req.prompt,
            width=req.width,
            height=req.height,
            num_frames=req.frames,
            fps=req.fps,
            seed=seed,
            output_path=out,
            input_image=req.image_b64,
        )
        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        log(f"job done -> {out} ({os.path.getsize(out)/1e6:.1f} MB)")
        return {"ok": True, "seed": seed, "video_b64": b64,
                "bytes": os.path.getsize(out)}
    except Exception as e:
        log(f"job failed: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=str(e)[:300])


@app.post("/shutdown")
def shutdown(token: str = ""):
    """OFF button — exit immediately so the GPU quota is released at once."""
    if JOB_TOKEN and token != JOB_TOKEN:
        raise HTTPException(status_code=401, detail="bad token")
    log("shutdown requested — exiting")
    threading.Timer(1.0, lambda: os._exit(0)).start()
    return {"ok": True, "stopping": True}


# ── tunnel ─────────────────────────────────────────────────────────────────
def start_tunnel(port=8000):
    """Expose the local server. Provider chosen by which token is present."""
    if TS_AUTHKEY and TS_FUNNEL_HOST:
        log("starting Tailscale Funnel")
        cmd = ["tailscale", "funnel", "--bg", f"--https={port}", TS_FUNNEL_HOST]
        subprocess.Popen(cmd, env={**os.environ}, stdout=subprocess.DEVNULL)
        STATE["tunnel"] = f"https://{TS_FUNNEL_HOST}"
        return STATE["tunnel"]

    if NGROK_AUTHTOKEN:
        log("starting ngrok")
        subprocess.Popen(["ngrok", "http", str(port), "--log", "stdout"],
                         env={**os.environ, "NGROK_AUTHTOKEN": NGROK_AUTHTOKEN})
        for _ in range(30):
            time.sleep(2)
            try:
                import urllib.request
                with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels") as r:
                    for t in json.load(r)["tunnels"]:
                        if t["proto"] == "https":
                            STATE["tunnel"] = t["public_url"]
                            return STATE["tunnel"]
            except Exception:
                continue

    log("WARNING: no tunnel token configured — worker is LAN-only")
    return None


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--probe", action="store_true",
                    help="run the internet-egress probe and exit")
    args = ap.parse_args()

    # ── EGRESS PROBE ───────────────────────────────────────────────────────
    # Everything downstream assumes a pushed kernel has internet. That is
    # UNVERIFIED on Kaggle: the enable_internet flag is provably accepted, but
    # no public report confirms egress actually happens. Settle it in 5 min.
    if args.probe:
        import urllib.request
        try:
            with urllib.request.urlopen("https://api.ipify.org", timeout=20) as r:
                ip = r.read().decode().strip()
            log(f"EGRESS OK — public ip = {ip}")
            print(json.dumps({"egress": True, "ip": ip}))
        except Exception as e:
            log(f"EGRESS FAILED — {type(e).__name__}: {e}")
            print(json.dumps({"egress": False, "error": str(e)}))
        sys.exit(0)

    import uvicorn

    threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0",
                                                port=args.port, log_level="warning"),
                     daemon=True).start()
    time.sleep(3)
    url = start_tunnel(args.port)
    log(f"tunnel ready: {url}")

    threading.Thread(target=load_model, args=("t2v",), daemon=True).start()
    time.sleep(30)
    STATE["ready"] = load_model("t2v") is not None
    log(f"READY={STATE['ready']}")

    # Belt-and-braces: exit even if nobody presses OFF. session_timeout_seconds
    # is the outer guarantee; this is the inner one.
    while True:
        time.sleep(30)
        if int(time.time() - STATE["started"]) > 12 * 3600:
            log("12h ceiling reached — exiting")
            os._exit(0)
