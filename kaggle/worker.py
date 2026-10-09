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
import shutil

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
# Kaggle kernel-metadata has no env-var field, so when this is pushed by the
# Vercel app the secrets arrive through the API's envVariables; for a direct
# `kaggle kernels push` they are written next to this file instead. Env wins.
try:
    import _secrets  # type: ignore
    for _k in ("NGROK_AUTHTOKEN", "NGROK_DOMAIN", "TS_AUTHKEY", "TS_FUNNEL_HOST",
               "JOB_TOKEN", "WEIGHTS_DIR"):
        os.environ.setdefault(_k, getattr(_secrets, _k, ""))
except Exception:
    pass

NGROK_AUTHTOKEN = os.environ.get("NGROK_AUTHTOKEN", "")
NGROK_DOMAIN = os.environ.get("NGROK_DOMAIN", "")               # pinned *.ngrok-free.dev
TS_AUTHKEY = os.environ.get("TS_AUTHKEY", "")
TS_FUNNEL_HOST = os.environ.get("TS_FUNNEL_HOST", "")       # name.tailnet.ts.net
JOB_TOKEN = os.environ.get("JOB_TOKEN", "")                 # shared secret with Vercel
DEFAULT_RES = os.environ.get("LTX_RESOLUTION", "768x512")
DEFAULT_FRAMES = int(os.environ.get("LTX_FRAMES", "25"))
DEFAULT_FPS = int(os.environ.get("LTX_FPS", "8"))

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
# Wan's modules/model.py calls flash_attention() directly and that function
# opens with `assert FLASH_ATTN_2_AVAILABLE`. flash-attn ships no wheel for
# torch 2.11/cu128 and compiling it outlives the kernel, so the import and the
# four call sites are rewritten to use the sibling attention(): same signature,
# and it falls through to torch.nn.functional.scaled_dot_product_attention
# when flash-attn is absent.
WANGIT = os.environ.get("WAN_GIT", "https://github.com/Wan-Video/Wan2.1")
WANPKG = "/kaggle/working/wanpkg"

_pipes = {}


def _patch_wan():
    mp = os.path.join(WANPKG, "wan/modules/model.py")
    src = open(mp).read()
    patched = (src.replace("from .attention import flash_attention",
                           "from .attention import attention")
                  .replace("x = flash_attention(", "x = attention("))
    open(mp, "w").write(patched)
    log(f"flash-attn patched: {patched.count('x = attention(')} attention call sites")


def _prepare_wan():
    if not os.path.isdir(os.path.join(WANPKG, "wan")):
        r = subprocess.run(f"git clone -q --depth 1 {WANGIT} {WANPKG}",
                           shell=True, capture_output=True, timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"wan clone failed: {r.stderr.decode('utf-8', 'replace')[:200]}")
        log("wan repo cloned")
    _patch_wan()


def find_weights():
    """Locate the staged checkpoint directory.

    Kernel output attaches under /kaggle/input/notebooks/<owner>/<kernel>/,
    not /kaggle/input/<kernel>/. Guessing the short path reports "weights not
    found" while the files sit right there, so the tree is walked instead.
    """
    for cand in (os.environ.get("WEIGHTS_DIR", ""), "/kaggle/working/wan"):
        if cand and os.path.exists(os.path.join(cand, "config.json")):
            return cand
    for root, dirs, _ in os.walk("/kaggle/input"):
        if "wan" in dirs and os.path.exists(os.path.join(root, "wan", "config.json")):
            return os.path.join(root, "wan")
        if os.path.basename(root) == "wan" and os.path.exists(os.path.join(root, "config.json")):
            return root
    raise FileNotFoundError("staged Wan checkpoint directory not found under /kaggle/input")


def load_model(kind: str = "t2v"):
    """Build the Wan pipeline once and keep it.

    t5_cpu=True is the difference between running and OOM: the T5-UMT5-XXL
    encoder is 11.36 GB and the transformer another 5.68 GB, while one T4
    exposes 14.56 GiB. device_id picks a single device for the whole pipeline,
    so both halves cannot share cuda:0, and t5_fsdp would need torch.distributed
    initialised across processes. The prompt is encoded once on CPU; every
    sampling step then runs on the GPU.
    """
    if kind in _pipes:
        return _pipes[kind]
    log(f"loading pipeline: {kind}")
    t0 = time.time()
    _prepare_wan()
    sys.path.insert(0, WANPKG)
    import wan
    from wan.configs import WAN_CONFIGS          # wan.configs — plural

    wdir = find_weights()
    log(f"weights: {wdir}")
    pipe = wan.WanT2V(config=WAN_CONFIGS["t2v-1.3B"], checkpoint_dir=wdir,
                      device_id=0, t5_fsdp=False, dit_fsdp=False, t5_cpu=True)
    import torch
    torch.cuda.empty_cache()
    log(f"pipeline ready in {time.time()-t0:.0f}s "
        f"(vram {torch.cuda.memory_allocated(0)/1e9:.1f} GB)")
    _pipes[kind] = pipe
    return pipe


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
    width: int = 480
    height: int = 320
    frames: int = DEFAULT_FRAMES
    fps: int = DEFAULT_FPS
    steps: int = 12
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
    log(f"job #{STATE['jobs']} {kind} prompt={req.prompt[:60]!r} "
        f"{req.width}x{req.height} frames={req.frames}")

    pipe = load_model(kind)
    if pipe is None:
        raise HTTPException(status_code=503, detail="pipeline unavailable on this GPU")

    out = f"/kaggle/working/out_{uuid.uuid4().hex[:8]}.mp4"
    try:
        import torch
        seed = req.seed if req.seed >= 0 else int(torch.randint(0, 2**31 - 1, (1,)).item())
        # WanT2V.generate takes input_prompt / frame_num / sampling_steps /
        # guide_scale / n_prompt - singular names, and no save_file. It decodes
        # the VAE itself and hands back frames shaped (C, T, H, W).
        # frame_num has to satisfy (n-1) % (vae_stride[0] * patch_size[0]) == 0,
        # so snap up to the next valid length instead of passing it raw.
        frame_num = req.frames + (4 - req.frames % 4) % 4
        video = pipe.generate(
            input_prompt=req.prompt or "a cinematic scene",
            size=(req.width, req.height),
            frame_num=frame_num,
            sampling_steps=max(1, min(req.steps, 50)),
            guide_scale=5.0,
            n_prompt="overexposed, static, blurry, low quality, worst quality",
            seed=seed,
            offload_model=True,
        )
        frames = (video.detach().float().cpu().clamp(-1, 1)
                  .add(1).div(2).mul(255).to(torch.uint8))
        if frames.ndim == 4:
            frames = frames.permute(1, 2, 3, 0)      # (C,T,H,W) -> (T,H,W,C)
        elif frames.ndim == 3:
            frames = frames.unsqueeze(-1)
        import imageio.v3 as iio
        iio.imwrite(out, frames.numpy(), fps=req.fps, codec="libx264", quality=8)

        with open(out, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        log(f"job done -> {os.path.getsize(out)/1e6:.2f} MB")
        return {"ok": True, "seed": seed, "video_b64": b64,
                "bytes": os.path.getsize(out), "frames": int(frames.shape[0])}
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
def _install_ngrok():
    """Kaggle ships no ngrok. Static binary download beats the apt repo here."""
    if shutil.which("ngrok"):
        return True
    log("downloading ngrok")
    url = ("https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip")
    try:
        subprocess.run(["curl", "-fsSL", "-o", "/tmp/ngrok.zip", url],
                       check=True, timeout=240)
        subprocess.run(["unzip", "-o", "-q", "/tmp/ngrok.zip", "-d", "/usr/local/bin"],
                       check=True, timeout=120)
        os.chmod("/usr/local/bin/ngrok", 0o755)
        return shutil.which("ngrok") is not None
    except Exception as e:
        log(f"ngrok install failed: {e}")
        return False


def _ngrok_url(tries=45):
    """Read the assigned public URL from ngrok's own local API.

    Discovering the URL at runtime instead of hardcoding it is deliberate: it
    makes the worker correct whether the tunnel uses a pinned dev domain or a
    per-session random one. Tailscale proved the hardcoded-name approach breaks
    here — successive sessions issued ltx-worker, ltx-worker-1 … ltx-worker-5.
    """
    import urllib.request
    for _ in range(tries):
        time.sleep(2)
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels",
                                        timeout=5) as r:
                for t in json.load(r).get("tunnels", []):
                    if t.get("proto") == "https":
                        return t["public_url"]
        except Exception:
            continue
    return None


def start_tunnel(port=8000):
    """Expose the local server. ngrok first; Tailscale kept as a fallback."""
    if NGROK_AUTHTOKEN and _install_ngrok():
        log("starting ngrok")
        cmd = ["ngrok", "http", str(port), "--log", "stdout"]
        if NGROK_DOMAIN:
            # pin the free static dev domain so the URL survives a reconnect
            cmd += ["--domain", NGROK_DOMAIN]
        subprocess.Popen(cmd,
                         env={**os.environ, "NGROK_AUTHTOKEN": NGROK_AUTHTOKEN})
        url = _ngrok_url()
        if url:
            STATE["tunnel"] = url
            # Vercel parses this exact line out of the kernel log, so the UI
            # never needs a hardcoded TUNNEL_URL.
            log(f"TUNNEL_URL_FOR_VERCEL: {url}")
            return url
        log("ngrok started but no https tunnel appeared")

    if TS_AUTHKEY:
        # Kaggle has no systemd, so tailscaled must be launched by hand.
        log("starting tailscaled (userspace, no systemd on Kaggle)")
        subprocess.Popen(
            ["tailscaled", "--tun=userspace-networking",
             "--socket=/tmp/ts/tailscaled.sock", "--state=/tmp/ts/state"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sock = "/tmp/ts/tailscaled.sock"
        for _ in range(20):
            time.sleep(2)
            if os.path.exists(sock):
                break
        subprocess.run(["sh", "-c",
                        f"tailscale --socket={sock} up --authkey={TS_AUTHKEY} "
                        f"--hostname={TS_FUNNEL_HOST.split('.')[0] or 'ltx-worker'} "
                        f"--reset"], capture_output=True, timeout=180)
        subprocess.run(["sh", "-c", f"tailscale --socket={sock} funnel --bg "
                                    f"--https={port}"], capture_output=True, timeout=90)
        out = subprocess.run(["sh", "-c", f"tailscale --socket={sock} status --json"],
                             capture_output=True, text=True, timeout=60).stdout
        try:
            dn = json.loads(out).get("Self", {}).get("DNSName", "").rstrip(".")
            if dn:
                STATE["tunnel"] = f"https://{dn}"
                log(f"TUNNEL_URL_FOR_VERCEL: {STATE['tunnel']}")
                return STATE["tunnel"]
        except Exception:
            pass

    log("WARNING: no working tunnel — worker is LAN-only")
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
    # Tunnel first, model second. Vercel reads TUNNEL_URL_FOR_VERCEL out of the
    # log and then polls /healthz, so the URL needs to exist long before the
    # ~200 s model build finishes. Loading in the foreground also avoids the
    # obvious trap: kicking off a loader thread and then calling load_model()
    # again a few seconds later starts a SECOND concurrent load of the same
    # 17 GB checkpoint, because the cache is only populated on return.
    url = start_tunnel(args.port)
    log(f"tunnel ready: {url}")

    t0 = time.time()
    try:
        STATE["ready"] = load_model("t2v") is not None
    except Exception as e:
        STATE["ready"] = False
        log(f"model load failed: {type(e).__name__}: {e}")
    log(f"READY={STATE['ready']} (model load {time.time()-t0:.0f}s)")

    # Belt-and-braces: exit even if nobody presses OFF. session_timeout_seconds
    # is the outer guarantee; this is the inner one.
    while True:
        time.sleep(30)
        if int(time.time() - STATE["started"]) > 12 * 3600:
            log("12h ceiling reached — exiting")
            os._exit(0)
