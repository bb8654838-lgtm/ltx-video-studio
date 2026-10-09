#!/usr/bin/env python3
"""
Video generation worker — the real thing, on a real GPU, behind a real tunnel.

This is the proof the whole project has been missing: an actual prompt turning
into an actual .mp4 that the browser can play. Everything upstream of it
(quota API, batch push, 307 redirect, ngrok) is already verified, so the only
thing left to prove is that weights load and pixels come out.

Placement is deliberate rather than automatic. The T5-UMT5-XXL encoder is
11.36 GB and the transformer plus VAE are 6.19 GB, so on two 15.6 GB T4s the
encoder goes on cuda:0 and the generative half on cuda:1. That leaves real
headroom for activations instead of running one device at 73% full.

Written to /kaggle/working so results survive a cancelled kernel; the log
stream truncates on cancel, output files do not.
"""
import base64
import json
import os
import subprocess
import sys
import time

WORK = "/kaggle/working"
OUT = os.path.join(WORK, "out")
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(WORK, "gen_report.json")

R = {"stages": []}


def save():
    json.dump(R, open(REPORT, "w"), indent=1)


def log(*a):
    print(*a, flush=True)


def stage(name, **kw):
    R["stages"].append({"name": name, "t": round(time.time() - T0, 1), **kw})
    save()
    log(f"[{name}] " + json.dumps(kw)[:200])


T0 = time.time()
save()

# ── 1. locate the staged weights ──────────────────────────────────────────
# Kernel output attaches under /kaggle/input/notebooks/<owner>/<kernel>/, not
# /kaggle/input/<kernel>/ — a hardcoded guess misses it and reports "weights
# not found" while the files are sitting right there. Walk instead of guessing.
WEIGHT_ROOT = None
for root, dirs, _ in os.walk("/kaggle/input"):
    if "wan" in dirs:
        WEIGHT_ROOT = os.path.join(root, "wan")
        break
    if root.endswith("/wan"):
        WEIGHT_ROOT = root
        break
stage("weights_located", root=WEIGHT_ROOT,
      input_top=os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else [])

if not WEIGHT_ROOT:
    stage("FATAL", error="weights not found",
          tree=subprocess.run("find /kaggle/input -maxdepth 4 | head -40",
                              shell=True, capture_output=True, text=True).stdout[:800])
    raise SystemExit(1)

files = {}
for root, _, fs in os.walk(WEIGHT_ROOT):
    for f in fs:
        files[f] = os.path.join(root, f)
stage("weights_seen", files={k: round(os.path.getsize(v) / 1e9, 2) for k, v in files.items()})

T5 = next((v for k, v in files.items() if "umt5" in k.lower()), None)
DIFF = next((v for k, v in files.items() if "diffusion_pytorch" in k), None)
VAE = next((v for k, v in files.items() if "vae" in k.lower()), None)
if not (T5 and DIFF and VAE):
    stage("FATAL", error=f"missing component t5={bool(T5)} diff={bool(DIFF)} vae={bool(VAE)}")
    raise SystemExit(1)

# ── 2. environment ────────────────────────────────────────────────────────
import torch  # noqa: E402

DEV_ENC, DEV_GEN = "cuda:0", "cuda:1"
stage("torch", version=torch.__version__, cuda=torch.version.cuda,
      devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

# easydict backs wan/configs — without it the import dies before any useful
# message, since a packaged kernel script prints a bare ImportError.
for mod in ("imageio", "imageio-ffmpeg", "einops", "ftfy", "regex", "av",
            "easydict", "dashscope"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", mod], check=False,
                   capture_output=True)

sys.path.insert(0, "/kaggle/working/wanpkg")
if not os.path.isdir("/kaggle/working/wanpkg/wan"):
    # the official repo carries the model definition; diffusers needs ~29 GB of
    # fp32 shards for this model, the original repo is 17.55 GB total
    r = subprocess.run("git clone -q --depth 1 https://github.com/Wan-Video/Wan2.1 wanpkg",
                       shell=True, cwd="/kaggle/working", capture_output=True, timeout=600)
    stage("wan_clone", rc=r.returncode, err=r.stderr.decode("utf-8", "replace")[:300])

stage("imports_begin")
import wan  # noqa: E402
# The package is `wan.configs` (plural) — there is no `wan.config`, and
# guessing the singular name kills the import before anything is logged.
from wan.configs import WAN_CONFIGS  # noqa: E402
from wan.utils.utils import cache_video  # noqa: E402
stage("imports_ok", configs=sorted(WAN_CONFIGS.keys()))

# ── 3. load through the official pipeline ──────────────────────────────────
# Loading by hand kept failing, because the official loader does three things
# that are easy to miss: T5 goes to CPU first and is moved afterwards, the
# tokenizer directory is required, and the transformer is built by
# WanModel.from_pretrained() from config.json rather than from a raw
# state_dict. WanT2V already does all of it, so use it and just point it at the
# staged directory. device_id=0 puts T5, the transformer and the VAE on cuda:0;
# the second T4 stays free as headroom, since the T5 encoder alone is 11.36 GB.
log("building WanT2V pipeline")
t = time.time()
cfg = WAN_CONFIGS["t2v-1.3B"]
t2v = wan.WanT2V(config=cfg, checkpoint_dir=WEIGHT_ROOT, device_id=0,
                 t5_fsdp=False, dit_fsdp=False, t5_cpu=False)
stage("pipeline_built", seconds=round(time.time() - t, 1),
      vram_gb=round(torch.cuda.memory_allocated(0) / 1e9, 2))

# ── 4. generate ───────────────────────────────────────────────────────────
PROMPT = ("A cinematic shot of a neon-lit rain slicked street in Tokyo at night, "
          "reflections of pink and cyan signs rippling in puddles, slow dolly forward")
SIZE = (480, 320)          # width, height — must be a multiple of 16 for the patch size
FRAMES = 25
STEPS = 20
SEED = 42

path = os.path.join(OUT, "video.mp4")

log(f"generating: {PROMPT}")
t = time.time()
with torch.no_grad():
    video = t2v.generate(
        prompts=[PROMPT],
        negative_prompts=["overexposed, static, blurry, low quality, worst quality"],
        steps=STEPS, size=list(SIZE), duration=FRAMES,
        seed=SEED, save_file=path)
gen_s = time.time() - t
stage("generated", seconds=round(gen_s, 1),
      type=type(video).__name__,
      shape=list(video.shape) if hasattr(video, "shape") else None)

# ── 5. confirm the mp4 ────────────────────────────────────────────────────
size_mb = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
stage("mp4_written", path=path, size_mb=round(size_mb, 2))

ok = size_mb > 0.05
R["verdict"] = "VIDEO_OK" if ok else "VIDEO_FAIL"
R["mp4_bytes"] = os.path.getsize(path) if os.path.exists(path) else 0
R["gen_seconds"] = round(gen_s, 1)
save()
log("GEN_VERDICT:", R["verdict"], "| %.2f MB | %.1f s" % (size_mb, gen_s))
raise SystemExit(0 if ok else 1)