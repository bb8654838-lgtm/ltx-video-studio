#!/usr/bin/env python3
"""
Stage the LTX-2.3 weights out of HuggingFace and persist them as this kernel's
output, so the worker can import them via `kernel_sources` instead of
downloading 21 GB on every single ON.

Why not just download in the worker? Because the working directory caps at
20 GB and the stack needs ~21 GB:

    transformer Q4_K_M   14.18 GB
    text encoder Q4_K_M   7.30 GB
    video VAE             1.45 GB
    projections           2.31 GB
    -------------------------
                         25.24 GB   >  20 GB limit

Kaggle persists ~20 GB of kernel output, which is the escape hatch: split the
stack across two kernels, each under the limit, and have the worker attach both.
Kernel output is also cheaper than a Dataset here because it needs no upload
from this side at all.

This run measures real throughput first, because HF bandwidth is the whole
feasibility question and it has swung wildly across sources (0.95 MB/s to
77 MB/s measured weeks apart).
"""
import json
import os
import shutil
import time

REPO = "ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8"

# Part A: everything except the two giants. Fits one session comfortably.
PART_A = [
    ("split/vae/ltxv23_uncensored_v1.4_video_vae.safetensors", "video_vae.safetensors"),
    ("split/vae/ltxv23_uncensored_v1.4_audio_vae.safetensors", "audio_vae.safetensors"),
    ("split/text_encoders/ltxv23_uncensored_v1.4_projections.safetensors", "projections.safetensors"),
]

OUT = "/kaggle/working/ltx"
REPORT = "/kaggle/working/stage_report.json"
os.makedirs(OUT, exist_ok=True)


def save(stage, **kw):
    json.dump({"stage": stage, **kw}, open(REPORT, "w"), indent=1)


def log(*a):
    print(*a, flush=True)


# Xet's storage backend hangs behind Kaggle's egress proxy and yields partial
# files, so it has to be off before anything imports huggingface_hub.
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["HF_HOME"] = "/kaggle/tmp/hf_home"

from huggingface_hub import hf_hub_download  # noqa: E402

# ── 1. throughput probe on the smallest real artifact ─────────────────────
log("=== speed probe (audio VAE, 0.36 GB) ===")
t0 = time.time()
try:
    p = hf_hub_download(REPO, PART_A[1][0], local_dir="/kaggle/tmp/probe")
    dt = time.time() - t0
    mb = os.path.getsize(p) / 1e6
    mbps = mb / dt if dt else 0
    log(f"  {mb:.0f} MB in {dt:.0f}s = {mbps:.2f} MB/s")
    shutil.rmtree("/kaggle/tmp/probe", ignore_errors=True)
    save("probe", mb=mb, seconds=round(dt), mb_per_s=round(mbps, 2))
    # 21 GB at this rate
    log(f"  -> 21 GB would take {21e3 / max(mbps, 0.01) / 60:.0f} min")
except Exception as e:
    log(f"  speed probe FAILED: {type(e).__name__}: {e}")
    save("probe_failed", error=f"{type(e).__name__}: {e}")
    raise SystemExit(1)

# ── 2. download part A for real ───────────────────────────────────────────
log("=== part A: VAEs + projections (~4.1 GB) ===")
saved = []
for src, dst in PART_A:
    t = time.time()
    try:
        got = hf_hub_download(REPO, src, local_dir="/kaggle/tmp/hfdl")
        target = os.path.join(OUT, dst)
        shutil.move(got, target)
        sz = os.path.getsize(target)
        log(f"  ✅ {dst} {sz/1e9:.2f} GB in {time.time()-t:.0f}s")
        saved.append({"file": dst, "bytes": sz})
        save("partA_partial", saved=saved)
    except Exception as e:
        log(f"  ❌ {dst}: {type(e).__name__}: {e}")
        saved.append({"file": dst, "error": f"{type(e).__name__}: {e}"})

shutil.rmtree("/kaggle/tmp", ignore_errors=True)
total = sum(s.get("bytes", 0) for s in saved)
log(f"=== part A done: {len(saved)} files, {total/1e9:.2f} GB ===")
save("partA_done", saved=saved, total_bytes=total,
     verdict="PART_A_OK" if total > 4e9 else "PART_A_INCOMPLETE")
log("STAGE_VERDICT:", "PART_A_OK" if total > 4e9 else "PART_A_INCOMPLETE")