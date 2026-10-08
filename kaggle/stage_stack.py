#!/usr/bin/env python3
"""
Stage a slice of the LTX-2.3 weight stack and persist it as kernel output.

The worker cannot download its own weights on every ON: the stack is ~21 GB
against a 20 GB working-directory cap, so it is split across three staging
kernels whose outputs the worker imports via `kernel_sources`. Kernel output
costs no upload from this side, which is why this is preferred over a Dataset.

Download method is pinned to `curl` against the plain resolve URL. Measured on
this account: 11.71 MB/s with a byte-exact result. `hf_hub_download` returned a
0-byte file here — the Xet backend and URL scheme do not survive Kaggle's
egress proxy — so it is deliberately not used.

Every file is verified against the size HuggingFace reports before being
accepted; a size mismatch is treated as a failure and retried, because a
silently truncated weight file produces garbage video rather than an error.
"""
import json
import os
import subprocess
import time

REPO = "ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8"
PART = "__PART__"          # substituted per run: "a" | "b" | "c"
OUT = "/kaggle/working/ltx"
WORK = "/kaggle/tmp/dl"
REPORT = "/kaggle/working/stage_report.json"

# Kept under the 20 GB persist/output cap per kernel.
SLICES = {
    "a": [
        ("split/vae/ltxv23_uncensored_v1.4_video_vae.safetensors", "video_vae.safetensors"),
        ("split/vae/ltxv23_uncensored_v1.4_audio_vae.safetensors", "audio_vae.safetensors"),
        ("split/text_encoders/ltxv23_uncensored_v1.4_projections.safetensors", "projections.safetensors"),
    ],
    "b": [
        ("split/diffusion_models/ltxv23_uncensored_v1.4_Q4_K_M.gguf", "transformer_Q4_K_M.gguf"),
    ],
    "c": [
        ("split/text_encoders/gemma-3-12b-it-ablit-norms-biproj-Q4_K_M.gguf", "text_encoder_Q4_K_M.gguf"),
    ],
}

FILES = SLICES[PART]
os.makedirs(OUT, exist_ok=True)
os.makedirs(WORK, exist_ok=True)

R = {"part": PART, "repo": REPO, "files": []}


def save():
    json.dump(R, open(REPORT, "w"), indent=1)


def log(*a):
    print(*a, flush=True)


def hf_size(path):
    try:
        r = subprocess.run(
            "curl -s -m 60 'https://huggingface.co/api/models/%s?blobs=true'" % REPO,
            shell=True, capture_output=True, text=True, timeout=90)
        for s in json.loads(r.stdout).get("siblings", []):
            if s.get("rfilename") == path:
                return s.get("size") or 0
    except Exception as e:
        log(f"  size lookup failed for {path}: {e}")
    return 0


def fetch(src, dst_name, expect, tries=3):
    # Download straight into the working directory. An earlier version staged
    # into /kaggle/tmp and then os.replace()d it, which fails with
    # "Invalid cross-device link" — /kaggle/tmp and /kaggle/working are
    # separate mounts. curl writes to a .part name instead so a partial
    # transfer is never mistaken for a finished one.
    dest = os.path.join(OUT, dst_name)
    part = dest + ".part"
    if os.path.exists(dest) and os.path.getsize(dest) == expect:
        log(f"  ✅ {dst_name} already staged")
        return True, os.path.getsize(dest), 0.0
    for attempt in range(1, tries + 1):
        t = time.time()
        # No ?download=true — the measured error was a malformed URL format
        # string, and the plain resolve path is what actually worked.
        rc = subprocess.run(
            "curl -sL --retry 3 --retry-delay 5 --fail -m 7200 "
            "-o '%s' 'https://huggingface.co/%s/resolve/main/%s'"
            % (part, REPO, src),
            shell=True, capture_output=True, timeout=7500)
        dt = time.time() - t
        got = os.path.getsize(part) if os.path.exists(part) else 0
        speed = got / 1e6 / max(dt, 0.01)
        log(f"  attempt {attempt}: {dst_name} {got/1e9:.2f}/{expect/1e9:.2f} GB "
            f"in {dt:.0f}s = {speed:.2f} MB/s rc={rc.returncode}")
        if expect and got == expect:
            os.replace(part, dest)
            log(f"  ✅ {dst_name} verified {got/1e9:.2f} GB @ {speed:.2f} MB/s")
            return True, got, speed
        # discard a wrong-sized transfer before retrying
        if os.path.exists(part):
            os.remove(part)
        if attempt < tries:
            time.sleep(10)
    return False, got if expect else 0, 0.0


t_start = time.time()
for src, name in FILES:
    exp = hf_size(src)
    log(f"=== {name} — expected {exp/1e9:.2f} GB ===")
    ok, got, speed = fetch(src, name, exp)
    R["files"].append({"file": name, "source": src, "expected_bytes": exp,
                       "bytes": got, "mb_per_s": round(speed, 2), "ok": ok})
    save()

total = sum(f["bytes"] for f in R["files"] if f["ok"])
R["total_bytes"] = total
R["verdict"] = "STAGE_OK" if all(f["ok"] for f in R["files"]) else "STAGE_INCOMPLETE"
R["elapsed_s"] = round(time.time() - t_start, 1)
save()
log(f"=== part {PART}: {total/1e9:.2f} GB in {R['elapsed_s']/60:.1f} min ===")
log("STAGE_VERDICT:", R["verdict"])

subprocess.run("rm -rf /kaggle/tmp", shell=True, capture_output=True)
raise SystemExit(0 if R["verdict"] == "STAGE_OK" else 1)