#!/usr/bin/env python3
"""
Stage Wan2.1-T2V-1.3B for the pipeline proof.

Why this model and not the 22B LTX stack: the goal here is to exercise the
whole control path — prompt to video, over the tunnel, into the browser — and
do it with a model whose loader is already known to work. LTX-2.3 ships only
Q4_K_M GGUF, which needs ComfyUI's loader; that is an untested dependency and a
bad thing to combine with a pipeline test.

Wan2.1-T2V-1.3B is 17.55 GB against a 20 GB working cap, and it splits across
the two T4s without pressure:

    T5-UMT5-XXL encoder  11.36 GB  -> GPU0 (15.6 GB)
    diffusion transformer 5.68 GB  -> GPU1
    VAE                   0.51 GB  -> GPU1

Download method is the same curl-against-plain-resolve path measured at
48-58 MB/s on this account, with every file size-checked before acceptance.
"""
import json
import os
import subprocess
import time

REPO = "Wan-AI/Wan2.1-T2V-1.3B"
OUT = "/kaggle/working/wan"
REPORT = "/kaggle/working/stage_report.json"

FILES = [
    ("models_t5_umt5-xxl-enc-bf16.pth", "models_t5_umt5-xxl-enc-bf16.pth"),
    ("diffusion_pytorch_model.safetensors", "diffusion_pytorch_model.safetensors"),
    ("Wan2.1_VAE.pth", "Wan2.1_VAE.pth"),
]

os.makedirs(OUT, exist_ok=True)
R = {"repo": REPO, "files": []}


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
        log(f"  size lookup failed: {e}")
    return 0


def fetch(src, name, expect, tries=3):
    dest = os.path.join(OUT, name)
    part = dest + ".part"          # staged in place: /kaggle/tmp is a separate
                                  # mount, so os.replace across it raises
                                  # "Invalid cross-device link"
    if os.path.exists(dest) and os.path.getsize(dest) == expect:
        log(f"  ✅ {name} already staged")
        return True, os.path.getsize(dest), 0.0
    for attempt in range(1, tries + 1):
        t = time.time()
        rc = subprocess.run(
            "curl -sL --retry 3 --retry-delay 5 --fail -m 7200 "
            "-o '%s' 'https://huggingface.co/%s/resolve/main/%s'"
            % (part, REPO, src),
            shell=True, capture_output=True, timeout=7500)
        dt = time.time() - t
        got = os.path.getsize(part) if os.path.exists(part) else 0
        speed = got / 1e6 / max(dt, 0.01)
        log(f"  attempt {attempt}: {name} {got/1e9:.2f}/{expect/1e9:.2f} GB "
            f"in {dt:.0f}s = {speed:.2f} MB/s rc={rc.returncode}")
        if expect and got == expect:
            os.replace(part, dest)
            log(f"  ✅ {name} verified @ {speed:.2f} MB/s")
            return True, got, speed
        if os.path.exists(part):
            os.remove(part)
        if attempt < tries:
            time.sleep(8)
    return False, got if expect else 0, 0.0


t0 = time.time()
for src, name in FILES:
    exp = hf_size(src)
    log(f"=== {name} — {exp/1e9:.2f} GB ===")
    ok, got, speed = fetch(src, name, exp)
    R["files"].append({"file": name, "bytes": got, "expected_bytes": exp,
                       "mb_per_s": round(speed, 2), "ok": ok})
    save()

total = sum(f["bytes"] for f in R["files"] if f["ok"])
R["total_bytes"] = total
R["verdict"] = "STAGE_OK" if all(f["ok"] for f in R["files"]) else "STAGE_INCOMPLETE"
R["elapsed_s"] = round(time.time() - t0, 1)
save()
log(f"=== {total/1e9:.2f} GB in {R['elapsed_s']/60:.1f} min ===")
log("STAGE_VERDICT:", R["verdict"])
raise SystemExit(0 if R["verdict"] == "STAGE_OK" else 1)