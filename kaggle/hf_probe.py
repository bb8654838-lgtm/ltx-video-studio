#!/usr/bin/env python3
"""
Find a HuggingFace download method that actually works from inside Kaggle.

The first staging attempt produced a 0-byte file: `hf_hub_download` returned a
path without raising, and the file was empty. That is the known Kaggle-egress
failure mode — the Xet storage backend and HF's URL scheme do not survive the
proxy, so the client silently yields nothing.

Rather than guess again, this tries each method, measures the bytes actually on
disk, and compares against the size HuggingFace reports for the same file. A
method only counts as working if the size matches.

The report is written before anything else runs, and re-written in a finally
block, so a hard crash still leaves a readable answer on disk.
"""
import json
import os
import subprocess
import time

REPO = "ChrisColeTech/LTX-2.3-uncensored-v1.4-FP8"
TEST_FILE = "split/vae/ltxv23_uncensored_v1.4_audio_vae.safetensors"
REPORT = "/kaggle/working/probe_report.json"
WORK = "/kaggle/tmp/w"
os.makedirs("/kaggle/working", exist_ok=True)
os.makedirs(WORK, exist_ok=True)

RESULTS = {"file": TEST_FILE, "attempts": []}


def write():
    json.dump(RESULTS, open(REPORT, "w"), indent=1)


write()  # placeholder so the file always exists


def log(*a):
    print(*a, flush=True)


def size(p):
    try:
        return os.path.getsize(p)
    except Exception:
        return 0


# ── expected size straight from the HF API ────────────────────────────────
EXPECTED = 0
try:
    r = subprocess.run(
        "curl -s -m 40 'https://huggingface.co/api/models/%s?blobs=true'" % REPO,
        shell=True, capture_output=True, text=True, timeout=60)
    for s in json.loads(r.stdout).get("siblings", []):
        if s.get("rfilename") == TEST_FILE:
            EXPECTED = s.get("size") or 0
except Exception as e:
    log(f"  expected-size lookup failed: {e}")
log(f"expected size from HF API: {EXPECTED/1e6:.1f} MB")
RESULTS["expected_bytes"] = EXPECTED
write()


def record(name, path, dt, err=""):
    got = size(path)
    ok = EXPECTED > 0 and got == EXPECTED
    RESULTS["attempts"].append({
        "method": name, "bytes_on_disk": got, "seconds": round(dt, 1),
        "mb_per_s": round(got / 1e6 / dt, 2) if dt else 0,
        "size_matches": ok, "error": err[:300]})
    log(f"  {name:28} {got/1e6:8.1f} MB in {dt:5.1f}s "
        f"= {got/1e6/max(dt,0.01):5.2f} MB/s  match={ok}")
    write()
    return ok


# ── A: curl with ?download=true (forces the CDN, avoids Xet) ─────────────
try:
    out = os.path.join(WORK, "a_vae.safetensors")
    t = time.time()
    subprocess.run(
        "curl -sL --retry 3 --retry-delay 5 -m 900 -w '%{http_code}' "
        "-o '%s' 'https://huggingface.co/%s/resolve/main/%s?download=true'"
        % (out, REPO, TEST_FILE),
        shell=True, capture_output=True, timeout=960)
    record("curl ?download=true", out, time.time() - t)
except Exception as e:
    RESULTS["attempts"].append({"method": "curl ?download=true",
                                "error": f"{type(e).__name__}: {e}"})

# ── B: plain resolve URL, no query string ────────────────────────────────
if not any(a.get("size_matches") for a in RESULTS["attempts"]):
    try:
        out = os.path.join(WORK, "b_vae.safetensors")
        t = time.time()
        subprocess.run(
            "curl -sL --retry 3 --retry-delay 5 -m 900 -o '%s' "
            "'https://huggingface.co/%s/resolve/main/%s'" % (out, REPO, TEST_FILE),
            shell=True, capture_output=True, timeout=960)
        record("curl plain resolve", out, time.time() - t)
    except Exception as e:
        RESULTS["attempts"].append({"method": "curl plain resolve",
                                    "error": f"{type(e).__name__}: {e}"})

# ── C: hf_hub_download with Xet disabled before import ───────────────────
if not any(a.get("size_matches") for a in RESULTS["attempts"]):
    try:
        os.environ["HF_HUB_DISABLE_XET"] = "1"
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
        os.environ["HF_HOME"] = "/kaggle/tmp/hf_home"
        from huggingface_hub import hf_hub_download
        t = time.time()
        p = hf_hub_download(REPO, TEST_FILE, local_dir=WORK)
        record("hf_hub_download (xet off)", p, time.time() - t)
    except Exception as e:
        RESULTS["attempts"].append({"method": "hf_hub_download",
                                    "error": f"{type(e).__name__}: {e}"})

# ── D: cdn-lfs mirror ─────────────────────────────────────────────────────
if not any(a.get("size_matches") for a in RESULTS["attempts"]):
    try:
        out = os.path.join(WORK, "d_vae.safetensors")
        t = time.time()
        subprocess.run(
            "curl -sL --retry 2 -m 900 -o '%s' "
            "'https://cdn-lfs.hf.co/repos/%s/resolve/main/%s' || true"
            % (out, REPO, TEST_FILE),
            shell=True, capture_output=True, timeout=960)
        record("cdn-lfs mirror", out, time.time() - t)
    except Exception as e:
        RESULTS["attempts"].append({"method": "cdn-lfs",
                                    "error": f"{type(e).__name__}: {e}"})

try:
    import shutil
    shutil.rmtree("/kaggle/tmp", ignore_errors=True)
except Exception:
    pass

win = [a for a in RESULTS["attempts"] if a.get("size_matches")]
RESULTS["verdict"] = "DOWNLOAD_OK" if win else "ALL_METHODS_FAILED"
if win:
    best = max(win, key=lambda a: a.get("mb_per_s", 0))
    RESULTS["best"] = best
    log(f"WINNER: {best['method']} @ {best['mb_per_s']} MB/s")
    if best["mb_per_s"] > 0.01:
        log(f"  -> 21 GB would take {21e3/best['mb_per_s']/60:.0f} min")
else:
    log("NO METHOD PRODUCED A CORRECT-SIZED FILE")
log("PROBE_VERDICT:", RESULTS["verdict"])
write()