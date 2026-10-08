#!/usr/bin/env python3
"""
Internet-egress probe for an API-pushed Kaggle kernel.

Every part of the design assumes a kernel started with `kaggle kernels push`
can reach the internet (to pull model weights and open a tunnel). That is
UNVERIFIED on Kaggle: `enable_internet: true` is provably *accepted* by the
API, but no public report confirms egress actually happens — the only
testimony is two 2022 non-staff claims that the flag is silently ignored.

This settles it in five minutes. Run it first; everything downstream assumes
the answer is yes.
"""
import os
import sys

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HOME", "/kaggle/tmp/hf_home")

import json
import subprocess
import time


def check(label, url, method="GET"):
    import urllib.request
    t0 = time.time()
    try:
        req = urllib.request.Request(url, method=method)
        if method == "GET":
            req.add_header("User-Agent", "probe/1.0")
        with urllib.request.urlopen(req, timeout=25) as r:
            body = r.read(200).decode("utf-8", "replace")
        return {"ok": True, "ms": int((time.time() - t0) * 1000),
                "body": body.strip()[:120]}
    except Exception as e:
        return {"ok": False, "ms": int((time.time() - t0) * 1000),
                "error": f"{type(e).__name__}: {e}"}


results = {}
results["public_ip"] = check("ipify", "https://api.ipify.org")
results["http_bin"] = check("httpbin", "https://httpbin.org/get")
results["github_raw"] = check("raw", "https://raw.githubusercontent.com/Lightricks/LTX-2/main/README.md")
results["huggingface"] = check("hf", "https://huggingface.co/api/models/Lightricks/LTX-2.3")

# GPU reality check — what did we actually get?
try:
    import torch
    n = torch.cuda.device_count()
    props = [torch.cuda.get_device_properties(i) for i in range(n)]
    results["gpu"] = {
        "ok": True,
        "count": n,
        "devices": [{
            "name": p.name,
            "total_gb": round(p.total_memory / 1e9, 1),
            "capability": f"{p.major}.{p.minor}",
        } for p in props],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
except Exception as e:
    results["gpu"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}

# Can we bind a port? (needed for the tunnel)
try:
    import socket
    s = socket.socket()
    s.bind(("0.0.0.0", 8000))
    s.listen(1)
    s.close()
    results["bind_8000"] = {"ok": True}
except Exception as e:
    results["bind_8000"] = {"ok": False, "error": str(e)}

# Download speed sample — Kaggle publishes no bandwidth figure, so measure.
try:
    import urllib.request
    t0 = time.time()
    req = urllib.request.Request(
        "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/README.md")
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read(2_000_000)
    dt = time.time() - t0
    results["download_speed"] = {
        "ok": True,
        "bytes": len(data),
        "mb_per_s": round(len(data) / 1e6 / max(dt, 0.01), 2),
    }
except Exception as e:
    results["download_speed"] = {"ok": False, "error": str(e)}

print("=" * 60)
print(json.dumps(results, indent=2))
print("=" * 60)
print("PROBE_VERDICT:", "EGRESS_OK" if results.get("public_ip", {}).get("ok") else "EGRESS_FAIL")
sys.exit(0)
