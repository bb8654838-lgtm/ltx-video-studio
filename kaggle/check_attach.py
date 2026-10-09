#!/usr/bin/env python3
"""
Verify the staged weights are actually visible to a kernel that attaches them
via `kernel_sources`.

Downloading kernel output locally showed 0-byte files, which may just mean the
CLI does not materialise large blobs — what actually matters is whether a GPU
kernel can mount them. This attaches all three staging kernels and reports the
real on-disk size of every file it finds, which also confirms the cross-device
and persistence issues are not silently truncating anything.
"""
import json
import os
import subprocess

REPORT = "/kaggle/working/attach_report.json"
R = {"inputs": {}, "scan": []}
os.makedirs("/kaggle/working", exist_ok=True)


def save():
    json.dump(R, open(REPORT, "w"), indent=1)


save()
print("=== /kaggle/input ===", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    for d in dirs:
        p = os.path.join(root, d)
        R["inputs"][os.path.relpath(p, "/kaggle/input")] = None
        print("  DIR  " + os.path.relpath(p, "/kaggle/input"), flush=True)
    for f in files:
        p = os.path.join(root, f)
        try:
            sz = os.path.getsize(p)
        except Exception:
            sz = -1
        rel = os.path.relpath(p, "/kaggle/input")
        R["inputs"][rel] = sz
        R["scan"].append({"path": rel, "bytes": sz})
        print(f"  FILE {sz/1e9:8.3f} GB  {rel}", flush=True)

big = [s for s in R["scan"] if s["bytes"] > 1e8]
print(f"\n=== {len(R['scan'])} files, {len(big)} over 100 MB ===", flush=True)

# disk headroom matters: the stack is ~25 GB and /kaggle/working caps at 20 GB
try:
    st = os.statvfs("/kaggle/working")
    R["working_free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 2)
    st2 = os.statvfs("/kaggle/input")
    R["input_free_gb"] = round(st2.f_bavail * st2.f_frsize / 1e9, 2)
    print(f"  /kaggle/working free: {R['working_free_gb']} GB", flush=True)
    print(f"  /kaggle/input  free: {R['input_free_gb']} GB", flush=True)
except Exception as e:
    print("  statvfs failed:", e, flush=True)

try:
    import torch
    R["gpu"] = {"count": torch.cuda.device_count(),
                "names": [torch.cuda.get_device_properties(i).name
                          for i in range(torch.cuda.device_count())],
                "total_gb": round(sum(torch.cuda.get_device_properties(i).total_memory
                                      for i in range(torch.cuda.device_count())) / 1e9, 1)}
    print("  GPU:", json.dumps(R["gpu"]), flush=True)
except Exception as e:
    R["gpu_error"] = str(e)
    print("  GPU check failed:", e, flush=True)

R["verdict"] = "ATTACH_OK" if big else "ATTACH_EMPTY"
print("ATTACH_VERDICT:", R["verdict"], flush=True)
save()