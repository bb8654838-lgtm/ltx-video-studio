#!/usr/bin/env python3
"""
LIVE end-to-end test: real GPU box + real ngrok tunnel, no model weights.

This proves the whole control path minus the one part that is expensive and
slow — loading 25 GB of LTX weights. Everything it *does* exercise is the part
that can still fail silently:

  1. a 2x T4 session actually starts
  2. ngrok installs and authenticates with the authtoken
  3. the tunnel gets an https URL
  4. that URL really carries traffic (round-tripped from inside the kernel)
  5. results survive kernel cancellation, which is why they are written to
     /kaggle/working and read back with `kaggle kernels output` — the log
     stream gets truncated on cancel

The tunnel logic here is a verbatim copy of worker.py's, so a pass here is a
real pass for the shipped worker.
"""
import json
import os
import shutil
import subprocess
import time
import urllib.request

NGROK_AUTHTOKEN = "__NGROK__"
NGROK_DOMAIN = ""          # leave empty: runtime discovery makes it optional
PORT = 8000
RESULTS = "/kaggle/working/results.json"
LOG = []
os.makedirs("/kaggle/working", exist_ok=True)


def log(msg):
    print(msg, flush=True)
    LOG.append(str(msg))


def save(stage, extra=None):
    json.dump({"stage": stage, "log": LOG, "extra": extra or {}},
              open(RESULTS, "w"), indent=1)


def sh(cmd, t=180):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=t)
        return {"rc": p.returncode, "out": p.stdout[-2000:], "err": p.stderr[-1000:],
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"rc": -1, "out": "", "err": "%s: %s" % (type(e).__name__, e), "ms": 0}


# ── 1. GPU ────────────────────────────────────────────────────────────────
log("=== GPU ===")
g = sh("python3 -c \"import torch,json;"
       "print(json.dumps({'n':torch.cuda.device_count(),"
       "'dev':[{'name':torch.cuda.get_device_properties(i).name,"
       "'gb':round(torch.cuda.get_device_properties(i).total_memory/1e9,1),"
       "'cap':torch.cuda.get_device_properties(i).major} for i in range(torch.cuda.device_count())]}))\"", 120)
GPU = {}
try:
    GPU = json.loads(g["out"].strip().splitlines()[-1])
except Exception:
    GPU = {"raw": g["out"][-200:]}
log(json.dumps(GPU))
save("gpu", {"gpu": GPU})

# ── 2. local HTTP server ──────────────────────────────────────────────────
log("=== local server ===")
# Write the server as its own file and format it in ONE pass. The previous
# attempt chained two `%` operators, so the second one was applied to the
# already-substituted string and raised "not all arguments converted".
SERVER_SRC = '''import http.server, socketserver, json
GPU_N = {gpu_n!r}
class H(http.server.BaseHTTPRequestHandler):
    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        self._send({{"ok": True, "from": "kaggle-live-test", "gpu_count": GPU_N}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n)
        try:
            body = json.loads(raw or b"{{}}")
        except Exception:
            body = {{"raw": raw[:200].decode("utf-8", "replace")}}
        self._send({{"ok": True, "echo": "post-received", "got": body}})
    def log_message(self, *a):
        pass
socketserver.TCPServer.allow_reuse_address = True
socketserver.TCPServer(("", {port}), H).serve_forever()
'''.format(gpu_n=GPU.get("n", 0), port=PORT)
with open("/tmp/srv.py", "w") as fh:
    fh.write(SERVER_SRC)

chk = sh("python3 -m py_compile /tmp/srv.py", 30)
log("server syntax rc=%s %s" % (chk["rc"], chk["err"][:200]))
save("srv_compile", {"rc": chk["rc"], "err": chk["err"][:400]})
sh("nohup python3 /tmp/srv.py > /tmp/srv.log 2>&1 &", 20)
time.sleep(4)
lc = sh("curl -s -m 10 http://127.0.0.1:%d/" % PORT, 20)
LOCAL_OK = "kaggle-live-test" in lc["out"]
log("local GET -> %s (%s)" % (lc["out"][:100], LOCAL_OK))
save("local", {"local_ok": LOCAL_OK, "body": lc["out"][:200]})

# ── 3. install ngrok ──────────────────────────────────────────────────────
log("=== ngrok install ===")
if not shutil.which("ngrok"):
    r = sh("curl -fsSL -o /tmp/ngrok.zip "
           "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip && "
           "unzip -o -q /tmp/ngrok.zip -d /usr/local/bin && chmod +x /usr/local/bin/ngrok", 300)
    log("install rc=%s err=%s" % (r["rc"], r["err"][:200]))
NG = shutil.which("ngrok")
v = sh("ngrok --version", 30)
log("ngrok binary=%s version=%s" % (NG, v["out"].strip()))
save("ngrok_install", {"binary": NG, "version": v["out"].strip()})

# ── 4. start tunnel ───────────────────────────────────────────────────────
log("=== start tunnel ===")
cmd = "ngrok http %d --log stdout" % PORT
if NGROK_DOMAIN:
    cmd += " --domain " + NGROK_DOMAIN
log("cmd: " + cmd)
subprocess.Popen(cmd, shell=True,
                 env={**os.environ, "NGROK_AUTHTOKEN": NGROK_AUTHTOKEN},
                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

URL = None
for i in range(45):
    time.sleep(2)
    try:
        with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as r:
            d = json.load(r)
        for t in d.get("tunnels", []):
            if t.get("proto") == "https":
                URL = t["public_url"]
                break
        if URL:
            log("TUNNEL_URL_FOR_VERCEL: %s" % URL)
            break
    except Exception:
        pass
save("tunnel", {"url": URL})

if not URL:
    err = sh("pgrep -a ngrok; tail -5 /tmp/srv.log", 30)
    log("NO TUNNEL. ngrok procs: %s" % err["out"][:300])
    save("failed", {"reason": "no https tunnel", "detail": err["out"][:500]})
    print("LIVE_VERDICT: TUNNEL_FAIL")
    raise SystemExit(1)

# ── 5. round-trip from inside the kernel ──────────────────────────────────
log("=== round-trip from inside ===")
RT = None
for i in range(8):
    time.sleep(10)
    a = sh("curl -s -m 40 %s" % URL, 70)
    if "kaggle-live-test" in a["out"]:
        RT = {"ok": True, "body": a["out"][:250], "attempt": i + 1}
        break
    log("attempt %d: %r" % (i + 1, a["out"][:100]))
save("roundtrip", {"rt": RT, "url": URL})

# POST too — the app sends JSON bodies, not just GETs
p = sh("curl -s -m 40 -X POST -H 'Content-Type: application/json' "
       "-d '{\"prompt\":\"hi\"}' %s" % URL, 70)
POST_OK = "post-received" in p["out"]
log("POST -> %s (%s)" % (p["out"][:100], POST_OK))

# ── 6. measure tunnel throughput (matters for video) ─────────────────────
log("=== throughput sample ===")
tp = sh("curl -s -m 30 -o /dev/null -w '%%{speed_download}' "
        "https://speed.cloudflare.com/__down?bytes=3000000", 60)
try:
    bps = float(tp["out"].strip())
    log("download: %.2f MB/s" % (bps / 1e6))
except Exception:
    bps = None
    log("throughput sample failed: %r" % tp["out"][:100])

save("done", {
    "url": URL,
    "gpu": GPU,
    "local_ok": LOCAL_OK,
    "roundtrip": RT,
    "post_ok": POST_OK,
    "download_mb_s": round(bps / 1e6, 2) if bps else None,
})

OK = bool(URL and RT and POST_OK)
print("=" * 60)
print("TUNNEL_URL_FOR_VERCEL: %s" % (URL or "NONE"))
print("GPU: %s" % json.dumps(GPU))
print("LIVE_VERDICT: %s" % ("LIVE_OK" if OK else "LIVE_FAIL"))
raise SystemExit(0 if OK else 1)