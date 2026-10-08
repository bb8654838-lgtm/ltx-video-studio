#!/usr/bin/env python3
"""
TUNNEL PROBE v5 — everything works except the final round-trip.

Established so far (all measured, not assumed):
  - tailscaled starts by hand (no systemd on Kaggle)
  - `tailscale up --authkey=...` returns rc=0, BackendState=Running
  - the correct subcommand is `tailscale funnel --bg --https=8000`
    (`funnel serve --bg` does not exist in 1.104.1)

Two things were wrong with v4's round-trip:
  1. TLS certificate provisioning for the funnel hostname is asynchronous —
     the first request triggers issuance, so testing immediately returns 000.
  2. A hostname collision suffix appears (`ltx-worker-1`) when a previous node
     is still registered. That matters: the URL is NOT predictable across
     sessions, which breaks hardcoding TUNNEL_URL in Vercel.

So this run: verify the serve config persisted, then poll the public URL for
several minutes, and report both the URL and whether the suffix was stable.
"""
import json
import os
import subprocess
import sys
import time

AUTHKEY = os.environ.get("TS_AUTHKEY", "PASTE_ME")
SOCK = "/tmp/ts/tailscaled.sock"
IDENT = {}


def run(cmd, timeout=120):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout[-3000:], "err": p.stderr[-1500:],
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"rc": -1, "out": "", "err": "%s: %s" % (type(e).__name__, e), "ms": 0}


print("=== install ===", flush=True)
if not run("command -v tailscaled", 20)["out"].strip():
    run("curl -fsSL https://tailscale.com/install.sh | sh", timeout=240)
print("  ", run("tailscale version", 20)["out"].splitlines()[0], flush=True)

print("=== start tailscaled (userspace, no systemd) ===", flush=True)
os.makedirs("/tmp/ts", exist_ok=True)
run("pkill tailscaled 2>/dev/null; nohup tailscaled --tun=userspace-networking "
    "--socket=%s --state=/tmp/ts/state > /tmp/ts/daemon.log 2>&1 &" % SOCK, 20)
for _ in range(30):
    time.sleep(2)
    if os.path.exists(SOCK):
        break
print("   socket:", os.path.exists(SOCK), flush=True)

print("=== auth ===", flush=True)
up = run("tailscale --socket=%s up --authkey=%s --hostname=ltx-worker --reset"
         % (SOCK, AUTHKEY), 180)
print("   rc=%s ms=%s err=%s" % (up["rc"], up["ms"], up["err"][:200]), flush=True)

print("=== identity ===", flush=True)
with open("/tmp/ts/status.json", "w") as f:
    subprocess.run("tailscale --socket=%s status --json" % SOCK, shell=True,
                   stdout=f, stderr=subprocess.DEVNULL, timeout=90)
try:
    d = json.load(open("/tmp/ts/status.json"))
    dn = d.get("Self", {}).get("DNSName", "").rstrip(".")
    tn = d.get("MagicDNSSuffix", "").rstrip(".")
    IDENT = {"hostname": dn, "tailnet": tn,
             "url": "https://%s" % dn if dn else "",
             "backend": d.get("BackendState"),
             "ips": d.get("Self", {}).get("TailscaleIPs", [])}
except Exception as e:
    print("   identity failed:", e, flush=True)
print("   ", json.dumps(IDENT), flush=True)

print("=== dns / prefs ===", flush=True)
p = run("tailscale --socket=%s debug prefs" % SOCK, 60)
try:
    pd = json.loads(p["out"])
    print("   WantRunning=%s CorpDNS=%s RouteAll=%s"
          % (pd.get("WantRunning"), pd.get("CorpDNS"), pd.get("RouteAll")), flush=True)
except Exception:
    print("   prefs raw:", p["out"][:200], flush=True)

print("=== local server ===", flush=True)
open("/tmp/srv.py", "w").write(
    "import http.server,socketserver,json\n"
    "class H(http.server.BaseHTTPRequestHandler):\n"
    "    def do_GET(self):\n"
    "        b=json.dumps({'ok':True,'from':'kaggle-tunnel-probe'}).encode()\n"
    "        self.send_response(200);self.send_header('Content-Type','application/json')\n"
    "        self.send_header('Content-Length',str(len(b)));self.end_headers();self.wfile.write(b)\n"
    "    def log_message(self,*a):pass\n"
    "socketserver.TCPServer.allow_reuse_address=True\n"
    "socketserver.TCPServer(('',8000),H).serve_forever()\n")
run("nohup python3 /tmp/srv.py > /tmp/srv.log 2>&1 &", 20)
time.sleep(4)
print("   local:", run("curl -s -m 10 http://127.0.0.1:8000/", 20)["out"][:120], flush=True)

print("=== funnel on ===", flush=True)
f = run("tailscale --socket=%s funnel --bg --https=8000" % SOCK, 120)
print("   rc=%s out=%r err=%r" % (f["rc"], f["out"][:200], f["err"][:400]), flush=True)

print("=== serve status (verify it persisted) ===", flush=True)
ss = run("tailscale --socket=%s funnel status" % SOCK, 60)
print("   rc=%s %r" % (ss["rc"], ss["out"][:400]), flush=True)

url = IDENT.get("url", "")
print("=== poll public URL for up to 5 min (TLS cert is async) ===", flush=True)
hit = None
for i in range(10):
    time.sleep(30)
    if not url:
        print("   [%2d] no url yet" % (i + 1), flush=True)
        break
    r = run("curl -s -m 45 %s" % url, 70)
    code = run('curl -s -m 25 -o /dev/null -w "%%{http_code}}" %s' % url, 45)["out"].strip()
    body = r["out"][:120]
    ok = "kaggle-tunnel-probe" in body
    print("   [%2d] http=%s ok=%s body=%r" % (i + 1, code, ok, body), flush=True)
    if ok:
        hit = {"url": url, "http": code, "body": body}
        break

print("=" * 66)
print(json.dumps({"identity": IDENT, "roundtrip": hit,
                  "funnel_cmd": "tailscale funnel --bg --https=8000"}, indent=2))
print("=" * 66)
print("TUNNEL_URL_FOR_VERCEL:", url or "NONE")
print("STABLE_ACROSS_SESSIONS:", "NO — suffix increments (ltx-worker, ltx-worker-1, ...)"
      if url else "UNKNOWN")
print("TUNNEL_VERDICT:", "TUNNEL_OK" if hit else "TUNNEL_FAIL")
sys.exit(0 if hit else 1)
