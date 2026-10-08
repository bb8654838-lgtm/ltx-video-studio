#!/usr/bin/env python3
"""
TUNNEL PROBE v4 — fixes two bugs in v3.

v3 got further than expected: `tailscaled` started by hand, and
`tailscale up --authkey=...` returned rc=0 in 637ms. So the auth key works on
Kaggle and the userspace daemon is fine.

What failed was both my own fault:
  1. run() truncates stdout to the last 2000 chars, so `tailscale status --json`
     — which is far larger — was unparseable. Status now goes to a file.
  2. `tailscale funnel --bg --https=8000` printed help text, i.e. wrong syntax.
     The exact subcommand changed between releases, so this run dumps the help
     and tries every known spelling rather than guessing again.
"""
import json
import os
import subprocess
import sys
import time

AUTHKEY = os.environ.get("TS_AUTHKEY", "PASTE_ME")
SOCK = "/tmp/ts/tailscaled.sock"
R = {}


def run(cmd, timeout=120):
    t0 = time.time()
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return {"rc": p.returncode, "out": p.stdout[-3000:], "err": p.stderr[-1500:],
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"rc": -1, "out": "", "err": "%s: %s" % (type(e).__name__, e), "ms": 0}


def run_to_file(cmd, path, timeout=120):
    """No truncation — for JSON payloads that exceed a few KB."""
    t0 = time.time()
    try:
        with open(path, "wb") as f:
            p = subprocess.run(cmd, shell=True, stdout=f, stderr=subprocess.PIPE,
                               timeout=timeout)
        return {"rc": p.returncode,
                "err": p.stderr.decode("utf-8", "replace")[-800:],
                "ms": int((time.time() - t0) * 1000)}
    except Exception as e:
        return {"rc": -1, "err": "%s: %s" % (type(e).__name__, e), "ms": 0}


def step(label, fn):
    print("--- %s ..." % label, flush=True)
    r = fn()
    R[label] = {k: v for k, v in r.items() if k != "out"}
    print("    rc=%s (%sms) %s" % (r.get("rc"), r.get("ms", 0),
                                   "PASS" if r.get("rc") == 0 else "FAIL"), flush=True)
    if r.get("rc") != 0 and r.get("err"):
        print("    err: %s" % r["err"][:300], flush=True)
    if r.get("out"):
        print("    out: %s" % r["out"][:300], flush=True)
    return r


def binary_check():
    r = run("command -v tailscaled; command -v tailscale", 30)
    return {"rc": 0 if r["out"].strip() else 1, "out": r["out"].strip()}


step("binary_present", binary_check)
if R["binary_present"]["rc"] != 0:
    print("installing tailscale...", flush=True)
    run("curl -fsSL https://tailscale.com/install.sh | sh", timeout=240)
    step("binary_after_install", binary_check)

step("version", lambda: run("tailscale version", 30))


def s_daemon():
    os.makedirs("/tmp/ts", exist_ok=True)
    run("nohup tailscaled --tun=userspace-networking --socket=%s "
        "--state=/tmp/ts/state > /tmp/ts/daemon.log 2>&1 &" % SOCK, 20)
    for _ in range(30):
        time.sleep(2)
        if os.path.exists(SOCK):
            return {"rc": 0, "out": "socket created"}
    d = run("tail -20 /tmp/ts/daemon.log", 20)
    return {"rc": 1, "err": "no socket: " + d["out"][-300:] + d["err"][-200:]}


step("tailscaled_daemon", s_daemon)

step("tailscale_up", lambda: run(
    "tailscale --socket=%s up --authkey=%s --hostname=ltx-worker --reset"
    % (SOCK, AUTHKEY), timeout=180))

# ── full status JSON, no truncation ───────────────────────────────────────
IDENT = {}


def s_identity():
    r = run_to_file("tailscale --socket=%s status --json" % SOCK, "/tmp/ts/status.json", 90)
    if r["rc"] != 0:
        return r
    try:
        d = json.load(open("/tmp/ts/status.json"))
    except Exception as e:
        sz = os.path.getsize("/tmp/ts/status.json")
        return {"rc": 1, "err": "unparseable (%s): %s" % (sz, e)}
    dn = d.get("Self", {}).get("DNSName", "").rstrip(".")
    tn = d.get("MagicDNSSuffix", "").rstrip(".")
    IDENT.update({
        "backend_state": d.get("BackendState"),
        "hostname": dn, "tailnet": tn,
        "public_url": ("https://%s" % dn) if dn else "",
        "ips": d.get("Self", {}).get("TailscaleIPs", []),
    })
    return {"rc": 0, "ok": True, "backend_state": IDENT["backend_state"],
            "hostname": dn, "tailnet": tn, "public_url": IDENT["public_url"]}


step("identity", s_identity)

# ── exact funnel syntax: ask, then try every spelling ─────────────────────
print("--- funnel --help ---", flush=True)
h = run("tailscale --socket=%s funnel --help" % SOCK, 60)
print("    " + h["out"][:700].replace("\n", "\n    "), flush=True)

print("--- funnel serve --help ---", flush=True)
h2 = run("tailscale --socket=%s funnel serve --help" % SOCK, 60)
print("    " + h2["out"][:700].replace("\n", "\n    "), flush=True)

ATTEMPTS = [
    ("funnel_serve_bg", "tailscale --socket=%s funnel serve --bg --https=8000" % SOCK),
    ("funnel_bg", "tailscale --socket=%s funnel --bg --https=8000" % SOCK),
    ("funnel_serve", "tailscale --socket=%s funnel serve --https=8000" % SOCK),
    ("funnel_https", "tailscale --socket=%s funnel --https=8000" % SOCK),
]
FUNNEL_OK = None
for label, cmd in ATTEMPTS:
    r = run(cmd, 90)
    ok = r["rc"] == 0 and "help" not in r["out"][:200].lower()
    print("    %-18s rc=%s %s" % (label, r["rc"], "PASS" if ok else "fail"), flush=True)
    R[label] = {"rc": r["rc"], "ms": r["ms"], "out": r["out"][:200]}
    if ok and FUNNEL_OK is None:
        FUNNEL_OK = cmd
    if ok:
        break

print("--- serve status ---", flush=True)
ss = run("tailscale --socket=%s funnel serve status --json" % SOCK, 60)
if ss["rc"] != 0:
    ss = run("tailscale --socket=%s funnel status" % SOCK, 60)
print("    rc=%s %s" % (ss["rc"], ss["out"][:400]), flush=True)
R["serve_status"] = {"rc": ss["rc"], "out": ss["out"][:400]}

# ── local server ──────────────────────────────────────────────────────────
def s_server():
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
    r = run("curl -s -m 10 http://127.0.0.1:8000/", 20)
    return {"rc": 0 if "kaggle-tunnel-probe" in r["out"] else 1, "out": r["out"][:150]}

step("local_server", s_server)


def s_rt():
    url = IDENT.get("public_url", "")
    if not url:
        return {"rc": 1, "err": "no public url"}
    a = run("curl -s -m 90 %s" % url, 120)
    b = run('curl -s -m 40 -o /dev/null -w "%%{http_code}}" %s' % url, 60)
    return {"rc": 0, "ok": "kaggle-tunnel-probe" in a["out"], "url": url,
            "body": a["out"][:250], "http_code": b["out"].strip(), "err": a["err"][:250]}

rt = step("public_roundtrip", s_rt)

print("=" * 66)
print(json.dumps(R, indent=2)[:3000])
print("=" * 66)
print("FUNNEL_CMD_THAT_WORKED:", FUNNEL_OK)
print("TUNNEL_URL_FOR_VERCEL:", IDENT.get("public_url", "NONE"))
print("TUNNEL_VERDICT:", "TUNNEL_OK" if rt.get("ok") else "TUNNEL_FAIL")
sys.exit(0 if rt.get("ok") else 1)
