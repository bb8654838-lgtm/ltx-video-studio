#!/usr/bin/env python3
"""Isolate the tunnel question from everything else.

The worker was reporting a healthy ngrok agent while probes against the pinned
dev domain came back ERR_NGROK_3200 ("endpoint offline"). Nothing in the live
kernel could be observed — Kaggle only materialises output files once a run
finishes, and the worker runs for hours by design — so this runs the tunnel
logic on its own, with no GPU and no model, and exits in a few minutes so the
log can actually be read.

Reports, for each strategy:
  - what the agent reports as its public URL
  - what an outside client gets from <url>/healthz
and leaves the answer in a report file as well as the log.
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import urllib.request

REPORT = "/kaggle/working/tunnel_report.json"
R = {"attempts": []}


def save():
    json.dump(R, open(REPORT, "w"), indent=1)


def log(*a):
    print(*a, flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def install_ngrok():
    if shutil.which("ngrok"):
        log("ngrok already present")
        return True
    url = "https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip"
    subprocess.run(["curl", "-fsSL", "-o", "/tmp/ngrok.zip", url], timeout=240)
    subprocess.run(["unzip", "-o", "-q", "/tmp/ngrok.zip", "-d", "/usr/local/bin"], timeout=120)
    os.chmod("/usr/local/bin/ngrok", 0o755)
    return shutil.which("ngrok") is not None


def agent_url(tries=25):
    for _ in range(tries):
        time.sleep(2)
        try:
            with urllib.request.urlopen("http://127.0.0.1:4040/api/tunnels", timeout=5) as r:
                for t in json.load(r).get("tunnels", []):
                    if t.get("proto") == "https":
                        return t["public_url"]
        except Exception:
            continue
    return None


def external_probe(url):
    """Hit the public URL from outside the agent, which is the only view that
    matters — the agent reports its URL even when the edge refuses to route."""
    if not url:
        return {"ok": False, "error": "no url"}
    out = {"ok": False}
    try:
        req = urllib.request.Request(url + "/healthz", headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=30) as r:
            out = {"ok": True, "status": r.status, "body": r.read().decode()[:120]}
    except Exception as e:
        body = ""
        try:
            body = getattr(e, "read", lambda: b"")().decode("utf-8", "replace")
        except Exception:
            pass
        code = ""
        for tok in body.split():
            if tok.startswith("ERR_NGROK"):
                code = tok.strip("()<>.")
        out = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:120]}",
               "ngrok_error": code}
    return out


def serve(port, stop):
    """Plain HTTP server — the model is irrelevant to whether the edge routes."""
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            body = b'{"ok":true,"source":"tunnel-test"}'
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    HTTPServer(("0.0.0.0", port), H).serve_forever()


PORT = free_port()
threading.Thread(target=serve, args=(PORT, None), daemon=True).start()
time.sleep(1)
log(f"local server on {PORT}")

R["port"] = PORT
save()

TOK = os.environ.get("NGROK_AUTHTOKEN", "")
DOM = os.environ.get("NGROK_DOMAIN", "")
if not TOK:
    R["verdict"] = "NO_TOKEN"
    save()
    log("NO_TOKEN — set NGROK_AUTHTOKEN")
    raise SystemExit(1)

log("installing ngrok:", install_ngrok())
v = subprocess.run(["ngrok", "--version"], capture_output=True, text=True).stdout.strip()
R["ngrok_version"] = v
log("ngrok version:", v)

if not R["ngrok_version"]:
    R["verdict"] = "INSTALL_FAILED"
    save()
    raise SystemExit(1)

for label, extra in (("pinned-domain", ["--domain", DOM] if DOM else []),
                     ("random-domain", [])):
    log(f"=== attempt: {label} {' '.join(extra)} ===")
    env = {**os.environ, "NGROK_AUTHTOKEN": TOK}
    p = subprocess.Popen(["ngrok", "http", str(PORT), "--log", "stdout", *extra],
                         env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(6)
    url = agent_url()
    log(f"  agent url: {url}")
    ext = external_probe(url)
    log(f"  external:  {json.dumps(ext)[:220]}")
    R["attempts"].append({"strategy": label, "extra": extra, "agent_url": url,
                          "external": ext})
    save()
    p.terminate()
    try:
        p.wait(timeout=15)
    except Exception:
        p.kill()
    time.sleep(3)

ok = [a for a in R["attempts"] if a["external"].get("ok")]
R["verdict"] = "TUNNEL_OK" if ok else "TUNNEL_FAIL"
R["working_strategy"] = ok[0]["strategy"] if ok else None
R["working_url"] = ok[0]["agent_url"] if ok else None
save()
log("TUNNEL_VERDICT:", R["verdict"], "|", R["working_strategy"])
raise SystemExit(0 if ok else 1)