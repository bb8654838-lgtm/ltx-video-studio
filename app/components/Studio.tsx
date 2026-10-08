"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type State = "offline" | "queued" | "running" | "complete" | "error";

type SessionInfo = {
  state: State;
  detail: string;
  tunnel: string | null;
  quota?: {
    usedH: number;
    reservedH: number;
    totalH: number;
    remainingH: number;
    refreshTime: string;
  };
};

export default function Studio() {
  const [info, setInfo] = useState<SessionInfo | null>(null);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");
  const [prompt, setPrompt] = useState("");
  const [imgB64, setImgB64] = useState<string | null>(null);
  const [working, setWorking] = useState(false);
  const [video, setVideo] = useState<string | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const fileRef = useRef<HTMLInputElement>(null);

  const tunnelRef = useRef<string | null>(null);

  const poll = useCallback(async () => {
    try {
      const r = await fetch("/api/session", { cache: "no-store" });
      const d = await r.json();
      setInfo(d);
      tunnelRef.current = d.tunnel ?? tunnelRef.current;
    } catch {
      /* transient */
    }
  }, []);

  useEffect(() => {
    poll();
    // 10s, not 3s: Vercel Hobby has a monthly invocation budget and one tab
    // open 24/7 at 3s burns ~86% of it and ends in a 30-day pause.
    const t = setInterval(poll, 10000);
    return () => clearInterval(t);
  }, [poll]);

  const on = async () => {
    setBusy(true);
    setMsg("");
    setLogs([]);
    const r = await fetch("/api/session", { method: "POST" });
    const d = await r.json();
    setBusy(false);
    if (d.error) setMsg(`❌ ${d.error}`);
    else setMsg("▶️ Starting — GPU queue + weight load takes a few minutes.");
    poll();
  };

  const probe = async () => {
    setBusy(true);
    setMsg("🧪 Running internet-egress probe…");
    setLogs([]);
    await fetch("/api/session", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ probe: true, timeoutSeconds: 900 }),
    });
    setBusy(false);
    setMsg("🧪 Probe submitted — results print in the Kaggle kernel log.");
    poll();
  };

  const off = async () => {
    setBusy(true);
    const r = await fetch("/api/session/off", { method: "POST" });
    const d = await r.json();
    setBusy(false);
    setMsg("⏹️ " + (d.note ?? "Stopping."));
    poll();
  };

  const pickImage = (f: File | null) => {
    if (!f) return;
    const rd = new FileReader();
    rd.onload = () => setImgB64(String(rd.result).split(",")[1] ?? null);
    rd.readAsDataURL(f);
  };

  const generate = async () => {
    if (!prompt && !imgB64) return setMsg("Enter a prompt or pick an image.");
    setWorking(true);
    setMsg("⏳ Generating…");
    setVideo(null);
    try {
      // Ask Vercel where to go, then talk to the tunnel directly. Vercel never
      // carries the video bytes.
      const r = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt, image_b64: imgB64 }),
      });
      const d = await r.json();
      if (d.error) throw new Error(d.error);

      const g = await fetch(d.url, {
        method: "POST",
        headers: d.headers,
        body: JSON.stringify(d.body),
      });
      if (!g.ok) throw new Error(`worker ${g.status}: ${(await g.text()).slice(0, 200)}`);
      const out = await g.json();
      if (out.video_b64) {
        setVideo(`data:video/mp4;base64,${out.video_b64}`);
        setMsg(`✅ Done — seed ${out.seed}`);
      } else {
        throw new Error("no video in response");
      }
    } catch (e: any) {
      setMsg(`❌ ${e.message}`);
    }
    setWorking(false);
  };

  const live = info?.state === "running" || info?.state === "queued";
  const q = info?.quota;

  return (
    <main style={{ maxWidth: 900, margin: "0 auto", padding: 24, fontFamily: "ui-sans-serif, system-ui" }}>
      <h1 style={{ marginBottom: 4 }}>LTX Video Studio</h1>
      <p style={{ color: "#666", marginTop: 0 }}>
        Personal text-to-video / image-to-video on Kaggle GPU.
      </p>

      {/* ── controls ─────────────────────────────────────────────── */}
      <section style={{ border: "1px solid #ddd", borderRadius: 10, padding: 16, marginBottom: 16 }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <button
            onClick={on}
            disabled={busy || live}
            style={{ padding: "10px 22px", borderRadius: 8, border: "none",
                     background: live ? "#bbb" : "#16a34a", color: "#fff", fontWeight: 600, cursor: live ? "default" : "pointer" }}
          >
            {live ? "● RUNNING" : "▶ ON"}
          </button>
          <button
            onClick={off}
            disabled={busy || !live}
            style={{ padding: "10px 22px", borderRadius: 8, border: "none",
                     background: live ? "#dc2626" : "#ddd", color: live ? "#fff" : "#888", fontWeight: 600, cursor: live ? "pointer" : "default" }}
          >
            ⏹ OFF
          </button>
          <button onClick={probe} disabled={busy || live}
                  style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid #ccc", background: "#fff", cursor: "pointer" }}>
            🧪 Egress probe
          </button>
          <button onClick={() => { setVideo(null); setMsg(""); setPrompt(""); setImgB64(null); }}
                  style={{ padding: "10px 16px", borderRadius: 8, border: "1px solid #ccc", background: "#fff" }}>
            Clear
          </button>
        </div>

        {q && (
          <div style={{ marginTop: 14, fontSize: 14 }}>
            <strong>Weekly GPU quota</strong>{" "}
            <span style={{ color: q.remainingH < 1 ? "#dc2626" : "#16a34a" }}>
              {q.remainingH}h left
            </span>{" "}
            of {q.totalH}h &nbsp;·&nbsp; used {q.usedH}h · reserved {q.reservedH}h
            <div style={{ height: 8, background: "#eee", borderRadius: 4, marginTop: 8, overflow: "hidden" }}>
              <div style={{
                width: `${Math.min(100, ((q.totalH - q.remainingH) / q.totalH) * 100)}%`,
                height: "100%",
                background: q.remainingH < 1 ? "#dc2626" : "#16a34a",
              }} />
            </div>
            <div style={{ color: "#888", marginTop: 4 }}>
              state: <code>{info?.state}</code> · resets {q.refreshTime?.slice(0, 10)}
            </div>
          </div>
        )}
      </section>

      {msg && <p style={{ padding: 10, background: "#f6f6f6", borderRadius: 8 }}>{msg}</p>}

      {/* ── generate ─────────────────────────────────────────────── */}
      <section style={{ border: "1px solid #ddd", borderRadius: 10, padding: 16, marginBottom: 16 }}>
        <h3 style={{ marginTop: 0 }}>Generate</h3>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the video… e.g. a neon city street at night, camera slowly pushes in"
          rows={4}
          style={{ width: "100%", padding: 12, borderRadius: 8, border: "1px solid #ccc", fontFamily: "inherit", boxSizing: "border-box" }}
        />
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginTop: 10, flexWrap: "wrap" }}>
          <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => pickImage(e.target.files?.[0] ?? null)} />
          <button onClick={() => fileRef.current?.click()}
                  style={{ padding: "8px 14px", borderRadius: 8, border: "1px solid #ccc", background: "#fff" }}>
            {imgB64 ? "🖼 Image chosen ✓" : "🖼 Image (image-to-video)"}
          </button>
          {imgB64 && <button onClick={() => setImgB64(null)} style={{ border: "none", background: "none", color: "#dc2626" }}>remove</button>}
          <button
            onClick={generate}
            disabled={working || !live}
            style={{ padding: "10px 20px", borderRadius: 8, border: "none", marginLeft: "auto",
                     background: working || !live ? "#bbb" : "#2563eb", color: "#fff", fontWeight: 600,
                     cursor: working || !live ? "default" : "pointer" }}
          >
            {working ? "⏳ Working…" : live ? "🎬 Generate" : "⚠ Turn ON first"}
          </button>
        </div>
      </section>

      {video && (
        <section>
          <h3>Result</h3>
          <video src={video} controls style={{ width: "100%", borderRadius: 10 }} />
          <a href={video} download="ltx-video.mp4"
             style={{ display: "inline-block", marginTop: 10, padding: "8px 16px", background: "#111", color: "#fff", borderRadius: 8, textDecoration: "none" }}>
            ⬇ Download
          </a>
        </section>
      )}

      {logs.length > 0 && (
        <pre style={{ background: "#111", color: "#0f0", padding: 12, borderRadius: 8, fontSize: 12, overflowX: "auto" }}>
          {logs.join("\n")}
        </pre>
      )}
    </main>
  );
}
