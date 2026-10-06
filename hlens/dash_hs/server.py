"""OLED live monitor for the H-space one-shot pod run (prereg paper/SCOPE-hspace.md).

    python interpcontrol/hlens/dash_hs/server.py 8797      -> http://127.0.0.1:8797

Polls the pod over ssh every ~6 s (runlog, GPU, download size), and pulls out/hspace_full.json every time it
changes so partial results are safe on the laptop even if the watchdog terminates the pod.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
HL = HERE.parent
LOCAL = HL / "out" / "pod"
LOCAL.mkdir(parents=True, exist_ok=True)
HOST, PORT = os.environ.get("HS_POD_HOST", ""), os.environ.get("HS_POD_PORT", "22")
POD_ID = os.environ.get("HS_POD_ID", "")
STARTED = float(os.environ.get("HS_POD_STARTED_EPOCH", "0"))
PRICE = 1.69
WATCHDOG_AT = datetime(2026, 10, 6, 11, 0, 0, tzinfo=timezone.utc).timestamp()   # moved 04:19 UTC (pod/watchdog11.sh)
KEY = os.environ.get("HS_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
SSH = next((p for p in (r"C:\Program Files\Git\usr\bin\ssh.exe", r"C:\Windows\System32\OpenSSH\ssh.exe") if Path(p).exists()), "ssh")
SCP = SSH.replace("ssh.exe", "scp.exe")
OPTS = ["-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR"]
D = "/workspace/mats/interpcontrol/hlens"
REMOTE = (f'cd {D}; echo "@@R"; grep -E "\\] \\[(hspace|ctrl)\\]|^\\[pod " runlog.md | tail -300; echo "@@G"; '
          'nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits; '
          'echo "@@D"; du -sb /workspace/hf 2>/dev/null | cut -f1; echo "@@J"; stat -c %Y out/hspace_full.json 2>/dev/null; echo "@@K"; stat -c %Y out/hspace2_full.json 2>/dev/null; echo "@@L"; stat -c %Y out/c6prime_full.json 2>/dev/null; echo "@@M"; stat -c %Y out/relcurv2_full.json 2>/dev/null; echo "@@N"; stat -c %Y out/c6defl_full.json 2>/dev/null; '
          'echo "@@E"; grep -E "Error|Traceback|OutOfMemory" out/hspace_full.log out/ctrl_full.log 2>/dev/null | tail -3')
STATE: dict = {"updated": 0, "ok": False}
ACC = {"wh": 0.0, "pflop": 0.0, "t": None}          # integrated GPU energy and estimated compute (peak ~110 TFLOP/s x util)
ACC_FILE = LOCAL / "dash_acc.json"
try:
    ACC.update({k: v for k, v in json.loads(ACC_FILE.read_text()).items() if k in ("wh", "pflop")})
    ACC["pflop"] = max(ACC["pflop"], 105.7); ACC["wh"] = max(ACC["wh"], 0.0)
except Exception:  # noqa: BLE001
    ACC["pflop"] = 105.7          # carried over from the previous dashboard process (measured 05:04 UTC)
CTR = []                                              # (t, phase, i) history for a live rate
JSON_MTIME = {"v": None, "v2": None, "c6": None, "rc": None, "c6d": None}


def sh(args, timeout=40):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL)


def parse(out: str) -> dict:
    sec, cur = {}, None
    for ln in out.splitlines():
        if re.fullmatch(r"@@\w", ln.strip()):
            cur = ln.strip()[2:]; sec[cur] = ""
        elif cur:
            sec[cur] += ln + "\n"
    lines = [l for l in sec.get("R", "").splitlines() if l.strip()]
    armed = [i for i, l in enumerate(lines) if "watchdog armed" in l]
    if armed:
        lines = lines[armed[-1]:]          # ignore the failed first launch
    st = {"lines": lines[-14:], "phase": "booting", "probe": None, "ctrl": None, "layer": None, "done": False}
    for l in lines:
        m = re.search(r"\[hspace\] L(\d+) (H|PC) probe (\d+)/(\d+)", l)
        if m:
            st["probe"] = {"layer": int(m[1]), "kind": m[2], "i": int(m[3]), "n": int(m[4])}
            st["phase"] = f"H-space L{m[1]}" if m[2] == "H" else f"positive control L{m[1]}"
        m = re.search(r"\[hspace\] L(\d+) (S\d)", l)
        if m:
            st["layer"] = int(m[1]); st["phase"] = f"L{m[1]} {m[2]}"
        m = re.search(r"S7 (labels|ablations window) (\d+)/(\d+)", l)
        if m:
            lm = re.search(r"L(\d+) S7", l); st["probe"] = {"layer": int(lm[1]) if lm else 0, "kind": "S7", "i": int(m[2]), "n": int(m[3])}
            st["phase"] = "S7 causal: " + ("labelling interaction tokens" if m[1] == "labels" else f"ablations L{st['probe']['layer']}")
        m = re.search(r"\[v2\] L(\d+) probe (\d+)/(\d+)", l)
        if m:
            st["probe"] = {"layer": int(m[1]), "kind": "v2", "i": int(m[2]), "n": int(m[3])}
            st["phase"] = f"run 2 · robust H-space L{m[1]}"
        m = re.search(r"\[v2\] (?:labels|L\d+ ablation window) (\d+)/(\d+)", l)
        if m:
            st["probe"] = {"layer": 0, "kind": "v2B", "i": int(m[1]), "n": int(m[2])}
            st["phase"] = "run 2 · causal tests"
        if "[v2] sweep" in l:
            st["phase"] = "run 2 · layer sweep"
        m = re.search(r"\[ctrl\] full (\d+)/(\d+)", l)
        if m:
            st["ctrl"] = {"i": int(m[1]), "n": int(m[2])}; st["phase"] = "control read"
        if "downloads done" in l:
            st["phase"] = st["phase"] if st["phase"] != "booting" else "loading model"
        if "FINAL DONE" in l:
            st["done"] = True; st["phase"] = "ALL DONE"
    g = sec.get("G", "").strip().split(",")
    if len(g) >= 3:
        try:
            st["gpu"] = {"util": float(g[0]), "mem": float(g[1]) / 1024, "watts": float(g[2])}
        except ValueError:
            pass
    try:
        st["hf_gb"] = int(sec.get("D", "0").strip() or 0) / 1e9
    except ValueError:
        st["hf_gb"] = 0
    st["errors"] = sec.get("E", "").strip()[-400:]
    st["json_mtime"] = sec.get("J", "").strip()
    st["json2_mtime"] = sec.get("K", "").strip()
    st["c6_mtime"] = sec.get("L", "").strip()
    st["rc_mtime"] = sec.get("M", "").strip()
    st["c6d_mtime"] = sec.get("N", "").strip()
    return st


def results() -> dict:
    p = LOCAL / "hspace_full.json"
    if not p.exists():
        return {}
    try:
        r = json.loads(p.read_text())
    except ValueError:
        return {}
    rows = {}
    for l, L in r.get("layers", {}).items():
        a = L.get("ablation", {})
        rand = [v["KL"] for k, v in a.items() if k.startswith("rand")]
        rr = [v["ratio"] for k, v in a.items() if k.startswith("rand")]
        rows[l] = {
            "C1_split": L.get("split_half_overlap"), "C2_energy": L.get("H25_energy_frac"),
            "C3_Jover": L.get("H25_vs_J25_overlap"), "C4_nat": (L.get("natural") or {}).get("ratio"),
            "C5_KL": (a["H25"]["KL"] / (sum(rand) / len(rand))) if a and rand else None,
            "C6_ratio": a.get("H25", {}).get("ratio") if a else None,
            "C6_perp": a.get("H25perpJ", {}).get("ratio") if a else None,
            "C6_lo": (a.get("H25", {}).get("ratio_ci95") or [None])[0] if a else None,
            "C6_bar": max([a["J25"]["ratio"], sum(rr) / len(rr)]) if a and rr else None,
            "PR": L.get("participation_ratio"), "J_in_M": L.get("J25_energy_frac_of_M"),
            "pc": (L.get("pos_control") or {}).get("recovery_top25_of_planted"),
        }
    return {"rows": rows, "labels": r.get("labels")}


def results3() -> dict:
    out = {}
    try:
        rc = json.loads((LOCAL / "relcurv2_full.json").read_text()) if (LOCAL / "relcurv2_full.json").exists() else {}
        cd = json.loads((LOCAL / "c6defl_full.json").read_text()) if (LOCAL / "c6defl_full.json").exists() else {"layers": {}}
    except ValueError:
        return out
    for l in sorted(set(rc) | set(cd.get("layers", {})), key=int):
        a, n = rc.get(l, {}).get("norm_eps0.1", {}), rc.get(l, {}).get("null_eps0.1", {})
        C = (cd.get("layers", {}).get(l) or {}).get("conds", {})
        g = lambda k: (C.get(k) or {}).get("S")
        out[l] = {"s1": (a.get("split") or {}).get("k1"), "s5": (a.get("split") or {}).get("k5"), "s25": (a.get("split") or {}).get("k25"),
                  "n25": (n.get("split") or {}).get("k25"), "pca": a.get("vs_pca25"), "C16": a.get("mass_in_C16_top25"),
                  "S_rel": g("rel25_norm"), "S_vm": g("vm_rand_rel"), "S_norm": g("loc_norm"), "S_pca": g("pca25"),
                  "ci_rel": (cd.get("layers", {}).get(l) or {}).get("S_rel25_minus_vm_rand_ci95"),
                  "ci_pca": (cd.get("layers", {}).get(l) or {}).get("S_loc_norm_minus_pca25_ci95")}
    return out


def results2() -> dict:
    out = {}
    try:
        r2 = json.loads((LOCAL / "hspace2_full.json").read_text()) if (LOCAL / "hspace2_full.json").exists() else {"layers": {}}
        r3 = json.loads((LOCAL / "c6prime_full.json").read_text()) if (LOCAL / "c6prime_full.json").exists() else {"layers": {}}
    except ValueError:
        return out
    for l, L in r2.get("layers", {}).items():
        x, nrm, ix = L.get("loc_x", {}), L.get("loc_norm", {}), L.get("int_x", {})
        ab = L.get("ablation", {})
        c3 = L.get("ablation", {}) and None
        out[l] = {"x25": x.get("split_half"), "x5": x.get("split_half_k5"), "xf": x.get("split_half_kfloor"), "kf": x.get("kfloor"),
                  "n25": nrm.get("split_half"), "i25": ix.get("split_half"), "PR": x.get("PR"),
                  "C3": L.get("C3_loc_x_vs_J25"), "KLx": (ab.get("loc_x") or {}).get("KL_x_rand"),
                  "S": ((r3.get("layers", {}).get(l) or {}).get("conds", {}).get("loc_x") or {}).get("S"),
                  "S_ci": ((r3.get("layers", {}).get(l) or {}).get("conds", {}).get("loc_x") or {}).get("S_ci95"),
                  "c6p": (r3.get("layers", {}).get(l) or {}).get("C6prime_pass_loc_x")}
    for l, L in r3.get("layers", {}).items():
        if l not in out:
            c = (L.get("conds", {}).get("loc_x") or {})
            out[l] = {"S": c.get("S"), "S_ci": c.get("S_ci95"), "c6p": L.get("C6prime_pass_loc_x")}
    return out


def loop():
    while True:
        try:
            out = sh([SSH, *OPTS, "-p", PORT, f"root@{HOST}", REMOTE])
            st = parse(out.stdout)
            st["ok"] = out.returncode == 0
            if st.get("json_mtime") and st["json_mtime"] != JSON_MTIME["v"]:
                r = sh([SCP, *OPTS, "-P", PORT, f"root@{HOST}:{D}/out/hspace_full.json", str(LOCAL / "hspace_full.json")])
                if r.returncode == 0:
                    JSON_MTIME["v"] = st["json_mtime"]
            for key_, fn, mt in (("v2", "hspace2_full.json", st.get("json2_mtime")), ("c6", "c6prime_full.json", st.get("c6_mtime")),
                                 ("rc", "relcurv2_full.json", st.get("rc_mtime")), ("c6d", "c6defl_full.json", st.get("c6d_mtime"))):
                if mt and mt != JSON_MTIME[key_]:
                    r = sh([SCP, *OPTS, "-P", PORT, f"root@{HOST}:{D}/out/{fn}", str(LOCAL / fn)])
                    if r.returncode == 0:
                        JSON_MTIME[key_] = mt
            st["res"] = results()
            st["res2"] = results2()
            st["res3"] = results3()
            now = time.time()
            g = st.get("gpu") or {}
            if ACC["t"] is not None and g:
                dt = now - ACC["t"]
                ACC["wh"] += g.get("watts", 0) * dt / 3600
                ACC["pflop"] += 0.110 * g.get("util", 0) / 100 * dt
            ACC["t"] = now
            st["gpu_wh"], st["pflop"] = ACC["wh"], ACC["pflop"]
            ACC_FILE.write_text(json.dumps({"wh": ACC["wh"], "pflop": ACC["pflop"]}))
            st["w_now"], st["pflops_now"] = g.get("watts", 0), 0.110 * g.get("util", 0) / 100
            pr = st.get("ctrl") if st.get("phase") == "control read" else (st.get("probe") or st.get("ctrl"))
            if pr:
                CTR.append((now, st.get("phase"), pr["i"]))
                same = [c for c in CTR if c[1] == st.get("phase")][-30:]
                st["ctr_rate"] = (same[-1][2] - same[0][2]) / max(same[-1][0] - same[0][0], 1e-6) if len(same) > 1 and same[-1][2] > same[0][2] else 0.0
            st.update(updated=now, elapsed=now - STARTED, usd=(now - STARTED) / 3600 * PRICE,
                      watchdog_left=WATCHDOG_AT - now, price=PRICE, pod=POD_ID)
            STATE.clear(); STATE.update(st)
        except Exception as e:  # noqa: BLE001
            STATE["err"] = f"{type(e).__name__}: {e}"
        time.sleep(6)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.startswith("/state"):
            body = json.dumps(STATE).encode(); ct = "application/json"
        else:
            body = (HERE / "index.html").read_bytes(); ct = "text/html; charset=utf-8"
        self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Cache-Control", "no-store")
        self.end_headers(); self.wfile.write(body)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8797
    threading.Thread(target=loop, daemon=True).start()
    print(f"H-space live monitor on http://127.0.0.1:{port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
