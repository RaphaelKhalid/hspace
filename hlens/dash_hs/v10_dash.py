"""OLED live dashboard for the v10 pod run.   python dash_hs/v10_dash.py 8798  ->  http://127.0.0.1:8798
Polls the pod over ssh every 20 s (hs_v10.log + nvidia-smi); read-only."""
import html
import json
import re
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST, PORT = "80.15.7.37", "49208"
STARTED = datetime(2026, 10, 6, 21, 30, 16, tzinfo=timezone.utc).timestamp()
PRICE, WATCHDOG_MIN = 1.69, 200
KEY = str(Path.home() / ".ssh" / "autolabs_runpod")
SSH = next((p for p in (r"C:\Program Files\Git\usr\bin\ssh.exe", r"C:\Windows\System32\OpenSSH\ssh.exe") if Path(p).exists()), "ssh")
D = "/workspace/mats/interpcontrol/hlens"
CMD = (f"cd {D}; echo @@L; grep -E '^\\[v10 |Traceback|Error' out/hs_v10.log 2>/dev/null | tail -400; echo @@P; grep '^\\[pod' runlog.md | tail -20; "
       "echo @@G; nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits")
S = {"lines": [], "pod": [], "gpu": "", "ok": False, "t": 0}


def poll():
    while True:
        try:
            r = subprocess.run([SSH, "-i", KEY, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10", "-o", "LogLevel=ERROR",
                                "-p", PORT, f"root@{HOST}", CMD], capture_output=True, text=True, timeout=40, encoding="utf-8", errors="replace",
                               stdin=subprocess.DEVNULL)
            sec, cur = {}, None
            for ln in r.stdout.splitlines():
                if ln.strip() in ("@@L", "@@P", "@@G"):
                    cur = ln.strip(); sec[cur] = []
                elif cur:
                    sec[cur].append(ln)
            S.update(lines=sec.get("@@L", []), pod=sec.get("@@P", []), gpu=" ".join(sec.get("@@G", [])), ok=r.returncode == 0, t=time.time())
        except Exception as e:  # noqa: BLE001
            S.update(ok=False, err=repr(e)[:120], t=time.time())
        time.sleep(20)


def stage_info(lines):
    txt = "\n".join(lines)
    st = {}
    st["S0"] = [l for l in lines if "] S0 " in l]
    m = re.findall(r"W window (\d+)/(\d+) \(([\d.]+)s/window", txt)
    st["Wprog"] = m[-1] if m else None
    st["W"] = [l for l in lines if re.search(r"\] W L\d+", l)]
    st["G"] = [l for l in lines if "] G " in l]
    st["Tprog"] = re.findall(r"T (\w+): item (\d+)/(\d+)", txt)[-1:] or None
    st["T"] = [l for l in lines if re.search(r"\] T \w+ L\d+", l)]
    st["R"] = [l for l in lines if "] R" in l]
    st["V"] = [l for l in lines if "VERDICT" in l or "verdict detail" in l]
    st["E"] = [l for l in lines if "Traceback" in l or "ERROR" in l]
    return st


def colorize(l):
    e = html.escape(re.sub(r"^\[v10 [^\]]*\] ", "", l))
    for k, c in (("PASS", "#3f6"), ("pass", "#3f6"), ("KILL", "#f44"), ("fail", "#f84"), ("inconclusive", "#fc4"), ("ok", "#3f6")):
        e = re.sub(rf"\b{k}\b", f'<b style="color:{c}">{k}</b>', e)
    return e


PAGE = """<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=15><title>H-space v10</title>
<style>body{{background:#000;color:#ddd;font:14px/1.45 ui-monospace,Consolas,monospace;margin:24px;max-width:1200px}}
h1{{font-size:22px;color:#fff;margin:0 0 4px}} .sub{{color:#888;margin-bottom:18px}} .card{{border:1px solid #222;border-radius:10px;padding:12px 16px;margin:10px 0;background:#050505}}
.h{{color:#9cf;font-weight:bold;margin-bottom:6px}} .bar{{height:8px;background:#111;border-radius:4px;overflow:hidden;margin:6px 0}} .bar>div{{height:100%;background:linear-gradient(90deg,#2a78d6,#1baf7a)}}
.k{{display:inline-block;margin-right:22px}} .big{{font-size:20px;color:#fff}} .err{{color:#f66}} .ln{{white-space:pre-wrap;color:#bbb;font-size:13px}}</style>
<h1>🔬 Finding H-space · v10 registered report · Qwen3.6-27B</h1><div class=sub>{sub}</div>
<div class=card><span class=k>⏱️ <span class=big>{el}</span> elapsed</span><span class=k>💸 <span class=big>${cost:.2f}</span> so far</span>
<span class=k>🛡️ watchdog in <span class=big>{wd}</span></span><span class=k>🖥️ GPU {gpu}</span><span class=k>{conn}</span></div>
{cards}"""


def render():
    now = time.time(); el = now - STARTED
    st = stage_info(S["lines"])
    cards = []

    def card(title, body, prog=None):
        b = f'<div class=bar><div style="width:{100*prog:.0f}%"></div></div>' if prog is not None else ""
        cards.append(f'<div class=card><div class=h>{title}</div>{b}{body}</div>')
    pod = [l for l in S["pod"]]
    card("📦 Pod", "<div class=ln>" + "<br>".join(html.escape(l) for l in pod[-4:]) + "</div>")
    card("🧮 S0 · class moments · M2 kill check · controls", "<div class=ln>" + "<br>".join(colorize(l) for l in st["S0"]) + "</div>")
    wp = st["Wprog"]
    card("📚 W · wikitext · M3 (vs 16 punct twins) · RMS-freeze · word positions · sink",
         (f"<div>windows {wp[0]}/{wp[1]} · {wp[2]} s/window</div>" if wp else "<div>waiting…</div>")
         + "<div class=ln>" + "<br>".join(colorize(l) for l in st["W"]) + "</div>", (int(wp[0]) / int(wp[1])) if wp else 0)
    card("🚦 G · H-blind task gates", "<div class=ln>" + "<br>".join(colorize(l) for l in st["G"]) + "</div>")
    tp = st["Tprog"]
    card("🧩 T · transfer (M6) · routing (M4) · ablation (M5) · gold rank (M7) · TOST",
         (f"<div>{tp[0][0]}: item {tp[0][1]}/{tp[0][2]}</div>" if tp else "<div>waiting…</div>")
         + "<div class=ln>" + "<br>".join(colorize(l) for l in st["T"]) + "</div>", (int(tp[0][1]) / int(tp[0][2])) if tp else 0)
    card("🎲 R · random-weights null", "<div class=ln>" + "<br>".join(colorize(l) for l in st["R"]) + "</div>")
    card("🏁 Verdict", "<div class=ln>" + "<br>".join(colorize(l) for l in st["V"]) + "</div>" if st["V"] else "<div>⏳ pending</div>")
    if st["E"]:
        card("⚠️ Errors", "<div class='ln err'>" + "<br>".join(html.escape(l) for l in st["E"][-6:]) + "</div>")
    fmt = lambda s: f"{int(s//3600)}h{int(s%3600//60):02d}m"
    return PAGE.format(sub=f"pod tf3rmygzca9296 (relaunch #2, healthy host) · RTX PRO 6000 · ${PRICE}/h · prereg SCOPE-hspace-v10.md (frozen 20:52 UTC)",
                       el=fmt(el), cost=PRICE * el / 3600, wd=fmt(max(0, WATCHDOG_MIN * 60 - (now - STARTED - 120))), gpu=html.escape(S["gpu"] or "?"),
                       conn=("🟢 live" if S["ok"] else "🔴 no ssh") + f" · {int(now - S['t'])}s ago", cards="".join(cards))


class Hd(BaseHTTPRequestHandler):
    def do_GET(self):
        b = render().encode("utf-8")
        self.send_response(200); self.send_header("Content-Type", "text/html; charset=utf-8"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    threading.Thread(target=poll, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 8798), Hd).serve_forever()
