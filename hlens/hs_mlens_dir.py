"""Directional (heavy-tail robust) context invariance of 27B curvature, from the run-2 / v8 dumps. CPU, $0, exploratory.
phi_dir = mean over probes of the between-window product of window means of UNIT rows u = y/|y| (y = (y1+y2)/2,
whitened, all axes). Per-row energy is 1, so phi_dir is the squared length of the context-invariant curvature direction.
Null: per-probe random-sign rows (same magnitude law, no shared direction) -> E = 0; we also report the 'same window,
different position' product as a within-context reference.
    python hs_mlens_dir.py
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def phi_dir(Y, probe, win, pos=None):
    U = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    vals, within = [], []
    for k in torch.unique(probe):
        mk = probe == k
        ws = torch.unique(win[mk])
        if len(ws) < 2:
            continue
        G = torch.stack([U[mk & (win == w)].mean(0) for w in ws])
        s = G.sum(0); n = G.shape[0]
        vals.append(float((s @ s - (G * G).sum()) / (n * (n - 1))))
        for w in ws:                                        # within one context: distinct positions
            Uw = U[mk & (win == w)]
            if Uw.shape[0] >= 2:
                sw = Uw.sum(0); m_ = Uw.shape[0]
                within.append(float((sw @ sw - (Uw * Uw).sum()) / (m_ * (m_ - 1))))
    v = np.array(vals)
    rng = np.random.default_rng(0)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(2000)]
    return float(v.mean()), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], float(np.mean(within))


def main():
    torch.set_num_threads(12)
    res = {}
    jobs = [("run2", l, OUT / f"pod/hf/out/hspace2_full_dump_L{l}.pt") for l in (16, 28, 40, 52)]
    jobs += [("v8", l, OUT / f"pod/hf/v8/out/hs_v8_full_L{l}.pt") for l in (16, 40)]
    for name, l, f in jobs:
        D = torch.load(f, map_location="cpu", weights_only=False)
        win = D["loc_win"] if "loc_win" in D else D["win"]
        Y = ((D["loc_y1"].float() + D["loc_y2"].float()) / 2).double()
        p, ci, wi = phi_dir(Y, D["loc_probe"], win)
        res[f"{name}_L{l}"] = {"phi_dir": p, "ci95": ci, "within_context_same_probe": wi}
        print(f"{name} L{l}: phi_dir (across contexts) = {p:.4f} CI {[round(x, 4) for x in ci]} | within-context positions {wi:.4f}", flush=True)
        del D
    (HERE.parent / "results" / "mlens").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HERE.parent / "results" / "mlens" / "mlens_dir.json", "w"), indent=1)


if __name__ == "__main__":
    main()
