"""Conditional mean-Hessian invariance on the 27B dumps: is curvature J-like ONCE YOU CONDITION on the current token?
CPU, $0, exploratory. For a class C of rows (token class, or one token id), per probe, the window means over the rows
of C give an unbiased between-window product (as in hs_mlens_offline / hs_mlens_dir):
  phi_C     = mean_k between-window <Ybar_w, Ybar_w'> / mean over C rows of <y1, y2>      (magnitude share)
  phi_dir_C = the same on unit rows                                                           (direction share)
Classes: hs_v9.klass of the decoded token (punctuation, digit, ...), and the 12 most frequent token ids per layer.
    python hs_mlens_cond.py
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from hs_v9 import klass

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def between(G):
    s = G.sum(0); n = G.shape[0]
    return float((s @ s - (G * G).sum()) / (n * (n - 1)))


def cond_phi(Y1, Y2, probe, win, mask):
    Y = (Y1 + Y2) / 2
    U = Y / Y.norm(dim=1, keepdim=True).clamp_min(1e-30)
    num, numd, nprobe = [], [], 0
    for k in torch.unique(probe[mask]):
        mk = mask & (probe == k)
        ws = torch.unique(win[mk])
        if len(ws) < 2:
            continue
        nprobe += 1
        num.append(between(torch.stack([Y[mk & (win == w)].mean(0) for w in ws])))
        numd.append(between(torch.stack([U[mk & (win == w)].mean(0) for w in ws])))
    den = float((Y1[mask] * Y2[mask]).sum(1).mean())
    if nprobe < 8:
        return None
    num, numd = np.array(num), np.array(numd)
    rng = np.random.default_rng(0)
    bd = [numd[rng.integers(0, len(numd), len(numd))].mean() for _ in range(1000)]
    return {"rows": int(mask.sum()), "probes": nprobe, "phi": float(num.mean() / den),
            "phi_dir": float(numd.mean()), "phi_dir_ci95": [float(np.quantile(bd, 0.025)), float(np.quantile(bd, 0.975))]}


def main():
    torch.set_num_threads(12)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    res = {}
    jobs = [("run2", l, OUT / f"pod/hf/out/hspace2_full_dump_L{l}.pt") for l in (16, 40, 52)]
    jobs += [("v8", l, OUT / f"pod/hf/v8/out/hs_v8_full_L{l}.pt") for l in (16, 40)]
    for name, l, f in jobs:
        D = torch.load(f, map_location="cpu", weights_only=False)
        win = D["loc_win"] if "loc_win" in D else D["win"]
        Y1, Y2 = D["loc_y1"].double(), D["loc_y2"].double()
        probe, tk = D["loc_probe"], D["loc_tok"]
        cache = {}

        def kl(t):
            if t not in cache:
                cache[t] = klass(tok.decode([int(t)]))
            return cache[t]
        cls = np.array([kl(int(t)) for t in tk])
        R = {"all": cond_phi(Y1, Y2, probe, win, torch.ones(len(tk), dtype=torch.bool))}
        for c in sorted(set(cls)):
            R[f"class:{c}"] = cond_phi(Y1, Y2, probe, win, torch.as_tensor(cls == c))
        for t, _ in Counter(tk.tolist()).most_common(12):
            R[f"tok:{tok.decode([t])!r}"] = cond_phi(Y1, Y2, probe, win, tk == t)
        res[f"{name}_L{l}"] = R
        print(f"== {name} L{l}", flush=True)
        for k, v in R.items():
            if v:
                print(f"   {k:28s} rows {v['rows']:6d} probes {v['probes']:4d} | phi {v['phi']:+.4f} | phi_dir {v['phi_dir']:.4f} "
                      f"CI {[round(x, 4) for x in v['phi_dir_ci95']]}", flush=True)
        del D
    (HERE.parent / "results" / "mlens").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HERE.parent / "results" / "mlens" / "mlens_cond.json", "w"), indent=1)


if __name__ == "__main__":
    main()
