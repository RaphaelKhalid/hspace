"""Token-conditioned version of hs_hpred: is curvature J-like ONCE THE CURRENT TOKEN IS FIXED? CPU, $0. Exploratory.

Partitions always raise a mean's share (more cells, fewer contexts each), so the comparison is RELATIVE: within one
token, does the held-out interaction share F_r rise toward the held-out single share F_single (curvature becomes
token-switched J-like), or do both rise together (generic partition effect)? For each target token t: n contexts
(distinct windows) whose position q holds t. Pooled control: n contexts drawn from the union of all targets (same n,
same estimator, token mixed). Moves, scales and held-out scoring exactly as hs_hpred (even/odd split, both ways).
    python hs_hpred_tok.py [--layer 18] [--n 48] [--K 6] [--scales 0.25 0.5]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hs_phi_cpu import SKIP, load, windows

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "mlens"
TARGETS = [" .", " ,", " the", " of", " and"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--scales", type=float, nargs="*", default=[0.25, 0.5])
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--tag", default="08b")
    a = ap.parse_args()
    torch.set_num_threads(12)
    m, tok = load(a.model)
    d, T, l = m.config.get_text_config().hidden_size, a.T, a.layer
    cov = windows(tok, "train", 24, T, seed=5)
    pool = windows(tok, "train", 1500, T, seed=11)
    tid = {t: tok(t, add_special_tokens=False)["input_ids"] for t in TARGETS}
    assert all(len(v) == 1 for v in tid.values()), tid
    rng = random.Random(4)
    order = list(range(len(pool))); rng.shuffle(order)
    used, sets = set(), {}
    for t in TARGETS:                                     # disjoint windows across sets
        got = []
        for w in order:
            if w in used:
                continue
            qs = [p for p in range(SKIP + 8, T - 1) if int(pool[w, p]) == tid[t][0]]
            if qs:
                got.append((w, rng.choice(qs))); used.add(w)
            if len(got) == a.n:
                break
        assert len(got) == a.n, (t, len(got))
        sets[t] = got
    union = [x for t in TARGETS for x in sets[t]]
    sets["pooled"] = rng.sample(union, a.n)

    layers = m.model.layers
    st = {"delta": None, "hL": None}

    def add(mod, inp, out):
        if st["delta"] is None:
            return out
        return (out[0] + st["delta"],) + tuple(out[1:]) if isinstance(out, tuple) else out + st["delta"]
    layers[l].register_forward_hook(add)
    layers[-1].register_forward_hook(lambda mod, inp, o: st.__setitem__("hL", o[0] if isinstance(o, tuple) else o))
    hs = []
    with torch.no_grad():
        for w in cov:
            o = m.model(input_ids=w[None], output_hidden_states=True, use_cache=False)
            hs.append(o.hidden_states[l + 1][0, SKIP:T - 1].double())
    S = torch.cov(torch.cat(hs).T)
    ev, V = torch.linalg.eigh(S)
    Sh = (V * ev.clamp_min(0).sqrt()) @ V.T
    g = torch.Generator().manual_seed(500 + l)
    v = torch.randn(d, a.K, generator=g, dtype=torch.float64); v /= v.norm(dim=0, keepdim=True)
    U = (Sh @ v).T
    U = U / U.norm(dim=1, keepdim=True) * float(torch.trace(S).sqrt())
    pairs = [(i, j) for i in range(a.K) for j in range(i + 1, a.K)]
    I = torch.tensor([p[0] for p in pairs]); J = torch.tensor([p[1] for p in pairs])
    cache, t0, out = {}, time.time(), {}

    def measure(w, q):
        if (w, q) in cache:
            return cache[(w, q)]
        ids = pool[w]

        def h(move):
            st["delta"] = None
            if move is not None:
                dl = torch.zeros(1, T, d); dl[0, q] = move.float(); st["delta"] = dl
            with torch.no_grad():
                m.model(input_ids=ids[None], use_cache=False)
            st["delta"] = None
            return st["hL"][0, q].double()
        h0 = h(None); rms = float(h0.pow(2).mean().sqrt())
        r = {}
        for s in a.scales:
            Ds = torch.stack([(h(s * U[i]) - h0) / rms for i in range(a.K)])
            Dp = torch.stack([(h(s * (U[i] + U[j])) - h0) / rms for (i, j) in pairs])
            r[s] = (Ds, Dp)
        cache[(w, q)] = r
        if len(cache) % 24 == 0:
            print(f"measured {len(cache)} contexts ({(time.time()-t0)/60:.1f} min)", flush=True)
        return r

    for name, ctx in sets.items():
        recs = [measure(w, q) for (w, q) in ctx]
        n = len(recs); tr = torch.arange(n) % 2 == 0; te = ~tr
        R = {}
        for s in a.scales:
            Ds = torch.stack([x[s][0] for x in recs]); Dp = torch.stack([x[s][1] for x in recs])
            rr_ = Dp - Ds[:, I] - Ds[:, J]
            hv = []
            for A, B in ((tr, te), (te, tr)):
                Dbar, rbar = Ds[A].mean(0), rr_[A].mean(0)
                PJ = Dbar[I] + Dbar[J]; tot = Dp[B].pow(2).sum(-1).mean()
                e_J = (Dp[B] - PJ).pow(2).sum(-1).mean(); e_JH = (Dp[B] - PJ - rbar).pow(2).sum(-1).mean()
                hv.append({"R2_J": float(1 - e_J / tot), "gain": float((e_J - e_JH) / tot),
                           "F_r": float(1 - (rr_[B] - rbar).pow(2).sum(-1).mean() / rr_[B].pow(2).sum(-1).mean()),
                           "F_single": float(1 - (Ds[B] - Dbar).pow(2).sum(-1).mean() / Ds[B].pow(2).sum(-1).mean()),
                           "share": float(rr_[B].pow(2).sum(-1).mean() / tot)})
            R[str(s)] = {k: float(np.mean([x[k] for x in hv])) for k in hv[0]}
            R[str(s)]["halves"] = hv
            print(f"{name!r:9s} s={s}: F_single {R[str(s)]['F_single']:.3f} F_r {R[str(s)]['F_r']:+.3f} "
                  f"ratio {R[str(s)]['F_r'] / R[str(s)]['F_single']:.3f} | R2_J {R[str(s)]['R2_J']:.3f} gain {R[str(s)]['gain']:+.4f} "
                  f"share {R[str(s)]['share']:.3f}", flush=True)
        out[name] = R
        RES.mkdir(parents=True, exist_ok=True)
        json.dump({"layer": l, "n": a.n, "K": a.K, "model": a.model, "sets": out}, open(RES / f"hpred_tok_{a.tag}_L{l}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
