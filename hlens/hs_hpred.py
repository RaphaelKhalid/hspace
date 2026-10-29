"""Does the mean Hessian help PREDICT interventions in unseen contexts? CPU, $0. Exploratory (not preregistered).

The operational J-lens claim: a context-averaged operator predicts what a vector does in a new context. The H analog:
the averaged cross-term T(u_i, u_j) predicts the interaction part of a joint move. At layer l, position q, directions u_i = Sigma^1/2 v_i (v_i unit, random)
rescaled to |u_i| = sqrt(tr Sigma) (the typical centred state norm), move size s = fraction of that norm:
  D_i(x)  = h_L(x + s u_i)_q - h_L(x)_q              (single effect, all orders)
  D_ij(x) = h_L(x + s(u_i + u_j))_q - h_L(x)_q        (joint effect)
  r_ij(x) = D_ij - D_i - D_j                          (interaction; = s^2 u_i^T H u_j + O(s^3))
Means are estimated on train contexts, scored on held-out test contexts (all / rms(h_L(x)_q)):
  R2_J   = 1 - E|D_ij - (Dbar_i + Dbar_j)|^2 / E|D_ij|^2           (averaged first-order lens, incl. self terms)
  R2_JH  = 1 - E|D_ij - (Dbar_i + Dbar_j + rbar_ij)|^2 / E|D_ij|^2   (+ averaged interaction)
  gain   = R2_JH - R2_J
  F_r    = 1 - E|r - rbar|^2 / E|r|^2   (held-out share of the interaction predicted by its mean)
  share  = E|r|^2 / E|D_ij|^2           (how large interactions are at all)
Kill: gain < 0.01 at s = 0.5 means the mean curvature is irrelevant to steering at realistic scales.
    python hs_hpred.py [--model Qwen/Qwen3.5-0.8B] [--layers 12 18] [--K 6] [--windows 96] [--scales 0.05 0.1 0.25 0.5]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--layers", type=int, nargs="*", default=[12, 18])
    ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--windows", type=int, default=96)
    ap.add_argument("--scales", type=float, nargs="*", default=[0.05, 0.1, 0.25, 0.5])
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--tag", default="08b")
    a = ap.parse_args()
    torch.set_num_threads(12)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    m, tok = load(a.model, dt)
    d, T = m.config.get_text_config().hidden_size, a.T
    cov = windows(tok, "train", 24, T, seed=5)
    ids_all = windows(tok, "train", a.windows, T, seed=8)
    assert ids_all.shape[0] == a.windows
    layers = m.model.layers
    st = {"delta": None, "hL": None}

    def add(mod, inp, out):
        if st["delta"] is None:
            return out
        return (out[0] + st["delta"],) + tuple(out[1:]) if isinstance(out, tuple) else out + st["delta"]
    layers[-1].register_forward_hook(lambda mod, inp, o: st.__setitem__("hL", o[0] if isinstance(o, tuple) else o))
    pairs = [(i, j) for i in range(a.K) for j in range(i + 1, a.K)]
    out = {}
    for l in a.layers:
        t0 = time.time()
        hk = layers[l].register_forward_hook(add)
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
        U = U / U.norm(dim=1, keepdim=True) * float(torch.trace(S).sqrt())   # [K, d], each = typical centred norm
        rng = random.Random(9 + l)
        Dsing = {s: [] for s in a.scales}; Dpair = {s: [] for s in a.scales}
        for wi, ids in enumerate(ids_all):
            q = rng.randrange(SKIP + 8, T - 1)

            def h(move):
                st["delta"] = None
                if move is not None:
                    dl = torch.zeros(1, T, d, dtype=dt); dl[0, q] = move.to(dt); st["delta"] = dl
                with torch.no_grad():
                    m.model(input_ids=ids[None], use_cache=False)
                st["delta"] = None
                return st["hL"][0, q].double()
            h0 = h(None); rms = float(h0.pow(2).mean().sqrt())
            for s in a.scales:
                Dsing[s].append(torch.stack([(h(s * U[i]) - h0) / rms for i in range(a.K)]))
                Dpair[s].append(torch.stack([(h(s * (U[i] + U[j])) - h0) / rms for (i, j) in pairs]))
            if (wi + 1) % 8 == 0:
                print(f"L{l}: window {wi+1}/{a.windows} ({(time.time()-t0)/60:.1f} min)", flush=True)
        hk.remove()
        n = a.windows; tr = torch.arange(n) % 2 == 0; te = ~tr
        R = {}
        for s in a.scales:
            Ds = torch.stack(Dsing[s]); Dp = torch.stack(Dpair[s])          # [n, K, d], [n, P, d]
            I = torch.tensor([p[0] for p in pairs]); J = torch.tensor([p[1] for p in pairs])
            r = Dp - Ds[:, I] - Ds[:, J]
            res = {}
            for name, (A, B) in {"split_even": (tr, te), "split_odd": (te, tr)}.items():
                Dbar, rbar = Ds[A].mean(0), r[A].mean(0)
                PJ = Dbar[I] + Dbar[J]
                tot = Dp[B].pow(2).sum(-1).mean()
                e_J = (Dp[B] - PJ).pow(2).sum(-1).mean(); e_JH = (Dp[B] - PJ - rbar).pow(2).sum(-1).mean()
                e_r = (r[B] - rbar).pow(2).sum(-1).mean(); rr = r[B].pow(2).sum(-1).mean()
                e_1 = (Ds[B] - Dbar).pow(2).sum(-1).mean(); s1 = Ds[B].pow(2).sum(-1).mean()
                res[name] = {"R2_J": float(1 - e_J / tot), "R2_JH": float(1 - e_JH / tot), "gain": float((e_J - e_JH) / tot),
                             "F_r": float(1 - e_r / rr), "F_single": float(1 - e_1 / s1), "share": float(rr / tot)}
            R[str(s)] = {k: float(np.mean([res[x][k] for x in res])) for k in res["split_even"]}
            R[str(s)]["halves"] = res
            print(f"L{l} s={s}: R2_J {R[str(s)]['R2_J']:.3f} R2_JH {R[str(s)]['R2_JH']:.3f} gain {R[str(s)]['gain']:+.4f} | "
                  f"F_r {R[str(s)]['F_r']:+.3f} F_single {R[str(s)]['F_single']:.3f} | interaction share {R[str(s)]['share']:.4f}", flush=True)
        out[str(l)] = {"scales": R, "K": a.K, "windows": n, "T": T, "model": a.model, "dtype": a.dtype, "minutes": (time.time() - t0) / 60}
        RES.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(RES / f"hpred_{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
