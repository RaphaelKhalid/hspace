"""Numerical check of the rectangle identity (theory Thm 3) on real Qwen3.5-0.8B at feature scale (lam = 1):
    I(a,b) = f(h) - f(h-a) - f(h-b) + f(h-a-b) = int_0^1 int_0^1 a^T H(h-a-b+sa+tb) b ds dt,
with an n x n Gauss-Legendre rule. Also reports the local (at h) and diagonal-only approximations.
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch

from common import OUT, Ctx, load_model, log, patch_deltanet
from hl2 import windows
from hs_anatomy import natural_cov, vocab_cov_sample
from hs_integrated import F_many, bilinear, T, SKIP


def main(l=12, nq=5, n_samp=8):
    m = load_model(); patch_deltanet(m)
    gen = torch.Generator(device="cuda").manual_seed(5)
    ctxs = [Ctx(m, w, l) for w in windows(m.tok, 40, T=T, seed=3)]
    for c in ctxs:
        c.h_all = {l: c.h_all[l]}
    mu, S, Shalf = natural_cov(m, ctxs[:32], l)
    Sinvh = torch.linalg.pinv(Shalf)
    Cs = vocab_cov_sample(m, 4, gen)
    x_, w_ = np.polynomial.legendre.leggauss(nq)
    x_, w_ = (x_ + 1) / 2, w_ / 2
    rows = []
    t0 = time.time()
    for i in range(n_samp):
        x = ctxs[32 + i]
        p = int(torch.randint(SKIP, T - 1, (1,), generator=gen, device="cuda"))
        Dw = Sinvh @ (x.h_all[l][0, p] - mu)
        Q, _ = torch.linalg.qr(torch.randn(m.d, m.d, device="cuda", generator=gen))
        aw = Q[:, : m.d // 2] @ (Q[:, : m.d // 2].T @ Dw)
        a, b = Shalf @ aw, Shalf @ (Dw - aw)
        h = x.h_all[l]
        X = h.repeat(4, 1, 1); X[1, p] -= a; X[2, p] -= b; X[3, p] -= a + b
        Fv = F_many(m, [x] * 4, l, X, Cs)
        true = (Fv[0] - Fv[1] - Fv[2] + Fv[3]).cpu().numpy()
        pos = torch.full((4,), p, device="cuda")
        rect, diag = np.zeros(4), np.zeros(4)
        for si, s in enumerate(x_):
            for ti, t in enumerate(x_):
                Xq = h.repeat(4, 1, 1); Xq[:, p] += -a - b + s * a + t * b
                rect += w_[si] * w_[ti] * bilinear(m, [x] * 4, l, Xq, pos, Cs, a.repeat(4, 1), b.repeat(4, 1)).cpu().numpy()
            Xd = h.repeat(4, 1, 1); Xd[:, p] += -(1 - s) * (a + b)
            diag += w_[si] * bilinear(m, [x] * 4, l, Xd, pos, Cs, a.repeat(4, 1), b.repeat(4, 1)).cpu().numpy()
        loc = bilinear(m, [x] * 4, l, h.repeat(4, 1, 1), pos, Cs, a.repeat(4, 1), b.repeat(4, 1)).cpu().numpy()
        rows.append({"true": true.tolist(), "rect": rect.tolist(), "diag": diag.tolist(), "local": loc.tolist()})
        log(f"[rect L{l}] sample {i+1}/{n_samp}: |true| {np.abs(true).mean():.3f} rect err {np.abs(rect-true).mean()/np.abs(true).mean():.3f} "
            f"diag err {np.abs(diag-true).mean()/np.abs(true).mean():.3f} local err {np.abs(loc-true).mean()/np.abs(true).mean():.3f} ({time.time()-t0:.0f}s)")
    T_ = np.array([r["true"] for r in rows]).ravel()
    out = {k: {"corr": float(np.corrcoef(np.array([r[k] for r in rows]).ravel(), T_)[0, 1]),
               "rel_err": float(np.abs(np.array([r[k] for r in rows]).ravel() - T_).sum() / np.abs(T_).sum())}
           for k in ("rect", "diag", "local")}
    log(f"[rect L{l}] {nq}x{nq} Gauss-Legendre: " + " | ".join(f"{k} r={v['corr']:.3f} relerr={v['rel_err']:.3f}" for k, v in out.items()))
    json.dump({"layer": l, "nq": nq, "summary": out, "rows": rows}, open(OUT / f"rectangle_08b_L{l}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
