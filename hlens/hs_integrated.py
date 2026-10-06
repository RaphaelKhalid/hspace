"""Local vs integrated H-lens on natural-scale interactions (Oct 5 2026, Qwen3.5-0.8B, $0).

Exact identity (rectangle theorem): for C^2 f and moves a, b,
    I(a,b) = f(h) - f(h-a) - f(h-b) + f(h-a-b) = int_0^1 int_0^1 a^T Hf(h-a-b+sa+tb) b ds dt.
So a finite interaction is the RECTANGLE-AVERAGED Hessian, not the Hessian at h. v1 found the local
Hessian at h fails at effective move sizes (T2). This test compares, on held-out contexts and natural
moves that remove parts of a token's deviation from the mean (Delta = h_p - mu, split into two random
whitened halves a, b, scaled by lam):
  local lens       L_loc = mean_x' H_x'(h_x')                           (the v1/v2 object)
  integrated lens  L_int = mean_x' int_0^1 H_x'(mu + s Delta_x') ds    (segment from the mean)
  per-context local       a^T H_x(h) b
  per-context integrated  int_0^1 a^T H_x(h-(1-s)(a+b)) b ds          (diagonal of the rectangle)
Objective per c: F_c = sum_{p' valid} c . h_L[p'] (J-lens convention), move at ONE position p.
Writes out/integrated_08b.json.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from common import OUT, Ctx, gpu_mem, load_model, log, patch_deltanet
from hl2 import windows
from hs_anatomy import natural_cov, run_subs, subs_after, vocab_cov_sample

T, SKIP = 96, 16
NODES, WTS = np.polynomial.legendre.leggauss(4)
NODES, WTS = (NODES + 1) / 2, WTS / 2  # Gauss-Legendre on [0,1]


def stack(ctxs, l):
    return torch.cat([c.h_all[l] for c in ctxs], 0)


def F_many(m, ctxs, l, X, Cs):
    """X [B,T,d] layer-l residuals, Cs [nc,d] -> [B,nc]."""
    with torch.no_grad():
        _, yL = run_subs(m, ctxs[0], X, subs_after(m, l), X.shape[0])
        return yL[:, ctxs[0].valid].sum(1) @ Cs.T


def bilinear(m, ctxs, l, X, pos, C, A, Bv):
    """per row b: A[b]^T H_b(X_b)[C_b] Bv[b] with the per-position block at pos[b]. Returns [B]."""
    B = X.shape[0]
    V = torch.zeros_like(X)
    V[torch.arange(B), pos] = Bv
    x = X.clone().requires_grad_(True)
    with torch.enable_grad():
        _, yL = run_subs(m, ctxs[0], x, subs_after(m, l), B)
        F = (yL[:, ctxs[0].valid] * C[:, None, :]).sum()
        (g,) = torch.autograd.grad(F, x, create_graph=True)
        (hv,) = torch.autograd.grad((g * V).sum(), x)
    return (hv[torch.arange(B), pos] * A).sum(-1).detach()


def main():
    l = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    m = load_model()
    patch_deltanet(m)
    gen = torch.Generator(device="cuda").manual_seed(1)
    wins = windows(m.tok, 64, T=T, seed=2)
    ctxs = [Ctx(m, w, l) for w in wins]
    for c in ctxs:
        c.h_all = {l: c.h_all[l]}
    mu, S, Shalf = natural_cov(m, ctxs[:32], l)
    Sinvh = torch.linalg.pinv(Shalf)
    fit, test = ctxs[32:48], ctxs[48:64]
    nc, BT = 4, 4
    Cs = vocab_cov_sample(m, nc, gen)
    lams = [0.1, 0.3, 1.0]
    # test samples: (context, position, a, b)
    samples = []
    for x in test:
        for _ in range(2):
            p = int(torch.randint(SKIP, T - 1, (1,), generator=gen, device="cuda"))
            D = x.h_all[l][0, p] - mu
            Dw = Sinvh @ D
            Q, _ = torch.linalg.qr(torch.randn(m.d, m.d, device="cuda", generator=gen))
            Pa = Q[:, : m.d // 2]
            aw = Pa @ (Pa.T @ Dw)
            samples.append((x, p, Shalf @ aw, Shalf @ (Dw - aw)))
    t0 = time.time()
    true = {lam: np.zeros((len(samples), nc)) for lam in lams}
    for i, (x, p, a, b) in enumerate(samples):
        h = x.h_all[l]
        for lam in lams:
            X = h.repeat(4, 1, 1)
            X[1, p] -= lam * a; X[2, p] -= lam * b; X[3, p] -= lam * (a + b)
            Fv = F_many(m, [x] * 4, l, X, Cs)
            true[lam][i] = (Fv[0] - Fv[1] - Fv[2] + Fv[3]).cpu().numpy()
    log(f"[integ L{l}] true interactions done {time.time()-t0:.0f}s; median |I| by lam "
        + str({lam: float(np.median(np.abs(true[lam]))) for lam in lams}))
    # per-context predictions
    pc_loc = np.zeros((len(samples), nc))
    pc_int = {lam: np.zeros((len(samples), nc)) for lam in lams}
    for i, (x, p, a, b) in enumerate(samples):
        h = x.h_all[l]
        pos = torch.full((nc,), p, device="cuda")
        pc_loc[i] = bilinear(m, [x] * nc, l, h.repeat(nc, 1, 1), pos, Cs, a.repeat(nc, 1), b.repeat(nc, 1)).cpu().numpy()
        for lam in lams:
            acc = 0
            for s, w in zip(NODES, WTS):
                X = h.repeat(nc, 1, 1)
                X[:, p] -= (1 - s) * lam * (a + b)
                acc = acc + w * bilinear(m, [x] * nc, l, X, pos, Cs, a.repeat(nc, 1), b.repeat(nc, 1)).cpu().numpy()
            pc_int[lam][i] = acc
    log(f"[integ L{l}] per-context preds done {time.time()-t0:.0f}s {gpu_mem()}")
    # averaged lenses (local and integrated), applied to each test pair
    lens_loc = np.zeros((len(samples), nc))
    lens_int = np.zeros((len(samples), nc))
    fit_pos = [int(p) for p in torch.randint(SKIP, T - 1, (len(fit),), generator=gen, device="cuda")]
    for i, (_, _, a, b) in enumerate(samples):
        for j, xf in enumerate(fit):
            pf = fit_pos[j]
            pos = torch.full((nc,), pf, device="cuda")
            h = xf.h_all[l]
            lens_loc[i] += bilinear(m, [xf] * nc, l, h.repeat(nc, 1, 1), pos, Cs, a.repeat(nc, 1), b.repeat(nc, 1)).cpu().numpy() / len(fit)
            Df = h[0, pf] - mu
            for s, w in zip(NODES, WTS):
                X = h.repeat(nc, 1, 1)
                X[:, pf] = mu + s * Df
                lens_int[i] += w * bilinear(m, [xf] * nc, l, X, pos, Cs, a.repeat(nc, 1), b.repeat(nc, 1)).cpu().numpy() / len(fit)
        if i % 8 == 7:
            log(f"[integ L{l}] lens preds {i+1}/{len(samples)} {time.time()-t0:.0f}s")

    def score(pred, tru):
        p, t = pred.ravel(), tru.ravel()
        return {"corr": float(np.corrcoef(p, t)[0, 1]), "expl": float(1 - ((t - p) ** 2).sum() / (t ** 2).sum()),
                "slope": float((p * t).sum() / (p * p).sum())}

    out = {"layer": l, "n": len(samples), "nc": nc, "median_absI": {}, "scores": {}}
    for lam in lams:
        tr = true[lam]
        out["median_absI"][lam] = float(np.median(np.abs(tr)))
        out["scores"][lam] = {
            "ctx_local": score(lam ** 2 * pc_loc, tr), "ctx_integrated": score(lam ** 2 * pc_int[lam], tr),
            "lens_local": score(lam ** 2 * lens_loc, tr), "lens_integrated": score(lam ** 2 * lens_int, tr)}
        log(f"[integ L{l}] lam {lam}: " + " | ".join(f"{k} r={v['corr']:.2f} expl={v['expl']:.2f}" for k, v in out["scores"][lam].items()))
    json.dump(out, open(OUT / f"integrated_08b_L{l}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
