"""J-adjoint local H-lens (goal 2: an H-lens at J-lens cost), Qwen3.5-0.8B, $0.

Exact H-space at layer l needs double backward through all L*-l downstream blocks. Theorem 1 + the anatomy
(curvature is generated mostly in the nearest blocks) suggest truncating to k blocks and replacing the true
adjoint at layer l+k by the J-lens adjoint of record:
    F_k = sum_{p valid} (J_{l+k}^T c) . h_{l+k}[p]          (k = L*-l recovers the exact objective)
Per-probe cost ~ k/(L*-l) of the exact. We measure, with the SAME probes (contexts, c, whitened v, position signs):
  overlap of the top-25 subspaces of M_k and M_exact, and the share of M_exact's energy captured by M_k's top-25.
"""
from __future__ import annotations

import json
import sys
import time

import torch

import bf16w  # noqa: F401
from hspace import CFG, SKIP, Model, log, overlap, sym_sqrt, top_eig, wikitext_windows


def main():
    layers = [int(a) for a in sys.argv[1:]] or [6, 12]
    cfg = CFG["small"]
    T, B, n_probe, k = 96, 2, 96, 25
    m = Model(cfg["model"], T)
    from huggingface_hub import hf_hub_download
    Jall = torch.load(hf_hub_download(cfg["lens"][0], cfg["lens"][2], revision=cfg["lens"][1]), map_location="cpu", weights_only=True)["J"]
    Lstar = max(Jall) + 1
    ids = wikitext_windows(m.tok, "validation", 96 + n_probe * B, T, seed=4).cuda()
    gen = torch.Generator(device="cuda").manual_seed(11)
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[SKIP:T - 1] = True
    d = m.d
    res = {}
    t0 = time.time()
    for l in layers:
        S = torch.zeros(d, d, device="cuda", dtype=torch.float64); mu = torch.zeros(d, device="cuda", dtype=torch.float64); n = 0
        for s in range(0, 96, 4):
            h = m.states(ids[s:s + 4], l)[:, valid].reshape(-1, d).double()
            mu += h.sum(0); S += h.T @ h; n += h.shape[0]
        mu /= n; S = S / n - torch.outer(mu, mu)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device="cuda", dtype=S.dtype)
        Sh = sym_sqrt(S.float())
        ks = [kk for kk in (1, 2, 4, 8) if l + kk < Lstar] + [Lstar - l]
        Ms = {kk: torch.zeros(d, d, device="cuda", dtype=torch.float64) for kk in ks}
        pool = ids[96:]
        for i in range(n_probe):
            X = m.states(pool[i * B:(i + 1) * B], l)
            eta = torch.randn(m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
            c = (eta.to(torch.bfloat16) @ m.W_U).float() * m.gain
            v = torch.randn(d, device="cuda", generator=gen) @ Sh
            sg = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
            for kk in ks:
                top = l + kk                                      # layer whose output carries the J-lens adjoint
                cvec = c if top == Lstar else (Jall[top].float().cuda().T @ c)
                x = X.clone().requires_grad_(True)
                with torch.enable_grad():
                    hk = m.run(x, l + 1, top + 1)
                    Fv = (hk[:, valid] @ cvec).sum()
                    (g,) = torch.autograd.grad(Fv, x, create_graph=True)
                    (hv,) = torch.autograd.grad((g * (sg[..., None] * v)).sum(), x)
                y = (sg[..., None] * hv)[:, valid].reshape(-1, d) @ Sh
                Ms[kk] += y.double().T @ y.double()
            if i % 24 == 0:
                log(f"[local L{l}] probe {i+1}/{n_probe} ({time.time()-t0:.0f}s)")
        exact = Ms[Lstar - l]
        _, Ue = top_eig(exact, k)
        tr = float(torch.trace(exact))
        out = {}
        for kk in ks:
            _, Uk = top_eig(Ms[kk], k)
            out[kk] = {"overlap_top25": overlap(Uk[:, :k], Ue[:, :k]),
                       "exact_energy_in_local_top25": float(torch.trace(Uk[:, :k].T @ exact.float() @ Uk[:, :k]) / tr),
                       "cost_frac": kk / (Lstar - l)}
        out["exact_top25_energy"] = float(torch.trace(Ue[:, :k].T @ exact.float() @ Ue[:, :k]) / tr)
        res[l] = out
        log(f"[local L{l}] " + " | ".join(f"k={kk} ({out[kk]['cost_frac']:.2f} cost): overlap {out[kk]['overlap_top25']:.2f}, "
                                          f"energy {out[kk]['exact_energy_in_local_top25']:.2f}" for kk in ks)
            + f" | exact top25 energy {out['exact_top25_energy']:.2f}")
        json.dump(res, open("out/local_hlens_08b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
