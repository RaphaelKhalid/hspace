"""Does a first-order (J-lens-cost) screen predict where H-lens curvature lives? (laptop, Qwen3.5-0.8B, $0; Oct 6 2026)

Uniform H-lens probing has a Kish effective sample size of 0.01-0.06% of rows at 27B: a rare-event estimation problem.
One HVP covers every position of a window, so the lever is WINDOW-level importance sampling: screen candidate windows
with a cheap first-order quantity and spend HVPs where the curvature is. Here, per probe (same c, v, s as hspace.py S4):
  g_p = dF/dx_p (one backward; the J-lens-objective gradient), proxy_p = |Sigma^1/2 g_p|^2 (natural-scale sensitivity)
  y_p = Sigma^1/2 s_p (H (s * v))_p (the H-lens row), e_p = |y_p|^2
Reported per layer:
  * heavy-tail check (Kish ESS of rows) at 0.8B;
  * Spearman(e_p, proxy_p) over rows, and over windows (sum_p e_p vs sum_p proxy_p);
  * exact per-draw Frobenius-variance reduction for estimating M = sum_rows y y' by sampling WINDOWS with
    q_w ~ sum_p proxy_p (proxy), q_w ~ ||Y_w||_F (oracle), vs uniform; Y_w = sum_{p in w} y_p y_p'.
    python hs_proxy08.py [n_probe] [layers...]
"""
import json
import sys
import time

import numpy as np
import torch

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hspace as H  # noqa: E402


def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main(n_probe, layers):
    cfg = H.CFG["small"]
    T, B = cfg["T"], 2
    m = H.Model(cfg["model"], T)
    d = m.d
    gen = torch.Generator(device="cuda").manual_seed(0)
    est = H.wikitext_windows(m.tok, "validation", n_probe * B + 96, T, seed=0).cuda()
    cov_ids, probe_ids = est[:96], est[96:]
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[H.SKIP:T - 1] = True

    def sample_c():
        eta = torch.randn(1, m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
        return ((eta.to(torch.bfloat16) @ m.W_U).float() * m.gain[None])[0]

    out = {}
    for l in layers:
        t0 = time.time()
        S = torch.zeros(d, d, device="cuda", dtype=torch.float64); mu = torch.zeros(d, device="cuda", dtype=torch.float64); n = 0
        for s in range(0, len(cov_ids), 4):
            h = m.states(cov_ids[s:s + 4], l)[:, valid].reshape(-1, d).double()
            mu += h.sum(0); S += h.T @ h; n += h.shape[0]
        mu /= n; S = S / n - torch.outer(mu, mu)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device="cuda", dtype=S.dtype)
        Sh = H.sym_sqrt(S.float())
        E, Pr, Wid, Ys = [], [], [], []
        for i in range(n_probe):
            ix = torch.arange(i * B, (i + 1) * B) % len(probe_ids)
            with torch.no_grad():
                X = m.run(m.embed(probe_ids[ix]), 0, l + 1)
            c = sample_c()
            v = torch.randn(d, device="cuda", generator=gen) @ Sh
            sg = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
            x = X.clone().requires_grad_(True)
            with torch.enable_grad():
                hL = m.run(x, l + 1)
                Fv = (hL[:, valid] @ c).sum()
                (g,) = torch.autograd.grad(Fv, x, create_graph=True)
                (hv,) = torch.autograd.grad((g * (sg[..., None] * v)).sum(), x)
            with torch.no_grad():
                y = ((sg[..., None] * hv)[:, valid] @ Sh)            # [B, P, d]
                pg = (g.detach()[:, valid] @ Sh).pow(2).sum(-1)      # [B, P]
                E.append(y.pow(2).sum(-1).cpu()); Pr.append(pg.cpu())
                Ys.append(y.cpu().to(torch.float32))
                Wid.append(ix.repeat_interleave(int(valid.sum())).view(B, -1))
            if i % 16 == 0:
                print(f"L{l} probe {i+1}/{n_probe} ({time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)", flush=True)
            del x, hL, g, hv
        e = torch.cat([t.reshape(-1) for t in E]).double()
        p = torch.cat([t.reshape(-1) for t in Pr]).double()
        Y = torch.cat([t.reshape(-1, d) for t in Ys]).double()             # rows
        Yw = torch.cat(Ys).double()                                         # [probes*B, P, d] one entry per window-probe
        kish = float(e.sum() ** 2 / (e ** 2).sum()) / len(e)
        ew, pw = Yw.pow(2).sum(-1).sum(-1), torch.cat(Pr).double().sum(-1)
        # window-level Frobenius norms ||Y_w||_F with Y_w = sum_p y_p y_p'  -> sqrt(sum_{p,p'} (y_p . y_p')^2)
        fw = torch.stack([(w @ w.T).pow(2).sum().sqrt() for w in Yw])
        M = Y.T @ Y
        F2 = float(M.pow(2).sum())
        nW = len(fw)

        def var(q):
            q = q / q.sum()
            return float((fw ** 2 / q).sum()) - F2

        vu = float(nW * (fw ** 2).sum()) - F2
        r = {"kish_ess_frac_rows": kish, "spearman_rows": spearman(e.numpy(), p.numpy()),
             "spearman_windows": spearman(ew.numpy(), pw.numpy()),
             "window_var_reduction_proxy_sum": vu / var(pw), "window_var_reduction_proxy_sum2": vu / var(pw ** 2),
             "window_var_reduction_oracle": vu / var(fw), "n_rows": len(e), "n_windows": nW}
        out[str(l)] = r
        print(f"L{l}: Kish ESS {100*kish:.2f}% of rows | Spearman(e, |Sh g|^2): rows {r['spearman_rows']:.2f}, windows {r['spearman_windows']:.2f} | "
              f"window-IS variance reduction: proxy {r['window_var_reduction_proxy_sum']:.2f}x (proxy^2 {r['window_var_reduction_proxy_sum2']:.2f}x), "
              f"oracle {r['window_var_reduction_oracle']:.2f}x ({time.time()-t0:.0f}s)", flush=True)
        del E, Pr, Ys, Y, Yw, M
        torch.cuda.empty_cache()
    json.dump(out, open(H.OUT / "hs_proxy08.json", "w"), indent=1)


if __name__ == "__main__":
    n = int(_a[0]) if _a else 96
    main(n, [int(x) for x in _a[1:]] or [6, 12, 18])
