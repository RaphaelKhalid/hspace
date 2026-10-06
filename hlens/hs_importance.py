"""How much cheaper can an H-lens get with importance-sampled positions? (laptop, $0; Oct 6 2026)

The raw H-lens energy operator is M = E_p[y_p y_p'] over (window, position) rows. Uniform Hutchinson sampling of positions
has an effective sample size ESS = (sum e)^2 / sum e^2 with e_i = |y_i|^2 (Kish), so with heavy tails most HVPs are wasted.
Sampling row i with probability q_i and weighting by 1/(n q_i) keeps the estimator unbiased; for the Frobenius error of a
sum of rank-one terms the variance-optimal q_i is proportional to |y_i|^2 (oracle), giving ESS = n.

Reported per layer (run-2 'loc' rows, which are themselves a uniform subsample of positions):
  * Kish ESS fraction under uniform sampling, and the oracle cost reduction n / ESS;
  * a PRE-HVP proxy available for free at inference: q_i proportional to the mean energy of the row's token class
    (digit / punctuation / word / whitespace), estimated on probe-parity half A and evaluated on half B (no leakage);
    its ESS and cost reduction;
  * a direct check: split-half reproducibility of the raw top-k eigenspace using m rows drawn uniformly vs by the
    class proxy (importance-weighted), for m in {256, 1024, 4096}.
    python hs_importance.py <dump_dir> 16 28 40 52
"""
import json
import sys

import torch
from transformers import AutoTokenizer


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def topk(M, k):
    ev, U = torch.linalg.eigh(M)
    return U[:, torch.argsort(ev, descending=True)[:k]]


def main(ddir, layers):
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    out = {}
    for l in layers:
        D = torch.load(f"{ddir}/hspace2_full_dump_L{l}.pt", map_location="cpu", weights_only=False)
        y = D["loc_y1"].float()
        toks, probe = D["loc_tok"], D["loc_probe"]
        del D
        e = y.pow(2).sum(1).double()
        n = len(e)
        ut = {int(t): klass(tok.decode([int(t)])) for t in torch.unique(toks)}
        cls = [ut[int(t)] for t in toks]
        A = (probe % 2 == 0)
        # class proxy fitted on half A, evaluated on half B
        cm = {}
        for c in set(cls):
            m = torch.tensor([x == c for x in cls]) & A
            cm[c] = float(e[m].mean()) if m.any() else float(e[A].mean())
        B = ~A
        eB = e[B]
        qB = torch.tensor([cm[c] for c, b in zip(cls, B.tolist()) if b], dtype=torch.float64)
        qB = qB / qB.sum()
        # Exact per-draw Frobenius variance of the unbiased one-row estimator of M_B = sum_i y_i y_i' (draw row i with
        # probability q_i, return y_i y_i' / q_i):  Var(q) = sum_i e_i^2 / q_i - ||M_B||_F^2.  Uniform q_i = 1/n,
        # oracle q_i ~ e_i, class proxy q_i ~ (half-A mean energy of the row's class). Cost reduction = Var_u / Var_q.
        YB = y[B].double()
        F2 = float((YB.T @ YB).pow(2).sum())
        nB = len(eB)
        var_u = float(nB * (eB ** 2).sum()) - F2
        var_p = float((eB ** 2 / qB).sum()) - F2
        var_o = float(eB.sum() ** 2) - F2
        del YB
        r = {"n_rows": n, "ESS_uniform_frac": ess(e) / n,
             "var_reduction_classproxy": var_u / var_p, "var_reduction_oracle": var_u / var_o}
        r["class_energy_share"] = {c: float(e[torch.tensor([x == c for x in cls])].sum() / e.sum()) for c in set(cls)}
        r["class_row_share"] = {c: sum(x == c for x in cls) / n for c in set(cls)}
        # direct check: split-half reproducibility of the raw top-5 with m sampled rows per half
        g = torch.Generator().manual_seed(l)
        q_all = torch.tensor([cm[c] for c in cls], dtype=torch.float64)
        rep = {}
        for mrows in (256, 1024, 4096):
            res = {}
            for mode in ("uniform", "classproxy"):
                Us = []
                for h in (A, B):
                    idx = torch.nonzero(h).squeeze(1)
                    if mode == "uniform":
                        pick = idx[torch.randint(0, len(idx), (mrows,), generator=g)]
                        wts = torch.ones(mrows, dtype=torch.float64)
                    else:
                        q = q_all[idx] / q_all[idx].sum()
                        j = torch.multinomial(q, mrows, replacement=True, generator=g)
                        pick = idx[j]
                        wts = 1.0 / (q[j] * len(idx))
                    Y = y[pick].double() * wts.sqrt()[:, None]
                    Us.append(topk(Y.T @ Y, 5))
                res[mode] = float((Us[0].T @ Us[1]).pow(2).sum() / 5)
            rep[str(mrows)] = res
        r["split_half_top5_raw_by_rows"] = rep
        out[str(l)] = r
        print(f"L{l}: uniform Kish ESS {100*r['ESS_uniform_frac']:.2f}% of rows | HVPs saved at equal Frobenius variance: "
              f"oracle IS {r['var_reduction_oracle']:.0f}x, token-class proxy IS {r['var_reduction_classproxy']:.1f}x (proxy fit on half A, scored on B); "
              f"raw top-5 split-half uniform/proxy: " + ", ".join(f"m={m}: {v['uniform']:.2f}/{v['classproxy']:.2f}" for m, v in rep.items()),
              flush=True)
        del y
    json.dump(out, open("out/hs_importance.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], [int(x) for x in sys.argv[2:]] or [16, 28, 40, 52])
