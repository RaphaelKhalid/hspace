"""Concept-level tests on the saved exact per-token Hessians (hs_fullh.py --toklist ... --center). Offline, no model.

Preregistered (written before the concept run, 2026-10-27). Clusters: FAKE = simulated, fake, virtual, artificial;
REAL = real, genuine; CTRL = river, music. v_a = top-|lambda| eigvec of token a's EVEN-context mean Hessian (whitened).
  (a) shared axis: mean |cos(v_a, v_b)| within FAKE > max over FAKE x CTRL pairs, and within-FAKE mean |cos| above the
      95th percentile of the same statistic under context sign flips (which kill every mean).
  (b) opposite curvature: r_ab = v_a^T Hbar_b(ODD contexts) v_a, held out. For a in FAKE: sign(r_a,REAL) opposite to
      sign(r_a,FAKE), with |z| >= 3 vs the per-context sign-flip null of b's odd Hessians, for both REAL tokens.
  (c) log-prob: from hs_fullh's causal_logprob (held-out, true log p): pushing +-v_a for a in FAKE raises mean
      log p(FAKE) more than log p(REAL) and more than log p(CTRL).
    python hs_fullh_concept.py --tag concept [--layer 18]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "fullh"
FAKE = [" simulated", " fake", " virtual", " artificial"]
REAL = [" real", " genuine"]
CTRL = [" river", " music"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="concept")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--flips", type=int, default=200)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main_json = json.load(open(RES / f"fullh_{a.tag}_L{a.layer}.json"))
    toks = list(main_json["tokens"])
    NT, n, d = len(toks), a.n, a.d
    assert set(FAKE + REAL + CTRL) <= set(toks), toks
    Hs = np.memmap(HERE / "out" / f"fullh_{a.tag}_L{a.layer}_H.f16", dtype=np.float16, mode="r", shape=(NT, n, d, d))
    even = torch.arange(n) % 2 == 0
    gen = torch.Generator().manual_seed(11)
    V, lam = [], []
    for ti in range(NT):
        H = torch.from_numpy(np.asarray(Hs[ti])).to(dev).float()
        e_, U_ = torch.linalg.eigh(H[even.to(dev)].mean(0))
        k = int(torch.argmax(e_.abs())); V.append(U_[:, k]); lam.append(float(e_[k]))
        del H
    V = torch.stack(V, 1)                                                     # [d, NT]
    # per-context quadratic forms q[a, b, x] = v_a^T H_b(x) v_a  (scalars: the flip null is then exact and cheap)
    Q = torch.zeros(NT, NT, n, dtype=torch.float64)
    Vf_null = {b: [] for b in range(NT)}
    for b in range(NT):
        H = torch.from_numpy(np.asarray(Hs[b])).to(dev).float()
        Q[:, b] = torch.einsum("ia,xij,ja->ax", V, H, V).double().cpu()
        for _ in range(a.flips):                                              # flipped-mean dial of b (cos null)
            eps = (torch.randint(0, 2, (n,), generator=gen).float() * 2 - 1).to(dev)
            e_, U_ = torch.linalg.eigh(torch.einsum("x,xij->ij", eps * even.to(dev), H))
            Vf_null[b].append(U_[:, int(torch.argmax(e_.abs()))].cpu())
        del H
    ix = {t: i for i, t in enumerate(toks)}
    C = (V.T @ V).abs().cpu().numpy()
    within = [C[ix[x], ix[y]] for i, x in enumerate(FAKE) for y in FAKE[i + 1:]]
    cross_ctrl = [C[ix[x], ix[y]] for x in FAKE for y in CTRL]
    null_within = []
    for f in range(a.flips):
        Vn = torch.stack([Vf_null[ix[x]][f] for x in FAKE], 1)
        Cn = (Vn.T @ Vn).abs().numpy()
        null_within.append(np.mean([Cn[i, j] for i in range(len(FAKE)) for j in range(i + 1, len(FAKE))]))
    A = {"mean_cos_within_fake": float(np.mean(within)), "max_cos_fake_ctrl": float(np.max(cross_ctrl)),
         "null_within_q95": float(np.quantile(null_within, 0.95)),
         "p_within": float((np.sum(np.array(null_within) >= np.mean(within)) + 1) / (a.flips + 1))}
    A["pass"] = bool(A["mean_cos_within_fake"] > A["max_cos_fake_ctrl"] and A["mean_cos_within_fake"] > A["null_within_q95"])
    odd = ~even
    Rr, Z = {}, {}
    rng = np.random.default_rng(5)
    for x in FAKE:
        for y in toks:
            qv = Q[ix[x], ix[y], odd].numpy()
            r = float(qv.mean())
            nulls = [float((qv * rng.choice([-1, 1], len(qv))).mean()) for _ in range(2000)]
            Rr[f"{x}->{y}"] = r; Z[f"{x}->{y}"] = r / (np.std(nulls) + 1e-30)
    B = {"r": Rr, "z": Z, "per_fake": {}}
    for x in FAKE:
        s_fake = np.sign(np.mean([Rr[f"{x}->{y}"] for y in FAKE]))
        ok = all(np.sign(Rr[f"{x}->{y}"]) == -s_fake and abs(Z[f"{x}->{y}"]) >= 3 for y in REAL)
        B["per_fake"][x] = {"sign_fake": float(s_fake), "pass": bool(ok)}
    B["pass"] = all(v["pass"] for v in B["per_fake"].values())
    Cc = {}
    for x in FAKE:
        ct = main_json["tokens"][x]["causal_logprob"]["cross_tokens_mean_top"]
        f_, r_, c_ = (np.mean([ct[y] for y in G]) for G in (FAKE, REAL, CTRL))
        Cc[x] = {"fake": float(f_), "real": float(r_), "ctrl": float(c_), "pass": bool(f_ > r_ and f_ > c_)}
    Cc["pass"] = all(v["pass"] for k, v in Cc.items() if k != "pass")
    out = {"tokens": toks, "lambda_even_top": dict(zip(toks, lam)), "cos": C.tolist(), "a_shared_axis": A,
           "b_opposite_curvature": B, "c_logprob": Cc}
    json.dump(out, open(RES / f"fullh_concept_tests_{a.tag}_L{a.layer}.json", "w"), indent=1)
    print("lambda_even_top:", {t: round(v, 4) for t, v in zip(toks, lam)})
    print("|cos| of top dials:\n" + "\n".join(f"{t!r:12s} " + " ".join(f"{c:.2f}" for c in row) for t, row in zip(toks, C)))
    print(f"(a) shared axis: within-FAKE {A['mean_cos_within_fake']:.3f} vs max FAKE-CTRL {A['max_cos_fake_ctrl']:.3f}, "
          f"null q95 {A['null_within_q95']:.3f} (p {A['p_within']:.3f}) -> {'PASS' if A['pass'] else 'FAIL'}")
    for x in FAKE:
        print(f"(b) {x!r}: " + " ".join(f"{y.strip()} {Rr[f'{x}->{y}']:+.3g} (z {Z[f'{x}->{y}']:+.1f})" for y in toks)
              + f" -> {'PASS' if B['per_fake'][x]['pass'] else 'FAIL'}")
    for x in FAKE:
        print(f"(c) {x!r}: dlogp FAKE {Cc[x]['fake']:+.4f} REAL {Cc[x]['real']:+.4f} CTRL {Cc[x]['ctrl']:+.4f} -> {'PASS' if Cc[x]['pass'] else 'FAIL'}")
    print(f"VERDICT (a) {A['pass']} (b) {B['pass']} (c) {Cc['pass']}")


if __name__ == "__main__":
    main()
