"""v7: token-conditional H-space (laptop CPU, $0; Oct 6 2026). Rule frozen in results/runlog_laptop.md before computing.

Motivation: the raw H-lens energy is token-sparse (L16: 0.1% of rows carry 97.5% of it; digit tokens are 7% of rows and
88% of the energy), so the global per-position operator is a mixture of token-class-specific operators. If an H-space
exists, it may live WITHIN a token class. Test with the reviewed v5 machinery (hs_flip5.run_layer: per-probe PCA sign
flips, axis-aligned drop of PCA 1-5 + massive-span axes, spatial-sign norm / xnorm, pattern split-halves by probe parity,
planted rank-3 control), restricted to rows of one token class.

PASS ("token-conditional H-space, digits (v7)") iff at >= 3 of the 4 layers, for BOTH norm and xnorm on DIGIT rows:
  lambda_1 > all 200 flips, split3 > all 200 flips, split3 >= 0.7, planted control detected; AND class-specificity:
  overlap of the digit top-10 patterns with the all-rows top-10 patterns (same retained coordinates, norm) <= 0.5.
word / punctuation classes: descriptive only.
    python hs_flip7.py <hspace2_full.pt> <dump_dir> 16 28 40 52
"""
import json
import sys
import time

import torch

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hs_flip5 as F  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def main(t2p, ddir, layers):
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    T2 = torch.load(t2p, map_location="cpu", weights_only=False)
    out = {}
    for l in layers:
        t0 = time.time()
        D = torch.load(f"{ddir}/hspace2_full_dump_L{l}.pt", map_location="cpu", weights_only=False)
        Sh, mu = T2[f"Sh{l}"].double(), T2[f"mu{l}"].double()
        ev, P = torch.linalg.eigh(Sh)
        P = P[:, torch.argsort(ev, descending=True)]
        d, K = P.shape[0], 1000
        C = F.conf_basis(mu, Sh, 4)
        wC = (P.T @ C).pow(2).sum(1)
        drop = torch.zeros(d, dtype=torch.bool); drop[:5] = True
        tot = float(wC[:K].sum())
        for i_ in torch.argsort(wC[:K], descending=True):
            if float(wC[:K][drop[:K]].sum()) >= 0.9 * tot:
                break
            drop[int(i_)] = True
        keep = (~drop).float()
        Pf = P.float()
        Y1 = D["loc_y1"].float() @ Pf
        Y2 = D["loc_y2"].float() @ Pf
        probe, toks = D["loc_probe"], D["loc_tok"]
        del D
        ut = {int(t): klass(tok.decode([int(t)])) for t in torch.unique(toks)}
        cls = [ut[int(t)] for t in toks]
        # all-rows reference patterns (variant d, no flips)
        half = (probe % 2 == 0)
        nrm = Y1.norm(dim=1)
        keep_raw = nrm <= torch.quantile(nrm, 0.95)
        ref = F.stats(Y1, Y2, half, K, keep, keep_raw)
        R = {"n_rows": len(cls), "dropped_axes": int(drop.sum()), "all_rows": {e: {"lam": ref[e]["lam"], "split": ref[e]["split"]} for e in ("norm", "xnorm")}}
        for c in ("digit", "punctuation", "word"):
            idx = torch.tensor([i for i, x in enumerate(cls) if x == c])
            if len(idx) < 200:
                R[c] = {"n_rows": len(idx), "skipped": "too few rows"}; continue
            pr = probe[idx]
            res = F.run_layer(Y1[idx], Y2[idx], pr, {"d": keep}, K, 200, 50, "cpu", 7000 + l, f"L{l} {c}")
            hc = (pr % 2 == 0)
            nr = Y1[idx].norm(dim=1)
            obs = F.stats(Y1[idx], Y2[idx], hc, K, keep, nr <= torch.quantile(nr, 0.95))
            spec = ov(obs["norm"]["_pat"]["A"][:, :10], ref["norm"]["_pat"]["A"][:, :10])
            R[c] = {"n_rows": len(idx), "n_probes": int(len(torch.unique(pr))),
                    "d": {e: {k: res["d"][e][k] for k in ("obs_lam", "null_lam1_max", "obs_split", "null_split3_max", "p_lam1", "p_split3")}
                          for e in ("norm", "xnorm", "raw_trim5")},
                    "planted": res["planted"], "overlap_top10_with_all_rows": spec}
            print(f"L{l} {c}: rows {len(idx)}; top-10 overlap with all-rows patterns {spec:.3f}", flush=True)
        R["seconds"] = time.time() - t0
        out[str(l)] = R
        json.dump(out, open("out/hs_flip7_digits.json", "w"), indent=1)
        del Y1, Y2
    ok_layers = 0
    for l, R in out.items():
        r = R.get("digit", {})
        if "d" not in r:
            continue
        ok = all(r["d"][e]["obs_lam"][0] > r["d"][e]["null_lam1_max"] and r["d"][e]["obs_split"]["k3"] > r["d"][e]["null_split3_max"]
                 and r["d"][e]["obs_split"]["k3"] >= 0.7 and r["planted"][e]["detected"] for e in ("norm", "xnorm"))
        ok = ok and r["overlap_top10_with_all_rows"] <= 0.5
        ok_layers += ok
        print(f"L{l}: digit-conditional rule {'PASS' if ok else 'fail'}", flush=True)
    verdict = "token-conditional H-space, digits (v7): FOUND" if ok_layers >= 3 else "token-conditional H-space, digits (v7): not found"
    out["VERDICT_v7"] = f"{verdict} ({ok_layers}/4 layers)"
    json.dump(out, open("out/hs_flip7_digits.json", "w"), indent=1)
    print(out["VERDICT_v7"], flush=True)


if __name__ == "__main__":
    main(_a[0], _a[1], [int(x) for x in _a[2:]] or [16, 28, 40, 52])
