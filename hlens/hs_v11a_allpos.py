"""v11a (laptop, $0): position-general H-space on the fresh v8 dumps. Prereg: paper/SCOPE-hspace-v11a.md.
Classes all (primary), nonpunct (secondary), punctuation (reference), at layers 16 and 40, with v10a's exact rule:
hs_flip5.run_layer (variant d, 200 flips, 50 planted; norm and xnorm) on the class rows of hs_v8_full_L{l}.pt, plus the
cross-run top-5 overlap (norm patterns) with the run-2 class rows (hspace2_full_dump_L{l}.pt).
Writes out/hs_v11a.json and out/v11a_subspaces.pt (top-5 patterns per class/layer, whitened run-2 frame).
    python hs_v11a_allpos.py [smoke]
"""
import json
import sys
import time

import torch
from transformers import AutoTokenizer

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hs_flip5 as F5  # noqa: E402
import hs_v8  # noqa: E402
from hs_v10a_classes import klass2, patterns  # noqa: E402

SMOKE = bool(_a and _a[0] == "smoke")
NFLIP, NPLANT = (6, 3) if SMOKE else (200, 50)
LAYERS = [16, 40]
CLASSES = ["all", "nonpunct", "punctuation"]


def member(c, k):
    return c == "all" or (c == "nonpunct" and k != "punctuation") or (c == "punctuation" and k == "punctuation")


def main():
    t0 = time.time()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    T2 = torch.load("out/pod/hf/out/hspace2_full.pt", map_location="cpu", weights_only=False)
    res, subs = {"smoke": SMOKE, "layers": {}}, {}
    for l in LAYERS:
        P, keep, K = hs_v8.drop_mask(T2[f"Sh{l}"].double(), T2[f"mu{l}"].double())
        Pf, keepd = P.float().to(dev), keep.to(dev)
        D = torch.load(f"out/pod/hf/v8/out/hs_v8_full_L{l}.pt", map_location="cpu", weights_only=False)
        D2 = torch.load(f"out/pod/hf/out/hspace2_full_dump_L{l}.pt", map_location="cpu", weights_only=False)
        ut = {int(t): klass2(tok.decode([int(t)])) for t in torch.unique(torch.cat([D["loc_tok"], D2["loc_tok"]]))}
        cls1 = [ut[t] for t in D["loc_tok"].tolist()]
        cls2 = [ut[t] for t in D2["loc_tok"].tolist()]
        L = {}
        for c in CLASSES:
            idx = torch.tensor([i for i, k in enumerate(cls1) if member(c, k)])
            i2 = torch.tensor([i for i, k in enumerate(cls2) if member(c, k)])
            if SMOKE:
                idx, i2 = idx[:600], i2[:600]
            Y1 = D["loc_y1"][idx].float().to(dev) @ Pf
            Y2 = D["loc_y2"][idx].float().to(dev) @ Pf
            prb = D["loc_probe"][idx]
            rr = F5.run_layer(Y1, Y2, prb, {"d": keepd}, K, NFLIP, NPLANT, dev, 11000 + l, f"v11a L{l} {c}")
            fresh_pat, cols = patterns(Y1.double(), keepd.double(), K)
            run2_pat, _ = patterns(D2["loc_y1"][i2].float().to(dev).double() @ Pf.double(), keepd.double(), K)
            rep = float((fresh_pat.T @ run2_pat).pow(2).sum() / 5)
            B = {"n_rows": len(idx), "n_rows_run2": len(i2),
                 "d": {e_: {k: rr["d"][e_][k] for k in ("obs_lam", "null_lam1_max", "obs_split", "null_split3_max", "p_lam1", "p_split3")}
                       for e_ in ("norm", "xnorm", "raw_trim5")},
                 "planted": rr["planted"], "cross_run_top5_overlap": rep, "random_expect_top5": 5 / float(keep[:K].sum())}
            ok = all(B["d"][e_]["obs_lam"][0] > B["d"][e_]["null_lam1_max"] and B["d"][e_]["obs_split"]["k3"] > B["d"][e_]["null_split3_max"]
                     and B["d"][e_]["obs_split"]["k3"] >= 0.7 and B["planted"][e_]["detected"] for e_ in ("norm", "xnorm"))
            B["replicates"] = bool(ok and rep >= 0.5)
            L[c] = B
            subs[f"H_{c}_{l}"] = (P[:, cols.cpu()] @ fresh_pat.cpu()).float()            # whitened run-2 frame, [d, 5]
            print(f"[v11a] L{l} {c:11s}: rows {len(idx)} (run2 {len(i2)}) | norm lam1 {B['d']['norm']['obs_lam'][0]:.2f} (flip max "
                  f"{B['d']['norm']['null_lam1_max']:.2f}) split3 {B['d']['norm']['obs_split']['k3']:.2f} | xnorm split3 "
                  f"{B['d']['xnorm']['obs_split']['k3']:.2f} | cross-run {rep:.3f} -> {'REPLICATES' if B['replicates'] else 'no'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del Y1, Y2
            torch.cuda.empty_cache()
        res["layers"][str(l)] = L
        del D, D2
        json.dump(res, open("out/hs_v11a" + ("_smoke" if SMOKE else "") + ".json", "w"), indent=1)
    rep_all = {l: res["layers"][str(l)]["all"]["replicates"] for l in LAYERS}
    rep_np = {l: res["layers"][str(l)]["nonpunct"]["replicates"] for l in LAYERS}
    res["punctuation_reference_replicates"] = all(res["layers"][str(l)]["punctuation"]["replicates"] for l in LAYERS)
    if all(rep_all.values()):
        v = "position-general H-space candidate"
    elif any(rep_all.values()):
        v = "position-general at " + " and ".join(f"L{l}" for l in LAYERS if rep_all[l]) + " only"
    else:
        v = "no position-general H-space under this estimator"
    if any(rep_all.values()) and not any(rep_np[l] for l in LAYERS if rep_all[l]):
        v += " (punctuation-carried)"
    res["replicates_all"], res["replicates_nonpunct"] = rep_all, rep_np
    res["VERDICT_v11a"] = v
    print(f"[v11a] VERDICT: {v} | nonpunct {rep_np} | punctuation reference replicates: {res['punctuation_reference_replicates']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("out/hs_v11a" + ("_smoke" if SMOKE else "") + ".json", "w"), indent=1)
    torch.save(subs, "out/v11a_subspaces" + ("_smoke" if SMOKE else "") + ".pt")


if __name__ == "__main__":
    main()
