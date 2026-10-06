"""v10a (laptop, $0): token-class scan for H-space on the fresh v8 dumps. Prereg: paper/SCOPE-hspace-v10a.md.
For each class c in {function, content, digit, whitespace} (+ punctuation as a reference) and layer l in {16, 40}:
the v8 rule (hs_flip5.run_layer, variant d, 200 flips, 50 planted; norm and xnorm) on the class-c rows of
hs_v8_full_L{l}.pt, plus the cross-run top-5 overlap (norm patterns) with the run-2 class-c rows (hspace2_full_dump_L{l}.pt).
Writes out/hs_v10a.json and out/v10a_subspaces.pt (H-space_c top-5 patterns, whitened run-2 frame, for every class).
    python hs_v10a_classes.py [smoke]
"""
import json
import sys
import time

import torch
from transformers import AutoTokenizer

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hs_flip5 as F5  # noqa: E402
import hs_v8  # noqa: E402

SMOKE = bool(_a and _a[0] == "smoke")
NFLIP, NPLANT = (6, 3) if SMOKE else (200, 50)
LAYERS = [16, 40]
CLASSES = ["function", "content", "digit", "whitespace", "punctuation"]   # punctuation = reference, not in the family
FUNCTION_WORDS = set("""a an the this that these those my your his her its our their some any no every each either neither
all both few many much more most other another such what which whose whatever whichever
i me you he him she it we us they them myself yourself himself herself itself ourselves themselves who whom
of in on at by for with about against between into through during before after above below to from up down
out off over under again further than as like near since until upon within without along among around across
behind beyond toward towards onto via per
and or but nor so yet if because although though while whereas unless whether
is am are was were be been being have has had having do does did doing will would shall should can could may might must
not""".split())


def klass2(s):
    k = hs_v8.klass(s)
    if k == "word":
        return "function" if s.strip().lower() in FUNCTION_WORDS else "content"
    return k


def patterns(Ya, keep, K):
    cols = torch.nonzero(keep[:K] > 0).squeeze(1)
    U = (Ya * keep / (Ya * keep).norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols]
    M = U.T @ U / len(U)
    dg = torch.diagonal(M).clamp_min(1e-30)
    R = dg.rsqrt()[:, None] * M * dg.rsqrt()[None, :]
    ev, V = torch.linalg.eigh(R)
    V = V[:, torch.argsort(ev, descending=True)[:5]]
    return torch.linalg.qr(dg.sqrt()[:, None] * V)[0], cols


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
            idx = torch.tensor([i for i, k in enumerate(cls1) if k == c])
            i2 = torch.tensor([i for i, k in enumerate(cls2) if k == c])
            if SMOKE:
                idx, i2 = idx[:600], i2[:600]
            Y1 = D["loc_y1"][idx].float().to(dev) @ Pf
            Y2 = D["loc_y2"][idx].float().to(dev) @ Pf
            prb = D["loc_probe"][idx]
            rr = F5.run_layer(Y1, Y2, prb, {"d": keepd}, K, NFLIP, NPLANT, dev, 10000 + l, f"v10a L{l} {c}")
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
            subs[f"H_{c}_{l}"] = (P[:, cols] @ fresh_pat.cpu()).float()            # whitened run-2 frame, [d, 5]
            print(f"[v10a] L{l} {c:11s}: rows {len(idx)} (run2 {len(i2)}) | norm lam1 {B['d']['norm']['obs_lam'][0]:.2f} (flip max "
                  f"{B['d']['norm']['null_lam1_max']:.2f}) split3 {B['d']['norm']['obs_split']['k3']:.2f} | xnorm split3 "
                  f"{B['d']['xnorm']['obs_split']['k3']:.2f} | cross-run {rep:.3f} -> {'REPLICATES' if B['replicates'] else 'no'} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            del Y1, Y2
            torch.cuda.empty_cache()
        res["layers"][str(l)] = L
        del D, D2
        json.dump(res, open("out/hs_v10a" + ("_smoke" if SMOKE else "") + ".json", "w"), indent=1)
    fam = [c for c in CLASSES if c != "punctuation"]
    res["passing_classes"] = [c for c in fam if all(res["layers"][str(l)][c]["replicates"] for l in LAYERS)]
    res["punctuation_reference_replicates"] = all(res["layers"][str(l)]["punctuation"]["replicates"] for l in LAYERS)
    res["VERDICT_v10a"] = ("token-conditional family candidate: " + ", ".join(res["passing_classes"])) if res["passing_classes"] \
        else "no non-punctuation class replicates (v10a)"
    print(f"[v10a] VERDICT: {res['VERDICT_v10a']} | punctuation reference replicates: {res['punctuation_reference_replicates']} "
          f"({time.time()-t0:.0f}s)", flush=True)
    json.dump(res, open("out/hs_v10a" + ("_smoke" if SMOKE else "") + ".json", "w"), indent=1)
    torch.save(subs, "out/v10a_subspaces" + ("_smoke" if SMOKE else "") + ".pt")


if __name__ == "__main__":
    main()
