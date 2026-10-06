"""Correction check for v8 (laptop, $0; Oct 6 2026). hs_v8.py deduplicated its fresh windows against seed 0 (run 1),
but run-2 estimation used seed 10 (hspace2.py). Count how many of the 256 v8 windows are identical to run-2 estimation
windows (and probe windows), then recompute the v8 punctuation patterns WITHOUT rows from those windows and their cross-run
top-5 overlap with the run-2 punctuation patterns (same procedure as hs_v8.full_patterns).
    python hs_v8_dedup.py <hspace2_full.pt> <run-2 dump dir> <v8 dump dir> 16 40
"""
import json
import sys

import torch
from transformers import AutoTokenizer

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hspace as H  # noqa: E402
import hs_v8  # noqa: E402  (klass, drop_mask)


def main(t2p, r2dir, v8dir, layers):
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    T = 128
    run1 = H.wikitext_windows(tok, "train", 1280, T, seed=0)
    run2 = H.wikitext_windows(tok, "train", 1280, T, seed=10)
    fresh = H.wikitext_windows(tok, "train", 768, T, seed=5)
    s1 = {tuple(w.tolist()) for w in run1}
    keepw = [w for w in fresh if tuple(w.tolist()) not in s1][:256]
    s2_all = {tuple(w.tolist()) for w in run2}
    s2_probe = {tuple(w.tolist()) for w in run2[256:]}
    dup_all = [i for i, w in enumerate(keepw) if tuple(w.tolist()) in s2_all]
    dup_probe = [i for i, w in enumerate(keepw) if tuple(w.tolist()) in s2_probe]
    print(f"v8 kept windows identical to run-2 estimation windows: {len(dup_all)} (of which run-2 PROBE windows: {len(dup_probe)})", flush=True)
    T2 = torch.load(t2p, map_location="cpu", weights_only=False)
    out = {"n_v8_windows": len(keepw), "dup_run2_all": dup_all, "dup_run2_probe": dup_probe, "layers": {}}
    for l in layers:
        P, keep, K = hs_v8.drop_mask(T2[f"Sh{l}"].double(), T2[f"mu{l}"].double())
        Pf = P.float()
        cols = torch.nonzero(keep[:K] > 0).squeeze(1)

        def pats(Ya):
            U = (Ya * keep / (Ya * keep).norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols]
            M = U.T @ U / len(U)
            dg = torch.diagonal(M).clamp_min(1e-30)
            R = dg.rsqrt()[:, None] * M * dg.rsqrt()[None, :]
            ev, V = torch.linalg.eigh(R)
            V = V[:, torch.argsort(ev, descending=True)[:5]]
            return torch.linalg.qr(dg.sqrt()[:, None] * V)[0]
        D8 = torch.load(f"{v8dir}/hs_v8_full_L{l}.pt", map_location="cpu", weights_only=False)
        ut = {int(t): hs_v8.klass(tok.decode([int(t)])) for t in torch.unique(D8["loc_tok"])}
        isp = torch.tensor([ut[int(t)] == "punctuation" for t in D8["loc_tok"]])
        bad = torch.zeros(len(isp), dtype=torch.bool)
        for i in dup_all:
            bad |= (D8["win"] == i)
        f_all = pats(D8["loc_y1"][isp].float() @ Pf)
        f_clean = pats(D8["loc_y1"][isp & ~bad].float() @ Pf)
        D2 = torch.load(f"{r2dir}/hspace2_full_dump_L{l}.pt", map_location="cpu", weights_only=False)
        ut2 = {int(t): hs_v8.klass(tok.decode([int(t)])) for t in torch.unique(D2["loc_tok"])}
        i2 = torch.tensor([i for i, t in enumerate(D2["loc_tok"].tolist()) if ut2[t] == "punctuation"])
        r2 = pats(D2["loc_y1"][i2].float() @ Pf)
        ov = lambda A, B: float((A.T @ B).pow(2).sum() / A.shape[1])
        L = {"rows_removed": int((isp & bad).sum()), "punct_rows_clean": int((isp & ~bad).sum()),
             "overlap_all_vs_run2": ov(f_all, r2), "overlap_clean_vs_run2": ov(f_clean, r2), "overlap_clean_vs_all": ov(f_clean, f_all)}
        out["layers"][str(l)] = L
        print(f"L{l}: removed {L['rows_removed']} punctuation rows from duplicated windows; cross-run top-5 overlap: all {L['overlap_all_vs_run2']:.3f} -> "
              f"clean {L['overlap_clean_vs_run2']:.3f} (clean vs all {L['overlap_clean_vs_all']:.3f})", flush=True)
        del D8, D2
    json.dump(out, open("out/hs_v8_dedup.json", "w"), indent=1)


if __name__ == "__main__":
    main(_a[0], _a[1], _a[2], [int(x) for x in _a[3:]] or [16, 40])
