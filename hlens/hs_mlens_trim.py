"""Is the 27B phi_H ~ 0 a heavy-tail artifact? Trimmed magnitude phi on the run-2 / v8 dumps. CPU, $0, exploratory.
Rows ranked by |y1| |y2| (the per-row energy scale); the top q fraction is removed from BOTH the window means and the
denominator, then phi = mean_k between-window <Ybar_w, Ybar_w'> / mean <y1, y2> on kept rows (hs_mlens_cond.cond_phi).
Selection on the magnitudes is a descriptive cut, not an unbiased estimator: compare across q, not to an absolute bar.
Also reports the share of y1.y2 energy carried by the top 0.1% / 1% rows (the heavy-tail measure).
    python hs_mlens_trim.py              (27B dumps)
    python hs_mlens_trim.py out/phi_08b_dump_rows_L12.pt ...   (hs_phi_cpu --dump rows; keys y1 y2 probe win)
"""
import json
import sys
from pathlib import Path

import torch

SMALL, sys.argv = sys.argv[1:], sys.argv[:1]          # hs_v9 (imported via hs_mlens_cond) reads argv[1] as its TAG
from hs_mlens_cond import cond_phi  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"


def main():
    torch.set_num_threads(12)
    res = {}
    jobs = [("run2", l, OUT / f"pod/hf/out/hspace2_full_dump_L{l}.pt") for l in (16, 28, 40, 52)]
    jobs += [("v8", l, OUT / f"pod/hf/v8/out/hs_v8_full_L{l}.pt") for l in (16, 40)]
    small = SMALL
    if small:
        jobs = [(Path(f).stem.replace("phi_", "").replace("_rows", ""), "", Path(f)) for f in small]
    for name, l, f in jobs:
        D = torch.load(f, map_location="cpu", weights_only=False)
        if small:
            D = {"loc_y1": D["y1"], "loc_y2": D["y2"], "loc_probe": D["probe"], "loc_win": D["win"]}
        win = D["loc_win"] if "loc_win" in D else D["win"]
        Y1, Y2, probe = D["loc_y1"].double(), D["loc_y2"].double(), D["loc_probe"]
        e = (Y1 * Y2).sum(1); sc = Y1.norm(dim=1) * Y2.norm(dim=1)
        order = torch.argsort(sc, descending=True); n = len(sc)
        R = {"rows": n, "energy_share_top0.1pct": float(e[order[: max(1, n // 1000)]].sum() / e.sum()),
             "energy_share_top1pct": float(e[order[: max(1, n // 100)]].sum() / e.sum())}
        for q in (0.0, 0.001, 0.01, 0.05, 0.2):
            keep = torch.ones(n, dtype=torch.bool); keep[order[: int(q * n)]] = False
            c = cond_phi(Y1, Y2, probe, win, keep)
            R[f"trim{q}"] = {"phi": c["phi"], "phi_dir": c["phi_dir"]} if c else None
        res[f"{name}_L{l}"] = R
        print(f"{name} L{l}: top0.1% {R['energy_share_top0.1pct']:.3f} top1% {R['energy_share_top1pct']:.3f} | " +
              " ".join(f"q={q}: {R[f'trim{q}']['phi']:+.4f}" for q in (0.0, 0.001, 0.01, 0.05, 0.2)), flush=True)
        del D
    json.dump(res, open(HERE.parent / "results" / "mlens" / ("mlens_trim_small.json" if small else "mlens_trim.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
