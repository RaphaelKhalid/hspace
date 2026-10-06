"""CPU-only analysis of run-2 per-position sample dumps (runs ON THE POD; pod->laptop link is ~40 KB/s, so only the
small JSON comes home). For each layer dump out/hspace2_<tag>_dump_L{l}.pt (paired rows y1, y2 with probe/pos/win/tok):
  1. heavy tail: norm quantiles; share of trace(raw second moment) carried by the top 0.1/1/5% rows
  2. who carries it: top tokens (decoded) and positions by row norm; share of energy by token class
  3. split-half (probe parity) on the dumped rows for raw, trimmed (drop top 1%/5% norms), spatial-sign, cross-moment x,
     and trimmed x, at k = 5, 10, 25
    python hs_dump_analysis.py small|full
"""
import json
import sys
from pathlib import Path

import torch

torch.set_num_threads(8)
tag = sys.argv[1] if len(sys.argv) > 1 else "small"
OUT = Path(__file__).resolve().parent / "out"
model = {"small": "Qwen/Qwen3.5-0.8B", "smoke": "Qwen/Qwen3.5-0.8B", "full": "Qwen/Qwen3.6-27B"}[tag]
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(model)


def top_eigvecs(M, k):
    ev, U = torch.linalg.eigh(M.double())
    return U[:, torch.argsort(ev, descending=True)[:k]]


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace/newline"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


res = {}
for f in sorted(OUT.glob(f"hspace2_{tag}_dump_L*.pt")):
    l = int(f.stem.split("_L")[-1])
    D = torch.load(f, map_location="cpu", weights_only=False)
    R = {}
    for kind in ("loc", "int"):
        if f"{kind}_y1" not in D:
            continue
        y1, y2 = D[f"{kind}_y1"].float(), D[f"{kind}_y2"].float()
        pr, toks, pos = D[f"{kind}_probe"], D[f"{kind}_tok"], D[f"{kind}_pos"]
        n1 = y1.norm(dim=-1)
        e = n1 ** 2
        order = torch.argsort(e, descending=True)
        tot = float(e.sum())
        r = {"n_rows": int(len(y1)),
             "norm_quantiles": {q: float(torch.quantile(n1, q)) for q in (0.5, 0.9, 0.99, 0.999)},
             "energy_share_top": {p: float(e[order[: max(1, int(p * len(e)))]].sum() / tot) for p in (0.001, 0.01, 0.05)}}
        top = order[:40]
        r["top_rows"] = [{"tok": tok.decode([int(toks[i])]), "pos": int(pos[i]), "norm": float(n1[i])} for i in top]
        cls = {}
        for i in range(len(e)):
            c = klass(tok.decode([int(toks[i])]))
            cls[c] = cls.get(c, 0.0) + float(e[i])
        r["energy_by_token_class"] = {c: v / tot for c, v in cls.items()}
        cnt = {}
        for i in range(len(e)):
            c = klass(tok.decode([int(toks[i])]))
            cnt[c] = cnt.get(c, 0) + 1
        r["rows_by_token_class"] = {c: v / len(e) for c, v in cnt.items()}
        half = (pr % 2 == 0)
        q99, q95 = torch.quantile(n1, 0.99), torch.quantile(n1, 0.95)

        def est(mask, kind_):
            a, b = y1[mask], y2[mask]
            if kind_ == "raw":
                return a.T @ a
            if kind_ == "norm":
                u = a / a.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                return u.T @ u
            x = a.T @ b
            return 0.5 * (x + x.T)

        sh = {}
        for name, kind_, keep in (("raw", "raw", None), ("trim1", "raw", q99), ("trim5", "raw", q95), ("norm", "norm", None),
                                  ("x", "x", None), ("x_trim1", "x", q99)):
            m0 = half if keep is None else (half & (n1 <= keep))
            m1 = (~half) if keep is None else ((~half) & (n1 <= keep))
            A, B = est(m0, kind_), est(m1, kind_)
            sh[name] = {f"k{k}": ov(top_eigvecs(A, k), top_eigvecs(B, k)) for k in (5, 10, 25)}
        r["split_half_from_dump"] = sh
        R[kind] = r
        print(f"L{l} {kind}: top1% rows carry {r['energy_share_top'][0.01]:.2f} of energy | split k25: "
              + " ".join(f"{k_}={v_['k25']:.2f}" for k_, v_ in sh.items())
              + " | top toks: " + " ".join(repr(x["tok"]) for x in r["top_rows"][:8]), flush=True)
    res[l] = R
json.dump(res, open(OUT / f"hs_dump_analysis_{tag}.json", "w"), indent=1)
print("done")
