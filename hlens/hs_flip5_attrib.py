"""What drives the v5 non-Sigma-commuting structure? (descriptive; Oct 6 2026)

For each layer: the top-5 v5 patterns (spatial-sign estimator, axis-aligned drop, exactly as hs_c6twin.ex_patterns(v5=True)),
each row's squared projection onto each pattern, aggregated by token (min 8 rows) and token class, and by position;
plus cross-layer overlap of the pattern spans in the raw residual frame (vector and covector conventions).
    python hs_flip5_attrib.py <hspace2_full.pt> <dump_dir> 40 52
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.argv, _a = sys.argv[:1], sys.argv[1:]
from hs_c6twin import ex_patterns  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace/newline"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


def main(t2p, ddir, layers):
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    T2 = torch.load(t2p, map_location="cpu", weights_only=False)
    out, EXs = {}, {}
    for l in layers:
        D = torch.load(Path(ddir) / f"hspace2_full_dump_L{l}.pt", map_location="cpu", weights_only=False)
        EX, _, _ = ex_patterns(T2, l, 5, D, v5=True)          # whitened frame, orthonormal, 5 columns
        EXs[l] = EX
        Y = D["loc_y1"].double()
        Sh = T2[f"Sh{l}"].double()
        ev, P = torch.linalg.eigh(Sh)
        P = P[:, torch.argsort(ev, descending=True)]
        keepcoords = (P @ (P.T @ EX))                         # EX already lies in the kept coords
        U = Y / Y.norm(dim=1, keepdim=True)
        sc = (U @ keepcoords).pow(2)                          # [rows, 5]
        base = sc.mean(0)
        toks, pos = D["loc_tok"], D["loc_pos"]
        R = {"pattern_mean_score": [float(x) for x in base]}
        for j in range(5):
            byt, cnt = defaultdict(float), defaultdict(int)
            for i in range(len(sc)):
                t = int(toks[i]); byt[t] += float(sc[i, j]); cnt[t] += 1
            avg = {t: byt[t] / cnt[t] for t in byt if cnt[t] >= 8}
            top = sorted(avg.items(), key=lambda x: -x[1])[:15]
            cls = defaultdict(float); ccnt = defaultdict(int)
            for t, v in byt.items():
                c = klass(tok.decode([t])); cls[c] += v; ccnt[c] += cnt[t]
            R[f"p{j}"] = {"top_tokens": [[tok.decode([t]), round(v / float(base[j]), 2), cnt[t]] for t, v in top],
                          "lift_by_class": {c: round(cls[c] / ccnt[c] / float(base[j]), 2) for c in cls},
                          "lift_by_position_bucket": {b: round(float(sc[(pos >= lo) & (pos < hi), j].mean() / base[j]), 2)
                                                      for b, (lo, hi) in {"16-31": (16, 32), "32-63": (32, 64), "64-95": (64, 96), "96-127": (96, 128)}.items()}}
            print(f"L{l} pattern {j}: classes {R[f'p{j}']['lift_by_class']} | top tokens "
                  + ", ".join(f"{s!r}x{lift}" for s, lift, _ in R[f'p{j}']['top_tokens'][:10]), flush=True)
        out[str(l)] = R
    if len(EXs) == 2:
        a, b = layers
        Sa, Sb = T2[f"Sh{a}"].double(), T2[f"Sh{b}"].double()
        for conv, (Ma, Mb) in (("vector", (Sa, Sb)), ("covector", (torch.linalg.inv(Sa), torch.linalg.inv(Sb)))):
            Xa, Xb = torch.linalg.qr(Ma @ EXs[a])[0], torch.linalg.qr(Mb @ EXs[b])[0]
            out[f"cross_layer_{conv}"] = float((Xa.T @ Xb).pow(2).sum() / 5)
            print(f"L{a} vs L{b} v5 top-5 span overlap ({conv} frame): {out[f'cross_layer_{conv}']:.3f} (random ~{5/5120:.4f})", flush=True)
    json.dump(out, open("out/hs_flip5_attrib.json", "w"), indent=1)


if __name__ == "__main__":
    main(_a[0], _a[1], [int(x) for x in _a[2:]] or [40, 52])
