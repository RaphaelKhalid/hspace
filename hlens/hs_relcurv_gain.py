"""Is the relative-curvature subspace a trivial weight-gain / dead-coordinate effect? (post-hoc diagnostic, CPU; Oct 6 2026)

For rel25_{norm,xnorm}_{l} (out/relcurv2_subs.pt, whitened frame), map to raw coordinates in both conventions
(vector: Sh w, covector: Sh^-1 w), and report:
  - coordinate participation ratio of the mass (few coordinates = axis-aligned, suspicious);
  - mass on the top-1% / top-0.25% coordinates by RMSNorm gain |1 + w| of layers l..l+3 (input and post-attention norms)
    and by low per-coordinate variance, each vs the random expectation;
  - the top coordinates.
    python hs_relcurv_gain.py <hspace2_full.pt> <relcurv2_subs.pt> [layers ...]
"""
import json
import sys

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

MODEL = "Qwen/Qwen3.6-27B"


def norm_weights(names):
    idx = json.load(open(hf_hub_download(MODEL, "model.safetensors.index.json")))["weight_map"]
    out = {}
    for n in names:
        key = next((k for k in idx if k.endswith(n)), None)
        if key is None:
            continue
        with safe_open(hf_hub_download(MODEL, idx[key]), "pt") as f:
            out[n] = f.get_tensor(key).float()
    return out


def main(t2p, rcp, layers):
    T2 = torch.load(t2p, map_location="cpu", weights_only=False)
    RC = torch.load(rcp, map_location="cpu", weights_only=False)
    res = {}
    for l in layers:
        Sh = T2[f"Sh{l}"].double()
        d = Sh.shape[0]
        var = torch.diagonal(Sh @ Sh)
        names = [f"layers.{i}.{nm}.weight" for i in range(l, l + 4) for nm in ("input_layernorm", "post_attention_layernorm")]
        W = norm_weights(names)
        gain = torch.stack([(1 + w.double()).abs() for w in W.values()]).max(0).values if W else None
        R = {}
        for var_ in ("norm", "xnorm"):
            key = f"rel25_{var_}_{l}"
            if key not in RC:
                continue
            Ww = RC[key].double()
            for conv, M in (("vector", Sh), ("covector", torch.linalg.inv(Sh))):
                X = M @ Ww
                X = X / X.norm(dim=0, keepdim=True)
                mass = X.pow(2).sum(1) / X.shape[1]                     # per raw coordinate, sums to 1
                r = {"coord_PR": round(float(1 / mass.pow(2).sum()), 1),
                     "top_coords": [[int(i), round(float(mass[i]), 3)] for i in torch.topk(mass, 8).indices]}
                for frac in (0.01, 0.0025):
                    m = max(1, int(frac * d))
                    lowv = torch.topk(-var, m).indices
                    r[f"mass_on_lowvar_top{frac}"] = round(float(mass[lowv].sum()), 3)
                    if gain is not None:
                        hig = torch.topk(gain, m).indices
                        r[f"mass_on_highgain_top{frac}"] = round(float(mass[hig].sum()), 3)
                    r[f"random_expect_{frac}"] = round(m / d, 4)
                R[f"{var_}_{conv}"] = r
                print(f"L{l} {var_:5s} {conv:8s}: coordPR {r['coord_PR']} | low-var 1%: {r['mass_on_lowvar_top0.01']} "
                      f"high-gain 1%: {r.get('mass_on_highgain_top0.01')} (rand 0.01) | top coords {r['top_coords'][:5]}", flush=True)
        res[str(l)] = R
    json.dump(res, open("out/hs_relcurv_gain_full.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]] or [16])
