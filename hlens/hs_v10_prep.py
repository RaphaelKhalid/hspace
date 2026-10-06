"""v10 prep (laptop, $0): build out/v10_inputs_<tag>.pt for hs_v10.py.
  full : H_16/H_40 = the frozen v8 H-space (out/v9_subspaces.pt "EX_l"), J_l = raw J-lens map of record (neuronpedia
         Qwen3.6-27B n1000) at l, v9_articles (validation articles used by v9, excluded from v10), and the v10a classes
         that passed (out/hs_v10a.json) with their H-space_c (out/v10a_subspaces.pt).
  small: Qwen3.5-0.8B dev inputs at layer 6: a random 5-dim "H-space", the 0.8B J-lens, a random J25, and one fake
         passing class (digit) so every code path runs.
    python hs_v10_prep.py full|small
"""
import json
import sys

import torch
from huggingface_hub import hf_hub_download

import hspace as H

TAG = sys.argv[1] if len(sys.argv) > 1 else "small"
cfg = H.CFG[TAG]
lens_path = hf_hub_download(cfg["lens"][0], cfg["lens"][2], revision=cfg["lens"][1])
lens = torch.load(lens_path, map_location="cpu", weights_only=True, mmap=True)
out = {}
if TAG == "full":
    SUB = torch.load(H.OUT / "v9_subspaces.pt", map_location="cpu", weights_only=False)
    for l in (16, 40):
        out[f"H_{l}"] = SUB[f"EX_{l}"].float()
        out[f"J_{l}"] = lens["J"][l].float().clone()
        out[f"G_{l}"] = SUB[f"tw_{l}"].float()                 # v9's exact global Sigma-orbit twins (continuity arm)
    raw9 = torch.load(H.OUT / "pod/live/hs_v9_full_raw.pt", map_location="cpu", weights_only=False)
    out["v9_articles"] = sorted({int(a) for (a, _, _) in raw9["plan"]})
    A = json.load(open(H.OUT / "hs_v10a.json"))
    out["classes_passing"] = list(A.get("passing_classes", []))
    S10a = torch.load(H.OUT / "v10a_subspaces.pt", map_location="cpu", weights_only=False)
    for c in out["classes_passing"]:
        for l in (16, 40):
            out[f"Hc_{c}_{l}"] = S10a[f"H_{c}_{l}"].float()
else:
    d = lens["J"][6].shape[0]
    g = torch.Generator().manual_seed(6)
    for l in (6, 12):
        out[f"H_{l}"] = torch.linalg.qr(torch.randn(d, 5, generator=g, dtype=torch.float64))[0].float()
        out[f"J_{l}"] = lens["J"][l].float().clone()
        out[f"J25w_{l}"] = torch.linalg.qr(torch.randn(d, 25, generator=g, dtype=torch.float64))[0].float()
    out["v9_articles"] = []
    out["classes_passing"] = []
torch.save(out, H.OUT / f"v10_inputs_{TAG}.pt")
print({k: (tuple(v.shape) if hasattr(v, "shape") else (len(v) if isinstance(v, list) else v)) for k, v in out.items()})
