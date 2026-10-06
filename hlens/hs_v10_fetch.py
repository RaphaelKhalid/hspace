"""Pull the v10 pod results from the private HF dataset into out/pod/v10/ and print a compact summary.
    python hs_v10_fetch.py
"""
import json
import sys
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

R = "RaphaelRaphaelRaphael/hspace-27b-results"
OUT = Path(__file__).resolve().parent / "out" / "pod" / "v10"
OUT.mkdir(parents=True, exist_ok=True)
api = HfApi()
have = {x.path for x in api.list_repo_tree(R, repo_type="dataset", path_in_repo="v10", recursive=True)}
for f in ("v10/out/hs_v10_full.json", "v10/out/hs_v10_full_raw.pt", "v10/out/hs_v10.log", "v10/runlog.md"):
    if f in have:
        hf_hub_download(R, f, repo_type="dataset", local_dir=str(OUT), force_download=True)
J = OUT / "v10/out/hs_v10_full.json"
if not J.exists():
    sys.exit("no hs_v10_full.json yet")
res = json.load(open(J))
V = res.get("verdict", {})
print("FINAL:", V.get("FINAL"))
for k in ("M2", "M3", "R", "M6", "M4", "M5", "M7", "TOST"):
    print(f"  {k:5s} {V.get(k)}")
W = res.get("stages", {}).get("W", {})
for l, Lw in W.items():
    if "I" in Lw:
        print(f"W L{l}: n {Lw['n']} | I T1 {Lw['I']['T1']:+.2f} {Lw['I']['ci95']} rank {Lw['I']['rank']} p_PI {Lw['I']['p_PI']:.3g} | "
              f"I_trunk {Lw['I_trunk']['T1']:+.2f} | global(v9) {Lw['I_global8_v9']['T1']:+.2f} | Jmatched {Lw['I_Jmatched8']['T1']:+.2f} | "
              f"freeze {Lw.get('freeze_retained')} | word {Lw['I_word']['T1']:+.2f} | M3 {Lw['M3']}")
print("errors:", list(res.get("errors", {}).keys()))
