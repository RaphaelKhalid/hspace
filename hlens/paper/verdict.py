"""Mechanical verdict for SCOPE-hspace.md section A (frozen thresholds).

    python paper/verdict.py out/pod/hspace_full.json
"""
import json
import sys

TH = dict(C0=0.8, C1=0.7, C2=0.5, C3=0.5, C4=10.0, C5=3.0, C6=2.0)


def layer_checks(L):
    a = L.get("ablation", {})
    rand_kl = [v["KL"] for k, v in a.items() if k.startswith("rand25")]
    rand_ratio = [v["ratio"] for k, v in a.items() if k.startswith("rand25")]
    c = {}
    c["C1"] = (L["split_half_overlap"], L["split_half_overlap"] >= TH["C1"])
    c["C2"] = (L["H25_energy_frac"], L["H25_energy_frac"] >= TH["C2"])
    c["C3"] = (L["H25_vs_J25_overlap"], L["H25_vs_J25_overlap"] < TH["C3"])
    c["C4"] = (L["natural"]["ratio"], L["natural"]["ratio"] >= TH["C4"])
    if a:
        kl_ratio = a["H25"]["KL"] / (sum(rand_kl) / len(rand_kl))
        c["C5"] = (kl_ratio, kl_ratio >= TH["C5"])
        r, rp, lo = a["H25"]["ratio"], a["H25perpJ"]["ratio"], a["H25"]["ratio_ci95"][0]
        bar = max(a["J25"]["ratio"], sum(rand_ratio) / len(rand_ratio))
        c["C6"] = ((r, rp, lo, bar), r >= TH["C6"] and rp >= TH["C6"] and lo > bar)
    else:
        c["C5"] = (None, False); c["C6"] = (None, False)
    return c


def main(path):
    R = json.load(open(path))
    pc = [L["pos_control"]["recovery_top25_of_planted"] for L in R["layers"].values() if "pos_control" in L]
    out = {"C0": (pc[0] if pc else None, bool(pc) and pc[0] >= TH["C0"]), "layers": {}}
    for l, L in sorted(R["layers"].items(), key=lambda kv: int(kv[0])):
        out["layers"][l] = layer_checks(L)
    names = ["C1", "C2", "C3", "C4", "C5", "C6"]
    n_all = sum(all(ch[n][1] for n in names) for ch in out["layers"].values())
    n_noC3 = sum(all(ch[n][1] for n in names if n != "C3") for ch in out["layers"].values())
    if not out["C0"][1]:
        v = "VOID (instrument failed the positive control)"
    elif n_all >= 2:
        v = "PASS: H-space found"
    elif n_noC3 >= 2:
        v = "H-space exists but coincides with J-space (all criteria except C3)"
    else:
        v = "FAIL"
    print(f"C0 positive control: {out['C0'][0]} -> {'ok' if out['C0'][1] else 'FAIL'}")
    for l, ch in out["layers"].items():
        print(f"L{l}: " + "  ".join(f"{n}={'PASS' if ch[n][1] else 'fail'}({_f(ch[n][0])})" for n in names))
    print(f"layers passing all six: {n_all}/4; all but C3: {n_noC3}/4\nVERDICT: {v}")
    return v, out


def _f(x):
    if x is None:
        return "-"
    if isinstance(x, tuple):
        return "/".join(f"{y:.2f}" for y in x)
    return f"{x:.2f}"


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "out/pod/hspace_full.json")
