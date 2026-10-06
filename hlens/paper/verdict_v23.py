"""Mechanical verdicts for run 2 (SCOPE-hspace-v2, exploratory) and v3 (SCOPE-hspace-v3: C1' + C6').

    python paper/verdict_v23.py out/pod/hf/out/hspace2_full.json out/pod/hf/out/c6prime_full.json
"""
import json
import sys


def f(x, d=2):
    return "-" if x is None else (f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x))


def main(p2, p3=None):
    R2 = json.load(open(p2))
    R3 = json.load(open(p3)) if p3 else {"layers": {}}
    layers = sorted(R2["layers"], key=int)
    print("RUN 2 (exploratory; SCOPE-hspace-v2) — per variant: C1 split@25, C2 debiased energy, C3 vs J25, C5 KLxrand, C6 ratio")
    v2_pass = {"loc_x": 0, "loc_norm": 0}
    for l in layers:
        L = R2["layers"][l]
        ab = L.get("ablation", {})
        for var in ("loc_raw", "loc_norm", "loc_x", "int_x"):
            if var not in L:
                continue
            s = L[var]
            c1 = s["split_half"] >= 0.7
            c2 = (s.get("top25_energy_debiased") or 0) >= 0.5
            c3v = L.get(f"C3_{var}_vs_J25")
            c3 = c3v is not None and c3v < 0.5
            a = ab.get(var, {})
            c5 = a.get("KL_x_rand", 0) >= 3
            rr = [ab[k]["ratio"] for k in ab if k.startswith("rand")]
            c6 = bool(a) and a["ratio"] >= 2 and a["ratio_ci95"][0] > max(ab.get("J25", {}).get("ratio", -1e9), sum(rr) / max(1, len(rr)))
            allp = c1 and c2 and c3 and c5 and c6
            if var in v2_pass and allp:
                v2_pass[var] += 1
            print(f"  L{l} {var:9s} C1={f(s['split_half'])}{'*' if c1 else ''} k5={f(s.get('split_half_k5'))} kfloor={s.get('kfloor')}:{f(s.get('split_half_kfloor'))} "
                  f"C2={f(s.get('top25_energy_debiased'))}{'*' if c2 else ''} PR={f(s.get('PR'),1)} C3={f(c3v)}{'*' if c3 else ''} "
                  f"C5={f(a.get('KL_x_rand'),1)}{'*' if c5 else ''} C6={f(a.get('ratio'))}{'*' if c6 else ''} {'ALL' if allp else ''}")
    print(f"RUN-2 exploratory claim 'robust H-space found' (loc_x or loc_norm all-pass at >=2 layers): "
          f"{'YES' if max(v2_pass.values()) >= 2 else 'NO'} {v2_pass}")
    print("\nV3 (SCOPE-hspace-v3) — loc_x: C1' (kfloor>=3 & split@kfloor>=0.7) [k5>=0.7 secondary], C6' (S>=0.2, CI>0, CI(S-ref)>0), C3")
    n3 = 0
    for l in layers:
        s = R2["layers"][l].get("loc_x", {})
        kf, shf, sh5 = s.get("kfloor"), s.get("split_half_kfloor"), s.get("split_half_k5")
        c1p = kf is not None and kf >= 3 and shf is not None and shf >= 0.7
        c3v = R2["layers"][l].get("C3_loc_x_vs_J25"); c3 = c3v is not None and c3v < 0.5
        L3 = R3["layers"].get(l)
        c6p = bool(L3 and L3.get("C6prime_pass_loc_x"))
        S = L3["conds"]["loc_x"] if L3 else {}
        ok = c1p and c6p and c3
        n3 += ok
        print(f"  L{l}: C1' kfloor={kf} split={f(shf)} k5={f(sh5)} -> {'PASS' if c1p else 'fail'} | C6' S={f(S.get('S'))} CI={S.get('S_ci95')} "
              f"CI(S-ref)={S.get('S_minus_ref_ci95')} -> {'PASS' if c6p else ('fail' if L3 else 'not run')} | C3={f(c3v)} -> {'ALL' if ok else ''}")
    print(f"V3 VERDICT: {'H-space found (v3)' if n3 >= 2 else 'not found (v3)'}  ({n3}/4 layers pass C1' & C6' & C3; C0 passed in run 1)")


if __name__ == "__main__":
    main(*sys.argv[1:3])
