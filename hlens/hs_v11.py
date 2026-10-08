"""v11: is position-general H-space (v11a's H_all) real, used, and distinct from J-space? Prereg: paper/SCOPE-hspace-v11.md.
Imports the frozen v10 machinery (hs_v10.py) unchanged: Runner, rect, eval_batch, twins_w, two_way_T1, cboot, windows_val.

Stages, in priority order (each guarded, results saved incrementally, R last because it destroys the weights):
  S0  all-position + punctuation class moments; objects H (v11a H_all), HJ (H with J25 projected out), EX (v10's punctuation
      H-space), J5; 8 all-position twins for H (tw*) and for HJ (tx*); M2' (descriptive: H's whitened mean-direction energy)
  L   H-blind localization on split A: suffix offsets o = 1..6 from the last token, DiD_site per (bank, layer, o); o* =
      argmax ratio; site gate (v10 thresholds) and behavioural gate
  T   used tests on split B at o*: M4 routing (projection deletion, D = removed_frac / energy share) and M5 ablation
      (4-corner-mean coordinates), for H vs tw*, HJ vs tx*, and J5
  W   fresh wikitext-validation windows (k >= 5), one random position from ALL tokens: M3 (H vs tw*, + RMS-freeze arm) and
      M3res (HJ vs tx*), with v10's M3 rules
  R   random-weights null (blocks > l shuffled, L40 then L16), H vs tw* on the first 24 W windows
Verdict (v10 aggregation): MUST = M3, R, M3res, TOST over both layers; M4, M5 >= 2 contexts at >= 1 layer.
    python hs_v11.py small|full [--minutes N]
"""
import hashlib
import json
import math
import os
import random
import sys
import time
import traceback

import numpy as np
import torch

_raw = sys.argv[1:]
TAG = _raw[0] if _raw else "small"
_r = list(_raw[1:])
MINUTES = 160.0
if "--minutes" in _r:
    k_ = _r.index("--minutes"); MINUTES = float(_r[k_ + 1]); del _r[k_:k_ + 2]
sys.argv = [sys.argv[0], TAG]
import hs_v10 as V  # noqa: E402  (frozen v10 machinery, imported unchanged)
import hspace as H  # noqa: E402
from hs_v9 import klass  # noqa: E402
sys.argv = sys.argv[:1]

FULL = TAG == "full"
assert V.TAG == TAG
LAYERS = V.LAYERS
MP, NEXT, T_W, T_T = V.MP, V.NEXT, V.T_W, V.T_T
N_COV = V.N_COV
N_W = 40 if FULL else 4
N_R = 24 if FULL else 2
N_LOC = 16 if FULL else 3
N_TEST = 32 if FULL else 3
N_NULL = 24 if FULL else 3
MIN_W, MIN_T, MIN_R = (24, 12, 12) if FULL else (2, 2, 2)
MAX_OFF = 6
NT, NTP = 16, 8
BANKS, BACKUPS, NULLB = V.BANKS, V.BACKUPS, V.NULLB
BUDGET = {"S0": 10, "L": 20, "T": 50, "W": 50, "R": 14} if FULL else {"S0": 6, "L": 6, "T": 8, "W": 6, "R": 6}
OUTF = H.OUT / f"hs_v11_{TAG}.json"
RAWF = H.OUT / f"hs_v11_{TAG}_raw.pt"
T0 = time.time()
DEADLINE = T0 + 60 * MINUTES
LOG2 = math.log(2)
TWH = [f"tw{j}" for j in range(NT)]
TWX = [f"tx{j}" for j in range(NT)]
TWP, TXP = TWH[:NTP], TWX[:NTP]          # 8-twin subsets (v10 TWP role: freeze arm, M5, R)
SUBS = ["H", "HJ", "J5", "EX"] + TWH + TWX                  # W arms and M4 routing
SUBS_A = ["H", "HJ", "J5", "EX"] + TWP + TXP               # M5 ablation
CORNERS = ("00", "01", "10", "11")


def log(msg):
    print(f"[v11 {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


def absI(r, arm, ci_, key="pairs"):
    return np.mean([abs(r[key][f"{arm}|{i}{j}"][ci_]) for (i, j) in MP])


def arm_stats2(recs, ex, names_tw, comp, clusters, key="pairs"):
    ci_ = {"I": 0, "I_trunk": 1}[comp]
    e = [absI(r, ex, ci_, key) for r in recs]
    tw = [[absI(r, t, ci_, key) for t in names_tw] for r in recs]
    return V.two_way_T1(e, tw, clusters)


def log_ratio2(recs, a, b, clusters):
    v = np.log(np.array([absI(r, a, 0) for r in recs]) + 1e-12) - np.log(np.array([absI(r, b, 0) for r in recs]) + 1e-12)
    return {"mean": float(v.mean()), "ci95": V.boot_ci(v, clusters)}


def specificity(recs, a):
    s = [np.mean([abs(r["pairs"][f"{a}|{i}{j}"][0]) for (i, j) in MP])
         / (np.mean([(abs(r["pairs"][f"{a}|{i}{j}"][3]) + abs(r["pairs"][f"{a}|{i}{j}"][4])) / 2 for (i, j) in MP]) + 1e-12) for r in recs]
    return float(np.exp(np.mean(np.log(np.array(s) + 1e-12))))


def strong(s):
    return s["T1"] > LOG2 and s["ci95"][0] > 0 and s["rank"] == 1 and s["p_PI"] < 0.05


def ctx_status(per):
    """v10's rule: per = {context: [status at each layer]}; a context passes if it passes at >= 1 layer;
    pass iff >= 2 contexts pass; inconclusive if < 2 contexts are evaluable (pass/fail at some layer); else fail."""
    n_pass = sum(any(s == "pass" for s in v) for v in per.values())
    n_eval = sum(any(s in ("pass", "fail") for s in v) for v in per.values())
    return "pass" if n_pass >= 2 else ("inconclusive" if n_eval < 2 else "fail")


# ------------------------------------------------------------------ main
def main():
    log(f"start {TAG}, layers {LAYERS}, budget {MINUTES} min")
    res = {"tag": TAG, "layers": LAYERS, "stages": {}, "errors": {}, "verdict": {}}
    raw = {}

    def save():
        tmp = str(OUTF) + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUTF)
        torch.save(raw, str(RAWF) + ".tmp"); os.replace(str(RAWF) + ".tmp", RAWF)

    MAN = json.load(open(H.HERE / "v10_banks" / "MANIFEST.json"))
    for b in BANKS + BACKUPS + [NULLB]:
        h = hashlib.sha256(open(H.HERE / "v10_banks" / f"{b}.jsonl", "rb").read()).hexdigest()
        assert h == MAN["banks"][b]["sha256"], f"bank {b} hash mismatch"
    T2 = torch.load(H.OUT / f"hspace2_{TAG}.pt", map_location="cpu", weights_only=False)
    INP = torch.load(H.OUT / f"v10_inputs_{TAG}.pt", map_location="cpu", weights_only=False)
    if FULL:
        sp = H.OUT / "v11a_subspaces.pt"
        res["v11a_subspaces_sha256"] = hashlib.sha256(open(sp, "rb").read()).hexdigest()
        assert res["v11a_subspaces_sha256"].startswith("29e0b7cb95991011"), "v11a subspace file hash mismatch"
        SUB = torch.load(sp, map_location="cpu", weights_only=False)
    R = V.Runner()
    m, tok = R.m, R.tok
    d = m.d
    cache = {}

    def cls(t):
        if t not in cache:
            cache[t] = klass(tok.decode([int(t)]))
        return cache[t]

    def f_scorer(Xt, Yt):
        def sc(rows_, b):
            lp = torch.log_softmax(m.logits(m.text.norm(rows_[-1:])).double()[0], -1)
            return float(lp[Xt] - lp[Yt])
        return sc

    # ================= S0
    st0 = time.time()
    S0 = {}
    res["stages"]["S0"] = S0
    covw = H.wikitext_windows(tok, "train", N_COV, T_W, seed=11)
    classes = ["all", "punctuation"]
    acc = {l: {c: [0, torch.zeros(d, dtype=torch.float64, device="cuda"), torch.zeros(d, d, dtype=torch.float64, device="cuda")]
               for c in classes} for l in LAYERS}
    for b0 in range(0, len(covw), 8):
        ids = covw[b0:b0 + 8].cuda()
        tk = ids[:, H.SKIP:T_W - 1].reshape(-1).tolist()
        sel = {"all": torch.arange(len(tk), device="cuda"),
               "punctuation": torch.tensor([i for i, t in enumerate(tk) if cls(t) == "punctuation"], dtype=torch.long, device="cuda")}
        with torch.no_grad():
            h = m.embed(ids); prev = 0
            for l in sorted(LAYERS):
                h = m.run(h, prev, l + 1); prev = l + 1
                Xl = h[:, H.SKIP:T_W - 1].reshape(-1, d).double()
                for c in classes:
                    if len(sel[c]):
                        Xs = Xl[sel[c]]
                        acc[l][c][0] += len(Xs); acc[l][c][1] += Xs.sum(0); acc[l][c][2] += Xs.T @ Xs
    log(f"S0: class moments from {len(covw)} windows")
    ctrl, kill = {}, {}
    for l in LAYERS:
        Mu, Sig = {}, {}
        for c in classes:
            n_, s1, s2 = acc[l][c]
            mu_c = s1 / n_
            S_ = (s2 / n_ - torch.outer(mu_c, mu_c)) * n_ / (n_ - 1)
            S_ = 0.95 * S_ + 0.05 * torch.trace(S_) / d * torch.eye(d, dtype=S_.dtype, device=S_.device)
            Mu[c], Sig[c] = mu_c.cpu(), S_.cpu()
        del acc[l]
        if f"Sh{l}" in T2:
            Sh = T2[f"Sh{l}"].double(); mu = T2[f"mu{l}"].double()
        else:                                                       # small-mode dev layers without a run-2 frame
            evs, Vs = V.eigh_desc(Sig["all"])
            Sh = Vs @ torch.diag(evs.clamp_min(1e-12).sqrt()) @ Vs.T; mu = Mu["all"]
        Shi = torch.linalg.inv(Sh.cuda()).cpu()
        Hw = SUB[f"H_all_{l}"].double() if FULL else INP[f"H_{l}"].double()     # small mode: smoke stand-in
        EXw = INP[f"H_{l}"].double()
        J25 = (INP[f"J25w_{l}"] if f"J25w_{l}" in INP else T2[f"J25w_{l}"]).double()
        QJ = V.orth(J25)
        J5w = V.orth(J25[:, :5])
        Hres = Hw - QJ @ (QJ.T @ Hw)
        HJ = V.orth(Hres)
        evA, Va = V.eigh_desc(Shi @ Sig["all"] @ Shi)
        evg, Pg = V.eigh_desc(Sh)
        L0 = {}
        mt = Shi @ mu; mt = mt / mt.norm()                          # whitened mean direction (outlier/massive-activation axis)
        refs = V.twins_w(Hw, Pg, 200, 7100 + l)
        eH = float((Hw.T @ mt).pow(2).sum()); eR = np.array([float((Q.T @ mt).pow(2).sum()) for Q in refs])
        kill[l] = False                                             # M2' is descriptive in v11 (cannot fire by construction; see SCOPE)
        L0["M2prime_descriptive"] = {"H_mean_energy": eH, "ref_max": float(eR.max()), "ref_q95": float(np.quantile(eR, 0.95)),
                                     "would_kill_at_0.25": bool(eH >= 0.25 and eH > eR.max())}
        tw = V.twins_w(Hw, Pg, NT, 11100 + l)                       # global Sigma-orbit twins (v9 construction): exact Gram, support,
        tx = V.twins_w(HJ, Pg, NT, 11200 + l)                       # whitened and raw 1-sigma norms
        rn = lambda Q: float((Sh @ Q).norm(dim=0).mean())          # noqa: E731
        L0["raw_norm"] = {"H": rn(Hw), "tw": [rn(Q) for Q in tw], "HJ": rn(HJ), "tx": [rn(Q) for Q in tx]}
        me_tw = [float((Q.T @ mt).pow(2).sum()) for Q in tw]
        L0["mean_energy"] = {"H": eH, "tw_max": max(me_tw)}
        L0["mean_norm_aligned"] = bool(eH > max(me_tw) or rn(Hw) > max(L0["raw_norm"]["tw"]) + 1e-9)
        Wa = Shi @ Sig["all"] @ Shi
        L0["twin_allpos_moment_maxdev"] = float(max((Q.T @ Wa @ Q - Hw.T @ Wa @ Hw).abs().max() for Q in tw))
        L0["H_in_J25"] = float((QJ.T @ Hw).pow(2).sum() / 5)
        L0["HJ_retained_per_col"] = [float(x) for x in Hres.pow(2).sum(0)]
        L0["H_vs_EX_overlap"] = float((V.orth(EXw).T @ Hw).pow(2).sum() / 5)
        C = {"H": Hw, "HJ": HJ, "EX": EXw, "J5": J5w}
        for j in range(NT):
            C[f"tw{j}"], C[f"tx{j}"] = tw[j], tx[j]
        ctrl[l] = {k: (Sh @ v).float().cuda() for k, v in C.items()}               # raw 1-sigma moves
        ctrl[l]["_Sh"] = Sh.float().cuda(); ctrl[l]["_Shi"] = Shi.float().cuda()
        S0[str(l)] = L0
        log(f"S0 L{l}: mean-energy {eH:.4f} (refs max {eR.max():.4f}, tw max {max(me_tw):.4f}) mean/norm-aligned {L0['mean_norm_aligned']} | H in J25 {L0['H_in_J25']:.3f} | "
            f"HJ retained {np.mean(L0['HJ_retained_per_col']):.3f} | H vs EX {L0['H_vs_EX_overlap']:.3f} | raw norm H {rn(Hw):.3f} vs tw max "
            f"{max(L0['raw_norm']['tw']):.3f}")
        save()
    live = [l for l in LAYERS if not kill[l]]
    log(f"S0 done ({(time.time()-st0)/60:.1f} min); live layers {live}")

    # ================= L (H-blind localization, split A)
    def prep(item):
        ids4 = {}
        for k in CORNERS:
            ids = tok(item["prompts"][k], add_special_tokens=False)["input_ids"]
            if len(ids) > T_T or len(ids) < 2:
                return None
            ids4[k] = ids
        Ls = 0
        while Ls < min(len(v) for v in ids4.values()) and all(v[-1 - Ls] == ids4["00"][-1 - Ls] for v in ids4.values()):
            Ls += 1
        if Ls < 1:
            return None
        X_ = tok(item["X"], add_special_tokens=False)["input_ids"][0]
        Y_ = tok(item["Y"], add_special_tokens=False)["input_ids"][0]
        if X_ == Y_:
            return None
        P = {k: {"ids": v + [v[-1]] * (T_T - len(v)), "ans": len(v) - 1, "n": len(v)} for k, v in ids4.items()}
        return {"P": P, "Ls": Ls, "X": X_, "Y": Y_, "cluster": item["cluster"], "item": item["item"]}

    def capture(pr):
        fb, Xl = {}, {l: {} for l in live}
        for k, Pk in pr["P"].items():
            ids = torch.tensor([Pk["ids"]], device="cuda")
            with torch.no_grad():
                h = m.embed(ids); prev = 0
                for l in sorted(live):
                    h = m.run(h, prev, l + 1); prev = l + 1
                    Xl[l][k] = h[0].clone()
                hL = m.run(h, prev)
            fb[k] = f_scorer(pr["X"], pr["Y"])(hL[0, Pk["ans"]:Pk["ans"] + 1], None)
        return fb, Xl

    def pos_of(pr, o):
        return {k: pr["P"][k]["n"] - o for k in CORNERS}

    banks_items = {}
    for bname in BANKS + BACKUPS + [NULLB]:
        items = [json.loads(x) for x in open(H.HERE / "v10_banks" / f"{bname}.jsonl", encoding="utf-8") if x.strip()]
        random.Random(43).shuffle(items)
        prs = [p for p in (prep(it) for it in items) if p is not None]
        banks_items[bname] = {"A": prs[:N_LOC], "B": prs[N_LOC:N_LOC + N_TEST]} if bname != NULLB else {"A": [], "B": prs[:N_NULL]}
    Lst, sel_site, gated, null_off = {}, {}, [], {}
    res["stages"]["L"] = Lst
    if live:
        try:
            R.use_T(T_T)
            stl = time.time()

            def localize(bname):
                recs = []
                for ii, pr in enumerate(banks_items[bname]["A"]):
                    if time.time() > stl + 60 * BUDGET["L"]:
                        log(f"L {bname}: budget reached after {ii} items"); break
                    fb, Xl = capture(pr)
                    g_ = {"cluster": pr["cluster"], "item": pr["item"], "DiD": fb["11"] - fb["10"] - fb["01"] + fb["00"],
                          "mA": fb["10"] - fb["00"], "mB": fb["01"] - fb["00"], "lay": {}}
                    for l in live:
                        X0 = Xl[l]["00"]; seqs, offs = [], []
                        for o in range(1, min(pr["Ls"], MAX_OFF) + 1):
                            ps = pos_of(pr, o); p0 = ps["00"]
                            dl_ = {c: Xl[l][c][ps[c]] - X0[p0] for c in ("10", "01", "11")}
                            for mv in (dl_["10"], dl_["01"], dl_["11"], dl_["10"] + dl_["01"]):
                                xx = X0.clone(); xx[p0] += mv; seqs.append(xx)
                            offs.append(o)
                        fs, _, _ = V.eval_batch(R, seqs, l, [[pr["P"]["00"]["ans"]]] * len(seqs), f_scorer(pr["X"], pr["Y"]))
                        g_["lay"][str(l)] = {str(o): {"dA": fs[4 * n] - fb["00"], "dB": fs[4 * n + 1] - fb["00"],
                                                      "DiD_site": fs[4 * n + 2] - fs[4 * n] - fs[4 * n + 1] + fb["00"],
                                                      "I_site": fs[4 * n + 3] - fs[4 * n] - fs[4 * n + 1] + fb["00"]}
                                             for n, o in enumerate(offs)}
                    recs.append(g_)
                    if (ii + 1) % 4 == 0:
                        log(f"L {bname}: item {ii+1}/{len(banks_items[bname]['A'])} ({(time.time()-stl)/60:.1f} min in L)")
                raw.setdefault("L", {})[bname] = recs
                if not recs:
                    return None
                cl = [r["cluster"] for r in recs]
                dd = np.array([r["DiD"] for r in recs])
                Gb = {"n": len(recs), "DiD_mean": float(dd.mean()), "DiD_abs_mean": float(np.abs(dd).mean()), "DiD_ci": V.boot_ci(dd, cl),
                      "mA_ci": V.boot_ci([r["mA"] for r in recs], cl), "mB_ci": V.boot_ci([r["mB"] for r in recs], cl), "lay": {}}
                Gb["gated"] = bool(Gb["DiD_ci"][0] > 0 or Gb["DiD_ci"][1] < 0) if bname != NULLB else None
                needA = Gb["mA_ci"][0] > 0 or Gb["mA_ci"][1] < 0
                needB = Gb["mB_ci"][0] > 0 or Gb["mB_ci"][1] < 0
                for l in live:
                    per_o = {}
                    for o in range(1, MAX_OFF + 1):
                        have = [r for r in recs if str(o) in r["lay"][str(l)]]
                        if len(have) < 0.75 * len(recs):
                            continue
                        dh = np.mean([r["DiD"] for r in have])
                        ratio = float(np.mean([r["lay"][str(l)][str(o)]["DiD_site"] for r in have]) / dh) if abs(dh) > 1e-12 else None
                        rA = float(np.mean([r["lay"][str(l)][str(o)]["dA"] for r in have]) / (np.mean([r["mA"] for r in have]) or 1e-12))
                        rB = float(np.mean([r["lay"][str(l)][str(o)]["dB"] for r in have]) / (np.mean([r["mB"] for r in have]) or 1e-12))
                        curv = float(np.mean([r["lay"][str(l)][str(o)]["I_site"] for r in have]) / dh) if abs(dh) > 1e-12 else None
                        thr = bool(ratio is not None and ratio >= 0.3 and (rA >= 0.3 or not needA) and (rB >= 0.3 or not needB))
                        per_o[o] = {"ratio": ratio, "curv_ratio": curv, "recA": rA, "recB": rB, "n": len(have), "meets_site_thresholds": thr}
                    ok_o = [o for o in per_o if per_o[o]["meets_site_thresholds"] and per_o[o]["curv_ratio"] is not None]
                    if not ok_o:                                      # nothing meets v10's site thresholds: report the best ratio, no gate
                        cand = [o for o in per_o if per_o[o]["ratio"] is not None]
                        o_rep = max(cand, key=lambda o: (per_o[o]["ratio"], -o)) if cand else None
                        Gb["lay"][str(l)] = {"per_o": per_o, "o_star": None, "o_best_ratio": o_rep, "site_gate": False}; continue
                    o_star = max(ok_o, key=lambda o: (per_o[o]["curv_ratio"], -o))     # where the interaction (curvature) lives
                    Gb["lay"][str(l)] = {"per_o": per_o, "o_star": o_star, "site_gate": bool(Gb["gated"])}
                Lst[bname] = Gb
                log(f"L {bname}: n {len(recs)} DiD {Gb['DiD_mean']:+.2f} CI {[round(x, 2) for x in Gb['DiD_ci']]} gated {Gb['gated']} | "
                    + " | ".join(f"L{l} o*={Gb['lay'][str(l)]['o_star']} ratio "
                                 f"{(Gb['lay'][str(l)]['per_o'].get(Gb['lay'][str(l)]['o_star']) or {}).get('ratio')} curv "
                                 f"{(Gb['lay'][str(l)]['per_o'].get(Gb['lay'][str(l)]['o_star']) or {}).get('curv_ratio')} "
                                 f"site-gate {Gb['lay'][str(l)]['site_gate']}" for l in live))
                save()
                return Gb
            for bname in BANKS:
                localize(bname)
            gated = [b for b in BANKS if Lst.get(b, {}).get("gated")]
            for b in BACKUPS:
                if len(gated) < 3 and localize(b) and Lst[b]["gated"]:
                    gated.append(b)
            for b in gated:
                for l in live:
                    Ly = Lst[b]["lay"][str(l)]
                    if Ly["site_gate"]:
                        sel_site[(b, l)] = Ly["o_star"]
            ls_null = min((p_["Ls"] for p_ in banks_items[NULLB]["B"]), default=0)
            for l in live:          # TOST null: most common o* among site-gated primary cells at l, clipped to the null suffix
                os_ = [min(sel_site[(b, l)], ls_null) for b in BANKS if (b, l) in sel_site]
                if os_ and ls_null >= 1:
                    null_off[l] = min(set(os_), key=lambda o: (-os_.count(o), o))
        except Exception:  # noqa: BLE001
            res["errors"]["L"] = traceback.format_exc(); log("L ERROR\n" + res["errors"]["L"])
    res["verdict"]["gated_contexts"] = gated
    res["verdict"]["site_gated_cells"] = [f"{b}|L{l}|o*={o}" for (b, l), o in sel_site.items()]
    save()

    # ================= T (used tests on split B at o*)
    ST, recT = {}, {}
    res["stages"]["T"] = ST
    run_banks = sorted({b for (b, l) in sel_site}, key=lambda b: (BANKS + BACKUPS).index(b)) + ([NULLB] if null_off else [])
    if run_banks:
        try:
            R.use_T(T_T)
            stt = time.time()
            for bname in run_banks:
                isnull = bname == NULLB
                cells = sorted(null_off) if isnull else [l for l in live if (bname, l) in sel_site]
                comps = ["H"] + TWH if isnull else SUBS
                recT[bname] = {l: [] for l in cells}
                for ii, pr in enumerate(banks_items[bname]["B"]):
                    if time.time() > stt + 60 * BUDGET["T"] or time.time() > DEADLINE - 60 * (BUDGET["W"] + BUDGET["R"]):
                        log(f"T {bname}: budget reached after {ii} items"); break
                    fb, Xl = capture(pr)
                    Xt, Yt = pr["X"], pr["Y"]
                    for l in cells:
                        o = null_off[l] if isnull else sel_site[(bname, l)]
                        if o > pr["Ls"]:
                            continue
                        ps = pos_of(pr, o)
                        Cl = ctrl[l]; Sh_, Shi_ = Cl["_Sh"], Cl["_Shi"]
                        X0 = Xl[l]["00"]; s0 = ps["00"]; ans0 = pr["P"]["00"]["ans"]
                        dA = Xl[l]["10"][ps["10"]] - X0[s0]
                        dB = Xl[l]["01"][ps["01"]] - X0[s0]
                        dAB = Xl[l]["11"][ps["11"]] - X0[s0]
                        Qw = {k: V.orth(Shi_ @ Cl[k]) for k in comps}

                        def proj(dv, Q):
                            return Sh_ @ (Q @ (Q.T @ (Shi_ @ dv)))
                        moves = [torch.zeros_like(dA), dA, dB, dA + dB, dAB]
                        for k in comps:
                            pa, pb = proj(dA, Qw[k]), proj(dB, Qw[k])
                            moves += [dA - pa, dB - pb, dA - pa + dB - pb]
                        seqs = []
                        for mv in moves:
                            xx = X0.clone(); xx[s0] += mv; seqs.append(xx)
                        fs, _, _ = V.eval_batch(R, seqs, l, [[ans0]] * len(seqs), f_scorer(Xt, Yt))
                        f00, fA, fB, fAB_, fAB = fs[:5]
                        rec = {"cluster": pr["cluster"], "item": pr["item"], "f_base": fb, "o": o,
                               "route": {"I": fAB_ - fA - fB + f00, "DiD_site": fAB - fA - fB + f00, "S": {}}, "ablate": {}}
                        wa_, wb_ = Shi_ @ dA, Shi_ @ dB
                        for n_, k in enumerate(comps):
                            b = 5 + 3 * n_; Q = Qw[k]
                            rec["route"]["S"][k] = {"I_del": fs[b + 2] - fs[b] - fs[b + 1] + f00,
                                                    "eA": float((Q.T @ wa_).pow(2).sum() / wa_.pow(2).sum()),
                                                    "eB": float((Q.T @ wb_).pow(2).sum() / wb_.pow(2).sum())}
                        if isnull:
                            recT[bname][l].append(rec); continue
                        seqs, meta = [], []
                        for k in SUBS_A:
                            Q = Qw[k]
                            coords = {c: Q.T @ (Shi_ @ Xl[l][c][ps[c]]) for c in CORNERS}
                            mean_c = sum(coords.values()) / 4
                            dose = float(sum((coords[c] - mean_c).pow(2).sum() for c in coords) / 4)
                            for c in CORNERS:
                                xx = Xl[l][c].clone(); xx[ps[c]] += Sh_ @ (Q @ (mean_c - coords[c]))
                                seqs.append(xx); meta.append((k, c, dose))
                        fs2, _, _ = V.eval_batch(R, seqs, l, [[pr["P"][c]["ans"]] for (_, c, _) in meta], f_scorer(Xt, Yt))
                        for n_, k in enumerate(SUBS_A):
                            rec["ablate"][k] = {"f": dict(zip(CORNERS, fs2[4 * n_:4 * n_ + 4])), "dose": meta[4 * n_][2]}
                        recT[bname][l].append(rec)
                    if (ii + 1) % 4 == 0:
                        raw["T"] = recT; save()
                        log(f"T {bname}: item {ii+1}/{len(banks_items[bname]['B'])} ({(time.time()-stt)/60:.1f} min in T)")
            raw["T"] = recT
            save()
            # ---- statistics (v10 formulas at the localized site)
            den_tost = float(np.mean([Lst[b]["DiD_abs_mean"] for b in gated])) if gated else None
            for bname in run_banks:
                ST[bname] = {}
                for l in recT[bname]:
                    rr = recT[bname][l]
                    if bname == NULLB:
                        Lt = {"n": len(rr), "o": null_off[l]}
                        if len(rr) >= MIN_T and den_tost:
                            cl = [r["cluster"] for r in rr]
                            dif = np.array([(r["route"]["I"] - r["route"]["S"]["H"]["I_del"])
                                            - np.mean([r["route"]["I"] - r["route"]["S"][t]["I_del"] for t in TWH]) for r in rr]) / den_tost
                            Lt["tost_diff"] = float(dif.mean()); Lt["tost_ci90"] = V.cboot(lambda ix: dif[ix].mean(), cl, q=(0.05, 0.95))
                            Lt["tost_equivalent"] = bool(Lt["tost_ci90"][0] > -0.10 and Lt["tost_ci90"][1] < 0.10)
                        ST[bname][str(l)] = Lt
                        log(f"T {bname} L{l} o={Lt['o']} (n={len(rr)}): TOST diff {Lt.get('tost_diff')} CI90 {Lt.get('tost_ci90')} "
                            f"-> equivalent {Lt.get('tost_equivalent')}")
                        continue
                    if len(rr) < MIN_T:
                        ST[bname][str(l)] = {"n": len(rr), "M4": "inconclusive", "M5": "inconclusive", "M4_HJ": "inconclusive",
                                             "M5_HJ": "inconclusive"}; continue
                    cl = [r["cluster"] for r in rr]
                    Lt = {"n": len(rr), "o": rr[0]["o"]}
                    I_ = np.array([r["route"]["I"] for r in rr]); Ds = np.array([r["route"]["DiD_site"] for r in rr])
                    Lt["DiD_site_ci"] = V.boot_ci(Ds, cl)
                    Lt["curv_share"] = float(I_.mean() / Ds.mean()) if abs(Ds.mean()) > 1e-12 else None
                    Lt["curv_share_ci"] = V.cboot(V.ratio_of_means(I_, Ds), cl)
                    Dall = {}
                    for k in SUBS:
                        rem = np.array([r["route"]["I"] - r["route"]["S"][k]["I_del"] for r in rr])
                        e = float(np.mean([(r["route"]["S"][k]["eA"] + r["route"]["S"][k]["eB"]) / 2 for r in rr]))
                        rf = float(rem.mean() / I_.mean()) if abs(I_.mean()) > 1e-12 else float("nan")
                        Dall[k] = {"removed_frac": rf, "e": e, "D": float(rf / e) if e > 0 else float("nan")}
                    Lt["route"] = Dall
                    evaluable = ((bname, l) in sel_site and (Lt["DiD_site_ci"][0] > 0 or Lt["DiD_site_ci"][1] < 0)
                                 and Lt["curv_share"] is not None and Lt["curv_share"] >= 0.25 and Lt["curv_share_ci"][0] > 0)
                    Lt["evaluable"] = bool(evaluable)

                    def m4(ex, twins, Lt=Lt, Dall=Dall, rr=rr, cl=cl, I_=I_, evaluable=evaluable):
                        Dtw = np.array([Dall[t]["D"] for t in twins])
                        med, mad = float(np.nanmedian(Dtw)), float(np.nanmedian(np.abs(Dtw - np.nanmedian(Dtw))))
                        z = float((Dall[ex]["D"] - med) / (1.4826 * mad)) if mad > 0 else (float("inf") if Dall[ex]["D"] > med else 0.0)
                        rk = int(1 + sum(Dall[k]["D"] > Dall[ex]["D"] for k in ["J5"] + twins))
                        rci = V.cboot(V.ratio_of_means([r["route"]["I"] - r["route"]["S"][ex]["I_del"] for r in rr], I_), cl)
                        out = {"D_robust_z": z, "rank_D": rk, "removed_frac_ci": rci}
                        if not evaluable:
                            out["status"] = "inconclusive"
                        else:
                            ok = Dall[ex]["removed_frac"] >= 0.10 and rci[0] > 0 and z >= 3 and rk == 1
                            out["status"] = "pass" if ok else "fail"
                        return out

                    def effs(f):
                        did = f["11"] - f["10"] - f["01"] + f["00"]
                        return did, 0.5 * ((f["10"] - f["00"]) + (f["11"] - f["01"])), 0.5 * ((f["01"] - f["00"]) + (f["11"] - f["10"]))
                    b_ = np.array([effs(r["f_base"]) for r in rr])
                    sd, sa, sb = (float(np.sign(b_[:, k].mean())) or 1.0 for k in range(3))

                    def losses(k, ix=None, rr=rr, b_=b_, sd=sd, sa=sa, sb=sb):
                        ix = np.arange(len(rr)) if ix is None else ix
                        a_ = np.array([effs(rr[i]["ablate"][k]["f"]) for i in ix]); bb = b_[ix]
                        il = 1 - (a_[:, 0] * sd).mean() / ((bb[:, 0] * sd).mean() or 1e-12)
                        ml = 1 - ((a_[:, 1] * sa).mean() + (a_[:, 2] * sb).mean()) / (((bb[:, 1] * sa).mean() + (bb[:, 2] * sb).mean()) or 1e-12)
                        return float(il), float(ml)
                    M5 = {}
                    for k in SUBS_A:
                        il, ml = losses(k)
                        M5[k] = {"int_loss": il, "main_loss": ml, "delta": il - ml, "dose": float(np.mean([r["ablate"][k]["dose"] for r in rr]))}
                    Lt["M5_detail"] = M5

                    def m5(ex, twins, M5=M5, cl=cl, losses=losses, site_gated=(bname, l) in sel_site):
                        ci = V.cboot(lambda ix: losses(ex, ix)[0] - np.mean([losses(t, ix)[0] for t in twins]), cl, B=400)
                        if not site_gated:
                            return {"status": "inconclusive", "int_loss_diff_ci": ci}
                        ok = (M5[ex]["int_loss"] > 0 and M5[ex]["delta"] > max(M5[t]["delta"] for t in twins)
                              and M5[ex]["delta"] > M5["J5"]["delta"] and ci[0] > 0)
                        return {"status": "pass" if ok else "fail", "int_loss_diff_ci": ci}
                    Lt["M4_H"], Lt["M4_HJ"] = m4("H", TWH), m4("HJ", TWX)
                    Lt["M5_H"], Lt["M5_HJ"] = m5("H", TWP), m5("HJ", TXP)
                    Lt["M4"], Lt["M5"] = Lt["M4_H"]["status"], Lt["M5_H"]["status"]
                    Lt["M4_HJ_status"], Lt["M5_HJ_status"] = Lt["M4_HJ"]["status"], Lt["M5_HJ"]["status"]
                    ST[bname][str(l)] = Lt
                    log(f"T {bname} L{l} o*={Lt['o']} (n={len(rr)}): DiD_site CI {[round(x, 3) for x in Lt['DiD_site_ci']]} | share "
                        f"{Lt['curv_share']} | H removes {Dall['H']['removed_frac']:.2f} (e {Dall['H']['e']:.3f}) z {Lt['M4_H']['D_robust_z']:.1f} "
                        f"rankD {Lt['M4_H']['rank_D']} -> M4 {Lt['M4']} | M5 dH {M5['H']['delta']:+.3f} vs J5 {M5['J5']['delta']:+.3f} "
                        f"twins max {max(M5[t]['delta'] for t in TWP):+.3f} -> M5 {Lt['M5']} | HJ: M4 {Lt['M4_HJ_status']} M5 {Lt['M5_HJ_status']}")
        except Exception:  # noqa: BLE001
            res["errors"]["T"] = traceback.format_exc(); log("T ERROR\n" + res["errors"]["T"])
        save()

    # ================= W (real + distinct on fresh windows, any position)
    plan, recW, SW = [], {l: [] for l in live}, {}
    res["stages"]["W"] = SW
    if live:
        try:
            R.use_T(T_W)
            stw = time.time()
            dl_w = stw + 60 * BUDGET["W"]
            cand, art = V.windows_val(tok, N_W * 3, T_W, seed=11, kmin=5)
            v10c, _ = V.windows_val(tok, 120, T_W, seed=10, kmin=2)
            v10set = {tuple(w) for w in v10c}
            rng = random.Random(11)
            n_excl = 0
            for w, a in zip(cand, art):
                if tuple(w) in v10set:
                    n_excl += 1; continue
                plan.append((torch.tensor(w), a, rng.randrange(H.SKIP + 8, T_W - NEXT - 4)))
                if len(plan) == N_W:
                    break
            res["W_windows_excluded_as_v10"] = n_excl
            log(f"W: {len(plan)} windows from {len(set(p_[1] for p_ in plan))} articles (k>=5; {n_excl} excluded as v10 windows)")
            for wi, (w, a, q) in enumerate(plan):
                if time.time() > dl_w or time.time() > DEADLINE - 60 * BUDGET["R"]:
                    log(f"W time budget reached after {wi} windows"); break
                ids = w[None].cuda(); tgt = ids[0]
                Xs = {}
                with torch.no_grad():
                    h = m.embed(ids); prev = 0
                    for l in sorted(live):
                        h = m.run(h, prev, l + 1); prev = l + 1
                        Xs[l] = h.clone()
                for l in live:
                    X = Xs[l][0]; Cl = ctrl[l]

                    def sc(rows_, b, q=q, tgt=tgt):
                        lg = m.logits(m.text.norm(rows_)).double()
                        return float(torch.log_softmax(lg, -1).gather(-1, tgt[q + 1:q + 1 + NEXT, None]).sum())
                    rec = {"article": int(a), "q": int(q), "tok_class": cls(int(w[q]))}
                    rec["pairs"], _ = V.rect(R, X, l, q, {k: Cl[k] for k in SUBS}, list(range(q, q + NEXT)), sc)
                    rec["freeze"], _ = V.rect(R, X, l, q, {"H": Cl["H"], **{t: Cl[t] for t in TWP}}, list(range(q, q + NEXT)), sc, freeze=True)
                    rec["freeze_identity_err"] = abs(rec["freeze"]["_f0"] - rec["pairs"]["_f0"])
                    rec["freeze_identity_tol"] = 1e-4 * max(1.0, abs(rec["pairs"]["_f0"]))
                    recW[l].append(rec)
                if (wi + 1) % 4 == 0 or wi == len(plan) - 1:
                    raw["W"] = recW; save()
                    log(f"W window {wi+1}/{len(plan)} ({(time.time()-stw)/(wi+1):.1f}s/window, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)")
            raw["W"] = recW
            for l in live:
                rr = recW[l]; cl = [r["article"] for r in rr]
                if len(rr) < MIN_W:
                    SW[str(l)] = {"n": len(rr), "M3": "inconclusive", "M3res": "inconclusive", "note": "too few windows"}; continue
                Lw = {"n": len(rr)}
                for comp in ("I", "I_trunk"):
                    Lw[comp] = arm_stats2(rr, "H", TWH, comp, cl)
                    Lw["HJ_" + comp] = arm_stats2(rr, "HJ", TWX, comp, cl)
                    Lw["EX_" + comp + "_vs_tw"] = arm_stats2(rr, "EX", TWH, comp, cl)
                    Lw[comp + "_plain8"] = arm_stats2(rr, "H", TWP, comp, cl)
                    Lw[comp + "_freeze"] = arm_stats2(rr, "H", TWP, comp, cl, key="freeze")
                Lw["log_H_over_J5"] = log_ratio2(rr, "H", "J5", cl)
                Lw["log_HJ_over_J5"] = log_ratio2(rr, "HJ", "J5", cl)
                Lw["log_EX_over_H"] = log_ratio2(rr, "EX", "H", cl)
                Lw["specificity"] = {k: specificity(rr, k) for k in ["H", "HJ", "J5", "EX"]}
                Lw["specificity"]["tw_median"] = float(np.median([specificity(rr, t) for t in TWH]))
                Lw["specificity"]["tx_median"] = float(np.median([specificity(rr, t) for t in TWX]))
                Lw["freeze_identity_err_max"] = float(max(r["freeze_identity_err"] for r in rr))
                Lw["freeze_valid"] = bool(max(r["freeze_identity_err"] - r["freeze_identity_tol"] for r in rr) <= 0)
                base = Lw["I_trunk_plain8"]["T1"]
                Lw["freeze_retained"] = float(Lw["I_trunk_freeze"]["T1"] / base) if base > LOG2 else None
                if not Lw["freeze_valid"]:
                    Lw["M3"] = "inconclusive"
                elif strong(Lw["I"]) and strong(Lw["I_trunk"]) and Lw["freeze_retained"] is not None and Lw["freeze_retained"] >= 0.5:
                    Lw["M3"] = "pass"
                else:
                    Lw["M3"] = "fail"
                Lw["M3res"] = "pass" if (strong(Lw["HJ_I"]) and strong(Lw["HJ_I_trunk"])) else "fail"
                pc = [r for r in rr if r["tok_class"] != "punctuation"]
                if len(pc) >= 9:
                    Lw["I_nonpunct_positions"] = arm_stats2(pc, "H", TWH, "I", [r["article"] for r in pc])
                    sn = Lw["I_nonpunct_positions"]
                    Lw["excess_at_nonpunct_positions"] = bool(sn["T1"] > LOG2 and sn["ci95"][0] > 0 and sn["rank"] == 1)
                SW[str(l)] = Lw
                log(f"W L{l} (n={len(rr)}): H T1 I {Lw['I']['T1']:+.2f} {[round(x, 2) for x in Lw['I']['ci95']]} rank {Lw['I']['rank']}/17 "
                    f"p_PI {Lw['I']['p_PI']:.3f} | I_trunk {Lw['I_trunk']['T1']:+.2f} | freeze kept {Lw['freeze_retained']} valid {Lw['freeze_valid']} "
                    f"-> M3 {Lw['M3']} | HJ T1 {Lw['HJ_I']['T1']:+.2f} {[round(x, 2) for x in Lw['HJ_I']['ci95']]} rank {Lw['HJ_I']['rank']}/17 "
                    f"-> M3res {Lw['M3res']} | log H/J5 {Lw['log_H_over_J5']['mean']:+.2f} HJ/J5 {Lw['log_HJ_over_J5']['mean']:+.2f}")
        except Exception:  # noqa: BLE001
            res["errors"]["W"] = traceback.format_exc(); log("W ERROR\n" + res["errors"]["W"])
        save()

    # ================= R (random-weights null)
    SR = {}
    res["stages"]["R"] = SR
    if live and plan and any(len(recW[l]) for l in live):
        try:
            R.use_T(T_W)
            for l in sorted(live, reverse=True):
                lo_blk = l + 1
                hi_blk = len(m.layers) if l == max(live) else (min(x for x in live if x > l) + 1)
                g = torch.Generator(device="cuda").manual_seed(31337 + l)
                nshuf = 0
                with torch.no_grad():
                    for bi in range(lo_blk, hi_blk):
                        for pn, prm in m.layers[bi].named_parameters():
                            if prm.dim() >= 2:
                                flat = prm.data.reshape(-1)
                                perm = torch.randperm(flat.numel(), device=flat.device, generator=g)
                                prm.data.copy_(flat[perm].reshape(prm.shape)); nshuf += 1
                log(f"R: shuffled {nshuf} weight tensors in blocks {lo_blk}..{hi_blk-1} (all blocks > {l} now shuffled)")
                rr = []
                for (w, a, q) in plan[:min(N_R, len(recW[l]))]:
                    if time.time() > DEADLINE:
                        break
                    ids = w[None].cuda(); tgt = ids[0]
                    X = R.to_layer(ids, l)[0]
                    Cl = ctrl[l]

                    def sc(rows_, b, q=q, tgt=tgt):
                        lg = m.logits(m.text.norm(rows_)).double()
                        return float(torch.log_softmax(lg, -1).gather(-1, tgt[q + 1:q + 1 + NEXT, None]).sum())
                    pairs, _ = V.rect(R, X, l, q, {"H": Cl["H"], **{t: Cl[t] for t in TWP}}, list(range(q, q + NEXT)), sc)
                    rr.append({"article": int(a), "pairs": pairs})
                raw.setdefault("R", {})[l] = rr
                tr_rec = recW.get(l, [])[:len(rr)]
                Lr = {"n": len(rr)}
                if len(rr) >= MIN_R and len(tr_rec) == len(rr):
                    st_r = arm_stats2(rr, "H", TWP, "I", [r["article"] for r in rr])
                    st_t = arm_stats2(tr_rec, "H", TWP, "I", [r["article"] for r in tr_rec])
                    abs_r = float(np.median([np.mean([abs(r["pairs"][f"{t}|{i}{j}"][0]) for t in TWP for (i, j) in MP]) for r in rr]))
                    abs_t = float(np.median([np.mean([abs(r["pairs"][f"{t}|{i}{j}"][0]) for t in TWP for (i, j) in MP]) for r in tr_rec]))
                    Lr.update({"I_random": st_r, "T1_trained_same_windows": st_t["T1"], "median_twin_absI_random": abs_r,
                               "median_twin_absI_trained": abs_t})
                    if not (math.isfinite(st_r["T1"]) and math.isfinite(st_t["T1"])):
                        Lr["R"] = "fail"
                    elif abs_t > 0 and abs_r / abs_t < 0.01:
                        Lr["R"] = "inconclusive"
                    else:
                        Lr["R"] = "fail" if st_r["T1"] >= 0.5 * st_t["T1"] else "pass"
                    log(f"R L{l} (n={len(rr)}): random-weights T1 {st_r['T1']:+.2f} vs trained {st_t['T1']:+.2f} on the same windows "
                        f"(median twin |I| {abs_r:.2e} vs {abs_t:.2e}) -> {Lr['R']}")
                else:
                    Lr["R"] = "inconclusive"
                SR[str(l)] = Lr
                save()
        except Exception:  # noqa: BLE001
            res["errors"]["R"] = traceback.format_exc(); log("R ERROR\n" + res["errors"]["R"])
        save()

    # ================= verdict (v10's aggregation: M3/R/M3res/TOST over both layers; M4/M5 >= 2 contexts at >= 1 layer)
    Vd = res["verdict"]
    checks = ["M3", "R", "M3res", "M4", "M5", "TOST"]
    Vd["M3_by_layer"] = {str(l): SW.get(str(l), {}).get("M3", "inconclusive") for l in LAYERS}
    Vd["R_by_layer"] = {str(l): SR.get(str(l), {}).get("R", "inconclusive") for l in LAYERS}
    Vd["M3res_by_layer"] = {str(l): SW.get(str(l), {}).get("M3res", "inconclusive") for l in LAYERS}
    for k in ("M3", "R", "M3res"):
        Vd[k] = V.status_agg(Vd[k + "_by_layer"].values())
    for key, nm in (("M4", "M4"), ("M5", "M5"), ("M4_HJ", "M4_HJ_status"), ("M5_HJ", "M5_HJ_status")):
        per = {b: [ST.get(b, {}).get(str(l), {}).get(nm, "inconclusive") for l in LAYERS] for b in gated}
        Vd[key + "_by_context"] = per
        Vd[key] = ctx_status(per) if per else "inconclusive"
    tl = sorted({l for (b, l) in sel_site if b in BANKS})          # TOST applies at every layer where a primary cell is site-gated
    tost = {str(l): ST.get(NULLB, {}).get(str(l), {}).get("tost_equivalent") for l in tl}
    Vd["TOST_by_layer"] = tost
    Vd["TOST"] = V.status_agg(["inconclusive" if v is None else ("pass" if v else "fail") for v in tost.values()]) if tost else "inconclusive"
    Vd["MUST_pass"] = all(Vd[c] == "pass" for c in checks)
    q = [f"localized sites: {', '.join(Vd.get('site_gated_cells', [])) or 'none'}", "subspace (k = 5 fixed)",
         "v11a object: position-general at L40 only, punctuation-carried at L40 (v11a frozen verdict)",
         "distinct = geometric (H in J25 overlap reported in S0) + functional (M4/M5 J5 contrasts)"]
    for l in LAYERS:
        if S0.get(str(l), {}).get("mean_norm_aligned"):
            q.append(f"mean/norm-aligned (L{l})")
        if str(l) in SW and "excess_at_nonpunct_positions" in SW[str(l)]:
            q.append(f"{'excess at non-punctuation positions' if SW[str(l)]['excess_at_nonpunct_positions'] else 'punctuation-carried in W'} (L{l})")
    Vd["qualifiers"] = q
    fails = [c for c in checks if Vd[c] == "fail"]; inc = [c for c in checks if Vd[c] == "inconclusive"]
    Vd["FINAL"] = ("I found H-space (v11 MUST tier passed); qualifiers: " + "; ".join(q)) if Vd["MUST_pass"] else         (f"MUST tier not passed. fail: {fails or 'none'}; inconclusive: {inc or 'none'}; qualifiers: " + "; ".join(q))
    save()
    log(f"VERDICT: {Vd['FINAL']}")
    log(f"verdict detail: {json.dumps({c: Vd[c] for c in checks + ['M4_HJ', 'M5_HJ']})}")
    log("DONE")


if __name__ == "__main__":
    main()
