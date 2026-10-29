"""AND battery runner: paper/PREREG-and-battery-2026-10-28.md + AMENDMENTS 1-2. Items: hs_and_items.py.
Pass 1: validate items, run the 4 unpatched cells (competence, true interaction), collect the real A / B changes at each layer.
Pass 2: patch the (0,0) prompt at layer l with real, Sigma-random and (within only) cross-class mismatched pairs.
Primary score: contrast logit c = z_ans11 - mean_k z_other_k (softmax-free). I_c = c(+A+B) - c(+A) - c(+B) + c(0).
    python hs_and_battery.py --model Qwen/Qwen3.5-0.8B --layers 6 12 18 --tag 08b            (fp32)
    python hs_and_battery.py --model Qwen/Qwen3.6-27B --layers 16 28 40 52 --bf16 --tag 27b  (pod)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import copy

import hs_and_items as ITEMS
import hs_and_items_t2 as ITEMS2
from hs_phi_cpu import SKIP, windows

LAST_ONLY = {"within", "lensB", "lensAB", "xferB", "xferAB"}
LENSLIKE = {"lensB", "lensAB", "xferB", "xferAB"}
FAST = {"on": True}               # amendment 5 fast path (cached prefix)


def items_of(cname):
    return ITEMS2.CLASSES_T2[cname[:-3]]() if cname.endswith("@t2") else ITEMS.CLASSES[cname]()

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "and_battery"
PROG = HERE / "out" / "progress"
ROWS = [(0, 0), (1, 0), (0, 1), (1, 1), (1, -1), (-1, 1), (-1, -1)]      # first 4: one-sided interaction; 4,5,6 + row 3: symmetric
LENS_AB_OK = {"negation_polarity", "quantifier_truth", "comparison_order", "number_comparison", "truth_frame", "xor_bits", "binding"}
VARS = ("within", "cross", "lensB", "lensAB")
SMALL = [(e * sa, e * sb) for e in (0.25, 0.5) for (sa, sb) in ((1, 1), (1, -1), (-1, 1), (-1, -1))]   # amendment 4: rows 7..14
XOR = {"negation_polarity", "quantifier_truth", "comparison_order", "number_comparison", "truth_frame", "binding", "xor_bits"}


def progress(done, total, note, t0, tag, status="running"):
    PROG.mkdir(parents=True, exist_ok=True)
    tmp = PROG / f"and_{tag}.json.tmp"
    json.dump({"job": f"and_{tag}", "title": f"AND battery ({tag})", "status": status, "stage": note.split(" |")[0], "stage_index": 1,
               "stage_count": 1, "done": done, "total": total, "unit": "item-layers", "started": t0, "updated": time.time(),
               "note": note}, open(tmp, "w"))
    for _ in range(5):
        try:
            tmp.replace(PROG / f"and_{tag}.json"); return
        except PermissionError:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layers", type=int, nargs="+", default=[6, 12, 18])
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--tag", default="08b")
    ap.add_argument("--classes", default="")
    ap.add_argument("--nrand", type=int, default=3)
    ap.add_argument("--transfer", action="store_true", help="also run the second phrasings (amendment 5)")
    ap.add_argument("--resume", action="store_true", help="keep classes already in the output JSON (pass 2 skipped for them)")
    a = ap.parse_args()
    t0 = time.time(); dev = "cuda"
    torch.backends.cudnn.enabled = False
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    dt = torch.bfloat16 if a.bf16 else torch.float32
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, device_map=dev, attn_implementation="eager", low_cpu_mem_usage=True)
    m = m.to(dt)                                   # from_pretrained(dtype=...) can leave bf16 modules: force it (found in the smoke run)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    layers = m.model.layers
    d = m.config.get_text_config().hidden_size
    st = {"delta": None}

    def make_hook(li):
        def hook(mod, inp, out):
            dl = st["delta"]
            if dl is None or dl[0] != li:
                return out
            h = out[0] if isinstance(out, tuple) else out
            h = h + dl[1].to(h.dtype)
            return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
        return hook
    for li in a.layers:
        layers[li].register_forward_hook(make_hook(li))
    cov = windows(tok, "train", 24, 64, seed=5)
    Sh = {}
    with torch.no_grad():
        hsl = {li: [] for li in a.layers}
        for w in cov:
            o = m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False)
            for li in a.layers:
                hsl[li].append(o.hidden_states[li + 1][0, SKIP:63].double())
    for li in a.layers:
        S = torch.cov(torch.cat(hsl[li]).T)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device=dev, dtype=S.dtype)
        ev, V = torch.linalg.eigh(S)
        Sh[li] = ((V * ev.clamp_min(1e-12).sqrt()) @ V.T).float()
    del hsl
    gen = torch.Generator(device=dev).manual_seed(11)
    pyr = random.Random(12)
    classes = a.classes.split(",") if a.classes else list(ITEMS.CLASSES)
    if a.transfer:
        classes = classes + [c + "@t2" for c in classes if not c.endswith("@t2")]
    # ---------------- pass 1
    P1 = {}
    for cname in classes:
        recs, n_bad, comp_hits, comp_n = [], 0, 0, 0
        for it in items_of(cname):
            ids = [[tok(it["p"][i][j], add_special_tokens=False)["input_ids"] for j in range(2)] for i in range(2)]
            L_ = {len(ids[i][j]) for i in range(2) for j in range(2)}
            atok = [[tok(it["a"][i][j], add_special_tokens=False)["input_ids"][0] for j in range(2)] for i in range(2)]
            dist = sorted({atok[i][j] for i in range(2) for j in range(2)})
            ok = len(L_) == 1 and len(dist) >= 2
            if ok:
                i00, i10, i01, i11 = (torch.tensor(x) for x in (ids[0][0], ids[1][0], ids[0][1], ids[1][1]))
                pA = torch.nonzero(i10 != i00).flatten(); pB = torch.nonzero(i01 != i00).flatten()
                exp11 = i00.clone(); exp11[pA] = i10[pA]; exp11[pB] = i01[pB]
                ok = len(pA) > 0 and len(pB) > 0 and not (set(pA.tolist()) & set(pB.tolist())) and bool((exp11 == i11).all())
            if not ok:
                n_bad += 1; continue
            T = len(ids[0][0]); last = T - 1
            batch = torch.stack([i00, i10, i01, i11]).to(dev)
            with torch.no_grad():
                o = m.model(input_ids=batch, output_hidden_states=True, use_cache=False)
                z = m.lm_head(o.last_hidden_state[:, last]).float()
            lp = torch.log_softmax(z, -1)
            dist_t = torch.tensor(dist, device=dev)
            cell = lp[:, dist_t].double().cpu()
            for k, (i, j) in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                comp_n += 1; comp_hits += int(dist[int(cell[k].argmax())] == atok[i][j])
            a11 = dist.index(atok[1][1]); oth = [k for k in range(len(dist)) if k != a11]
            cc = cell[:, a11] - cell[:, oth].mean(1)
            vec = {}
            for li in a.layers:
                H = o.hidden_states[li + 1]
                vec[li] = {"A_last": (H[1, last] - H[0, last]).float().cpu(), "B_last": (H[2, last] - H[0, last]).float().cpu(),
                           "A_pos": (H[1, pA] - H[0, pA]).float().cpu(), "B_pos": (H[2, pB] - H[0, pB]).float().cpu()}
            recs.append({"ids": i00, "T": T, "pA": pA, "pB": pB, "dist": dist, "a11": a11, "vec": vec,
                         "true_I_c": float(cc[3] - cc[1] - cc[2] + cc[0]), "true_I_logp": float(cell[3, a11] - cell[1, a11] - cell[2, a11] + cell[0, a11])})
        P1[cname] = {"recs": recs, "n_bad": n_bad, "competence": comp_hits / max(comp_n, 1)}
        progress(0, 1, f"pass1 {cname} | {(time.time()-t0)/60:.0f} min", t0, a.tag)
    pool = [(c, k) for c in P1 for k in range(len(P1[c]["recs"]))]
    total = sum(len(v["recs"]) for v in P1.values()) * len(a.layers); done = 0
    out = {"model": a.model, "layers": a.layers, "prereg": "PREREG-and-battery-2026-10-28.md + AMENDMENTS 1-5", "classes": {}}
    RES.mkdir(parents=True, exist_ok=True)
    if a.resume and (RES / f"and_battery_{a.tag}.json").exists():
        out["classes"] = json.load(open(RES / f"and_battery_{a.tag}.json"))["classes"]
        print(f"resume: keeping {list(out['classes'])}", flush=True)
    # ---------------- pass 2
    exact_log = {}
    for cname in classes:
        if cname in out["classes"]:
            done += len(P1[cname]["recs"]) * len(a.layers); continue
        R1 = P1[cname]; recs_out = []
        nC = len(R1["recs"])
        DEC = {}                                                             # (layer, lens var) -> summed vocab interactions
        sumA = {li: sum(x["vec"][li]["A_last"] for x in R1["recs"]) for li in a.layers} if nC > 1 else {}
        sumB = {li: sum(x["vec"][li]["B_last"] for x in R1["recs"]) for li in a.layers} if nC > 1 else {}
        base = cname[:-3] if cname.endswith("@t2") else None
        R0 = P1.get(base) if base else None                                  # first-phrasing class (transfer source)
        cvars = (("within", "xferB", "xferAB") if R0 and len(R0["recs"]) >= 3 else ("within",)) if base else VARS   # amendment 6
        if R0 and len(R0["recs"]) >= 3:
            n0 = len(R0["recs"])
            xA = {li: sum(x["vec"][li]["A_last"] for x in R0["recs"]) / n0 for li in a.layers}
            xB = {li: sum(x["vec"][li]["B_last"] for x in R0["recs"]) / n0 for li in a.layers}
        for ri, r in enumerate(R1["recs"]):
            pkv0 = None
            T, last, dist, a11 = r["T"], r["T"] - 1, r["dist"], r["a11"]
            oth = [k for k in range(len(dist)) if k != a11]
            dist_t = torch.tensor(dist, device=dev)
            others = [(c, k) for (c, k) in pool if c.split("@")[0] != cname.split("@")[0]]
            rec = {"true_I_c": r["true_I_c"], "true_I_logp": r["true_I_logp"], "layers": {}}
            for li in a.layers:
                res = {}
                for var in cvars:
                    if var.startswith("lens") and nC < 3:
                        continue
                    dA = torch.zeros(T, d, device=dev); dB = torch.zeros(T, d, device=dev)
                    if var == "within":
                        dA[last] = r["vec"][li]["A_last"].to(dev); dB[last] = r["vec"][li]["B_last"].to(dev)
                    elif var.startswith("lens"):                          # context-free leave-one-out class means (amendment 3)
                        Bbar = (sumB[li] - r["vec"][li]["B_last"]) / (nC - 1)
                        Abar = (sumA[li] - r["vec"][li]["A_last"]) / (nC - 1)
                        dA[last] = (r["vec"][li]["A_last"] if var == "lensB" else Abar).to(dev); dB[last] = Bbar.to(dev)
                    elif var.startswith("xfer"):                          # first-phrasing class means (amendment 5)
                        dA[last] = (r["vec"][li]["A_last"] if var == "xferB" else xA[li]).to(dev); dB[last] = xB[li].to(dev)
                    else:
                        dA[r["pA"]] = r["vec"][li]["A_pos"].to(dev); dB[r["pB"]] = r["vec"][li]["B_pos"].to(dev)
                    pairs = [(dA, dB)]
                    for _ in range(a.nrand):
                        rA = torch.randn(T, d, device=dev, generator=gen) @ Sh[li].T
                        rB = torch.randn(T, d, device=dev, generator=gen) @ Sh[li].T
                        rA = rA / rA.norm(dim=1, keepdim=True).clamp_min(1e-30) * dA.norm(dim=1, keepdim=True)
                        rB = rB / rB.norm(dim=1, keepdim=True).clamp_min(1e-30) * dB.norm(dim=1, keepdim=True)
                        pairs.append((rA, rB))
                    n_mis = 0
                    if var != "cross" and others:
                        for side in ("B", "A"):                              # swap in a real change from an unrelated class
                            c2, k2 = pyr.choice(others)
                            v2 = P1[c2]["recs"][k2]["vec"][li][f"{side}_last"].to(dev)
                            x = torch.zeros(T, d, device=dev); x[last] = v2
                            pairs.append((dA, x) if side == "B" else (x, dB)); n_mis += 1
                    RR = ROWS + SMALL if var in LENSLIKE else ROWS
                    deltas = torch.stack([cA * x + cB * y for (x, y) in pairs for (cA, cB) in RR])
                    ids1 = r["ids"][None].to(dev)

                    def full_path():
                        outs = []
                        for k0 in range(0, deltas.shape[0], 32):            # chunked (OOM on small GPUs)
                            st["delta"] = (li, deltas[k0:k0 + 32])
                            oo = m.model(input_ids=ids1.expand(deltas[k0:k0 + 32].shape[0], -1), use_cache=False)
                            st["delta"] = None
                            outs.append(m.lm_head(oo.last_hidden_state[:, last]).float())
                        return torch.cat(outs)

                    def fast_path():
                        nonlocal pkv0
                        if pkv0 is None:
                            pkv0 = m.model(input_ids=ids1[:, :last], use_cache=True).past_key_values
                        pk = copy.deepcopy(pkv0)
                        pk.reorder_cache(torch.zeros(deltas.shape[0], dtype=torch.long, device=dev))
                        st["delta"] = (li, deltas[:, last:last + 1])
                        oo = m.model(input_ids=ids1[:, last:last + 1].expand(deltas.shape[0], -1), past_key_values=pk, use_cache=True)
                        st["delta"] = None
                        return m.lm_head(oo.last_hidden_state[:, -1]).float()
                    with torch.no_grad():
                        if var in LAST_ONLY and FAST["on"]:
                            zf = fast_path()
                            if ri == 0 and li == a.layers[0] and var == "within":      # exactness check, once per class
                                zs = full_path()
                                cf = lambda zz: (torch.log_softmax(zz, -1)[:, dist_t][:, a11] - torch.log_softmax(zz, -1)[:, dist_t][:, oth].mean(1))
                                Iq = lambda cc: float(cc[3] - cc[1] - cc[2] + cc[0])         # the measured interaction, real pair
                                If, Is_ = Iq(cf(zf)), Iq(cf(zs))
                                err = abs(If - Is_)
                                exact_log[cname] = {"I_fast": If, "I_full": Is_, "abs_err": err}
                                tol = max(0.15, 0.25 * abs(Is_)) if a.bf16 else 1e-3      # bf16: two valid rounding orders differ ~0.06
                                if err > tol:
                                    print(f"FAST PATH MISMATCH {cname}: {err:.3g} > {tol}; switching to the full path for the rest of the run", flush=True)
                                    FAST["on"] = False; zf = zs
                        else:
                            zf = full_path()
                        lq = torch.log_softmax(zf, -1)[:, dist_t]
                    lq = lq.view(len(pairs), len(RR), len(dist)).double().cpu()
                    c = lq[:, :, a11] - lq[:, :, oth].mean(2)                 # [pairs, rows] contrast logit
                    Ic = c[:, 3] - c[:, 1] - c[:, 2] + c[:, 0]
                    Ics = c[:, 3] - c[:, 4] - c[:, 5] + c[:, 6]
                    Ilp = lq[:, 3, a11] - lq[:, 1, a11] - lq[:, 2, a11] + lq[:, 0, a11]
                    nr = 1 + a.nrand
                    small = None
                    if var in LENSLIKE:                                         # amendment 4: second order + vocab decode
                        zc = (zf - zf.mean(-1, keepdim=True)).view(len(pairs), len(RR), -1)
                        v1 = (zc[0, 3] - zc[0, 1] - zc[0, 2] + zc[0, 0]).double()
                        vs = (zc[0, 7] - zc[0, 8] - zc[0, 9] + zc[0, 10]).double()
                        vn = (zc[nr, 3] - zc[nr, 1] - zc[nr, 2] + zc[nr, 0]).double() if len(pairs) > nr else torch.zeros_like(v1)
                        tgt = dist[a11]
                        key = (li, var)
                        if key not in DEC:
                            DEC[key] = [torch.zeros_like(v1), torch.zeros_like(vs), torch.zeros_like(vn)]
                        DEC[key][0] += v1; DEC[key][1] += vs; DEC[key][2] += vn
                        S25 = c[:, 7] - c[:, 8] - c[:, 9] + c[:, 10]; S50 = c[:, 11] - c[:, 12] - c[:, 13] + c[:, 14]
                        small = {"S25": float(S25[0]), "S50": float(S50[0]), "S25_rand": [float(x) for x in S25[1:nr]],
                                 "S25_mis": [float(x) for x in S25[nr:]], "rank_onesided": int((v1 > v1[tgt]).sum()),
                                 "rank_small": int((vs > vs[tgt]).sum()), "rank_null": int((vn > vn[tgt]).sum())}
                    res[var] = {"small": small, "I_c": float(Ic[0]), "I_c_rand": [float(x) for x in Ic[1:nr]], "I_c_mis": [float(x) for x in Ic[nr:]],
                                "I_c_sym": float(Ics[0]), "I_c_sym_rand": [float(x) for x in Ics[1:nr]], "I_logp": float(Ilp[0]),
                                "dA_c": float(c[0, 1] - c[0, 0]), "dB_c": float(c[0, 2] - c[0, 0]),
                                "complete_c": float(c[0, 3] - c[0, 0])}
                rec["layers"][str(li)] = res
                done += 1
                progress(done, total, f"pass2 {cname} | {(time.time()-t0)/60:.0f} min", t0, a.tag)
            recs_out.append(rec)
        comp = R1["competence"]; n = len(recs_out)
        C = {"logic": "XOR" if cname.split("@")[0] in XOR else "AND", "n_items": n, "n_invalid": R1["n_bad"], "competence": comp,
             "competent": bool(comp >= 0.7), "true_I_c_mean": float(np.mean([x["true_I_c"] for x in recs_out])) if n else None, "cells": {}}
        for li in a.layers:
            for var in cvars:
                if not n or var in LENSLIKE and n < 3:
                    continue
                g = lambda k: np.array([x["layers"][str(li)][var][k] for x in recs_out])
                gr = lambda k: np.array(sum((x["layers"][str(li)][var][k] for x in recs_out), []))
                Ic, rnd, mis = g("I_c"), gr("I_c_rand"), gr("I_c_mis")
                sd_null = max(rnd.std(), mis.std() if len(mis) else 0.0)
                z = float(Ic.mean() / (sd_null + 1e-12))
                frac = np.abs(Ic) / (np.abs(g("dA_c")) + np.abs(g("dB_c")) + np.abs(Ic) + 1e-12)
                cell = {"pos_frac": float((Ic > 0).mean()), "z": z, "mean_I_c": float(Ic.mean()),
                        "null": "sigma-random only (weaker)" if var == "cross" else "max(sigma-random, mismatched)",
                        "z_vs_random": float(Ic.mean() / (rnd.std() + 1e-12)),
                        "z_vs_mismatched": float(Ic.mean() / (mis.std() + 1e-12)) if len(mis) else None,
                        "interactive_frac": float(frac.mean()),
                        "sym_z": float(g("I_c_sym").mean() / (gr("I_c_sym_rand").std() + 1e-12)),
                        "logp_I_mean": float(g("I_logp").mean()),
                        "completeness_c": float(np.mean(g("complete_c")))}
                cell["pass"] = bool(C["competent"] and n >= 12 and cell["pos_frac"] >= 0.8 and z >= 3)
                if var in LENSLIKE:
                    sm = [x["layers"][str(li)][var]["small"] for x in recs_out]
                    sgn = 1.0 if Ic.mean() >= 0 else -1.0
                    S25 = np.array([q["S25"] for q in sm]) * sgn; S50 = np.array([q["S50"] for q in sm]) * sgn
                    nul = max(np.std(sum((q["S25_rand"] for q in sm), [])),
                              np.std(sum((q["S25_mis"] for q in sm), [])) if sm[0]["S25_mis"] else 0.0)
                    ratio = float(S50.mean() / S25.mean()) if abs(S25.mean()) > 1e-12 else float("nan")
                    zs = float(S25.mean() / (nul + 1e-12))
                    med1 = float(np.median([q["rank_onesided"] for q in sm])); medn = float(np.median([q["rank_null"] for q in sm]))
                    D1, Ds, Dn = DEC[(li, var)]
                    cell["small"] = {"pos_frac": float((S25 > 0).mean()), "z": zs, "scaling_ratio": ratio,
                                     "scaling_ok": bool(2.5 <= ratio <= 6),
                                     "pass": bool(cell["pass"] and (S25 > 0).mean() >= 0.8 and zs >= 3 and 2.5 <= ratio <= 6)}
                    cell["decode"] = {"median_rank_ans11_onesided": med1, "median_rank_ans11_null": medn,
                                      "median_rank_ans11_small": float(np.median([q["rank_small"] for q in sm])),
                                      "top10_onesided": [tok.decode([i]) for i in torch.topk(D1, 10).indices.tolist()],
                                      "top10_small": [tok.decode([i]) for i in torch.topk(Ds * sgn, 10).indices.tolist()],
                                      "top10_null": [tok.decode([i]) for i in torch.topk(Dn, 10).indices.tolist()],
                                      "verbalisable": bool(med1 < 10 and medn >= 10)}
                    wi = C["cells"].get(f"L{li}_within")
                    cell["context_free_ratio"] = float(Ic.mean() / wi["mean_I_c"]) if wi and abs(wi["mean_I_c"]) > 1e-12 else None
                C["cells"][f"L{li}_{var}"] = cell
        C["interaction_class"] = bool(any(c_["pass"] for k_, c_ in C["cells"].items() if k_.endswith(("within", "cross"))))
        bname = cname.split("@")[0]
        tier_of = lambda cs: (3 if any(c_["small"]["pass"] and c_["decode"]["verbalisable"] for c_ in cs)
                              else 2 if any(c_["small"]["pass"] for c_ in cs) else 1 if any(c_["pass"] for c_ in cs) else 0)
        xfer_cells = [c_ for k_, c_ in C["cells"].items() if k_.endswith("xferB") or (k_.endswith("xferAB") and bname in LENS_AB_OK)]
        C["tier_transfer"] = tier_of(xfer_cells) if (xfer_cells and C["competent"] and n >= 12) else (0 if cname.endswith("@t2") else None)
        lens_cells = [c_ for k_, c_ in C["cells"].items() if k_.endswith("lensB") or (k_.endswith("lensAB") and bname in LENS_AB_OK)]
        C["lens_entry"] = bool(any(c_["pass"] for c_ in lens_cells))
        C["tier"] = (3 if any(c_["small"]["pass"] and c_["decode"]["verbalisable"] for c_ in lens_cells)
                     else 2 if any(c_["small"]["pass"] for c_ in lens_cells) else 1 if C["lens_entry"] else 0)
        if not C["competent"] or n < 12:
            C["tier"] = 0
        C["items"] = recs_out
        out["classes"][cname] = C
        best = max(C["cells"].items(), key=lambda kv: kv[1]["z"]) if C["cells"] else ("-", {"z": float("nan"), "pos_frac": float("nan"), "interactive_frac": float("nan")})
        print(f"{cname:24s} [{C['logic']}] items {n:2d} (bad {R1['n_bad']}) comp {comp:.2f} {'OK ' if C['competent'] else 'LOW'} "
              f"| true I_c {C['true_I_c_mean'] if n else float('nan'):+.2f} | best {best[0]} z {best[1]['z']:+.1f} pos {best[1]['pos_frac']:.2f} "
              f"frac {best[1]['interactive_frac']:.2f} | {'INTERACTION' if C['interaction_class'] else '-'} TIER {C['tier']} ({(time.time()-t0)/60:.1f} min)", flush=True)
        for k_, c_ in C["cells"].items():
            if "decode" in c_ and c_["pass"]:
                print(f"      {k_}: lens z {c_['z']:+.1f} | small z {c_['small']['z']:+.1f} ratio {c_['small']['scaling_ratio']:.2f} "
                      f"| ans11 median rank {c_['decode']['median_rank_ans11_onesided']:.0f} (null {c_['decode']['median_rank_ans11_null']:.0f}) "
                      f"| writes {c_['decode']['top10_onesided'][:6]}", flush=True)
        json.dump(out, open(RES / f"and_battery_{a.tag}.json", "w"), indent=1)
    comp_cls = [k for k, c in out["classes"].items() if c["competent"] and c["n_items"] >= 12]
    out["summary"] = {"classes": len(out["classes"]), "competent": comp_cls,
                      "interaction_classes": [k for k, c in out["classes"].items() if c["interaction_class"]],
                      "lens_classes": [k for k, c in out["classes"].items() if c["lens_entry"] and k in comp_cls],
                      "competent_XOR_without_interaction": [k for k in comp_cls if out["classes"][k]["logic"] == "XOR" and not out["classes"][k]["interaction_class"]]}
    tiers = {k: c["tier"] for k, c in out["classes"].items() if not k.endswith("@t2")}
    out["summary"]["tiers"] = tiers
    xt = {k[:-3]: c["tier_transfer"] for k, c in out["classes"].items() if k.endswith("@t2")}
    out["summary"]["transfer_tiers"] = xt
    if xt:
        tx = [k for k, t in xt.items() if t == 3 and k in XOR]; ta = [k for k, t in xt.items() if t == 3 and k not in XOR]
        out["summary"]["HEART_TRANSFER"] = ("SUPPORTED across phrasings" if tx and ta else "PARTIAL" if any(t >= 2 for t in xt.values())
                                            else "TIER-1 ONLY" if any(t >= 1 for t in xt.values()) else "NOT context-free across phrasings")
        print(f"HEART_TRANSFER {a.tag}: {out['summary']['HEART_TRANSFER']} | transfer tiers {xt}", flush=True)
    out["fast_path_exactness"] = exact_log
    lx = [k for k, t in tiers.items() if t == 3 and k in XOR]; la = [k for k, t in tiers.items() if t == 3 and k not in XOR]
    out["summary"]["HEART"] = ("SUPPORTED" if lx and la else "PARTIAL" if any(t >= 2 for t in tiers.values()) else
                               "TIER-1 ONLY (context-free interactions, not shown second-order)" if any(t >= 1 for t in tiers.values()) else
                               "REFUTED (ANDs contextual)" if out["summary"]["interaction_classes"] else "NO INTERACTIONS FOUND")
    json.dump(out, open(RES / f"and_battery_{a.tag}.json", "w"), indent=1)
    print(f"HEART {a.tag}: H-space in the J-space sense -> {out['summary']['HEART']} | tier-3 XOR {lx} AND {la} | tiers {tiers}", flush=True)
    print(f"VERDICT {a.tag}: {len(out['summary']['interaction_classes'])} interaction classes of {len(comp_cls)} competent "
          f"({len(out['classes'])} total): {out['summary']['interaction_classes']} | competent XOR w/o interaction at tested sites: "
          f"{out['summary']['competent_XOR_without_interaction']} ({(time.time()-t0)/60:.1f} min)", flush=True)
    progress(total, total, "done", t0, a.tag, status="done")


if __name__ == "__main__":
    main()
