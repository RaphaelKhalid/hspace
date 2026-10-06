"""v9: does the replicated punctuation-conditional H-subspace carry excess second-order (trunk) interaction?
Prereg: paper/SCOPE-hspace-v9.md (frozen and pushed before any pod). Revised after the v9 design review (wf_695de7ae).

EX (5 directions, whitened run-2 frame) = punctuation-conditional patterns from the FRESH v8 data (hs_v9_prep.py).
Controls: 8 Sigma-orbit twins of EX (identical raw variance and PCA-axis profile; overlap with EX 0.01-0.03).
Windows: wikitext-103 VALIDATION split, seed 9 (never used at 27B). Per window: punctuation position p, word position q.
All trunk forwards in exact fp32 (bf16w.MODE='fp32', TF32 off); float64 log-softmax.
For a pair (u, w) of 1-sigma moves at position pos (x_pos += Sh u):
  f(.) = sum_{k=1..16} log p(token[pos+k] | ...)        (16 next tokens)
  I = f(u+w) - f(u) - f(w) + f(0)                          (total pairwise interaction)
  I_trunk = f(u+w) - f_add,  f_add = f evaluated on h_u + h_w - h_0 (pre-norm last-layer states)
  I_soft  = f_add - f(u) - f(w) + f(0)                     (final-norm/softmax part)
Pairs: EX at p: all 10 (descriptive) and the matched pairs MP = (0,1),(2,3),(1,4); each twin at p: MP.
(The EX-at-q arm was dropped before freezing to secure >= 60 windows within the compute cap; it was descriptive only.)
Per window, matched statistics (for X in {I, I_trunk}):
  ex3 = mean_{MP} |X_EX|, tw3_j = mean_{MP} |X_twj|, T1 = mean_j log(ex3 / tw3_j)
  T3 = mean_j log((ex3 / mEX3) / (tw3_j / mTWj3)),  m* = mean_{MP} (|M_u| + |M_w|)
REAL9 at a layer iff for BOTH X = I and X = I_trunk: mean T1 > log 2 with article-bootstrap 95% CI low > 0,
  T3 CI low > 0, and EX ranks 1 of 9 by mean ex3 vs mean tw3_j.
Validity outcomes: n < 60 windows at a layer -> INCOMPLETE; projected < 60 windows after 3 -> ABORTED (throughput);
  a failing layer whose median noise floor >= 0.25 x median twin |I| -> UNINFORMATIVE (precision).
VERDICT "EX carries excess within-span trunk pairwise interaction vs Sigma-orbit twins (v9)" iff REAL9 at both layers.
I_soft is descriptive only.
    python hs_v9.py small|full [layers...] [--minutes N] [--null] [--plant G]
"""
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

_raw = sys.argv[1:]
sys.argv = sys.argv[:1]
import bf16w  # noqa: E402
import hspace as H  # noqa: E402

TAG = _raw[0] if _raw else "small"
_r = list(_raw[1:])
MINUTES, NULL, PLANT_G = 60.0, False, None
if "--minutes" in _r:
    k_ = _r.index("--minutes"); MINUTES = float(_r[k_ + 1]); del _r[k_:k_ + 2]
if "--plant" in _r:
    k_ = _r.index("--plant"); PLANT_G = float(_r[k_ + 1]); del _r[k_:k_ + 2]
if "--null" in _r:
    NULL = True; _r.remove("--null")
LAYERS = [int(x) for x in _r if x.isdigit()] or ([16, 40] if TAG == "full" else [6])
assert TAG != "full" or set(LAYERS) == {16, 40}, LAYERS
N_WIN = {"full": 96, "small": 12}[TAG]
NEXT = 16
MP = [(0, 1), (2, 3), (1, 4)]
ALLP = [(i, j) for i in range(5) for j in range(i + 1, 5)]
CHUNK = 24


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


def windows_with_articles(tok, split, n, T, seed):
    import glob
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=[f"wikitext-103-raw-v1/{split}-*"])
    fs = sorted(glob.glob(os.path.join(root, "wikitext-103-raw-v1", f"{split}-*.parquet")))
    lines = []
    for f in fs:
        lines += pq.read_table(f).column("text").to_pylist()
    arts, cur = [], []
    for ln in lines:
        if ln.startswith(" = ") and not ln.startswith(" = = ") and cur:
            arts.append("".join(cur)); cur = []
        cur.append(ln)
    if cur:
        arts.append("".join(cur))
    order = list(range(len(arts)))
    random.Random(seed).shuffle(order)
    out, art = [], []
    for k in range(8):
        for ai in order:
            a = arts[ai]
            if len(a) < 2000:
                continue
            ids = tok(a[:20000], add_special_tokens=False)["input_ids"]
            s = 100 + k * (T + 50)
            if len(ids) >= s + T:
                out.append(ids[s:s + T]); art.append(ai)
                if len(out) == n:
                    return torch.tensor(out), art
    return torch.tensor(out), art


def boot_ci(vals, arts, B=2000, seed=0):
    vals, arts = np.asarray(vals, dtype=float), np.asarray(arts)
    ua = np.unique(arts)
    rb = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        pick = rb.choice(ua, len(ua))
        idx = np.concatenate([np.where(arts == u)[0] for u in pick])
        bs.append(vals[idx].mean())
    return [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]


def main():
    t0 = time.time()
    cfg = H.CFG[TAG]
    T = cfg["T"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    T2 = torch.load(H.OUT / f"hspace2_{TAG}.pt", map_location="cpu", weights_only=False)
    if TAG == "full" and not NULL:
        SUB = torch.load(H.OUT / "v9_subspaces.pt", map_location="cpu", weights_only=False)
    else:                                                       # smoke / null: random EX + its Sigma-orbit twins
        SUB = {}
        for l in LAYERS:
            Sh_ = T2[f"Sh{l}"].double(); ev_, P_ = torch.linalg.eigh(Sh_); P_ = P_[:, torch.argsort(ev_, descending=True)]
            g_ = torch.Generator().manual_seed(l)
            EX_ = torch.linalg.qr(torch.randn(Sh_.shape[0], 5, generator=g_, dtype=torch.float64))[0]
            SUB[f"EX_{l}"] = EX_.float()
            SUB[f"tw_{l}"] = torch.stack([P_ @ ((torch.randint(0, 2, (Sh_.shape[0], 1), generator=g_).double() * 2 - 1) * (P_.T @ EX_)) for _ in range(8)]).float()
    assert all(f"Sh{l}" in T2 and f"EX_{l}" in SUB and f"tw_{l}" in SUB for l in LAYERS)
    m = H.Model(cfg["model"], T)
    bf16w.MODE = "fp32"
    tok = m.tok
    cand, art = windows_with_articles(tok, "validation", N_WIN * 4, T, seed=9)
    rng = random.Random(9)
    cache = {}

    def cls(t):
        if t not in cache:
            cache[t] = klass(tok.decode([t]))
        return cache[t]
    plan = []
    for w, a in zip(cand, art):
        lo, hi = H.SKIP + 8, T - NEXT - 4
        P_ = [i for i in range(lo, hi) if cls(int(w[i])) == "punctuation"]
        Q_ = [i for i in range(lo, hi) if cls(int(w[i])) == "word"]
        if P_ and Q_:
            plan.append((w, a, rng.choice(P_), rng.choice(Q_)))
        if len(plan) == N_WIN:
            break
    ptoks = {}
    for w, a, p, q in plan:
        s_ = tok.decode([int(w[p])]); ptoks[s_] = ptoks.get(s_, 0) + 1
    print(f"[v9] {TAG}: {len(plan)} fresh validation windows from {len(set(a for _, a, _, _ in plan))} articles; layers {LAYERS}; "
          f"null={NULL} plant={PLANT_G} ({time.time()-t0:.0f}s)", flush=True)
    dirs, plants = {}, {}
    for l in LAYERS:
        Sh = T2[f"Sh{l}"].cuda().float()
        EX, TW = SUB[f"EX_{l}"].cuda(), SUB[f"tw_{l}"].cuda()
        dirs[l] = {"EX": Sh @ EX, **{f"tw{j}": Sh @ TW[j] for j in range(TW.shape[0])}}
        if PLANT_G is not None:
            mu = T2[f"mu{l}"].cuda().float()
            Sih = torch.linalg.inv(Sh.double()).float()
            wdir = m.W_U[tok(" the", add_special_tokens=False)["input_ids"][0]].float()
            wdir = wdir / wdir.norm() * float(Sh.diagonal().mean()) * 4
            plants[l] = {"block": l + 1, "mu": mu, "Sinvh": Sih, "a": EX[:, 0], "b": EX[:, 1], "w": wdir, "G": PLANT_G}
    deadline = time.time() + 60 * MINUTES
    t_start = time.time()
    rec = {l: [] for l in LAYERS}
    res = {"tag": TAG, "null": NULL, "plant": PLANT_G, "n_plan": len(plan), "p_token_mix": dict(sorted(ptoks.items(), key=lambda x: -x[1])[:20]),
           "layers": {}}
    lay_sorted = sorted(LAYERS)

    def evaluate(Xl, l, pos_list, pos_of_seq, hrows_keep):
        """Xl [N,T,d] layer-l states (perturbed). Returns f per seq (float64) and pre-norm rows for kept seqs."""
        fs, rows_out = [], {}
        for c0 in range(0, len(Xl), CHUNK):
            hL = m.run(Xl[c0:c0 + CHUNK], l + 1, plant=plants.get(l))
            idx = list(range(c0, min(c0 + CHUNK, len(Xl))))
            R = torch.stack([hL[b - c0, pos_of_seq[b]:pos_of_seq[b] + NEXT] for b in idx])        # [n,16,d] pre-norm
            for b in idx:
                if b in hrows_keep:
                    rows_out[b] = hL[b - c0, pos_of_seq[b]:pos_of_seq[b] + NEXT].clone()
            lg = m.logits(m.text.norm(R.reshape(-1, R.shape[-1]))).double()
            ll = torch.log_softmax(lg, -1).view(len(idx), NEXT, -1)
            tg = torch.stack([tgt_ids[pos_of_seq[b] + 1:pos_of_seq[b] + 1 + NEXT] for b in idx])
            fs += ll.gather(-1, tg[..., None])[..., 0].sum(-1).tolist()
        return fs, rows_out

    def f_of_rows(Rr, pos):
        lg = m.logits(m.text.norm(Rr)).double()
        ll = torch.log_softmax(lg, -1)
        return float(ll.gather(-1, tgt_ids[pos + 1:pos + 1 + NEXT, None]).sum())

    for wi, (w, a, p, q) in enumerate(plan):
        if time.time() > deadline:
            print(f"[v9] time budget reached after {wi} windows", flush=True); break
        ids = w[None].cuda()
        tgt_ids = ids[0]
        with torch.no_grad():
            Xs = {}
            h = m.embed(ids)
            prev = 0
            for l in lay_sorted:
                h = m.run(h, prev, l + 1, plant=None)          # plant acts at block l+1 only, i.e. after layer l
                Xs[l] = h.clone(); prev = l + 1
            for l in lay_sorted:
                X = Xs[l][0]
                D = dirs[l]
                seqs, pos_of, key = [X], [p], {("base", p): 0}
                seqs.append(X); pos_of.append(q); key[("base", q)] = 1
                arms = [("EX", p, ALLP)] + [(f"tw{j}", p, MP) for j in range(len(D) - 1)]   # EX@q arm dropped (budget; descriptive only)
                for name, pos, pairs in arms:
                    V = D["EX" if name == "EXq" else name]
                    need = sorted({i for pr in pairs for i in pr})
                    for i in need:
                        xx = X.clone(); xx[pos] += V[:, i]; key[(name, "s", i)] = len(seqs); seqs.append(xx); pos_of.append(pos)
                    for (i, j) in pairs:
                        xx = X.clone(); xx[pos] += V[:, i] + V[:, j]; key[(name, "d", i, j)] = len(seqs); seqs.append(xx); pos_of.append(pos)
                keep_rows = {key[("base", p)], key[("base", q)]} | {v for k_, v in key.items() if k_[1] == "s"}
                fs, rows = evaluate(torch.stack(seqs), l, None, pos_of, keep_rows)
                out = {}
                for name, pos, pairs in arms:
                    b0 = key[("base", pos)]
                    for (i, j) in pairs:
                        ku, kw, kd = key[(name, "s", i)], key[(name, "s", j)], key[(name, "d", i, j)]
                        f0, fu, fw_, fd = fs[b0], fs[ku], fs[kw], fs[kd]
                        fadd = f_of_rows(rows[ku] + rows[kw] - rows[b0], pos)
                        out[f"{name}|{i}{j}"] = (fd - fu - fw_ + f0, fd - fadd, fadd - fu - fw_ + f0, fu - f0, fw_ - f0)
                # in-run noise floor: recompute EX(0,1) and tw0(0,1) in a separate batch of 7
                V0, V1 = D["EX"], D["tw0"]
                fl = [X]
                for V in (V0, V1):
                    for mv in (V[:, 0], V[:, 1], V[:, 0] + V[:, 1]):
                        xx = X.clone(); xx[p] += mv; fl.append(xx)
                f2, _ = evaluate(torch.stack(fl), l, None, [p] * 7, set())
                Ir_ex = f2[3] - f2[1] - f2[2] + f2[0]; Ir_tw = f2[6] - f2[4] - f2[5] + f2[0]
                floor = 0.5 * (abs(Ir_ex - out["EX|01"][0]) + abs(Ir_tw - out["tw0|01"][0]))
                rec[l].append({"article": int(a), "p": int(p), "q": int(q), "pairs": out, "floor": floor})
        el = time.time() - t_start
        if wi == 2:
            proj = 3 * (deadline - t_start) / max(el, 1e-6)
            if proj < (60 if TAG == "full" else 4):
                res["VERDICT_v9"] = f"ABORTED (throughput: projected {proj:.0f} windows)"
                print(f"[v9] {res['VERDICT_v9']}", flush=True)
                break
        if (wi + 1) % 8 == 0 or wi == len(plan) - 1:
            tmp = H.OUT / f"hs_v9_{TAG}_raw.pt.tmp"
            torch.save({"rec": rec, "plan": [(int(a_), int(p_), int(q_)) for (_, a_, p_, q_) in plan]}, tmp)
            os.replace(tmp, H.OUT / f"hs_v9_{TAG}_raw.pt")
            print(f"[v9] window {wi+1}/{len(plan)} ({time.time()-t0:.0f}s, {el/(wi+1):.1f}s/window, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)", flush=True)
    # ---------------- statistics
    min_n = 60 if TAG == "full" else 4
    for l in LAYERS:
        R = rec[l]
        L = {"n_windows": len(R)}
        res["layers"][str(l)] = L
        if len(R) < min_n:
            L["status"] = "INCOMPLETE"; continue
        TWN = sorted({k.split("|")[0] for k in R[0]["pairs"] if k.startswith("tw")})
        arts = [r["article"] for r in R]
        for comp, ci_ in (("I", 0), ("I_trunk", 1)):
            T1, T3, ex3s, tw3s = [], [], [], []
            for r in R:
                P_ = r["pairs"]
                ex3 = np.mean([abs(P_[f"EX|{i}{j}"][ci_]) for (i, j) in MP])
                mex = np.mean([abs(P_[f"EX|{i}{j}"][3]) + abs(P_[f"EX|{i}{j}"][4]) for (i, j) in MP])
                tw3 = [np.mean([abs(P_[f"{t_}|{i}{j}"][ci_]) for (i, j) in MP]) for t_ in TWN]
                mtw = [np.mean([abs(P_[f"{t_}|{i}{j}"][3]) + abs(P_[f"{t_}|{i}{j}"][4]) for (i, j) in MP]) for t_ in TWN]
                eps = 1e-12
                T1.append(np.mean([math.log((ex3 + eps) / (t + eps)) for t in tw3]))
                T3.append(np.mean([math.log(((ex3 + eps) / (mex + eps)) / ((t + eps) / (mt + eps))) for t, mt in zip(tw3, mtw)]))
                ex3s.append(ex3); tw3s.append(tw3)
            rank = int(1 + sum(np.mean([t[j] for t in tw3s]) > np.mean(ex3s) for j in range(len(TWN))))
            L[comp] = {"T1": float(np.mean(T1)), "T1_ci": boot_ci(T1, arts), "T3": float(np.mean(T3)), "T3_ci": boot_ci(T3, arts),
                       "rank": rank, "mean_ex3": float(np.mean(ex3s)), "mean_tw3": float(np.mean([np.mean(t) for t in tw3s]))}
            L[comp]["pass"] = bool(L[comp]["T1"] > math.log(2) and L[comp]["T1_ci"][0] > 0 and L[comp]["T3_ci"][0] > 0 and rank == 1)
        floors = [r["floor"] for r in R]
        twabs = [np.mean([abs(r["pairs"][f"{t_}|{i}{j}"][0]) for t_ in TWN for (i, j) in MP]) for r in R]
        L["median_floor"] = float(np.median(floors)); L["median_twin_absI"] = float(np.median(twabs))
        exp_ = [np.mean([abs(r["pairs"][f"EX|{i}{j}"][0]) for (i, j) in ALLP]) for r in R]
        L["descriptive"] = {"mean_absI_EX_p_allpairs": float(np.mean(exp_)),
                            "mean_abs_Isoft_EX_MP": float(np.mean([np.mean([abs(r["pairs"][f"EX|{i}{j}"][2]) for (i, j) in MP]) for r in R]))}
        L["REAL9"] = bool(L["I"]["pass"] and L["I_trunk"]["pass"])
        L["status"] = "REAL9" if L["REAL9"] else ("UNINFORMATIVE (precision)" if L["median_floor"] >= 0.25 * L["median_twin_absI"] else "not real")
        print(f"[v9] L{l} (n={len(R)}): I: T1 {L['I']['T1']:+.3f} CI {[round(x,3) for x in L['I']['T1_ci']]} T3 CI {[round(x,3) for x in L['I']['T3_ci']]} rank {L['I']['rank']} | "
              f"I_trunk: T1 {L['I_trunk']['T1']:+.3f} CI {[round(x,3) for x in L['I_trunk']['T1_ci']]} T3 CI {[round(x,3) for x in L['I_trunk']['T3_ci']]} rank {L['I_trunk']['rank']} | "
              f"floor {L['median_floor']:.2e} vs twin |I| {L['median_twin_absI']:.2e} -> {L['status']}", flush=True)
    if "VERDICT_v9" not in res:
        st = [res["layers"][str(l)].get("status") for l in LAYERS]
        if any(s == "INCOMPLETE" for s in st):
            res["VERDICT_v9"] = "INCOMPLETE (v9)"
        elif all(s == "REAL9" for s in st):
            res["VERDICT_v9"] = "EX carries excess within-span trunk pairwise interaction vs Sigma-orbit twins (v9)"
        elif any(s.startswith("UNINFORMATIVE") for s in st):
            res["VERDICT_v9"] = "UNINFORMATIVE (precision) (v9)"
        else:
            res["VERDICT_v9"] = "NOT shown (v9)"
    print(f"[v9] VERDICT: {res['VERDICT_v9']}", flush=True)
    json.dump(res, open(H.OUT / f"hs_v9_{TAG}{'_null' if NULL else ''}{'_plant' if PLANT_G is not None else ''}.json", "w"), indent=1)
    print(f"[v9] DONE ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
