"""v12: is H-space's joint (second-order) content context-invariant beyond matched decoys? Prereg: paper/SCOPE-hspace-v12.md.
Imports the frozen v10 machinery (hs_v10.py) unchanged: Runner, Freeze, orth, twins_w, eigh_desc, windows_val, cboot.

J-space is convincing because a direction's first-order effect on the output is context-invariant and readable. v12 asks
the same one order up. For a pair of an arm's directions (u_i, u_j) moved at layer l, token q (1 sigma):
    r_ij(x) = [h_L(x+u_i+u_j) - h_L(x+u_i) - h_L(x+u_j) + h_L(x)]_q / rms(h_L(x)_q),  m_i(x) = [h_L(x+u_i) - h_L(x)]_q / rms
projected per window off n = h_L(x)_q / |h_L(x)_q| (the final norm removes it), read in the centred logit metric
M = G W_U^T (I - 11^T/V) W_U G (G = final-norm gain). Any fixed pair has SOME context-invariant joint effect, so the
statistic is the SIZE of the invariant part relative to the whole (review 2026-10-26: identification saturates):
    F_det(arm) = mean over the 10 pairs of the unbiased invariant fraction  (||sum r||^2 - sum ||r||^2) / (n(n-1)) / mean ||r||^2
of r after projecting off span{template (k=1), mean m_i, mean m_j}. The template (norm curvature's pair-shared vector)
is estimated in stage K on SEPARATE windows from a SEPARATE pool of 48 Sigma-orbit twins.
Arms (plain): H (v11a all-position H-space), HJ (H minus J25), J5, tw0-7 (Sigma-orbit twins), dm0-7 (pool twins matched
to H on interaction magnitude AND whitened mean energy, stage K), tx0-7 (twins of HJ). Frozen (every RMSNorm / gated norm /
l2norm scale fixed at the clean pass, norm curvature exactly zero): H, J5, tw0-7, no template.
Stages: S0 frame/objects; K decoy screening + template; P planted positive control (move-only bilinear gate); W main.
    python hs_v12.py small|full [--minutes N]
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
MINUTES = 115.0
N_W_OVERRIDE = None
if "--minutes" in _r:
    k_ = _r.index("--minutes"); MINUTES = float(_r[k_ + 1]); del _r[k_:k_ + 2]
if "--nw" in _r:                                  # dev only (small mode); the full run uses the frozen N_W
    k_ = _r.index("--nw"); N_W_OVERRIDE = int(_r[k_ + 1]); del _r[k_:k_ + 2]
sys.argv = [sys.argv[0], TAG]
import hs_v10 as V  # noqa: E402  (frozen v10 machinery, imported unchanged)
import hspace as H  # noqa: E402
sys.argv = sys.argv[:1]

FULL = TAG == "full"
assert V.TAG == TAG
assert not (FULL and N_W_OVERRIDE), "--nw is for small-mode development only"
LAYERS = V.LAYERS
T12 = 64
N_COV = V.N_COV
N_W = 40 if FULL else (N_W_OVERRIDE or 8)
N_P = 8
N_K = 4
N_POOL = 48
MIN_W = 24 if FULL else 4
NT = 8
K_TMPL = 1
PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]
PK = [f"{i}{j}" for (i, j) in PAIRS]
N_BOOT = 2000
BUDGET = {"K": 10, "P": 8, "W": 70} if FULL else {"K": 6, "P": 6, "W": 60}
OUTF = H.OUT / f"hs_v12_{TAG}.json"
RAWF = H.OUT / f"hs_v12_{TAG}_raw.pt"
PROG = H.OUT / "progress" / "v12.json"
PLANT_TOKEN = " banana"
T0 = time.time()
DEADLINE = T0 + 60 * MINUTES
LOG2 = math.log(2)


def log(msg):
    print(f"[v12 {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


def progress(stage, idx, done, total, unit, note="", status="running"):
    try:
        PROG.parent.mkdir(parents=True, exist_ok=True)
        tmp = PROG.with_suffix(".tmp")
        tmp.write_text(json.dumps({"job": "v12", "title": f"v12 second-order lens ({TAG})", "status": status, "stage": stage,
                                   "stage_index": idx, "stage_count": 4, "done": done, "total": total, "unit": unit,
                                   "started": T0, "updated": time.time(), "note": note}))
        os.replace(tmp, PROG)
    except OSError:
        pass


# ------------------------------------------------------------------ statistics (float64 numpy; vectors already in
# M-metric coordinates y = r @ Lf with Lf Lf^T = M, so every ||.||_M and <.,.>_M below is Euclidean)
def proj_off(R, basis):
    """R [n,d] or [d]; basis list of [d] vectors (ignored if empty). Removes span(basis)."""
    if not basis:
        return R
    Q = np.linalg.qr(np.stack(basis, 1))[0]
    return R - (R @ Q) @ Q.T


def gram(R):
    return R @ R.T


def F_from_gram(G, ix=None):
    """Unbiased invariant fraction from a window Gram matrix: mean <r_a, r_b> over pairs of DISTINCT windows divided by
    mean ||r||^2. On a bootstrap index multiset, pairs of copies of the same window are excluded too."""
    ix = np.arange(G.shape[0]) if ix is None else np.asarray(ix)
    sub = G[np.ix_(ix, ix)]
    off = ix[:, None] != ix[None, :]
    den = float(np.diag(sub).mean())
    return float(sub[off].mean() / den) if den > 0 and off.any() else float("nan")


def arm_grams(st, arm, tmpl, extra=()):
    """Per pair: Gram of de-templated r (template, the arm's full-sample mean m_i, m_j, and optional extra vectors)."""
    out = {}
    mm = st[arm]["m"].mean(0)
    for p, (i, j) in zip(PK, PAIRS):
        out[p] = gram(proj_off(st[arm][p], list(tmpl) + [mm[i], mm[j]] + list(extra)))
    return out


def F_arm(grams, ix=None):
    return float(np.mean([F_from_gram(g, ix) for g in grams.values()]))


def compare_F(Gs, ex, decoys, clusters, min_lr, seed):
    """Frozen rule: F_det(ex) vs decoys. LR = log F(ex) - log median F(decoys); CI by article-cluster bootstrap."""
    f_ex = F_arm(Gs[ex])
    f_d = [F_arm(Gs[t]) for t in decoys]
    rank = 1 + sum(f >= f_ex for f in f_d)

    def lr(ix):                                       # F floored at 1e-4 before logs (F is unbiased, can dip below 0)
        a = F_arm(Gs[ex], ix); b = float(np.median([F_arm(Gs[t], ix) for t in decoys]))
        return math.log(max(a, 1e-4)) - math.log(max(b, 1e-4)) if math.isfinite(a) and math.isfinite(b) else float("nan")
    full_ = lr(None)
    ci = V.cboot(lr, clusters, B=N_BOOT, seed=seed)
    ok = rank == 1 and full_ >= min_lr and ci[0] > 0
    return {"F": f_ex, "F_decoys": f_d, "rank": int(rank), "K": len(decoys), "log_ratio": full_, "ci95": ci,
            "min_log_ratio": min_lr, "pass": bool(ok) if math.isfinite(full_) else False}


# ------------------------------------------------------------------ main
def main():
    log(f"start {TAG}, layers {LAYERS}, N_W {N_W}, budget {MINUTES} min")
    progress("S0 setup", 1, 0, 1, "steps", "loading model")
    res = {"tag": TAG, "layers": LAYERS, "scale_sigma": 1.0, "N_W": N_W, "stages": {}, "errors": {}, "verdict": {}}
    raw = {}

    def save(with_raw=False):
        tmp = str(OUTF) + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUTF)
        if with_raw:
            torch.save(raw, str(RAWF) + ".tmp"); os.replace(str(RAWF) + ".tmp", RAWF)

    T2 = torch.load(H.OUT / f"hspace2_{TAG}.pt", map_location="cpu", weights_only=False)
    INP = torch.load(H.OUT / f"v10_inputs_{TAG}.pt", map_location="cpu", weights_only=False)
    if FULL:
        sp = H.OUT / "v11a_subspaces.pt"
        res["v11a_subspaces_sha256"] = hashlib.sha256(open(sp, "rb").read()).hexdigest()
        assert res["v11a_subspaces_sha256"].startswith("29e0b7cb95991011"), "v11a subspace file hash mismatch"
        SUB = torch.load(sp, map_location="cpu", weights_only=False)
    R_ = V.Runner()
    m, tok = R_.m, R_.tok
    d = m.d
    R_.use_T(T12)
    # freeze must bind every norm: the gated DeltaNet norm must be the patched torch class, not a fused kernel
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q35
    gated = [type(b.linear_attn.norm).__name__ for b in m.layers if hasattr(b, "linear_attn")]
    assert all(isinstance(b.linear_attn.norm, q35.Qwen3_5RMSNormGated) for b in m.layers if hasattr(b, "linear_attn")), gated
    res["gated_norm_classes"] = sorted(set(gated))

    # ---- logit metric M and its factor Lf (Lf Lf^T = M)
    gain = m.gain.double()
    WtW = torch.zeros(d, d, dtype=torch.float64, device="cuda"); s1 = torch.zeros(d, dtype=torch.float64, device="cuda")
    for s in range(0, m.W_U.shape[0], 8192):
        W = m.W_U[s:s + 8192].double()
        WtW += W.T @ W; s1 += W.sum(0)
    Vn = m.W_U.shape[0]
    Mg = gain[:, None] * (WtW - torch.outer(s1, s1) / Vn) * gain[None, :]
    evM, QM = torch.linalg.eigh(Mg)
    Lf = (QM * evM.clamp_min(0).sqrt()[None, :]).contiguous()
    del WtW, s1, Mg, QM, evM

    def logits_of(vec):
        v_ = torch.as_tensor(np.asarray(vec), dtype=torch.float32, device="cuda") * m.gain
        return torch.cat([v_ @ m.W_U[s:s + 16384].float().T for s in range(0, Vn, 16384)])

    def decode(vec, k=15):
        lg = logits_of(vec)
        lg = lg - lg.mean()
        top = torch.topk(lg, k).indices.tolist(); bot = torch.topk(-lg, k).indices.tolist()
        return {"top": [tok.decode([t]) for t in top], "bottom": [tok.decode([t]) for t in bot]}

    # ================= S0: whitening frame, objects, twins, pool
    st0 = time.time()
    S0 = {}
    res["stages"]["S0"] = S0
    covw = H.wikitext_windows(tok, "train", N_COV, T12, seed=12)
    acc = {l: [0, torch.zeros(d, dtype=torch.float64, device="cuda"), torch.zeros(d, d, dtype=torch.float64, device="cuda")]
           for l in LAYERS}
    for b0 in range(0, len(covw), 8):
        ids = covw[b0:b0 + 8].cuda()
        with torch.no_grad():
            h = m.embed(ids); prev = 0
            for l in sorted(LAYERS):
                h = m.run(h, prev, l + 1); prev = l + 1
                Xl = h[:, H.SKIP:T12 - 1].reshape(-1, d).double()
                acc[l][0] += len(Xl); acc[l][1] += Xl.sum(0); acc[l][2] += Xl.T @ Xl
    ctrl, pool, aux = {}, {}, {}
    for l in LAYERS:
        n_, s1_, s2_ = acc[l]
        mu_c = s1_ / n_
        Sig = ((s2_ / n_ - torch.outer(mu_c, mu_c)) * n_ / (n_ - 1))
        Sig = (0.95 * Sig + 0.05 * torch.trace(Sig) / d * torch.eye(d, dtype=Sig.dtype, device=Sig.device)).cpu()
        if f"Sh{l}" in T2:
            Sh = T2[f"Sh{l}"].double(); mu = T2[f"mu{l}"].double()
        else:
            evs, Vs = V.eigh_desc(Sig)
            Sh = Vs @ torch.diag(evs.clamp_min(1e-12).sqrt()) @ Vs.T; mu = mu_c.cpu()
        Shi = torch.linalg.inv(Sh.cuda()).cpu()
        Hw = SUB[f"H_all_{l}"].double() if FULL else INP[f"H_{l}"].double()        # small mode: smoke stand-in
        J25 = (INP[f"J25w_{l}"] if f"J25w_{l}" in INP else T2[f"J25w_{l}"]).double()
        QJ = V.orth(J25)
        J5w = V.orth(J25[:, :5])
        HJ = V.orth(Hw - QJ @ (QJ.T @ Hw))
        evg, Pg = V.eigh_desc(Sh)
        tw = V.twins_w(Hw, Pg, NT, 12100 + l)
        tx = V.twins_w(HJ, Pg, NT, 12200 + l)
        pl = V.twins_w(Hw, Pg, N_POOL, 12300 + l)
        mt = Shi @ mu; mt = mt / mt.norm()                                          # whitened mean direction
        me = lambda Q: float((Q.T @ mt).pow(2).sum())                              # noqa: E731
        C = {"H": Hw, "HJ": HJ, "J5": J5w}
        for j in range(NT):
            C[f"tw{j}"], C[f"tx{j}"] = tw[j], tx[j]
        ctrl[l] = {k: (Sh @ v).float().cuda() for k, v in C.items()}               # raw 1-sigma moves [d,5]
        pool[l] = {f"pl{j}": (Sh @ pl[j]).float().cuda() for j in range(N_POOL)}
        aux[l] = {"Sh": Sh, "Shi": Shi, "mean_energy": {**{k: me(v) for k, v in C.items()}, **{f"pl{j}": me(pl[j]) for j in range(N_POOL)}}}
        S0[str(l)] = {"H_in_J25": float((QJ.T @ Hw).pow(2).sum() / 5),
                      "mean_energy": {"H": me(Hw), "tw_max": max(me(t) for t in tw), "pool_max": max(me(t) for t in pl),
                                      "J5": me(J5w)},
                      "raw_norm": {k: float(v.norm(dim=0).mean()) for k, v in ctrl[l].items()}}
        log(f"S0 L{l}: H in J25 {S0[str(l)]['H_in_J25']:.3f} | whitened mean energy H {me(Hw):.4f} vs twins max "
            f"{S0[str(l)]['mean_energy']['tw_max']:.4f}, pool max {S0[str(l)]['mean_energy']['pool_max']:.4f}")
    del acc
    save()
    log(f"S0 done ({(time.time()-st0)/60:.1f} min)")

    # ------------------------------------------------------------------ the measurement
    def joint(X, l, q, arms, plant=None, freeze=False):
        """X [T,d] layer-l residual. Returns ({arm: (r [10,d], m [5,d])} float32 CPU, h0 [d]) at final layer, token q,
        divided by rms(h_L(x)_q) and projected off n = h0/|h0|."""
        seqs, key = [X], {}
        for a in arms:
            Vm = arms[a]
            for i in range(5):
                xx = X.clone(); xx[q] += Vm[:, i]; key[(a, i)] = len(seqs); seqs.append(xx)
            for (i, j) in PAIRS:
                xx = X.clone(); xx[q] += Vm[:, i] + Vm[:, j]; key[(a, i, j)] = len(seqs); seqs.append(xx)
        hq = []
        try:
            if freeze:
                V.Freeze.mode, V.Freeze.store = "record", []
                with torch.no_grad():
                    m.run(X[None], l + 1)
                V.Freeze.mode = "use"
            for c0 in range(0, len(seqs), V.CHUNK):
                V.Freeze.k = 0
                with torch.no_grad():
                    hL = m.run(torch.stack(seqs[c0:c0 + V.CHUNK]), l + 1, plant=plant)
                if freeze and V.Freeze.k != len(V.Freeze.store):
                    raise RuntimeError(f"freeze did not bind: used {V.Freeze.k} of {len(V.Freeze.store)} stored scales")
                hq.append(hL[:, q].double())
        finally:
            V.Freeze.mode, V.Freeze.store, V.Freeze.k = None, [], 0
        hq = torch.cat(hq)
        h0 = hq[0]
        rms = float(h0.pow(2).mean().sqrt())
        nrm = h0 / h0.norm()
        po = lambda v: v - (v @ nrm)[..., None] * nrm                              # noqa: E731
        out = {}
        for a in arms:
            mm = torch.stack([hq[key[(a, i)]] - h0 for i in range(5)])
            rr = torch.stack([hq[key[(a, i, j)]] - hq[key[(a, i)]] - hq[key[(a, j)]] + h0 for (i, j) in PAIRS])
            out[a] = (po(rr / rms).cpu().float(), po(mm / rms).cpu().float())
        return out, h0.cpu()

    def finite(jt):
        return all(bool(torch.isfinite(v[0]).all() and torch.isfinite(v[1]).all()) for v in jt.values())

    W_SEED, P_SEED, K_SEED = 1202, 1201, 1203
    wsW, artW = V.windows_val(tok, N_W, T12, seed=W_SEED, kmin=2)
    W_arts = set(artW)

    def windows_excl(n, seed):
        ws, art = V.windows_val(tok, n + 400, T12, seed=seed, kmin=2)
        keep = [(w, a) for w, a in zip(ws, art) if a not in W_arts][:n]
        rng = random.Random(seed)
        return [(torch.tensor(w), a, rng.randrange(H.SKIP + 8, T12 - 1)) for w, a in keep]

    rngW = random.Random(W_SEED)
    planW = [(torch.tensor(w), a, rngW.randrange(H.SKIP + 8, T12 - 1)) for w, a in zip(wsW, artW)]

    def layer_states(w):
        Xs = {}
        with torch.no_grad():
            h = m.embed(w[None].cuda()); prev = 0
            for l in sorted(LAYERS):
                h = m.run(h, prev, l + 1); prev = l + 1
                Xs[l] = h[0].clone()
        return Xs

    def stack_metric(recs, arms):
        """recs: list of {arm: (r [10,d], m [5,d])}. -> {arm: {pair: [n,d], 'm': [n,5,d]}} float64 numpy, M coords."""
        o = {}
        for a in arms:
            R = torch.stack([r[a][0] for r in recs]).double().cuda()                  # [n,10,d]
            Mm = torch.stack([r[a][1] for r in recs]).double().cuda()                 # [n,5,d]
            Ry, My = (R @ Lf).cpu().numpy(), (Mm @ Lf).cpu().numpy()
            o[a] = {p: Ry[:, k] for k, p in enumerate(PK)}
            o[a]["m"] = My
        return o

    def raw_means(recs, arms):
        return {a: {"r": torch.stack([r[a][0] for r in recs]).double().mean(0), "m": torch.stack([r[a][1] for r in recs]).double().mean(0)}
                for a in arms}

    # ================= K: decoy screening (magnitude + mean energy) and the pair-shared template, separate windows
    SK = {}
    res["stages"]["K"] = SK
    tmpl, tmpl_raw = {}, {}
    try:
        stk = time.time()
        planK = windows_excl(N_K, K_SEED)
        res["K_articles_overlap_W"] = len({a for _, a, _ in planK} & W_arts)
        for l in LAYERS:
            arms = {"H": ctrl[l]["H"], **pool[l]}
            recs = []
            for wi, (w, a, q) in enumerate(planK):
                jt, _ = joint(layer_states(w)[l], l, q, arms)
                if finite(jt):
                    recs.append(jt)
                progress(f"K screening L{l}", 2, wi + 1, len(planK), "windows")
            stK = stack_metric(recs, list(arms))
            mag = {k: float(np.median([np.linalg.norm(stK[k][p], axis=1) for p in PK])) for k in arms}
            me = aux[l]["mean_energy"]
            lm = np.array([math.log(mag[f"pl{j}"]) for j in range(N_POOL)])
            ee = np.array([me[f"pl{j}"] for j in range(N_POOL)])
            dist = np.abs(lm - math.log(mag["H"])) / (lm.std() + 1e-12) + np.abs(ee - me["H"]) / (ee.std() + 1e-12)
            pick = [int(i) for i in np.argsort(dist)[:NT]]
            for j, pj in enumerate(pick):
                ctrl[l][f"dm{j}"] = pool[l][f"pl{pj}"]
                aux[l]["mean_energy"][f"dm{j}"] = me[f"pl{pj}"]
            # template: top right-singular vectors of the pool's unit-normalised pair means (M coords and raw)
            A = np.stack([stK[f"pl{j}"][p].mean(0) for j in range(N_POOL) for p in PK])
            A = A / np.linalg.norm(A, axis=1, keepdims=True)
            sv = np.linalg.svd(A, full_matrices=False)
            tmpl[l] = [sv[2][k] for k in range(10)]
            rm = raw_means(recs, [f"pl{j}" for j in range(N_POOL)])
            Ar = torch.stack([rm[f"pl{j}"]["r"][k] for j in range(N_POOL) for k in range(10)])
            Ar = Ar / Ar.norm(dim=1, keepdim=True)
            tmpl_raw[l] = torch.linalg.svd(Ar, full_matrices=False)[2][0].numpy()
            SK[str(l)] = {"n": len(recs), "magnitude": {"H": mag["H"], "pool_median": float(np.median([mag[f"pl{j}"] for j in range(N_POOL)]))},
                          "mean_energy": {"H": me["H"], "pool_median": float(np.median(ee))},
                          "picked": pick, "picked_magnitude": [mag[f"pl{p}"] for p in pick], "picked_mean_energy": [me[f"pl{p}"] for p in pick],
                          "template_sv_share": [float(x) for x in (sv[1][:10] ** 2 / (sv[1] ** 2).sum())]}
            log(f"K L{l}: magnitude H {mag['H']:.3e} vs pool median {SK[str(l)]['magnitude']['pool_median']:.3e}; mean energy H "
                f"{me['H']:.4f} vs pool median {np.median(ee):.4f}; matched decoys magnitude {[round(x / mag['H'], 2) for x in SK[str(l)]['picked_magnitude']]} "
                f"x H; template top-SV share {SK[str(l)]['template_sv_share'][0]:.2f}")
        save()
        log(f"K done ({(time.time()-stk)/60:.1f} min)")
    except Exception:  # noqa: BLE001
        res["errors"]["K"] = traceback.format_exc(); log("K ERROR\n" + res["errors"]["K"])
    k_ok = all(str(l) in SK for l in LAYERS)

    TW = [f"tw{j}" for j in range(NT)]
    TX = [f"tx{j}" for j in range(NT)]
    DM = [f"dm{j}" for j in range(NT)]

    # ================= P: planted positive control. Move-only bilinear gate at block l+1 (plant mu = clean state, so
    # z = whitened move at q and 0 elsewhere): adds G (z.a)(z.b) w with J_l w ~ c (lstsq), c = readout of PLANT_TOKEN.
    # P runs a 5-direction arm (a, b, 3 random) planted and unplanted, plus 4 Sigma-orbit twins of it (planted model).
    SP = {}
    res["stages"]["P"] = SP
    try:
        stp = time.time()
        target = tok(PLANT_TOKEN, add_special_tokens=False)["input_ids"][0]
        planP = windows_excl(N_P, P_SEED)
        res["P_articles_overlap_W"] = len({a for _, a, _ in planP} & W_arts)
        for l in LAYERS:
            Sh, Shi = aux[l]["Sh"], aux[l]["Shi"]
            g = torch.Generator().manual_seed(9100 + l)
            Pw = V.orth(torch.randn(d, 5, generator=g, dtype=torch.float64))
            c = (m.gain.double().cpu() * m.W_U[target].double().cpu())
            Jl = INP.get(f"J_{l}")
            w_ = torch.linalg.lstsq(Jl.double(), c[:, None]).solution[:, 0] if Jl is not None else c.clone()
            w_ = w_ / w_.norm()
            twP = V.twins_w(Pw, V.eigh_desc(Sh)[1], 4, 9200 + l)
            armsP = {"P": (Sh @ Pw).float().cuda(), **{f"pt{j}": (Sh @ twP[j]).float().cuda() for j in range(4)}}
            base = {"block": l + 1, "Sinvh": Shi.float().cuda(), "a": Pw[:, 0].float().cuda(), "b": Pw[:, 1].float().cuda(),
                    "w": w_.float().cuda(), "G": 1.0}
            # G: planted per-window term (planted - unplanted, P pair 01, M norm) = 0.5 x median de-templated null r norm
            w0, _, q0 = planP[0]
            X0 = layer_states(w0)[l]
            j_un, _ = joint(X0, l, q0, armsP)
            j_pl, _ = joint(X0, l, q0, {"P": armsP["P"]}, plant={**base, "mu": X0})
            y = lambda v: (v.double().cuda() @ Lf).cpu().numpy()                     # noqa: E731
            per_G = float(np.linalg.norm(y(j_pl["P"][0][0] - j_un["P"][0][0])))
            nulln = float(np.median([np.linalg.norm(proj_off(y(j_un[f"pt{j}"][0][k]), tmpl.get(l, [])[:K_TMPL]))
                                     for j in range(4) for k in range(10)]))
            Gv = 0.5 * nulln / per_G if per_G > 0 else 1.0
            recs_pl, recs_un = [], []
            for wi, (w, a, q) in enumerate(planP):
                if time.time() > stp + 60 * BUDGET["P"]:
                    break
                Xl = layer_states(w)[l]
                jp, _ = joint(Xl, l, q, armsP, plant={**base, "mu": Xl, "G": Gv})
                ju, _ = joint(Xl, l, q, {"P": armsP["P"]})
                if finite(jp) and finite(ju):
                    recs_pl.append(jp); recs_un.append({"P0": ju["P"]})
                progress(f"P planted control L{l}", 3, wi + 1, len(planP), "windows")
            stP = stack_metric(recs_pl, list(armsP)); stU = stack_metric(recs_un, ["P0"])
            stP.update(stU)
            t_ = tmpl.get(l, [])[:K_TMPL]
            GsP = {k: arm_grams(stP, k, t_) for k in stP}
            f01 = F_from_gram(GsP["P"]["01"])
            f01_un = F_from_gram(GsP["P0"]["01"])
            f_null = max(F_from_gram(GsP[f"pt{j}"][p]) for j in range(4) for p in PK)
            f_other = max(F_from_gram(GsP["P"][p]) for p in PK if p != "01")
            diff = (torch.stack([r["P"][0][0] for r in recs_pl]) - torch.stack([r["P0"][0][0] for r in recs_un])).double().mean(0)
            lg = logits_of(diff)
            rank_t = int((lg > lg[target]).sum()) + 1
            realized = float(np.linalg.norm(y((recs_pl[0]["P"][0][0] - recs_un[0]["P0"][0][0]).float()))) / nulln if nulln > 0 else float("nan")
            SP[str(l)] = {"n": len(recs_pl), "G": Gv, "realized_ratio_window0": realized, "F_P01": f01, "F_P01_unplanted": f01_un,
                          "F_null_max": f_null, "F_P_other_max": f_other, "target_rank": rank_t, "decode_planted_minus_unplanted": decode(diff),
                          "pass": bool(f01 > f_null and f01 > f_other and f01_un <= f_null and rank_t <= 10)}
            log(f"P L{l}: F planted pair {f01:.3f} (unplanted {f01_un:.3f}, null max {f_null:.3f}, other P pairs max {f_other:.3f}) "
                f"realized {realized:.2f}x | '{PLANT_TOKEN}' rank {rank_t} -> {'PASS' if SP[str(l)]['pass'] else 'FAIL'}")
            save()
    except Exception:  # noqa: BLE001
        res["errors"]["P"] = traceback.format_exc(); log("P ERROR\n" + res["errors"]["P"])
    save()

    # ================= W: the main measurement on fresh windows (articles disjoint from K and P), one random position each
    SW = {}
    res["stages"]["W"] = SW
    recW = {l: [] for l in LAYERS}
    ARMS_PLAIN = ["H", "HJ", "J5"] + TW + DM + TX
    ARMS_FROZEN = ["H", "J5"] + TW
    n_drop = {l: 0 for l in LAYERS}
    try:
        if not k_ok:
            raise RuntimeError("stage K failed: no matched decoys or template; W skipped")
        stw = time.time()
        res["W_articles"] = len(W_arts)
        for wi, (w, a, q) in enumerate(planW):
            if time.time() > stw + 60 * BUDGET["W"] or time.time() > DEADLINE - 180:
                log(f"W time budget reached after {wi} windows"); break
            Xs = layer_states(w)
            for l in LAYERS:
                jp, h0p = joint(Xs[l], l, q, {k: ctrl[l][k] for k in ARMS_PLAIN})
                jf, h0f = joint(Xs[l], l, q, {k: ctrl[l][k] for k in ARMS_FROZEN}, freeze=True)
                base_err = float((h0p - h0f).abs().max() / h0p.abs().max())
                if not (finite(jp) and finite(jf)):
                    n_drop[l] += 1; log(f"W L{l} window {wi}: non-finite values, dropped for all conditions"); continue
                recW[l].append({"article": int(a), "q": int(q), "freeze_base_rel_err": base_err, "plain": jp, "frozen": jf})
            el = time.time() - stw
            progress("W joint content", 4, wi + 1, len(planW), "windows",
                     f"{el / (wi + 1):.0f}s/window, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
            if (wi + 1) % 4 == 0 or wi == len(planW) - 1:
                log(f"W window {wi+1}/{len(planW)} ({el/(wi+1):.1f}s/window, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)")
            if (wi + 1) % 10 == 0:
                raw["W"] = recW; save(with_raw=True)
        raw["W"] = recW
        res["W_dropped_nonfinite"] = n_drop
        for l in LAYERS:
            rr = recW[l]
            L = {"n": len(rr), "freeze_base_rel_err_max": float(max([r["freeze_base_rel_err"] for r in rr] or [float("nan")]))}
            L["freeze_valid"] = bool(L["freeze_base_rel_err_max"] <= 1e-4)
            if len(rr) < MIN_W:
                L["verdict"] = "inconclusive (too few windows)"; SW[str(l)] = L; continue
            cl = [r["article"] for r in rr]
            stp_ = stack_metric([r["plain"] for r in rr], ARMS_PLAIN)
            stf_ = stack_metric([r["frozen"] for r in rr], ARMS_FROZEN)
            t1 = tmpl[l][:K_TMPL]
            Gp = {k: arm_grams(stp_, k, t1) for k in ARMS_PLAIN}
            Gf = {k: arm_grams(stf_, k, []) for k in ARMS_FROZEN}
            c1 = compare_F(Gp, "H", TW, cl, LOG2, 100 + l)
            c2 = compare_F(Gp, "H", DM, cl, math.log(1.5), 200 + l)
            c3 = compare_F(Gf, "H", TW, cl, 0.0, 300 + l)
            dj = compare_F(Gp, "HJ", TX, cl, LOG2, 400 + l)
            # output-side distinctness: also project r off the full-sample mean single effects of J5 (its output span)
            j5out = list(stp_["J5"]["m"].mean(0))
            Gq = {k: arm_grams(stp_, k, t1, extra=j5out) for k in ["H"] + TW}
            do = compare_F(Gq, "H", TW, cl, LOG2, 500 + l)
            content = bool(c1["pass"] and c2["pass"] and c3["pass"] and L["freeze_valid"])
            distinct = bool(dj["pass"] and do["pass"])
            # descriptive: template depth k = 3, 10; heaviest window dropped; J5
            desc = {}
            for kk in (3, 10):
                Gk = {k: arm_grams(stp_, k, tmpl[l][:kk]) for k in ["H"] + TW + DM}
                desc[f"k{kk}"] = {"F_H": F_arm(Gk["H"]), "F_tw_median": float(np.median([F_arm(Gk[t]) for t in TW])),
                                  "F_dm_median": float(np.median([F_arm(Gk[t]) for t in DM]))}
            heavy = int(np.argmax([float(np.mean([np.linalg.norm(stp_[k][p][i]) for k in ARMS_PLAIN for p in PK])) for i in range(len(rr))]))
            keep = [i for i in range(len(rr)) if i != heavy]
            desc["drop_heaviest"] = {"F_H": F_arm(Gp["H"], keep), "F_tw_median": float(np.median([F_arm(Gp[t], keep) for t in TW])),
                                     "F_dm_median": float(np.median([F_arm(Gp[t], keep) for t in DM]))}
            desc["J5"] = {"F_plain": F_arm(Gp["J5"]), "F_frozen": F_arm(Gf["J5"]),
                          "F_per_pair": {p: F_from_gram(Gp["J5"][p]) for p in PK}}
            desc["F_per_pair_H"] = {p: F_from_gram(Gp["H"][p]) for p in PK}
            desc["mean_energy"] = {k: aux[l]["mean_energy"][k] for k in ["H", "J5"] + TW + DM}
            L.update({"H_vs_sigma_twins": c1, "H_vs_matched_decoys": c2, "H_frozen_vs_twins": c3, "HJ_vs_tx": dj,
                      "H_minus_J5_output_vs_twins": do, "content_pass": content, "distinct_pass": distinct, "descriptive": desc})
            # decodes (raw space, de-templated with the raw pool template and each arm's own mean single effects)
            rm = raw_means([r["plain"] for r in rr], ["H", "HJ", "J5"] + DM)
            tr = torch.as_tensor(tmpl_raw[l])
            dec = {}
            for k in ["H", "HJ", "J5"] + DM:
                dec[k] = {}
                for kk, (i, j) in enumerate(PAIRS):
                    v = proj_off(rm[k]["r"][kk].numpy(), [tr.numpy(), rm[k]["m"][i].numpy(), rm[k]["m"][j].numpy()])
                    dec[k][PK[kk]] = decode(v)
            L["decode_pairs_detemplated"] = dec
            L["decode_singles"] = {k: [decode(rm[k]["m"][i]) for i in range(5)] for k in ("H", "HJ", "J5")}
            log(f"W L{l} (n={len(rr)}): F_det H {c1['F']:.4f} | vs Sigma-twins median {np.median(c1['F_decoys']):.4f} LR {c1['log_ratio']:+.2f} "
                f"CI {[round(x, 2) for x in c1['ci95']]} rank {c1['rank']}/9 -> {'PASS' if c1['pass'] else 'fail'} | vs matched decoys "
                f"LR {c2['log_ratio']:+.2f} CI {[round(x, 2) for x in c2['ci95']]} rank {c2['rank']}/9 -> {'PASS' if c2['pass'] else 'fail'} | "
                f"frozen LR {c3['log_ratio']:+.2f} rank {c3['rank']}/9 -> {'PASS' if c3['pass'] else 'fail'} (freeze valid {L['freeze_valid']}) | "
                f"HJ LR {dj['log_ratio']:+.2f} -> {'PASS' if dj['pass'] else 'fail'}; H-minus-J5out LR {do['log_ratio']:+.2f} -> "
                f"{'PASS' if do['pass'] else 'fail'} | J5 F {desc['J5']['F_plain']:.4f}")
            SW[str(l)] = L
            save()
    except Exception:  # noqa: BLE001
        res["errors"]["W"] = traceback.format_exc(); log("W ERROR\n" + res["errors"]["W"])
    save(with_raw=True)

    # ================= verdict (frozen rule, SCOPE-hspace-v12.md)
    p_ok = all(SP.get(str(l), {}).get("pass") for l in LAYERS)
    content = [SW.get(str(l), {}).get("content_pass") for l in LAYERS]
    distinct = [SW.get(str(l), {}).get("distinct_pass") for l in LAYERS]
    if "P" in res["errors"] or "K" in res["errors"] or "W" in res["errors"]:
        v = "INCONCLUSIVE (stage error: " + ",".join(k for k in ("K", "P", "W") if k in res["errors"]) + ")"
    elif not p_ok:
        v = "INSTRUMENT FAILED (planted control)"
    elif any(c is None for c in content):
        v = "INCONCLUSIVE (too few windows)"
    elif all(content) and all(distinct):
        v = "STAGE-1 PASS: H-space's joint content is context-invariant beyond matched decoys, at both layers, and distinct from J-space (readability: stage 2)"
    elif all(content):
        v = "PARTIAL: invariant joint content at both layers, not distinct from J-space"
    elif any(content):
        v = "PARTIAL: invariant joint content at one layer only"
    else:
        v = "FAIL: H-space's joint content is not more context-invariant than matched decoys"
    res["verdict"] = {"planted_control_ok": p_ok, "content_by_layer": content, "distinct_by_layer": distinct, "verdict": v}
    log(f"VERDICT: {v}")
    progress("done", 4, 1, 1, "steps", v, status="done")
    save(with_raw=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        progress("crashed", 4, 0, 1, "steps", traceback.format_exc()[-180:], status="failed")
        raise
