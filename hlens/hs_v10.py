"""v10: is the frozen H-space (v8 punctuation patterns; "EX" in v8/v9 files) real, used, and distinct from J-space?
Prereg: paper/SCOPE-hspace-v10.md (frozen and pushed before the pod). Plan: research/proposals/40-hspace-shopping-list.md.
Revised after the pre-freeze adversarial review (wf_12488a90): see SCOPE "Deviations".

Stages, in priority order (results saved incrementally; W, G, T are each guarded so R and the verdict always run):
  S0  class moments (fresh wikitext-train windows), M2 punctuation-moment gate, controls: 16 whitened-frame punctuation
      twins (8 plain + 8 J-matched), 8 word twins, 8 global twins (v9's), J5, PC5; twin moment diagnostics
  W   wikitext validation windows at offsets k>=2 (text v9 never used): M3 (H vs 16 punctuation twins, + twin
      prediction-interval test), RMS-freeze arm (H + 8 plain twins), M8a (word position vs 8 word twins), sink scope
  G   H-blind task gates: behavioural DiD gate (M6 family), site gate = patched DiD_site/DiD >= 0.3 (+ main-effect
      recovery where that main effect is non-zero) for M4/M5
  T   per gated context and the null: transfer T1 (M6), gold rank (M7), routing/deletion/sufficiency/AtP (M4, A3, A1),
      per-item-mean ablation (M5), natural-site routing (M8c), TOST in units of real task DiD
  R   random-weights null: shuffle every >=2-D parameter of blocks > l (L40 first, then L16), rerun the M3 rectangle
    python hs_v10.py small|full [--minutes N]
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
import torch.nn.functional as F
from scipy.stats import t as tdist
from scipy.stats import wilcoxon

_raw = sys.argv[1:]
sys.argv = sys.argv[:1]
import bf16w  # noqa: E402
import hspace as H  # noqa: E402
from hs_v9 import boot_ci, klass  # noqa: E402

TAG = _raw[0] if _raw else "small"
_r = list(_raw[1:])
MINUTES = 170.0
if "--minutes" in _r:
    k_ = _r.index("--minutes"); MINUTES = float(_r[k_ + 1]); del _r[k_:k_ + 2]
FULL = TAG == "full"
LAYERS = [16, 40] if FULL else [6, 12]
MP = [(0, 1), (2, 3), (1, 4)]
CHUNK = 24 if FULL else 8
NEXT = 16
T_W, T_T = 128, 32
N_COV = 768 if FULL else 40
N_W = 40 if FULL else 4
N_R = 24 if FULL else 2
N_ITEMS = 40 if FULL else 3
N_NULL = 24 if FULL else 3
MIN_W, MIN_T, MIN_R = (24, 12, 12) if FULL else (2, 2, 2)
N_TW, N_TWJ, N_TWW, N_TG = 8, 8, 8, 8
BANKS = ["negation_truth", "entity_attribute", "relational_composition"]
BACKUPS = ["binding_backup"]
NULLB = "null_unrelated"
BUDGET = {"S0": 12, "W": 50, "G": 8, "T": 70, "R": 14} if FULL else {"S0": 6, "W": 6, "G": 6, "T": 8, "R": 6}
OUTF = H.OUT / f"hs_v10_{TAG}.json"
RAWF = H.OUT / f"hs_v10_{TAG}_raw.pt"
T0 = time.time()
DEADLINE = T0 + 60 * MINUTES
LOG2 = math.log(2)
TW = [f"tw{j}" for j in range(N_TW)] + [f"tj{j}" for j in range(N_TWJ)]
TWP = TW[:N_TW]


def log(msg):
    print(f"[v10 {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


# ------------------------------------------------------------------ norm freezing (RMS-freeze arm)
class Freeze:
    mode = None          # None | "record" | "use"
    store = []
    k = 0


def _frozen_scale(r):
    if Freeze.mode is None:
        return r
    if Freeze.mode == "record":
        Freeze.store.append(r.detach())
        return r
    s = Freeze.store[Freeze.k]; Freeze.k += 1
    if s.shape == r.shape:
        return s
    if s.shape[0] == 1 and s.shape[1:] == r.shape[1:]:
        return s.expand_as(r)
    if r.shape[0] % s.shape[0] == 0 and s.shape[1:] == r.shape[1:]:
        return s.repeat(r.shape[0] // s.shape[0], *([1] * (s.dim() - 1)))
    raise RuntimeError(f"freeze shape mismatch {tuple(s.shape)} vs {tuple(r.shape)}")


def install_freeze_patches():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    def rms_norm(self, x):
        return x * _frozen_scale(torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))
    q.Qwen3_5RMSNorm._norm = rms_norm

    def gated_fwd(self, hidden_states, gate=None):
        dt = hidden_states.dtype
        hs = hidden_states.to(torch.float32)
        hs = hs * _frozen_scale(torch.rsqrt(hs.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
        hs = self.weight * hs.to(dt)
        hs = hs * F.silu(gate.to(torch.float32))
        return hs.to(dt)
    q.Qwen3_5RMSNormGated.forward = gated_fwd
    _l2 = q.l2norm

    def l2norm_frozen(x, dim=-1, eps=1e-6):
        if Freeze.mode is None:
            return _l2(x, dim=dim, eps=eps)
        return x * _frozen_scale(torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps))
    q.l2norm = l2norm_frozen


# ------------------------------------------------------------------ helpers
def orth(A):
    return torch.linalg.qr(A)[0]


def eigh_desc(S):
    ev, V = torch.linalg.eigh(S.cuda() if torch.cuda.is_available() else S)
    o = torch.argsort(ev, descending=True)
    return ev[o].cpu(), V[:, o].cpu()


def twins_w(Hw, Vc, n, seed):
    """Whitened-frame class-orbit twins Vc diag(s) Vc^T Hw (Vc = eigvecs of the WHITENED class covariance): exact global
    moments (1-sigma norms, Gram) and exact class-c moments."""
    g = torch.Generator().manual_seed(seed)
    C = Vc.T @ Hw
    return [Vc @ ((torch.randint(0, 2, (Vc.shape[0], 1), generator=g).double() * 2 - 1) * C) for _ in range(n)]


def cboot(fn, clusters, B=2000, seed=0, q=(0.025, 0.975)):
    cl = np.asarray(clusters); ua = np.unique(cl); io = {u: np.where(cl == u)[0] for u in ua}
    rb = np.random.default_rng(seed)
    bs = [fn(np.concatenate([io[u] for u in rb.choice(ua, len(ua))])) for _ in range(B)]
    bs = np.array([b for b in bs if np.isfinite(b)])
    return [float(np.quantile(bs, q[0])), float(np.quantile(bs, q[1]))] if len(bs) else [float("nan")] * 2


def ratio_of_means(num, den):
    num, den = np.asarray(num, float), np.asarray(den, float)

    def f(ix):
        d = den[ix].mean()
        return num[ix].mean() / d if abs(d) > 1e-12 else float("nan")
    return f


def two_way_T1(ex, tw, clusters, B=2000, seed=0):
    """T1 = mean_i mean_j log(ex_i / tw_ij); 95% CI from a bootstrap over clusters AND twins; p_PI = one-sided
    prediction-interval p of H against the leave-one-out twin T1_j distribution (t, K-1 df)."""
    ex, tw = np.asarray(ex, float), np.asarray(tw, float)
    eps = 1e-12
    a, b = np.log(ex + eps), np.log(tw + eps)
    L = a[:, None] - b
    T1 = float(L.mean())
    cl = np.asarray(clusters); ua = np.unique(cl)
    idx_of = {u: np.where(cl == u)[0] for u in ua}
    rb = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        pick = np.concatenate([idx_of[u] for u in rb.choice(ua, len(ua))])
        tj = rb.integers(0, L.shape[1], L.shape[1])
        bs.append(L[np.ix_(pick, tj)].mean())
    bs = np.array(bs)
    K = b.shape[1]
    T1j = np.array([float((b[:, [j]] - b[:, [x for x in range(K) if x != j]]).mean()) for j in range(K)])
    sdj = float(T1j.std(ddof=1)) if K > 2 else float("nan")
    tstat = (T1 - T1j.mean()) / (sdj * math.sqrt(1 + 1 / K)) if sdj and sdj > 0 else float("inf")
    p_pi = float(tdist.sf(tstat, K - 1)) if math.isfinite(tstat) else 0.0
    rank = int(1 + sum(tw[:, j].mean() > ex.mean() for j in range(K)))
    p_boot = float((1 + (bs <= 0).sum()) / (B + 1))
    return {"T1": T1, "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))], "p_le0": p_boot,
            "p_PI": p_pi, "p": max(p_boot, p_pi), "twin_T1_mean": float(T1j.mean()), "twin_T1_sd": sdj,
            "rank": rank, "K": int(K), "mean_ex": float(ex.mean()), "mean_tw": float(tw.mean()), "n": int(len(ex))}


def holm(pvals, alpha=0.05):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    ok = [False] * len(pvals)
    for r, i in enumerate(order):
        if pvals[i] <= alpha / (len(pvals) - r):
            ok[i] = True
        else:
            break
    return ok


def windows_val(tok, n, T, seed, kmin=2):
    """hs_v9.windows_with_articles at window offsets k >= kmin only (v9 used k = 0, 1): validation text never used."""
    import glob
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/validation-*"])
    lines = []
    for f in sorted(glob.glob(os.path.join(root, "wikitext-103-raw-v1", "validation-*.parquet"))):
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
    for k in range(kmin, 8):
        for ai in order:
            a = arts[ai]
            if len(a) < 2000:
                continue
            ids = tok(a[:20000], add_special_tokens=False)["input_ids"]
            s = 100 + k * (T + 50)
            if len(ids) >= s + T:
                out.append(ids[s:s + T]); art.append(ai)
                if len(out) == n:
                    return out, art
    return out, art


def status_agg(sts):
    sts = [s for s in sts if s is not None]
    if not sts:
        return "inconclusive"
    if any(s == "fail" for s in sts):
        return "fail"
    if all(s == "pass" for s in sts):
        return "pass"
    return "inconclusive"


# ------------------------------------------------------------------ model wrapper
class Runner:
    def __init__(self):
        cfg = H.CFG[TAG]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        install_freeze_patches()
        self.m = H.Model(cfg["model"], T_W)
        bf16w.MODE = "fp32"
        self.tok = self.m.tok
        self.curT = T_W
        WUf = self.m.W_U.float()                       # cached fp32 unembedding (lm_head is never shuffled)
        self.m.logits = lambda x: F.linear(x.float(), WUf)
        self.sink_on, self.sink_store = False, []
        self.full_layers = [i for i, b in enumerate(self.m.layers) if getattr(b, "layer_type", "") == "full_attention"]
        for i in self.full_layers:
            self.m.layers[i].self_attn.register_forward_hook(self._sink_hook)

    def _sink_hook(self, mod, inp, out):
        if self.sink_on and isinstance(out, tuple) and len(out) > 1 and out[1] is not None:
            self.sink_store.append((mod.layer_idx, out[1].detach().float()))

    def use_T(self, T):
        if T != self.curT:
            self.m._capture_kwargs(T); self.curT = T

    def to_layer(self, ids, l):
        with torch.no_grad():
            return self.m.run(self.m.embed(ids), 0, l + 1)

    def run_from(self, X, l):
        Freeze.k = 0
        out = self.m.run(X, l + 1)
        if Freeze.mode == "use" and not (len(Freeze.store) > 0 and Freeze.k == len(Freeze.store)):
            raise RuntimeError(f"freeze did not bind: used {Freeze.k} of {len(Freeze.store)} stored scales")
        return out


# ------------------------------------------------------------------ rectangle evaluation
def eval_batch(R, seqs, l, rowpos, scorer, keep_rows=(), want_logits=()):
    fs, rows_out, lg_out = [], {}, {}
    m = R.m
    for c0 in range(0, len(seqs), CHUNK):
        Xb = torch.stack(seqs[c0:c0 + CHUNK])
        with torch.no_grad():
            hL = R.run_from(Xb, l)
        mode_, Freeze.mode = Freeze.mode, None                    # the final norm/softmax are never frozen
        for bi in range(hL.shape[0]):
            b = c0 + bi
            rows = hL[bi, rowpos[b]]
            if b in keep_rows:
                rows_out[b] = rows.clone()
            fs.append(scorer(rows, b))
            if b in want_logits:
                lg_out[b] = m.logits(m.text.norm(rows[-1:])).double()[0]
        Freeze.mode = mode_
    return fs, rows_out, lg_out


def rect(R, X, l, pos, arms, rowpos_fn, scorer, freeze=False, gold=None):
    seqs, key = [X], {"base": 0}
    for name, V in arms.items():
        for i in sorted({i for pr in MP for i in pr}):
            xx = X.clone(); xx[pos] += V[:, i]; key[(name, "s", i)] = len(seqs); seqs.append(xx)
        for (i, j) in MP:
            xx = X.clone(); xx[pos] += V[:, i] + V[:, j]; key[(name, "d", i, j)] = len(seqs); seqs.append(xx)
    rp = [rowpos_fn] * len(seqs)
    keep = {0} | {v for k, v in key.items() if isinstance(k, tuple) and k[1] == "s"}
    want = set(range(len(seqs))) if gold is not None else set()
    if freeze:
        Freeze.mode, Freeze.store = "record", []
        with torch.no_grad():
            R.run_from(X[None], l)
        Freeze.mode = "use"
    try:
        fs, rows, lgs = eval_batch(R, seqs, l, rp, scorer, keep, want)
    finally:
        Freeze.mode, Freeze.store = None, []
    out, ranks = {"_f0": fs[0]}, {}
    for name in arms:
        for (i, j) in MP:
            ku, kw, kd = key[(name, "s", i)], key[(name, "s", j)], key[(name, "d", i, j)]
            f0, fu, fw_, fd = fs[0], fs[ku], fs[kw], fs[kd]
            fadd = scorer(rows[ku] + rows[kw] - rows[0], None)
            out[f"{name}|{i}{j}"] = (fd - fu - fw_ + f0, fd - fadd, fadd - fu - fw_ + f0, fu - f0, fw_ - f0)
            if gold is not None:
                v = lgs[kd] - lgs[ku] - lgs[kw] + lgs[0]
                ranks[f"{name}|{i}{j}"] = int((v > v[gold]).sum())
    return out, ranks


def arm_stats(recs, names_tw, comp, clusters):
    ci_ = {"I": 0, "I_trunk": 1}[comp]
    ex = [np.mean([abs(r["pairs"][f"H|{i}{j}"][ci_]) for (i, j) in MP]) for r in recs]
    tw = [[np.mean([abs(r["pairs"][f"{t}|{i}{j}"][ci_]) for (i, j) in MP]) for t in names_tw] for r in recs]
    return two_way_T1(ex, tw, clusters)


def log_ratio_vs(recs, other, comp, clusters):
    ci_ = {"I": 0, "I_trunk": 1}[comp]
    ex = np.array([np.mean([abs(r["pairs"][f"H|{i}{j}"][ci_]) for (i, j) in MP]) for r in recs])
    ee = np.array([np.mean([abs(r["pairs"][f"{other}|{i}{j}"][ci_]) for (i, j) in MP]) for r in recs])
    v = np.log((ex + 1e-12) / (ee + 1e-12))
    return {"mean": float(v.mean()), "ci95": boot_ci(v, clusters)}


# ------------------------------------------------------------------ main
def main():
    log(f"start {TAG}, layers {LAYERS}, budget {MINUTES} min")
    res = {"tag": TAG, "layers": LAYERS, "stages": {}, "errors": {}, "verdict": {}}
    raw = {}

    def save():
        tmp = str(OUTF) + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, OUTF)
        torch.save(raw, str(RAWF) + ".tmp"); os.replace(str(RAWF) + ".tmp", RAWF)

    # bank integrity (hashes frozen in v10_banks/MANIFEST.json)
    MAN = json.load(open(H.HERE / "v10_banks" / "MANIFEST.json"))
    for b in BANKS + BACKUPS + [NULLB]:
        h = hashlib.sha256(open(H.HERE / "v10_banks" / f"{b}.jsonl", "rb").read()).hexdigest()
        assert h == MAN["banks"][b]["sha256"], f"bank {b} hash mismatch"
    T2 = torch.load(H.OUT / f"hspace2_{TAG}.pt", map_location="cpu", weights_only=False)
    INP = torch.load(H.OUT / f"v10_inputs_{TAG}.pt", map_location="cpu", weights_only=False)
    R = Runner()
    m, tok = R.m, R.tok
    d = m.d
    cache, cache2 = {}, {}

    def cls(t):
        if t not in cache:
            cache[t] = klass(tok.decode([int(t)]))
        return cache[t]
    import hs_v10a_classes as A10

    def cls2(t):
        if t not in cache2:
            cache2[t] = A10.klass2(tok.decode([int(t)]))
        return cache2[t]

    # ================= S0
    st0 = time.time()
    S0 = {}
    res["stages"]["S0"] = S0
    covw = H.wikitext_windows(tok, "train", N_COV, T_W, seed=11)
    lay_sorted = sorted(LAYERS)
    classes = ["punctuation", "word", "function", "content", "digit", "whitespace", "all"]
    acc = {l: {c: [0, torch.zeros(d, dtype=torch.float64, device="cuda"), torch.zeros(d, d, dtype=torch.float64, device="cuda")]
               for c in classes} for l in LAYERS}
    for b0 in range(0, len(covw), 8):
        ids = covw[b0:b0 + 8].cuda()
        tk = ids[:, H.SKIP:T_W - 1].reshape(-1).tolist()
        c1 = [cls(t) for t in tk]; c2 = [cls2(t) for t in tk]
        sel = {}
        for c in classes:
            if c == "all":
                ix = list(range(len(tk)))
            elif c in ("punctuation", "word", "digit", "whitespace"):
                ix = [i for i in range(len(tk)) if c1[i] == c]
            else:
                ix = [i for i in range(len(tk)) if c2[i] == c]
            sel[c] = torch.tensor(ix, dtype=torch.long, device="cuda")
        with torch.no_grad():
            h = m.embed(ids); prev = 0
            for l in lay_sorted:
                h = m.run(h, prev, l + 1); prev = l + 1
                Xl = h[:, H.SKIP:T_W - 1].reshape(-1, d).double()
                for c in classes:
                    if len(sel[c]):
                        Xs = Xl[sel[c]]
                        acc[l][c][0] += len(Xs); acc[l][c][1] += Xs.sum(0); acc[l][c][2] += Xs.T @ Xs
    log(f"S0: class moments from {len(covw)} windows")
    ctrl, m2_kill, quals = {}, {}, {}
    for l in LAYERS:
        Mu, Sig, nrows = {}, {}, {}
        for c in classes:
            n_, s1, s2 = acc[l][c]
            if n_ > 10:
                nrows[c] = int(n_)
                mu_c = s1 / n_
                S_ = (s2 / n_ - torch.outer(mu_c, mu_c)) * n_ / (n_ - 1)
                S_ = 0.95 * S_ + 0.05 * torch.trace(S_) / d * torch.eye(d, dtype=S_.dtype, device=S_.device)
                Mu[c], Sig[c] = mu_c.cpu(), S_.cpu()
        del acc[l]
        if f"Sh{l}" in T2:
            Sh = T2[f"Sh{l}"].double(); mu = T2[f"mu{l}"].double()
        else:                                                       # small-mode dev layers without a run-2 frame
            evs, Vs = eigh_desc(Sig["all"])
            Sh = Vs @ torch.diag(evs.clamp_min(1e-12).sqrt()) @ Vs.T; mu = Mu["all"]
        Shi = torch.linalg.inv(Sh.cuda()).cpu()
        Hw = INP[f"H_{l}"].double()
        L0 = {"n_rows": nrows}
        Cw = Shi @ Sig["punctuation"] @ Shi
        ev, V = eigh_desc(Cw)
        mt = Shi @ (Mu["punctuation"] - mu); mt = mt / mt.norm()
        evg, Pg = eigh_desc(Sh)
        refs = twins_w(Hw, Pg, 200, 7000 + l)

        def geo(Q):
            Q = orth(Q)
            return (float((V[:, :5].T @ Q).pow(2).sum() / 5), float((V[:, :25].T @ Q).pow(2).sum() / 5), float((Q.T @ mt).pow(2).sum()))
        gH = geo(Hw); gR = np.array([geo(Q) for Q in refs])
        kill_ov = bool(gH[1] >= 0.25 and gH[1] > gR[:, 1].max())
        kill_me = bool(gH[2] >= 0.25 and gH[2] > gR[:, 2].max())
        m2_kill[l] = kill_ov or kill_me
        L0["M2"] = {"H_ov_punctPC5": gH[0], "H_ov_punctPC25": gH[1], "H_mean_energy": gH[2],
                    "ref_q95": [float(np.quantile(gR[:, k], 0.95)) for k in range(3)], "ref_max": [float(gR[:, k].max()) for k in range(3)],
                    "KILL": m2_kill[l], "moment_aligned": bool(gH[1] > np.quantile(gR[:, 1], 0.95) or gH[2] > np.quantile(gR[:, 2], 0.95))}
        # ---- controls
        NC = 4000 if FULL else 200
        gq = torch.Generator().manual_seed(9100 + l)
        Ssg = (torch.randint(0, 2, (d, NC), generator=gq).float() * 2 - 1)
        Cc_ = (V.T @ Hw).float()
        JP = (INP[f"J_{l}"].float().cuda() @ Sh.float().cuda()) @ V.float().cuda()

        def jen_batch(Sb):
            Z = torch.einsum("de,enk->dnk", JP, (Sb.cuda()[:, :, None] * Cc_.cuda()[:, None, :]))
            e1 = Z.pow(2).sum(0)
            e2 = torch.stack([(Z[:, :, i] + Z[:, :, j]).pow(2).sum(0) for (i, j) in MP], 1)
            return torch.cat([e1, e2], 1).cpu()
        eH = jen_batch(torch.ones(d, 1))[0]
        E = torch.cat([jen_batch(Ssg[:, c0:c0 + 250]) for c0 in range(0, NC, 250)])
        devs = ((E - eH).abs() / eH).max(1).values.numpy()

        def mkw(j):
            return V @ (Ssg[:, j:j + 1].double() * (V.T @ Hw))
        plain_w = [mkw(j) for j in range(N_TW)]
        order = [int(i) for i in (np.argsort(devs[N_TW:]) + N_TW)][:N_TWJ]
        jm_w = [mkw(j) for j in order]
        L0["J_match_maxreldev"] = [float(devs[j]) for j in order]
        L0["J_energy_ratio_plain_twins_vs_H"] = [float((E[j] / eH).mean()) for j in range(N_TW)]
        del JP
        evw, Vw = eigh_desc(Shi @ Sig["word"] @ Shi)
        word_w = twins_w(Hw, Vw, N_TWW, 9200 + l)
        glob_w = [INP[f"G_{l}"][j].double() for j in range(N_TG)] if f"G_{l}" in INP else twins_w(Hw, Pg, N_TG, 9090 + l)
        J25 = INP[f"J25w_{l}"] if f"J25w_{l}" in INP else T2[f"J25w_{l}"]
        J5w = orth(J25.double()[:, :5])
        PC5w = V[:, :5]

        def diag(Q):                     # punctuation-mean energy and raw 1-sigma column norms (not matched by the twins)
            return float((Q.T @ mt).pow(2).sum()), float((Sh @ Q).norm(dim=0).mean())
        dH = diag(Hw); dT = [diag(Q) for Q in plain_w + jm_w]
        me = np.array([x[0] for x in dT]); rn = np.array([x[1] for x in dT])
        L0["twin_diag"] = {"H_mean_energy": dH[0], "H_raw_norm": dH[1], "tw_mean_energy": me.tolist(), "tw_raw_norm": rn.tolist()}
        quals[l] = {"mean_norm_aligned": bool(dH[0] > me.max() or dH[1] > rn.max()), "moment_aligned": L0["M2"]["moment_aligned"]}
        zz = np.abs(me - dH[0]) / (me.std() + 1e-12) + np.abs(rn - dH[1]) / (rn.std() + 1e-12)
        L0["closest8"] = [TW[int(i)] for i in np.argsort(zz)[:8]]
        C = {"H": Hw, "J5": J5w, "PC5": PC5w}
        for j in range(N_TW):
            C[f"tw{j}"] = plain_w[j]
        for j in range(N_TWJ):
            C[f"tj{j}"] = jm_w[j]
        for j in range(N_TWW):
            C[f"tw_w{j}"] = word_w[j]
        for j in range(N_TG):
            C[f"tg{j}"] = glob_w[j]
        ctrl[l] = {k: (Sh @ v).float().cuda() for k, v in C.items()}               # raw 1-sigma moves
        ctrl[l]["_Sh"] = Sh.float().cuda(); ctrl[l]["_Shi"] = Shi.float().cuda()
        S0[str(l)] = L0
        log(f"S0 L{l}: rows {nrows} | M2 ov25 {gH[1]:.3f} (ref max {gR[:, 1].max():.3f}) mean-energy {gH[2]:.3f} "
            f"(ref max {gR[:, 2].max():.3f}) -> {'KILL' if m2_kill[l] else 'ok'} | J-match maxdev {max(L0['J_match_maxreldev']):.2f} | "
            f"H mean-energy {dH[0]:.4f} vs twins max {me.max():.4f}, raw norm {dH[1]:.3f} vs twins max {rn.max():.3f}")
    res["verdict"]["M2_kill"] = {str(l): m2_kill[l] for l in LAYERS}
    save()
    live = [l for l in LAYERS if not m2_kill[l]]
    log(f"S0 done ({(time.time()-st0)/60:.1f} min); live layers {live}")
    plan, recW = [], {l: [] for l in live}
    SW = {}
    Gst, prepped, gated = {}, {}, []
    ST = {}
    SR = {}

    # ================= W (wikitext)
    if live:
        try:
            stw = time.time()
            dl_w = stw + 60 * BUDGET["W"]
            cand, art = windows_val(tok, N_W * 3, T_W, seed=10, kmin=2)
            rng = random.Random(10)
            for w, a in zip(cand, art):
                lo, hi = H.SKIP + 8, T_W - NEXT - 4
                P_ = [i for i in range(lo, hi) if cls(int(w[i])) == "punctuation"]
                Q_ = [i for i in range(lo, hi) if cls(int(w[i])) == "word"]
                if P_ and Q_:
                    plan.append((torch.tensor(w), a, rng.choice(P_), rng.choice(Q_)))
                if len(plan) == N_W:
                    break
            log(f"W: {len(plan)} windows from {len(set(p_[1] for p_ in plan))} articles (validation offsets k>=2)")
            if FULL:
                assert len(plan) == N_W, len(plan)
            R.use_T(T_W)
            for wi, (w, a, p, q) in enumerate(plan):
                if time.time() > dl_w or time.time() > DEADLINE - 60 * BUDGET["R"]:
                    log(f"W time budget reached after {wi} windows"); break
                ids = w[None].cuda(); tgt = ids[0]
                Xs = {}
                with torch.no_grad():
                    h = m.embed(ids); prev = 0
                    R.sink_on, R.sink_store = True, []
                    for l in sorted(live):
                        h = m.run(h, prev, l + 1); prev = l + 1
                        Xs[l] = h.clone()
                    m.run(h, prev)
                    R.sink_on = False
                for l in live:
                    X = Xs[l][0]; Cl = ctrl[l]

                    def scorer_at(pos):
                        def sc(rows_, b):
                            lg = m.logits(m.text.norm(rows_)).double()
                            return float(torch.log_softmax(lg, -1).gather(-1, tgt[pos + 1:pos + 1 + NEXT, None]).sum())
                        return sc
                    sinks = [A_[0, :, p + 1:, p].mean().item() for li, A_ in R.sink_store if li > l]
                    rec = {"article": int(a), "p": int(p), "q": int(q), "sink": float(np.mean(sinks)) if sinks else None}
                    arms = {"H": Cl["H"], "J5": Cl["J5"], "PC5": Cl["PC5"], **{t: Cl[t] for t in TW}, **{f"tg{j}": Cl[f"tg{j}"] for j in range(N_TG)}}
                    rec["pairs"], _ = rect(R, X, l, p, arms, list(range(p, p + NEXT)), scorer_at(p))
                    fz = {"H": Cl["H"], **{t: Cl[t] for t in TWP}}
                    rec["freeze"], _ = rect(R, X, l, p, fz, list(range(p, p + NEXT)), scorer_at(p), freeze=True)
                    rec["freeze_identity_err"] = abs(rec["freeze"]["_f0"] - rec["pairs"]["_f0"])
                    rec["freeze_identity_tol"] = 1e-4 * max(1.0, abs(rec["pairs"]["_f0"]))
                    wa = {"H": Cl["H"], **{f"tw_w{j}": Cl[f"tw_w{j}"] for j in range(N_TWW)}}
                    rec["word"], _ = rect(R, X, l, q, wa, list(range(q, q + NEXT)), scorer_at(q))
                    recW[l].append(rec)
                if (wi + 1) % 4 == 0 or wi == len(plan) - 1:
                    raw["W"] = recW; save()
                    log(f"W window {wi+1}/{len(plan)} ({(time.time()-stw)/(wi+1):.1f}s/window, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)")
            raw["W"] = recW
            for l in live:
                rr = recW[l]; cl = [r["article"] for r in rr]
                if len(rr) < MIN_W:
                    SW[str(l)] = {"n": len(rr), "M3": "inconclusive", "note": "too few windows"}; continue
                Lw = {"n": len(rr)}
                for comp in ("I", "I_trunk"):
                    Lw[comp] = arm_stats(rr, TW, comp, cl)
                    Lw[comp + "_plain8"] = arm_stats(rr, TWP, comp, cl)
                    Lw[comp + "_Jmatched8"] = arm_stats(rr, TW[N_TW:], comp, cl)
                    Lw[comp + "_global8_v9"] = arm_stats(rr, [f"tg{j}" for j in range(N_TG)], comp, cl)
                    Lw[comp + "_closest8"] = arm_stats(rr, S0[str(l)]["closest8"], comp, cl)
                    Lw[comp + "_vs_J5"] = log_ratio_vs(rr, "J5", comp, cl)
                    Lw[comp + "_vs_PC5"] = log_ratio_vs(rr, "PC5", comp, cl)
                    Lw[comp + "_freeze"] = arm_stats([{"pairs": r["freeze"]} for r in rr], TWP, comp, cl)
                    Lw[comp + "_word"] = arm_stats([{"pairs": r["word"]} for r in rr], [f"tw_w{j}" for j in range(N_TWW)], comp, cl)
                errs = [r["freeze_identity_err"] - r["freeze_identity_tol"] for r in rr]
                Lw["freeze_identity_err_max"] = float(max(r["freeze_identity_err"] for r in rr))
                Lw["freeze_valid"] = bool(max(errs) <= 0)
                base = Lw["I_trunk_plain8"]["T1"]
                Lw["freeze_retained"] = float(Lw["I_trunk_freeze"]["T1"] / base) if base > LOG2 else None
                sk = np.array([r["sink"] if r["sink"] is not None else np.nan for r in rr])
                if np.isfinite(sk).sum() >= 9:
                    lo_ = sk <= np.nanquantile(sk, 1 / 3)
                    sub = [r for r, k in zip(rr, lo_) if k]
                    Lw["I_lowsink"] = arm_stats(sub, TW, "I", [r["article"] for r in sub])
                Lw["sink_scope"] = bool("I_lowsink" in Lw and Lw["I_lowsink"]["T1"] < 0.5 * Lw["I"]["T1"])
                Lw["median_twin_absI"] = float(np.median([np.mean([abs(r["pairs"][f"{t}|{i}{j}"][0]) for t in TW for (i, j) in MP]) for r in rr]))

                def strong(s):
                    return s["T1"] > LOG2 and s["ci95"][0] > 0 and s["rank"] == 1 and s["p_PI"] < 0.05
                if not Lw["freeze_valid"]:
                    st_ = "inconclusive"
                elif strong(Lw["I"]) and strong(Lw["I_trunk"]) and Lw["freeze_retained"] is not None and Lw["freeze_retained"] >= 0.5:
                    st_ = "pass"
                else:
                    st_ = "fail"
                Lw["M3"] = st_
                Lw["M8a_position_general"] = bool(Lw["I_word"]["T1"] > LOG2 and Lw["I_word"]["ci95"][0] > 0 and Lw["I_word"]["rank"] == 1)
                SW[str(l)] = Lw
                log(f"W L{l} (n={len(rr)}): T1 I {Lw['I']['T1']:+.2f} {[round(x, 2) for x in Lw['I']['ci95']]} rank {Lw['I']['rank']}/17 "
                    f"p_PI {Lw['I']['p_PI']:.3f} | I_trunk {Lw['I_trunk']['T1']:+.2f} | global(v9) {Lw['I_global8_v9']['T1']:+.2f} | "
                    f"freeze kept {Lw['freeze_retained']} valid {Lw['freeze_valid']} | J5 {Lw['I_vs_J5']['mean']:+.2f} PC5 {Lw['I_vs_PC5']['mean']:+.2f} | "
                    f"word {Lw['I_word']['T1']:+.2f} | sink-scope {Lw['sink_scope']} | M3 {st_}")
        except Exception:  # noqa: BLE001
            res["errors"]["W"] = traceback.format_exc(); log("W ERROR\n" + res["errors"]["W"])
        res["stages"]["W"] = SW
        save()

    # ================= G (task gates)
    def f_scorer(Xt, Yt):
        def sc(rows_, b):
            lp = torch.log_softmax(m.logits(m.text.norm(rows_[-1:])).double()[0], -1)
            return float(lp[Xt] - lp[Yt])
        return sc
    if live:
        try:
            R.use_T(T_T)
            banks = {}
            for bname in BANKS + [NULLB] + BACKUPS:
                items = [json.loads(x) for x in open(H.HERE / "v10_banks" / f"{bname}.jsonl", encoding="utf-8") if x.strip()]
                random.Random(42).shuffle(items)
                banks[bname] = items[:(N_NULL if bname == NULLB else N_ITEMS)]

            def prep(item):
                P = {}
                for k in ("00", "01", "10", "11"):
                    ids = tok(item["prompts"][k], add_special_tokens=False)["input_ids"]
                    if len(ids) > T_T:
                        return None
                    n = len(ids)
                    site = [i for i in range(n - 1) if cls(ids[i]) == "punctuation"]
                    if not site:
                        return None
                    ns = item["natural_site"][k] if isinstance(item["natural_site"], dict) else item["natural_site"]
                    pre = item["prompts"][k][:item["prompts"][k].index(ns) + len(ns)]
                    P[k] = {"ids": ids + [ids[-1]] * (T_T - n), "ans": n - 1, "site": site[-1],
                            "nat": len(tok(pre, add_special_tokens=False)["input_ids"]) - 1}
                X_ = tok(item["X"], add_special_tokens=False)["input_ids"][0]
                Y_ = tok(item["Y"], add_special_tokens=False)["input_ids"][0]
                if X_ == Y_:
                    return None
                return {"P": P, "X": X_, "Y": Y_, "gold11": X_ if item["gold"]["11"] == "X" else Y_, "cluster": item["cluster"], "item": item["item"]}
            for bname, items in banks.items():
                stg = time.time()
                recs = []
                for it in items:
                    if time.time() > stg + 60 * BUDGET["G"] / 5:
                        break
                    pr = prep(it)
                    if pr is None:
                        continue
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
                    g_ = {"DiD": fb["11"] - fb["10"] - fb["01"] + fb["00"], "mA": fb["10"] - fb["00"], "mB": fb["01"] - fb["00"],
                          "cluster": pr["cluster"], "lay": {}}
                    for l in live:
                        P0 = pr["P"]["00"]; s0 = P0["site"]
                        dA = Xl[l]["10"][pr["P"]["10"]["site"]] - Xl[l]["00"][s0]
                        dB = Xl[l]["01"][pr["P"]["01"]["site"]] - Xl[l]["00"][s0]
                        dAB = Xl[l]["11"][pr["P"]["11"]["site"]] - Xl[l]["00"][s0]
                        seqs = []
                        for dv in (dA, dB, dAB):
                            xx = Xl[l]["00"].clone(); xx[s0] += dv; seqs.append(xx)
                        fs, _, _ = eval_batch(R, seqs, l, [[P0["ans"]]] * 3, f_scorer(pr["X"], pr["Y"]))
                        g_["lay"][str(l)] = {"dA": fs[0] - fb["00"], "dB": fs[1] - fb["00"], "DiD_site": fs[2] - fs[0] - fs[1] + fb["00"]}
                    recs.append(g_)
                    prepped.setdefault(bname, []).append((pr, Xl, fb))
                if not recs:
                    continue
                cl = [r["cluster"] for r in recs]
                dd = np.array([r["DiD"] for r in recs])
                Gb = {"n": len(recs), "DiD_mean": float(dd.mean()), "DiD_ci": boot_ci(dd, cl), "DiD_abs_mean": float(np.abs(dd).mean()),
                      "mA_ci": boot_ci([r["mA"] for r in recs], cl), "mB_ci": boot_ci([r["mB"] for r in recs], cl)}
                Gb["gated"] = bool(Gb["DiD_ci"][0] > 0 or Gb["DiD_ci"][1] < 0) if bname != NULLB else None
                for l in live:
                    ds = [r["lay"][str(l)]["DiD_site"] for r in recs]
                    Gb[f"site_ratio_L{l}"] = float(np.mean(ds) / dd.mean()) if abs(dd.mean()) > 1e-12 else None
                    rA = float(np.mean([r["lay"][str(l)]["dA"] for r in recs]) / (np.mean([r["mA"] for r in recs]) or 1e-12))
                    rB = float(np.mean([r["lay"][str(l)]["dB"] for r in recs]) / (np.mean([r["mB"] for r in recs]) or 1e-12))
                    Gb[f"recov_A_L{l}"], Gb[f"recov_B_L{l}"] = rA, rB
                    needA = Gb["mA_ci"][0] > 0 or Gb["mA_ci"][1] < 0
                    needB = Gb["mB_ci"][0] > 0 or Gb["mB_ci"][1] < 0
                    Gb[f"site_gate_L{l}"] = bool(Gb["gated"] and Gb[f"site_ratio_L{l}"] is not None and Gb[f"site_ratio_L{l}"] >= 0.3
                                                 and (rA >= 0.3 or not needA) and (rB >= 0.3 or not needB))
                Gst[bname] = Gb
                log(f"G {bname}: n {len(recs)} DiD {Gb['DiD_mean']:+.2f} CI {[round(x, 2) for x in Gb['DiD_ci']]} gated {Gb['gated']} | "
                    + " ".join(f"L{l} site {Gb[f'site_ratio_L{l}']} recA {Gb[f'recov_A_L{l}']:.2f} recB {Gb[f'recov_B_L{l}']:.2f} "
                               f"site-gate {Gb[f'site_gate_L{l}']}" for l in live))
            gated = [b for b in BANKS if Gst.get(b, {}).get("gated")]
            for b in BACKUPS:
                if len(gated) < 3 and Gst.get(b, {}).get("gated"):
                    gated.append(b)
        except Exception:  # noqa: BLE001
            res["errors"]["G"] = traceback.format_exc(); log("G ERROR\n" + res["errors"]["G"])
        res["stages"]["G"] = Gst
        res["verdict"]["gated_contexts"] = gated
        save()

    # ================= T (task arms)
    recT = {}
    if live and prepped:
        try:
            R.use_T(T_T)
            stt = time.time()
            dl_t = stt + 60 * BUDGET["T"]
            run_ctx = gated + ([NULLB] if NULLB in prepped else [])
            per_ctx = 60 * BUDGET["T"] / max(1, len(run_ctx))
            for bname in run_ctx:
                t_ctx = time.time()
                recT[bname] = {l: [] for l in live}
                for ii, (pr, Xl, fb0) in enumerate(prepped[bname]):
                    if time.time() > min(dl_t, t_ctx + per_ctx) or time.time() > DEADLINE - 60 * BUDGET["R"]:
                        log(f"T {bname}: budget reached after {ii} items"); break
                    Xt, Yt = pr["X"], pr["Y"]
                    for l in live:
                        Cl = ctrl[l]; Sh_, Shi_ = Cl["_Sh"], Cl["_Shi"]
                        rec = {"cluster": pr["cluster"], "item": pr["item"], "f_base": fb0}
                        P1 = pr["P"]["11"]
                        arms = {"H": Cl["H"], **{t: Cl[t] for t in TW}}
                        rec["transfer"], rec["gold_rank"] = rect(R, Xl[l]["11"], l, P1["site"], arms, [P1["ans"]], f_scorer(Xt, Yt), gold=pr["gold11"])
                        P0 = pr["P"]["00"]; s0 = P0["site"]; X0 = Xl[l]["00"]
                        dA = Xl[l]["10"][pr["P"]["10"]["site"]] - X0[s0]
                        dB = Xl[l]["01"][pr["P"]["01"]["site"]] - X0[s0]
                        dAB = Xl[l]["11"][P1["site"]] - X0[s0]
                        comps = ["H", "J5"] + TW
                        Qw = {k: orth(Shi_ @ Cl[k]) for k in comps}

                        def proj(dv, Q):
                            return Sh_ @ (Q @ (Q.T @ (Shi_ @ dv)))
                        moves = [torch.zeros_like(dA), dA, dB, dA + dB, dAB]
                        for k in comps:
                            pa, pb = proj(dA, Qw[k]), proj(dB, Qw[k])
                            moves += [dA - pa, dB - pb, dA - pa + dB - pb, pa, pb, pa + pb]
                        seqs = []
                        for mv in moves:
                            xx = X0.clone(); xx[s0] += mv; seqs.append(xx)
                        fs, _, _ = eval_batch(R, seqs, l, [[P0["ans"]]] * len(seqs), f_scorer(Xt, Yt))
                        f00, fA, fB, fAB_, fAB = fs[:5]
                        I = fAB_ - fA - fB + f00
                        rec["route"] = {"I": I, "DiD_site": fAB - fA - fB + f00, "S": {}}
                        x = X0.clone()[None].requires_grad_(True)
                        with torch.enable_grad():
                            hL = R.run_from(x, l)
                            lp = torch.log_softmax(m.logits(m.text.norm(hL[0, P0["ans"]:P0["ans"] + 1])).double()[0], -1)
                            (gr,) = torch.autograd.grad(lp[Xt] - lp[Yt], x)
                        gs = gr[0, s0].detach()
                        rec["route"]["atp_err"] = fAB_ - f00 - float(gs @ (dA + dB))
                        wa_, wb_ = Shi_ @ dA, Shi_ @ dB
                        for n_, k in enumerate(comps):
                            b = 5 + 6 * n_; Q = Qw[k]
                            pdA, pdB = dA - proj(dA, Q), dB - proj(dB, Q)
                            rec["route"]["S"][k] = {"I_del": fs[b + 2] - fs[b] - fs[b + 1] + f00, "I_keep": fs[b + 5] - fs[b + 3] - fs[b + 4] + f00,
                                                    "eA": float((Q.T @ wa_).pow(2).sum() / wa_.pow(2).sum()),
                                                    "eB": float((Q.T @ wb_).pow(2).sum() / wb_.pow(2).sum()),
                                                    "atp_err_del": fs[b + 2] - f00 - float(gs @ (pdA + pdB))}
                        rec["ablate"] = {}
                        abl_S = ["H", "J5"] + TWP
                        seqs, meta = [], []
                        for k in abl_S:
                            Q = Qw[k]
                            coords = {c: Q.T @ (Shi_ @ Xl[l][c][pr["P"][c]["site"]]) for c in ("00", "01", "10", "11")}
                            mean_c = sum(coords.values()) / 4
                            dose = float(sum((coords[c] - mean_c).pow(2).sum() for c in coords) / 4)
                            for c in ("00", "01", "10", "11"):
                                xx = Xl[l][c].clone(); xx[pr["P"][c]["site"]] += Sh_ @ (Q @ (mean_c - coords[c]))
                                seqs.append(xx); meta.append((k, c, dose))
                        fs2, _, _ = eval_batch(R, seqs, l, [[pr["P"][c]["ans"]] for (_, c, _) in meta], f_scorer(Xt, Yt))
                        for n_, k in enumerate(abl_S):
                            rec["ablate"][k] = {"f": dict(zip(("00", "01", "10", "11"), fs2[4 * n_:4 * n_ + 4])), "dose": meta[4 * n_][2]}
                        Pn0 = P0["nat"]
                        dAn = Xl[l]["10"][pr["P"]["10"]["nat"]] - X0[Pn0]
                        dBn = Xl[l]["01"][pr["P"]["01"]["nat"]] - X0[Pn0]
                        nat_S = ["H", "J5"] + TWP
                        moves = [torch.zeros_like(dAn), dAn, dBn, dAn + dBn]
                        for k in nat_S:
                            pa, pb = proj(dAn, Qw[k]), proj(dBn, Qw[k])
                            moves += [dAn - pa, dBn - pb, dAn - pa + dBn - pb]
                        seqs = []
                        for mv in moves:
                            xx = X0.clone(); xx[Pn0] += mv; seqs.append(xx)
                        fs3, _, _ = eval_batch(R, seqs, l, [[P0["ans"]]] * len(seqs), f_scorer(Xt, Yt))
                        rec["nat"] = {"I": fs3[3] - fs3[1] - fs3[2] + fs3[0],
                                      "S": {k: fs3[4 + 3 * n_ + 2] - fs3[4 + 3 * n_] - fs3[4 + 3 * n_ + 1] + fs3[0] for n_, k in enumerate(nat_S)}}
                        recT[bname][l].append(rec)
                    if (ii + 1) % 4 == 0:
                        raw["T"] = recT; save()
                        log(f"T {bname}: item {ii+1}/{len(prepped[bname])} ({(time.time()-t_ctx)/(ii+1):.1f}s/item)")
            raw["T"] = recT
            save()
            # ---- task statistics
            den_tost = float(np.mean([Gst[b]["DiD_abs_mean"] for b in gated])) if gated else None
            for bname in run_ctx:
                ST[bname] = {}
                for l in live:
                    rr = recT[bname][l]
                    if len(rr) < MIN_T:
                        ST[bname][str(l)] = {"n": len(rr), "M4": "inconclusive", "M5": "inconclusive"}; continue
                    cl = [r["cluster"] for r in rr]
                    Lt = {"n": len(rr)}
                    tr = [{"pairs": r["transfer"]} for r in rr]
                    Lt["transfer_I"] = arm_stats(tr, TW, "I", cl)
                    Lt["transfer_Itrunk"] = arm_stats(tr, TW, "I_trunk", cl)
                    diff = np.array([np.mean([r["gold_rank"][f"H|{i}{j}"] for (i, j) in MP])
                                     - np.median([np.mean([r["gold_rank"][f"{t}|{i}{j}"] for (i, j) in MP]) for t in TW]) for r in rr])
                    ucl = sorted(set(cl))
                    cm = np.array([diff[[k for k, c in enumerate(cl) if c == u]].mean() for u in ucl])
                    Lt["M7"] = {"median_rank_diff": float(np.median(diff)), "n_clusters": len(ucl),
                                "p_wilcoxon_less": float(wilcoxon(cm, alternative="less").pvalue) if (len(cm) >= 6 and np.any(cm != 0)) else 1.0}
                    I_ = np.array([r["route"]["I"] for r in rr]); Ds = np.array([r["route"]["DiD_site"] for r in rr])
                    Lt["DiD_site_ci"] = boot_ci(Ds, cl)
                    Lt["curv_share"] = float(I_.mean() / Ds.mean()) if abs(Ds.mean()) > 1e-12 else None
                    Lt["curv_share_ci"] = cboot(ratio_of_means(I_, Ds), cl)
                    Dall = {}
                    for k in ["H", "J5"] + TW:
                        rem = np.array([r["route"]["I"] - r["route"]["S"][k]["I_del"] for r in rr])
                        kept = np.array([r["route"]["S"][k]["I_keep"] for r in rr])
                        e = float(np.mean([(r["route"]["S"][k]["eA"] + r["route"]["S"][k]["eB"]) / 2 for r in rr]))
                        rf = float(rem.mean() / I_.mean()) if abs(I_.mean()) > 1e-12 else float("nan")
                        atp_den = float(np.mean([r["route"]["atp_err"] for r in rr]))
                        atp = np.array([r["route"]["atp_err"] - r["route"]["S"][k]["atp_err_del"] for r in rr])
                        Dall[k] = {"removed_frac": rf, "e": e, "D": float(rf / e) if e > 0 else float("nan"),
                                   "kept_frac": float(kept.mean() / I_.mean()) if abs(I_.mean()) > 1e-12 else float("nan"),
                                   "atp_err_removed": float(atp.mean() / atp_den) if abs(atp_den) > 1e-12 else None}
                    Dall["H"]["removed_frac_ci"] = cboot(ratio_of_means([r["route"]["I"] - r["route"]["S"]["H"]["I_del"] for r in rr], I_), cl)
                    Dtw = np.array([Dall[t]["D"] for t in TW])
                    med, mad = float(np.nanmedian(Dtw)), float(np.nanmedian(np.abs(Dtw - np.nanmedian(Dtw))))
                    Lt["route"] = Dall
                    Lt["D_robust_z"] = float((Dall["H"]["D"] - med) / (1.4826 * mad)) if mad > 0 else (float("inf") if Dall["H"]["D"] > med else 0.0)
                    Lt["M4_rank_D"] = int(1 + sum(Dall[k]["D"] > Dall["H"]["D"] for k in ["J5"] + TW))
                    Lt["site_gated"] = bool(Gst.get(bname, {}).get(f"site_gate_L{l}"))
                    evaluable = Lt["site_gated"] and (Lt["DiD_site_ci"][0] > 0 or Lt["DiD_site_ci"][1] < 0)
                    if not evaluable:
                        Lt["M4"] = "inconclusive"
                    else:
                        ok4 = (Lt["curv_share"] is not None and Lt["curv_share"] >= 0.25 and Lt["curv_share_ci"][0] > 0
                               and Dall["H"]["removed_frac"] >= 0.10 and Dall["H"]["removed_frac_ci"][0] > 0
                               and Lt["D_robust_z"] >= 3 and Lt["M4_rank_D"] == 1)
                        Lt["M4"] = "pass" if ok4 else "fail"

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
                    for k in ["H", "J5"] + TWP:
                        il, ml = losses(k)
                        M5[k] = {"int_loss": il, "main_loss": ml, "delta": il - ml, "phi": il / ml if abs(ml) > 1e-9 else None,
                                 "dose": float(np.mean([r["ablate"][k]["dose"] for r in rr]))}
                    Lt["M5_detail"] = M5
                    Lt["M5_int_loss_diff_ci"] = cboot(lambda ix: losses("H", ix)[0] - np.mean([losses(t, ix)[0] for t in TWP]), cl, B=400)
                    if not Lt["site_gated"]:
                        Lt["M5"] = "inconclusive"
                    else:
                        ok5 = (M5["H"]["int_loss"] > 0 and M5["H"]["delta"] > max(M5[t]["delta"] for t in TWP)
                               and M5["H"]["delta"] > M5["J5"]["delta"] and Lt["M5_int_loss_diff_ci"][0] > 0)
                        Lt["M5"] = "pass" if ok5 else "fail"
                    Lt["M5_double_dissociation"] = bool(M5["H"]["int_loss"] > M5["J5"]["int_loss"] and M5["J5"]["main_loss"] > M5["H"]["main_loss"])
                    In = np.array([r["nat"]["I"] for r in rr])
                    Lt["nat_removed_frac"] = {k: (float(np.mean([r["nat"]["I"] - r["nat"]["S"][k] for r in rr]) / In.mean()) if abs(In.mean()) > 1e-12 else None)
                                              for k in ["H", "J5"] + TWP}
                    if bname == NULLB and den_tost:
                        dif = np.array([(r["route"]["I"] - r["route"]["S"]["H"]["I_del"])
                                        - np.mean([r["route"]["I"] - r["route"]["S"][t]["I_del"] for t in TW]) for r in rr]) / den_tost
                        Lt["tost_diff"] = float(dif.mean()); Lt["tost_ci90"] = cboot(lambda ix: dif[ix].mean(), cl, q=(0.05, 0.95))
                        Lt["tost_equivalent"] = bool(Lt["tost_ci90"][0] > -0.10 and Lt["tost_ci90"][1] < 0.10)
                    ST[bname][str(l)] = Lt
                    log(f"T {bname} L{l} (n={len(rr)}): transfer T1 {Lt['transfer_I']['T1']:+.2f} {[round(x, 2) for x in Lt['transfer_I']['ci95']]} "
                        f"rank {Lt['transfer_I']['rank']}/17 p {Lt['transfer_I']['p']:.3f} | share {Lt['curv_share']} | H removes "
                        f"{Dall['H']['removed_frac']:.2f} z {Lt['D_robust_z']:.1f} rankD {Lt['M4_rank_D']} -> M4 {Lt['M4']} | "
                        f"M5 dH {M5['H']['delta']:+.3f} vs J5 {M5['J5']['delta']:+.3f} -> {Lt['M5']} | M7 p {Lt['M7']['p_wilcoxon_less']:.3f}"
                        + (f" | TOST {Lt.get('tost_diff')} {Lt.get('tost_ci90')}" if bname == NULLB else ""))
        except Exception:  # noqa: BLE001
            res["errors"]["T"] = traceback.format_exc(); log("T ERROR\n" + res["errors"]["T"])
        res["stages"]["T"] = ST
        save()

    # ================= R (random-weights null)
    if live and plan:
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
                for (w, a, p, q) in plan[:N_R]:
                    if time.time() > DEADLINE:
                        break
                    ids = w[None].cuda(); tgt = ids[0]
                    X = R.to_layer(ids, l)[0]
                    Cl = ctrl[l]

                    def sc(rows_, b, p=p, tgt=tgt):
                        lg = m.logits(m.text.norm(rows_)).double()
                        return float(torch.log_softmax(lg, -1).gather(-1, tgt[p + 1:p + 1 + NEXT, None]).sum())
                    pairs, _ = rect(R, X, l, p, {"H": Cl["H"], **{t: Cl[t] for t in TWP}}, list(range(p, p + NEXT)), sc)
                    rr.append({"article": int(a), "pairs": pairs})
                raw.setdefault("R", {})[l] = rr
                tr_rec = recW.get(l, [])[:len(rr)]
                Lr = {"n": len(rr)}
                if len(rr) >= MIN_R and len(tr_rec) == len(rr):
                    st_r = arm_stats(rr, TWP, "I", [r["article"] for r in rr])
                    st_t = arm_stats(tr_rec, TWP, "I", [r["article"] for r in tr_rec])
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
                res["stages"]["R"] = SR
                save()
        except Exception:  # noqa: BLE001
            res["errors"]["R"] = traceback.format_exc(); log("R ERROR\n" + res["errors"]["R"])
        res["stages"]["R"] = SR
        save()

    # ================= verdict
    V = res["verdict"]
    V["M2"] = "fail" if any(m2_kill.values()) else "pass"
    V["M3_by_layer"] = {str(l): SW.get(str(l), {}).get("M3", "inconclusive") for l in LAYERS}
    V["M3"] = status_agg(V["M3_by_layer"].values())
    V["R_by_layer"] = {str(l): SR.get(str(l), {}).get("R", "inconclusive") for l in LAYERS}
    V["R"] = status_agg(V["R_by_layer"].values())
    cells = [(b, l) for b in gated for l in LAYERS]
    pv = [ST.get(b, {}).get(str(l), {}).get("transfer_I", {}).get("p", 1.0) for (b, l) in cells]
    hk = holm(pv) if pv else []
    m6 = {}
    for (b, l), ok in zip(cells, hk):
        s = ST.get(b, {}).get(str(l), {}).get("transfer_I")
        m6.setdefault(b, {})[str(l)] = bool(ok and s is not None and s["T1"] > LOG2 and s["rank"] == 1)
    V["M6_by_context"] = m6
    n_ctx = sum(all(m6.get(b, {}).get(str(l), False) for l in LAYERS) for b in gated)
    V["M6"] = "inconclusive" if len(gated) < 3 else ("pass" if n_ctx >= 3 else "fail")

    def ctx_status(key):
        per = {b: [ST.get(b, {}).get(str(l), {}).get(key, "inconclusive") for l in LAYERS] for b in gated}
        n_pass = sum(any(s == "pass" for s in v) for v in per.values())
        n_eval = sum(any(s in ("pass", "fail") for s in v) for v in per.values())
        return per, ("pass" if n_pass >= 2 else ("inconclusive" if n_eval < 2 else "fail"))
    V["M4_by_context"], V["M4"] = ctx_status("M4")
    V["M5_by_context"], V["M5"] = ctx_status("M5")
    pv7 = [ST.get(b, {}).get(str(l), {}).get("M7", {}).get("p_wilcoxon_less", 1.0) for (b, l) in cells]
    h7 = holm(pv7) if pv7 else []
    V["M7_holm"] = {f"{b}|L{l}": bool(ok) for (b, l), ok in zip(cells, h7)}
    V["M7"] = "inconclusive" if not gated else ("pass" if any(h7) else "fail")
    tost = {str(l): ST.get(NULLB, {}).get(str(l), {}).get("tost_equivalent") for l in LAYERS}
    V["TOST_by_layer"] = tost
    V["TOST"] = "inconclusive" if any(v is None for v in tost.values()) else ("pass" if all(tost.values()) else "fail")
    checks = ["M2", "M3", "R", "M6", "M4", "M5", "M7", "TOST"]
    V["MUST_pass"] = all(V[c] == "pass" for c in checks)
    q = []
    for l in LAYERS:
        s = SW.get(str(l), {})
        if s.get("sink_scope"):
            q.append(f"attention-sink curvature (L{l})")
        q.append(f"{'position-general' if s.get('M8a_position_general') else 'punctuation-specific'} (L{l})")
        if quals.get(l, {}).get("mean_norm_aligned"):
            q.append(f"mean/norm-aligned (L{l})")
        if quals.get(l, {}).get("moment_aligned"):
            q.append(f"punctuation-moment-aligned (L{l})")
    q.append("subspace (k = 5 fixed)")
    V["qualifiers"] = q
    fails = [c for c in checks if V[c] == "fail"]; inc = [c for c in checks if V[c] == "inconclusive"]
    V["FINAL"] = (("I found H-space (v10 MUST tier passed); qualifiers: " + "; ".join(q)) if V["MUST_pass"] else
                  f"MUST tier not passed. fail: {fails or 'none'}; inconclusive: {inc or 'none'}; qualifiers: " + "; ".join(q))
    res["verdict"] = V
    save()
    log(f"VERDICT: {V['FINAL']}")
    log(f"verdict detail: {json.dumps({k: V[k] for k in checks})}")
    log("DONE")


if __name__ == "__main__":
    main()
