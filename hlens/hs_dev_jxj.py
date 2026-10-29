"""hs_dev_jxj.py: J x J interaction map on Qwen3.5-0.8B (dev, laptop, $0). Oct 26 2026.

    python hs_dev_jxj.py smoke      # timing + code check (few windows, 1 layer, tiny flip null)
    python hs_dev_jxj.py full       # layers 12 (primary) and 6 (secondary)

Question. On Qwen3.6-27B, J5 carries 6-13x more same-position pairwise interaction than H-space, in the trunk
(hs_v11_softmax.py). Hypothesis: second-order computation largely lives in J-space; an H-space candidate is what is
left over. Everything below is in the project's whitened frame (Sigma^1/2, shrinkage 0.05 toward tr/d), as in
hspace.py, unless marked "raw".

S1  J-space. J25 = top-25 eigenvectors of the count-weighted Gram of the whitened ACTIVE J-lens vectors (hspace.py S3
    recipe: top-25 lens tokens per position, vectors u_t^T J Sigma^1/2, unit-normalized). Order = eigenvalue order; J5,
    J10 = first 5, 10. Decode: J-lens readout W_U (g * J Sigma^1/2 z), top tokens for +z and -z. Raw (unwhitened)
    J-space for the sensitivity check: same recipe with u_t^T J (no Sigma^1/2).
    First-order gain of a whitened direction v: gain(v)^2 = v^T Gj v, Gj = Sigma^1/2 J^T Sigma_U J Sigma^1/2 (the
    whitened J-lens Gram, hspace.py S3).
    Decoys. Sigma-orbit twins (hs_v10.twins_w with the PCA basis of Sigma): exact PCA profile per direction.
    Gain twins (hs_v10.twins_w with the eigenbasis of Gj): exact per-direction gain, exact pairwise cross-gain
    v_i^T Gj v_j and exact Gram. Gain twins are the MAIN control (literature kill-check input, adopted before freezing).
S2  Energy split. A = Sigma^1/2 H_pp[c] Sigma^1/2, the whitened per-position block of the H-lens objective
    F_c = sum_{p' valid} c . h_23[p'], c ~ N(0, Sigma_U) (exact reverse-over-reverse HVPs in fp32, bf16-stored weights).
    Per probe: B = 2 windows, one c, one whitened Gaussian xi, two independent position-sign vectors s1, s2 (hspace2
    loc_x: the cross moment is unbiased for M^diag = E A^2). Tangents: xi (full) and P_S xi for each subspace S.
    With a = rows of P_S xi and b = full - a: ||PAP||^2 = E (Pa1).(Pa2), ||QAP||^2 = E (Qa1).(Qa2),
    ||PAQ||^2 = E (Pb1).(Pb2), ||QAQ||^2 = E (Qb1).(Qb2). Fractions of the sum of the four (ratio of sums over all rows):
    JJ, JX = QAP + PAQ, XX; touch = JJ + JX = 1 - XX. Subspaces: J25, J10, J5, 8 gain twins of J25 (first 3 also
    truncated to k = 5, 10), 8 Sigma twins of J25, PCA25, uniform random 25. Raw family (whitening off): raw xi, raw
    rows, raw J25/J10/J5 and random raw 25. CIs: 2000 article-cluster bootstrap resamples. Robustness: the same split
    with the heaviest 5% of rows (by full-row norm) removed; massive/sink diagnostics (hs_v8.drop_mask axes).
S3  Leftover. Full-probe rows y1, y2 in PCA coordinates, the project's massive-axis drop mask applied (hs_v8.drop_mask:
    PCA axes 1-5 plus the top massive/radial-mass axes to 90%; axis-aligned, commutes with the flips), then J25 (its
    kept-coordinate part) deflated: variant "dJ" of hs_flip5.stats. Flip test = hs_flip5.run_layer (200 PCA sign flips,
    50 planted; norm and xnorm). Halves = probe parity = article-disjoint window halves. Leftover L5 = top-5 norm
    patterns (hs_v10a_classes.patterns logic) of the dJ rows. H5 = same with the mask only (variant d), reference.
S4  Wikitext interactions (hs_v10.rect semantics, vectorized; equality with hs_v10.rect checked in smoke). 1-sigma move
    Sigma^1/2 z at one position p (uniform in [24, 76]); f = sum of next-16-token log-likelihoods; I = f(a+b) - f(a) -
    f(b) + f(0); I_trunk = f(a+b) - f(rows_a + rows_b - rows_0); I_head = I - I_trunk; A, B main effects. Arms: J25
    (300 pairs), 3 gain-twin sets + 1 Sigma-twin set of J25 (300 pairs each), L5 (10 pairs), 12 gain twins + 6 Sigma
    twins of L5, H5. J5 = J25 pairs among directions 0-4. Per unit main effect: mean|I| / mean(|A| + |B|).
S5  Banks (v10_banks, sha256 checked vs MANIFEST): one variant with a defined gold answer per composition item
    (round-robin), two per null item. Move at the LAST prompt token; f = log p(first token of the gold answer).
    Arms: J25 (300 pairs) + 2 gain-twin sets of J25 (600 decoy pairs).

FROZEN DECISION RULE (written before any run on real data; script sha256 logged to out/devlog_2026-10-26.md first).
Layer 12 decides. Layer 6 is a replication: "replicates" iff its combined verdict string (before qualifiers) matches.
  E_major     touch(J25) >= 0.5.
  E_specific  touch(J25) > touch of all 8 gain twins AND the 95% CI of log(touch(J25) / median gain-twin touch) > 0.
  IJ_specific (wikitext, J25 300 pairs vs the 3 gain-twin sets, hs_v10.two_way_T1 on per-window mean |I|):
              T1 > log 2 AND CI low > 0 AND rank 1 of 4.
  LR          leftover reproducible: for BOTH norm and xnorm (variant dJ): obs lambda1 > flip-null max, obs split k3 >
              flip-null split3 max, split k3 >= 0.7, planted detected (the v10a/v11a criterion).
  LI          leftover interaction-specific: hs_v10 "strong" on I for L5 vs its 12 gain twins: T1 > log 2, CI low > 0,
              rank 1 of 13, p_PI < 0.05.
  LC          leftover comparable to J: mean over windows of log(mean10|I_L5| / mean10|I_J5|) >= log 0.5.
  Combined:
    "SUBSTANTIAL LEFTOVER"            iff LR and LI and LC.
    else "LIVES IN J-SPACE"           iff E_major; qualifier "(J-specific)" iff E_specific and IJ_specific, else
                                      "(not J-specific: matched by gain twins)".
    else "NEITHER"                    (J25 touches < 50% of the energy and no substantial leftover).
    Note "weak leftover" appended iff LR and LI but not LC.
  Validity: fewer than 20 wikitext windows at L12 -> "INCONCLUSIVE (too few windows)".
  Secondary, not in the verdict: Sigma-twin versions of E_specific and IJ_specific; k = 5, 10 and raw-coordinate splits;
  trimmed split; I_trunk versions; per-unit-main-effect versions.
  Bank selectivity (secondary, thresholds frozen): s = log mean|I|(bank) - log mean|I|(null) per pair. A J pair is
  selective on bank b iff s > max decoy s (600 gain-twin pairs) AND its item-cluster bootstrap 95% CI low > the decoy
  95th percentile. Decoy false-positive rate: same rule per decoy against the other decoys. "Selective composition
  pairs present on b" iff >= 3 J pairs selective AND J rate >= 3 x (decoy rate + 1/600).
"""
from __future__ import annotations

import sys

_ARGS = sys.argv[1:]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
sys.argv = sys.argv[:1]

import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from pathlib import Path  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flagship" / "src"))   # common.py expects flagship one level up
import bf16w  # noqa: E402
import hspace as H  # noqa: E402
import hs_v10 as V10  # noqa: E402
import hs_flip5 as F5  # noqa: E402
import hs_v8  # noqa: E402

TAG = _ARGS[0] if _ARGS else "smoke"
SMOKE = TAG == "smoke"
MINUTES = float(_ARGS[_ARGS.index("--minutes") + 1]) if "--minutes" in _ARGS else (20.0 if SMOKE else 65.0)
LAYERS = [12]                         # deviation (timing, logged): L6 dropped
T = 96
NEXT = 16
SKIP = H.SKIP
B = 2
N_PROBE = 2 if SMOKE else 30
N_WIN = 2 if SMOKE else 22
MIN_WIN = 20
N_FLIP, N_PLANT = (6, 3) if SMOKE else (200, 50)
N_GT, N_ST = 8, 2                     # energy-split twins of J25 (Sigma twins 8 -> 2: timing deviation, secondary only)
N_GTJ_W, N_STJ_W = 3, 1               # wikitext J25 twin sets (on SUB50)
N_LG, N_LS = 12, 3                    # L5 twins (Sigma 6 -> 3: timing deviation, secondary)
N_GTJ_B = 1                           # bank J25 twin sets (2 -> 1: timing deviation; 50 decoy pairs)
BANK_N = 2 if SMOKE else 16           # items per composition bank (null: BANK_N // 2 items x 2 variants)
COMP = ["binding_backup", "entity_attribute", "negation_truth", "relational_composition"]
NULLB = "null_unrelated"
K_SUB = 25
OUTD = H.OUT / "dev_jxj"
OUTD.mkdir(parents=True, exist_ok=True)
OUTF = OUTD / f"hs_dev_jxj_{TAG}.json"
RAWF = OUTD / f"hs_dev_jxj_{TAG}_raw.pt"
SUMF = H.HERE.parent / "results" / "dev" / "hs_dev_jxj.json"
LOGF = OUTD / f"run_{TAG}.log"
PROGF = H.OUT / "progress" / "jxj.json"
T0 = time.time()
DEADLINE = T0 + 60 * MINUTES
LOG2 = math.log(2)
ALLP25 = [(i, j) for i in range(25) for j in range(i + 1, 25)]
ALLP5 = [(i, j) for i in range(5) for j in range(i + 1, 5)]
SUB50 = sorted(random.Random(50).sample(ALLP25, 50))   # deviation (timing): J25-twin arms use this fixed 50-pair subset
SUB50_IDX = [ALLP25.index(p_) for p_ in SUB50]
STAGES = ["stats+J", "energy split", "leftover flips", "wikitext pairs", "bank pairs"]


# ------------------------------------------------------------------ logging + live progress
def log(msg):
    line = f"[jxj {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:.1f}m] {msg}"
    print(line, flush=True)
    with open(LOGF, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def _started():
    try:
        r = json.load(open(PROGF, encoding="utf-8"))
        if r.get("status") == "running" and r.get("job") == "jxj":
            return float(r.get("started", T0))
    except Exception:  # noqa: BLE001
        pass
    return T0


PROG = {"job": "jxj", "title": "J×J interaction map (0.8B)", "status": "running", "stage": "start", "stage_index": 1,
        "stage_count": len(STAGES) * len(LAYERS), "done": 0, "total": 1, "unit": "steps", "started": _started(),
        "updated": time.time(), "note": f"{TAG} run starting"}
_PLOCK = threading.Lock()


def _pwrite():
    with _PLOCK:
        PROG["updated"] = time.time()
        PROGF.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(PROGF) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(PROG, f, ensure_ascii=False)
        os.replace(tmp, PROGF)


def prog(**kw):
    PROG.update(kw)
    try:
        _pwrite()
    except Exception:  # noqa: BLE001
        pass


def _heartbeat():
    while PROG["status"] == "running":
        try:
            _pwrite()
        except Exception:  # noqa: BLE001
            pass
        time.sleep(5)


def set_stage(li, si, total, unit, note=""):
    prog(stage=f"{STAGES[si]} L{LAYERS[li]}", stage_index=li * len(STAGES) + si + 1, done=0, total=int(total), unit=unit,
         note=note or f"starting {STAGES[si]} L{LAYERS[li]}")


# ------------------------------------------------------------------ helpers
def orth(A):
    return torch.linalg.qr(A)[0]


def ov(A, Bm):
    return float((A.T @ Bm).pow(2).sum() / A.shape[1])


def eig_desc(S):
    ev, U = torch.linalg.eigh(S.double())
    o = torch.argsort(ev, descending=True)
    return ev[o], U[:, o]


def use_T(m, Tn):
    if m.T != Tn:
        m._capture_kwargs(Tn)
        m.T = Tn


def sample_c(m, gen):
    eta = torch.randn(1, m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
    return ((eta.to(torch.bfloat16) @ m.W_U).float() * m.gain[None])[0]


def score_rows(m, R, tgt, vchunk=32768, rchunk=512):
    """R [n, r, d] last-block rows, tgt [r] target ids -> [n] float64: sum_r log p(tgt_r) (fp32 logits, float64 softmax,
    the hs_v10 scorer computed in vocab chunks)."""
    n, r, d = R.shape
    x = m.text.norm(R.reshape(-1, d).float())
    tg = tgt.repeat(n)
    out = torch.empty(n * r, dtype=torch.float64, device="cuda")
    V = m.W_U.shape[0]
    for r0 in range(0, n * r, rchunk):
        xc, tc = x[r0:r0 + rchunk], tg[r0:r0 + rchunk]
        lse = torch.full((len(xc),), -float("inf"), dtype=torch.float64, device="cuda")
        tl = torch.zeros(len(xc), dtype=torch.float64, device="cuda")
        for v0 in range(0, V, vchunk):
            lg = F.linear(xc, m.W_U[v0:v0 + vchunk].float()).double()
            lse = torch.logaddexp(lse, torch.logsumexp(lg, -1))
            inr = torch.nonzero((tc >= v0) & (tc < v0 + lg.shape[1])).squeeze(1)
            if len(inr):
                tl[inr] = lg[inr, tc[inr] - v0]
        out[r0:r0 + rchunk] = tl - lse
    return out.view(n, r).sum(1)


def rect_vec(m, X, l, pos, arms, rowpos, tgt, chunk):
    """Vectorized hs_v10.rect over all pairs. X [Tn, d] layer-l states; arms {name: (V [d, n] raw moves, pairs)}.
    Returns {name: np.array [n_pairs, 5] = (I, I_trunk, I_head, A, B)}."""
    deltas, key = [torch.zeros(X.shape[1], device="cuda")], {}
    for name, (Vm, pairs) in arms.items():
        for i in sorted({i for pr in pairs for i in pr}):
            key[(name, "s", i)] = len(deltas); deltas.append(Vm[:, i])
        for (i, j) in pairs:
            key[(name, "d", i, j)] = len(deltas); deltas.append(Vm[:, i] + Vm[:, j])
    D = torch.stack(deltas)
    rp = torch.as_tensor(rowpos, device="cuda")
    rows = torch.empty(len(D), len(rowpos), X.shape[1], device="cuda")
    with torch.no_grad():
        c0 = 0
        while c0 < len(D):
            try:
                Xb = X[None].repeat(min(chunk, len(D) - c0), 1, 1)
                Xb[:, pos] += D[c0:c0 + chunk]
                rows[c0:c0 + len(Xb)] = m.run(Xb, l + 1)[:, rp]
                c0 += len(Xb)
            except torch.cuda.OutOfMemoryError:
                del Xb
                torch.cuda.empty_cache()
                assert chunk > 1
                chunk = max(1, chunk // 2)
                log(f"OOM in rect forward; chunk -> {chunk}")
        f = score_rows(m, rows, tgt)
        add_rows, add_key = [], {}
        for name, (Vm, pairs) in arms.items():
            for (i, j) in pairs:
                add_key[(name, i, j)] = len(add_rows)
                add_rows.append((key[(name, "s", i)], key[(name, "s", j)]))
        ia = torch.tensor([a for a, _ in add_rows], device="cuda"); ib = torch.tensor([b for _, b in add_rows], device="cuda")
        fadd = torch.cat([score_rows(m, rows[ia[s:s + 256]] + rows[ib[s:s + 256]] - rows[0:1], tgt) for s in range(0, len(ia), 256)])
    f, fadd = f.cpu().numpy(), fadd.cpu().numpy()
    out = {}
    for name, (Vm, pairs) in arms.items():
        arr = np.zeros((len(pairs), 5))
        for k, (i, j) in enumerate(pairs):
            fu, fw, fd, fa = f[key[(name, "s", i)]], f[key[(name, "s", j)]], f[key[(name, "d", i, j)]], fadd[add_key[(name, i, j)]]
            arr[k] = (fd - fu - fw + f[0], fd - fa, fa - fu - fw + f[0], fu - f[0], fw - f[0])
        out[name] = arr
    return out


def hvp_rows(m, X, l, c, tangents, S, Sh_out, valid, hb):
    """X [B, T, d] states (no grad), tangents [nt, d] raw vectors, S [2, B, T] signs (valid-masked).
    Returns y [nt, 2, B, n_valid, d]: y = s_p (H (s (x) v))_p, multiplied by Sh_out (whitened) when given."""
    nt = tangents.shape[0]
    jobs = [(t, si, b) for t in range(nt) for si in range(2) for b in range(X.shape[0])]
    nv = int(valid.sum())
    y = torch.empty(nt, 2, X.shape[0], nv, X.shape[2], device="cuda")
    s0 = 0
    while s0 < len(jobs):
        ch = jobs[s0:s0 + hb[0]]
        try:
            bi = torch.tensor([b for _, _, b in ch], device="cuda")
            sg = torch.stack([S[si, b] for _, si, b in ch])
            V = sg[..., None] * torch.stack([tangents[t] for t, _, _ in ch])[:, None, :]
            x = X[bi].clone().requires_grad_(True)
            with torch.enable_grad():
                hL = m.run(x, l + 1)
                Fv = (hL[:, valid] @ c).sum()
                (g,) = torch.autograd.grad(Fv, x, create_graph=True)
                (hv,) = torch.autograd.grad((g * V).sum(), x)
            r = (sg[..., None] * hv)[:, valid]
            if Sh_out is not None:
                r = r @ Sh_out
            for k, (t, si, b) in enumerate(ch):
                y[t, si, b] = r[k]
            del hL, g, hv, x, Fv
            s0 += len(ch)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if hb[0] == 1:
                raise
            hb[0] = max(1, hb[0] // 2)
            log(f"OOM in HVP batch; batch -> {hb[0]}")
    return y


def comps(a1, a2, f1, f2, U):
    """per-row block energies [n, 5] = (PAP, QAP, PAQ, QAQ, full) from rows of P xi (a) and xi (f)."""
    b1, b2 = f1 - a1, f2 - a2
    pa = ((a1 @ U) * (a2 @ U)).sum(1); aa = (a1 * a2).sum(1)
    pb = ((b1 @ U) * (b2 @ U)).sum(1); bb = (b1 * b2).sum(1)
    return torch.stack([pa, aa - pa, pb, bb - pb, (f1 * f2).sum(1)], 1)


def split_stats(C, keep_rows=None):
    """C [n, 5] -> fractions of the 4-block sum."""
    if keep_rows is not None:
        C = C[keep_rows]
    s = C.sum(0)
    tot = s[:4].sum()
    return {"JJ": float(s[0] / tot), "JX": float((s[1] + s[2]) / tot), "XX": float(s[3] / tot),
            "touch": float((s[0] + s[1] + s[2]) / tot), "sum4_over_full": float(tot / s[4]),
            "QAP_over_PAQ": float(s[1] / s[2]) if s[2] != 0 else None}


def boot_touch(Cs, names, art, ref_names, nb=2000, seed=0):
    """Article-cluster bootstrap. Cs {name: [n, 5] rows}; art [n]. Returns CI of touch for each name and of
    log(touch(J25) / median touch(ref_names))."""
    ua = np.unique(art)
    A = {nm: np.stack([Cs[nm][art == u].sum(0) for u in ua]) for nm in names}   # [n_art, 5]
    rb = np.random.default_rng(seed)
    res = {nm: [] for nm in names}
    lr = []
    for _ in range(nb):
        pick = rb.choice(len(ua), len(ua))
        t = {}
        for nm in names:
            s = A[nm][pick].sum(0)
            t[nm] = (s[0] + s[1] + s[2]) / s[:4].sum()
            res[nm].append(t[nm])
        lr.append(math.log(t["J25"] / np.median([t[r] for r in ref_names])))
    q = lambda v: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]  # noqa: E731
    return {nm: q(v) for nm, v in res.items()}, q(lr)


def decode(m, J, Sh, z, k=8):
    vec = (J @ (Sh @ z)) * m.gain if Sh is not None else (J @ z) * m.gain
    out = {}
    for sgn, nm in ((1, "+"), (-1, "-")):
        lg = F.linear((sgn * vec).to(torch.bfloat16)[None], m.W_U)[0].float()
        out[nm] = m.tok.convert_ids_to_tokens(lg.topk(k).indices.tolist())
    return out


def label(dec, i, n=3):
    clean = lambda ts: ",".join(t.replace("Ġ", "_").replace("Ċ", "\\n") for t in ts[:n])  # noqa: E731
    return f"J{i}[+{clean(dec['+'])} | -{clean(dec['-'])}]"


# ------------------------------------------------------------------ main
def main():
    threading.Thread(target=_heartbeat, daemon=True).start()
    prog(note=f"{TAG}: loading model")
    sha = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    log(f"start {TAG}; layers {LAYERS}; budget {MINUTES} min; script sha256 {sha}")
    res = {"tag": TAG, "sha256": sha, "layers": {}, "config": {
        "T": T, "NEXT": NEXT, "B": B, "N_PROBE": N_PROBE, "N_WIN": N_WIN, "N_FLIP": N_FLIP, "N_PLANT": N_PLANT,
        "N_GT": N_GT, "N_ST": N_ST, "N_GTJ_W": N_GTJ_W, "N_STJ_W": N_STJ_W, "N_LG": N_LG, "N_LS": N_LS, "N_GTJ_B": N_GTJ_B,
        "BANK_N": BANK_N}, "errors": {}}
    raw = {}

    def save():
        tmp = str(OUTF) + ".tmp"
        json.dump(res, open(tmp, "w", encoding="utf-8"), indent=1, ensure_ascii=False); os.replace(tmp, OUTF)
        torch.save(raw, str(RAWF) + ".tmp"); os.replace(str(RAWF) + ".tmp", RAWF)

    MAN = json.load(open(H.HERE / "v10_banks" / "MANIFEST.json"))
    banks = {}
    for b in COMP + [NULLB]:
        p = H.HERE / "v10_banks" / f"{b}.jsonl"
        assert hashlib.sha256(open(p, "rb").read()).hexdigest() == MAN["banks"][b]["sha256"], b
        banks[b] = [json.loads(x) for x in open(p, encoding="utf-8") if x.strip()]

    cfg = H.CFG["small"]
    m = H.Model(cfg["model"], T)
    m.T = T
    bf16w.MODE = "fp32"
    tok, d = m.tok, m.d
    from huggingface_hub import hf_hub_download
    Jall = torch.load(hf_hub_download(cfg["lens"][0], cfg["lens"][2], revision=cfg["lens"][1]), map_location="cpu", weights_only=True)["J"]
    assert max(Jall) + 1 == m.n_layers - 1
    log(f"model + lens loaded ({time.time()-T0:.0f}s); {H.torch.cuda.memory_allocated()/2**30:.2f} GB")
    U2 = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    for s in range(0, m.W_U.shape[0], 16384):
        Wg = m.W_U[s:s + 16384].float() * m.gain[None]
        U2 += (Wg.T @ Wg).double()
    U2 = (U2 / m.W_U.shape[0]).float()

    # data: distinct (article, offset) validation windows, partitioned by article id
    pool, arts = V10.windows_val(tok, 480, T, seed=26, kmin=0)
    groups = {g: [(torch.tensor(w), a) for w, a in zip(pool, arts) if a % 4 == g] for g in range(4)}
    statw = groups[0]
    hvpw = {0: [x for x in groups[1] if (x[1] // 4) % 2 == 0], 1: [x for x in groups[1] if (x[1] // 4) % 2 == 1]}
    intw = groups[2] + groups[3]
    random.Random(26).shuffle(intw)
    log(f"windows: pool {len(pool)} from {len(set(arts))} articles | stats {len(statw)} | hvp halves {len(hvpw[0])}/{len(hvpw[1])} "
        f"| interaction {len(intw)}")
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[SKIP:T - 1] = True
    gen = torch.Generator(device="cuda").manual_seed(1026)
    hb = [8]
    vdone = False

    for li, l in enumerate(LAYERS):
        L = {}
        res["layers"][str(l)] = L
        RW = {}
        raw[l] = RW
        # ================= S1 stats + J-space
        set_stage(li, 0, 3, "steps")
        use_T(m, T)
        ids_s = torch.stack([w for w, _ in statw[:48 if not SMOKE else 12]]).cuda()
        S = torch.zeros(d, d, device="cuda", dtype=torch.float64); mu = torch.zeros(d, device="cuda", dtype=torch.float64); n = 0
        with torch.no_grad():
            for s in range(0, len(ids_s), 4):
                h = m.states(ids_s[s:s + 4], l)[:, valid].reshape(-1, d).double()
                mu += h.sum(0); S += h.T @ h; n += h.shape[0]
        mu /= n; S = S / n - torch.outer(mu, mu)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device="cuda", dtype=S.dtype)
        mu, S = mu.float(), S.float()
        Sh = H.sym_sqrt(S)
        evS, Pg = eig_desc(Sh)
        Pg = Pg.float()
        prog(done=1, note=f"Sigma from {n} tokens")
        J = Jall[l].float().cuda()
        Gj = Sh @ J.T @ U2 @ J @ Sh
        evG, UG = eig_desc(Gj)
        UG = UG.float()
        acc = torch.zeros(d, d, device="cuda", dtype=torch.float64); accr = torch.zeros_like(acc)
        cnt = 0
        JS = J @ Sh
        ids_j = torch.stack([w for w, _ in statw[48 if not SMOKE else 12:]][:8]).cuda()
        with torch.no_grad():
            for s in range(0, len(ids_j), 2):
                if cnt >= 512:
                    break
                h = m.states(ids_j[s:s + 2], l)[:, valid].reshape(-1, d)
                lg = ((h @ J.T) * m.gain[None]).to(torch.bfloat16) @ m.W_U.T
                top = lg.float().topk(25, dim=-1).indices
                uniq, counts = torch.unique(top, return_counts=True)
                uu = m.u(uniq)
                vt = uu @ JS; vt = vt / vt.norm(dim=-1, keepdim=True)
                vr = uu @ J; vr = vr / vr.norm(dim=-1, keepdim=True)
                acc += (vt.T.double() * counts[None].double()) @ vt.double()
                accr += (vr.T.double() * counts[None].double()) @ vr.double()
                cnt += h.shape[0]
        evJ, UJ = eig_desc(acc); J25 = UJ[:, :25].float()
        _, UJr = eig_desc(accr); J25r = UJr[:, :25].float()
        Jglob = UG[:, :25]
        gain = lambda Z: (((Z.T @ Gj) * Z.T).sum(1)).clamp_min(0).sqrt()  # noqa: E731
        decs = [decode(m, J, Sh, J25[:, i]) for i in range(25)]
        labels = [label(dc, i) for i, dc in enumerate(decs)]
        L["J"] = {"positions": cnt, "whitened": True, "active_eigs_top25_share": float(evJ[:25].sum() / evJ.sum()),
                  "J25_vs_Jglob_overlap": ov(J25, Jglob), "J25_vs_PCA25_overlap": ov(J25, Pg[:, :25]),
                  "gain_J25": [float(x) for x in gain(J25)], "gain_Jglob_top25": [float(x) for x in evG[:25].sqrt()],
                  "gain_median_random": float(np.median(gain(orth(torch.randn(d, 200, device="cuda", generator=gen))).cpu().numpy())),
                  "decodes": decs, "labels": labels}
        try:
            T1 = torch.load(H.OUT / "hspace_small.pt", map_location="cpu", weights_only=False)
            if f"J25_{l}" in T1:
                Jw_old = orth(Sh @ torch.linalg.solve(T1[f"Sh{l}"].cuda().float(), T1[f"J25_{l}"].cuda().float()))
                L["J"]["overlap_with_hspace_small_J25"] = ov(J25, Jw_old)
        except Exception as e:  # noqa: BLE001
            L["J"]["overlap_with_hspace_small_J25"] = f"n/a: {e}"
        P_, keepm, Kd = hs_v8.drop_mask(Sh.double().cpu(), mu.double().cpu())
        keepm = keepm.cuda(); Kd = int(Kd)
        L["mask"] = {"n_dropped_axes": int((keepm[:Kd] == 0).sum()), "K": Kd,
                     "dropped_axes": [int(i) for i in torch.nonzero(keepm == 0).squeeze(1).tolist()][:40],
                     "J25_mass_on_dropped_axes": float(((Pg.T @ J25).pow(2) * (1 - keepm)[:, None]).sum() / 25)}
        log(f"L{l} S1: J25 from {cnt} positions; vs Jglob {L['J']['J25_vs_Jglob_overlap']:.2f}, vs PCA25 "
            f"{L['J']['J25_vs_PCA25_overlap']:.2f}, vs run-1 J25 {L['J'].get('overlap_with_hspace_small_J25')}; "
            f"gain J0 {L['J']['gain_J25'][0]:.3g} J24 {L['J']['gain_J25'][24]:.3g} rand {L['J']['gain_median_random']:.3g}; "
            f"mask drops {L['mask']['n_dropped_axes']} axes (J25 mass there {L['mask']['J25_mass_on_dropped_axes']:.3f})")
        for i in range(10):
            log(f"   {labels[i]}  gain {L['J']['gain_J25'][i]:.3g}")
        # subspaces
        GT = [orth(Z.float()) for Z in V10.twins_w(J25.double().cpu(), UG.double().cpu(), N_GT, 5000 + l)]
        ST = [orth(Z.float()) for Z in V10.twins_w(J25.double().cpu(), Pg.double().cpu(), N_ST, 6000 + l)]
        GT = [Z.cuda() for Z in GT]; ST = [Z.cuda() for Z in ST]
        L["decoys"] = {"gain_twin_overlap_with_J25": [ov(J25, Z) for Z in GT], "sigma_twin_overlap_with_J25": [ov(J25, Z) for Z in ST],
                       "gain_twin_gain_maxreldev": [float(((gain(Z) - gain(J25)).abs() / gain(J25)).max()) for Z in GT],
                       "sigma_twin_gain_ratio_median": [float((gain(Z) / gain(J25)).median()) for Z in ST]}
        log(f"L{l} decoys: gain-twin overlap with J25 {np.mean(L['decoys']['gain_twin_overlap_with_J25']):.3f} (max gain dev "
            f"{max(L['decoys']['gain_twin_gain_maxreldev']):.1e}); Sigma-twin overlap {np.mean(L['decoys']['sigma_twin_overlap_with_J25']):.3f}, "
            f"Sigma-twin gain ratio {np.median(L['decoys']['sigma_twin_gain_ratio_median']):.2f}")
        prog(done=3, note=f"J25 ready; gain-twin overlap {np.mean(L['decoys']['gain_twin_overlap_with_J25']):.2f}")
        save()

        # ================= S2 energy split (+ rows for S3)
        set_stage(li, 1, N_PROBE, "probes")
        wsub = {"J25": J25, "J10": J25[:, :10], "J5": J25[:, :5], "PCA25": Pg[:, :25],
                "rand25": orth(torch.randn(d, 25, device="cuda", generator=gen))}     # (gain-twin k=5/10 truncations dropped: timing)
        for t in range(N_GT):
            wsub[f"gt{t}"] = GT[t]
        for t in range(N_ST):
            wsub[f"st{t}"] = ST[t]
        rsub = {"rawJ25": J25r, "rawJ5": J25r[:, :5], "rawrand25": orth(torch.randn(d, 25, device="cuda", generator=gen))}
        wn, rn = list(wsub), list(rsub)
        Cw = {nm: [] for nm in wn}; Cr = {nm: [] for nm in rn}
        rows_art, Y1s, Y2s, prb, fn_w, fn_r = [], [], [], [], [], []
        st2 = time.time()
        for pi in range(N_PROBE):
            half = pi % 2
            hw = hvpw[half]
            sel = [hw[(pi // 2 * B + k) % len(hw)] for k in range(B)]
            X = torch.cat([m.states(w[None].cuda(), l) for w, _ in sel], 0)
            c = sample_c(m, gen)
            xi = torch.randn(d, device="cuda", generator=gen)
            xr = torch.randn(d, device="cuda", generator=gen)
            Sg = (torch.randint(0, 2, (2, B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None, None]
            tw_ = [Sh @ xi] + [Sh @ (U @ (U.T @ xi)) for U in wsub.values()]
            tr_ = [xr] + [U @ (U.T @ xr) for U in rsub.values()]
            yw = hvp_rows(m, X, l, c, torch.stack(tw_), Sg, Sh, valid, hb)       # [nt, 2, B, nv, d] whitened
            yr = hvp_rows(m, X, l, c, torch.stack(tr_), Sg, None, valid, hb)     # raw
            f1, f2 = yw[0, 0].reshape(-1, d), yw[0, 1].reshape(-1, d)
            for k, nm in enumerate(wn):
                Cw[nm].append(comps(yw[1 + k, 0].reshape(-1, d), yw[1 + k, 1].reshape(-1, d), f1, f2, wsub[nm]).cpu())
            g1, g2 = yr[0, 0].reshape(-1, d), yr[0, 1].reshape(-1, d)
            for k, nm in enumerate(rn):
                Cr[nm].append(comps(yr[1 + k, 0].reshape(-1, d), yr[1 + k, 1].reshape(-1, d), g1, g2, rsub[nm]).cpu())
            nv = f1.shape[0] // B
            rows_art += [a for _, a in sel for _ in range(nv)]
            Y1s.append((f1 @ Pg).cpu()); Y2s.append((f2 @ Pg).cpu()); prb += [pi] * f1.shape[0]
            fn_w.append(f1.norm(dim=1).cpu()); fn_r.append(g1.norm(dim=1).cpu())
            del yw, yr
            el = time.time() - st2
            tj = split_stats(torch.cat(Cw["J25"]).double())["touch"]
            prog(done=pi + 1, note=f"probe {pi+1}/{N_PROBE}: running touch(J25) {tj:.3f}; {el/(pi+1):.1f}s/probe")
            if pi % max(1, N_PROBE // 8) == 0 or pi == N_PROBE - 1:
                log(f"L{l} S2 probe {pi+1}/{N_PROBE} ({el/(pi+1):.1f}s/probe, HVP batch {hb[0]}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB)")
        art = np.array(rows_art)
        Cw = {nm: torch.cat(v).double().numpy() for nm, v in Cw.items()}
        Cr = {nm: torch.cat(v).double().numpy() for nm, v in Cr.items()}
        fnw = torch.cat(fn_w).numpy(); fnr = torch.cat(fn_r).numpy()
        keep95 = fnw <= np.quantile(fnw, 0.95)
        keep95r = fnr <= np.quantile(fnr, 0.95)
        E = {"n_rows": int(len(art)), "n_articles": int(len(np.unique(art))), "split": {}, "split_trim5": {}, "raw_split": {},
             "raw_split_trim5": {}}
        for nm in wn:
            E["split"][nm] = split_stats(torch.tensor(Cw[nm]))
            E["split_trim5"][nm] = split_stats(torch.tensor(Cw[nm]), torch.tensor(keep95))
        for nm in rn:
            E["raw_split"][nm] = split_stats(torch.tensor(Cr[nm]))
            E["raw_split_trim5"][nm] = split_stats(torch.tensor(Cr[nm]), torch.tensor(keep95r))
        gtn = [f"gt{t}" for t in range(N_GT)]; stn = [f"st{t}" for t in range(N_ST)]
        cis, lr_g = boot_touch(Cw, ["J25"] + gtn + stn + ["PCA25", "rand25", "J10", "J5"], art, gtn, seed=1)
        _, lr_s = boot_touch(Cw, ["J25"] + stn, art, stn, seed=2)
        E["touch_ci95"] = cis
        E["log_touch_J25_over_median_gain_twin"] = {"mean": math.log(E["split"]["J25"]["touch"] / np.median([E["split"][t]["touch"] for t in gtn])), "ci95": lr_g}
        E["log_touch_J25_over_median_sigma_twin"] = {"mean": math.log(E["split"]["J25"]["touch"] / np.median([E["split"][t]["touch"] for t in stn])), "ci95": lr_s}
        e2 = np.sort(fnw ** 2)[::-1]
        E["heavy_tail"] = {"top0.1pct_row_energy_share": float(e2[:max(1, len(e2) // 1000)].sum() / e2.sum()),
                           "top1pct_row_energy_share": float(e2[:max(1, len(e2) // 100)].sum() / e2.sum()),
                           "kish_ess_frac": float(e2.sum() ** 2 / (e2 ** 2).sum() / len(e2))}
        Y1 = torch.cat(Y1s); Y2 = torch.cat(Y2s); probe_t = torch.tensor(prb)
        E["row_energy_share_on_dropped_axes"] = float((Y1.pow(2) * (1 - keepm.cpu())).sum() / Y1.pow(2).sum())
        Jt = E["split"]["J25"]
        E["random25_expected_touch"] = float(1 - (1 - 25 / d) ** 2)
        E["E_major"] = bool(Jt["touch"] >= 0.5)
        E["E_specific"] = bool(all(Jt["touch"] > E["split"][t]["touch"] for t in gtn) and lr_g[0] > 0)
        E["E_specific_sigma"] = bool(all(Jt["touch"] > E["split"][t]["touch"] for t in stn) and lr_s[0] > 0)
        L["energy"] = E
        RW["energy_rows"] = {"Cw": Cw, "Cr": Cr, "art": art}
        log(f"L{l} S2 (n rows {len(art)}): J25 JJ {Jt['JJ']:.3f} JX {Jt['JX']:.3f} XX {Jt['XX']:.3f} touch {Jt['touch']:.3f} "
            f"{[round(x, 3) for x in cis['J25']]} | gain twins touch median {np.median([E['split'][t]['touch'] for t in gtn]):.3f} "
            f"log-ratio CI {[round(x, 3) for x in lr_g]} | Sigma twins {np.median([E['split'][t]['touch'] for t in stn]):.3f} | "
            f"PCA25 {E['split']['PCA25']['touch']:.3f} rand25 {E['split']['rand25']['touch']:.3f} | J10 {E['split']['J10']['touch']:.3f} "
            f"J5 {E['split']['J5']['touch']:.3f} | raw J25 {E['raw_split']['rawJ25']['touch']:.3f} rawrand {E['raw_split']['rawrand25']['touch']:.3f} | "
            f"trim5 J25 {E['split_trim5']['J25']['touch']:.3f} | top0.1% rows {E['heavy_tail']['top0.1pct_row_energy_share']:.3f} | "
            f"E_major {E['E_major']} E_specific {E['E_specific']}")
        save()

        # ================= S3 leftover (flip test)
        set_stage(li, 2, N_FLIP + N_PLANT, "flip replicates", note="flip test running (progress updates per stage only)")
        dev = "cuda"
        Y1d = (Y1.to(dev) * keepm); Y2d = (Y2.to(dev) * keepm)
        QJ = orth(((Pg.T @ J25) * keepm[:, None]))                     # kept-coordinate part of J25, PCA coords
        t3 = time.time()
        rr = F5.run_layer(Y1d, Y2d, probe_t, {"dJ": QJ}, Kd, N_FLIP, N_PLANT, dev, 11000 + l, f"jxj L{l}")
        half_m = (probe_t % 2 == 0).to(dev)
        nrm = Y1d.norm(dim=1)
        keep_raw = nrm <= torch.quantile(nrm.float().cpu(), 0.95).to(dev)
        obs_d = F5.stats(Y1d, Y2d, half_m, Kd, keepm, keep_raw)

        def pats(Yp, Kc):
            cols = torch.arange(Kc, device=dev)
            U = (Yp / Yp.norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols].double()
            Mn = U.T @ U / len(U)
            dg = torch.diagonal(Mn).clamp_min(1e-30)
            R_ = dg.rsqrt()[:, None] * Mn * dg.rsqrt()[None, :]
            ev, Vv = torch.linalg.eigh(R_)
            Vv = Vv[:, torch.argsort(ev, descending=True)[:5]]
            pat = torch.linalg.qr(dg.sqrt()[:, None] * Vv)[0]
            full = torch.zeros(d, 5, dtype=torch.float64, device=dev); full[:Kc] = pat
            return (Pg.double() @ full).float()                          # whitened [d, 5]
        Y1dJ = Y1d - (Y1d @ QJ) @ QJ.T
        L5 = pats(Y1dJ, Kd)
        H5 = pats(Y1d, Kd)
        LG = [orth(Z.float()).cuda() for Z in V10.twins_w(L5.double().cpu(), UG.double().cpu(), N_LG, 7000 + l)]
        LS = [orth(Z.float()).cuda() for Z in V10.twins_w(L5.double().cpu(), Pg.double().cpu(), N_LS, 8000 + l)]
        LR = all(rr["dJ"][e]["obs_lam"][0] > rr["dJ"][e]["null_lam1_max"] and rr["dJ"][e]["obs_split"]["k3"] > rr["dJ"][e]["null_split3_max"]
                 and rr["dJ"][e]["obs_split"]["k3"] >= 0.7 and rr["planted"][e]["detected"] for e in ("norm", "xnorm"))
        Lf = {"flip": {e: {k: rr["dJ"][e][k] for k in ("obs_lam", "obs_split", "null_lam1_max", "null_split3_max", "null_split5_max",
                                                        "p_lam1", "p_split3")} for e in ("norm", "xnorm", "raw_trim5")},
              "planted": rr["planted"], "LR": bool(LR),
              "H5_mask_only_split": {e: obs_d[e]["split"] for e in ("norm", "xnorm")},
              "L5_in_J25": ov(L5, J25), "H5_in_J25": ov(H5, J25), "L5_vs_H5": ov(L5, H5),
              "gain_L5": [float(x) for x in gain(L5)], "gain_H5": [float(x) for x in gain(H5)],
              "L5_gain_twin_overlap": [ov(L5, Z) for Z in LG], "L5_sigma_twin_gain_ratio": [float((gain(Z) / gain(L5)).median()) for Z in LS],
              "decodes_L5": [decode(m, J, Sh, L5[:, i]) for i in range(5)], "seconds": time.time() - t3}
        L["leftover"] = Lf
        RW["L5"], RW["H5"], RW["J25"], RW["Sh"], RW["mu"] = L5.cpu(), H5.cpu(), J25.cpu(), Sh.cpu(), mu.cpu()
        log(f"L{l} S3: leftover norm split k3 {Lf['flip']['norm']['obs_split']['k3']:.2f} (null max {Lf['flip']['norm']['null_split3_max']:.2f}) "
            f"lam1 {Lf['flip']['norm']['obs_lam'][0]:.3f} (null max {Lf['flip']['norm']['null_lam1_max']:.3f}) | xnorm k3 "
            f"{Lf['flip']['xnorm']['obs_split']['k3']:.2f} (null max {Lf['flip']['xnorm']['null_split3_max']:.2f}) | planted "
            f"{rr['planted']['norm']['detected']}/{rr['planted']['xnorm']['detected']} | LR {LR} | L5 in J25 {Lf['L5_in_J25']:.3f} | "
            f"H5 in J25 {Lf['H5_in_J25']:.3f} | gain L5 {np.median(Lf['gain_L5']):.3g} vs J5 {np.median(L['J']['gain_J25'][:5]):.3g} ({Lf['seconds']:.0f}s)")
        prog(done=N_FLIP + N_PLANT, note=f"leftover split3 {Lf['flip']['norm']['obs_split']['k3']:.2f} vs null max {Lf['flip']['norm']['null_split3_max']:.2f}")
        del Y1d, Y2d, Y1dJ
        torch.cuda.empty_cache()
        save()

        # ================= S4 wikitext rectangles
        set_stage(li, 3, N_WIN, "windows")
        mv = lambda Z: Sh @ Z  # noqa: E731
        armsW = {"J": (mv(J25), ALLP25)}
        for t in range(N_GTJ_W):
            armsW[f"gJ{t}"] = (mv(GT[t]), SUB50)
        for t in range(N_STJ_W):
            armsW[f"sJ{t}"] = (mv(ST[t]), SUB50)
        armsW["L"] = (mv(L5), ALLP5); armsW["H5"] = (mv(H5), ALLP5)
        for t in range(N_LG):
            armsW[f"Lg{t}"] = (mv(LG[t]), ALLP5)
        for t in range(N_LS):
            armsW[f"Ls{t}"] = (mv(LS[t]), ALLP5)
        recs = []
        rng = random.Random(100 + l)
        st4 = time.time()
        dl4 = st4 + 60 * (MINUTES / len(LAYERS)) * 0.33
        for wi, (w, a) in enumerate(intw[:N_WIN]):
            if time.time() > dl4 or time.time() > DEADLINE - 60 * 4:
                log(f"L{l} S4 time budget reached after {wi} windows"); break
            p = rng.randint(SKIP + 8, T - NEXT - 4)
            Tn = p + NEXT + 1                                      # causal: later positions cannot affect rows p..p+15
            use_T(m, Tn)
            ids = w[:Tn][None].cuda()
            X = m.states(ids, l)[0]
            tgt = ids[0, p + 1:p + 1 + NEXT]
            assert len(tgt) == NEXT
            rowpos = list(range(p, p + NEXT))
            if SMOKE and not vdone:
                # equality check against hs_v10.rect (MP pairs, J arm only)
                class _R:
                    pass
                Rr = _R(); Rr.m = m; Rr.run_from = lambda Xb, l_: m.run(Xb, l_ + 1)

                def sc(rows_, b, tgt=tgt):
                    lg_ = m.logits(m.text.norm(rows_)).double()
                    return float(torch.log_softmax(lg_, -1).gather(-1, tgt[:, None]).sum())
                ref, _ = V10.rect(Rr, X, l, p, {"J": mv(J25)[:, :5]}, rowpos, sc)
                mine = rect_vec(m, X, l, p, {"J": (mv(J25)[:, :5], V10.MP)}, rowpos, tgt, 32)["J"]
                diffs = [float(np.max(np.abs(np.array(ref[f"J|{i}{j}"]) - mine[k]))) for k, (i, j) in enumerate(V10.MP)]
                mags = [float(np.max(np.abs(mine[k][:3]))) for k in range(len(V10.MP))]
                res["rect_check"] = {"max_abs_diff": max(diffs), "max_abs_I": max(mags)}
                log(f"rect_vec vs hs_v10.rect: max abs diff {max(diffs):.2e} (|I| up to {max(mags):.2e})")
                vdone = True
            out = rect_vec(m, X, l, p, armsW, rowpos, tgt, 48)
            ok = all(np.isfinite(v).all() for v in out.values())
            if not ok:
                log(f"L{l} S4 window {wi}: non-finite values, dropped"); continue
            recs.append({"article": int(a), "p": int(p), "pairs": out})
            el = time.time() - st4
            jI = np.mean([np.abs(r['pairs']['J'][:, 0]).mean() for r in recs])
            prog(done=wi + 1, note=f"window {wi+1}: mean |I| J25 pairs {jI:.2e}; {el/(wi+1):.0f}s/window")
            if (wi + 1) % 4 == 0 or wi < 2:
                RW["W"] = recs; save()
                log(f"L{l} S4 window {wi+1}/{N_WIN} ({el/(wi+1):.1f}s/window, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB)")
        use_T(m, T)
        RW["W"] = recs
        L["wikitext"] = analyze_w(recs, labels)
        save()

        # ================= S5 banks
        seqs = []
        for b in COMP:
            for ii, it in enumerate(banks[b][:BANK_N]):
                vs = [v for v in ("00", "01", "10", "11") if it["gold"][v] in ("X", "Y")]
                seqs.append((b, it, vs[ii % len(vs)]))
        for ii, it in enumerate(banks[NULLB][:max(1, BANK_N // 2)]):
            vs = [v for v in ("00", "01", "10", "11") if it["gold"][v] in ("X", "Y")]
            seqs += [(NULLB, it, vs[ii % len(vs)]), (NULLB, it, vs[(ii + 2) % len(vs)])]
        set_stage(li, 4, len(seqs), "prompts")
        armsB = {"J": (mv(J25), ALLP25)}
        for t in range(N_GTJ_B):
            armsB[f"gJ{t}"] = (mv(GT[t]), SUB50)
        brec = []
        st5 = time.time()
        enc = []
        for (b, it, v) in seqs:
            ids = tok(it["prompts"][v], add_special_tokens=False)["input_ids"]
            g = tok(it[it["gold"][v]], add_special_tokens=False)["input_ids"][0]
            enc.append((len(ids), b, it, v, ids, g))
        enc.sort(key=lambda x: x[0])
        for si, (Tn, b, it, v, ids, g) in enumerate(enc):
            if time.time() > DEADLINE - 60 * 1.5:
                log(f"L{l} S5 deadline reached after {si} prompts"); break
            use_T(m, Tn)
            idt = torch.tensor(ids, device="cuda")[None]
            X = m.states(idt, l)[0]
            out = rect_vec(m, X, l, Tn - 1, armsB, [Tn - 1], torch.tensor([g], device="cuda"), 96)
            if not all(np.isfinite(x).all() for x in out.values()):
                log(f"S5 {it['item']} non-finite, dropped"); continue
            brec.append({"bank": b, "item": it["item"], "cluster": f"{b}:{it['cluster']}", "variant": v, "len": Tn, "pairs": out})
            if (si + 1) % 8 == 0 or si == len(enc) - 1:
                el = time.time() - st5
                prog(done=si + 1, note=f"prompt {si+1}/{len(enc)}; {el/(si+1):.1f}s/prompt")
            if (si + 1) % 40 == 0:
                RW["banks"] = brec; save()
                log(f"L{l} S5 prompt {si+1}/{len(enc)} ({(time.time()-st5)/(si+1):.2f}s/prompt)")
        use_T(m, T)
        RW["banks"] = brec
        L["banks"] = analyze_b(brec, labels)
        L["verdict"] = verdict(L)
        log(f"L{l} VERDICT: {L['verdict']['combined']} | flags {L['verdict']['flags']}")
        save()
    # overall
    v12 = res["layers"].get("12", {}).get("verdict", {})
    v6 = res["layers"].get("6", {}).get("verdict", {})
    res["verdict"] = {"L12": v12.get("combined"), "L6": v6.get("combined"),
                      "L6_replicates": (v12.get("core") == v6.get("core")) if v6 else None}
    res["runtime_min"] = (time.time() - T0) / 60
    save()
    if not SMOKE:
        SUMF.parent.mkdir(parents=True, exist_ok=True)
        json.dump(summary(res), open(SUMF, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    log(f"DONE {res['runtime_min']:.1f} min; verdict {res['verdict']}")
    prog(status="done", done=PROG["total"], note=f"done in {res['runtime_min']:.0f} min: L12 {res['verdict']['L12']}")


# ------------------------------------------------------------------ analysis
def per_win(recs, arm, comp=0, pairs=None):
    return np.array([np.abs(r["pairs"][arm][pairs if pairs is not None else slice(None), comp]).mean() for r in recs])


def per_win_main(recs, arm, pairs=None):
    sl = pairs if pairs is not None else slice(None)
    return np.array([(np.abs(r["pairs"][arm][sl, 3]) + np.abs(r["pairs"][arm][sl, 4])).mean() for r in recs])


def analyze_w(recs, labels):
    out = {"n": len(recs)}
    if len(recs) < 2:
        return out
    cl = [r["article"] for r in recs]
    gJ = [k for k in recs[0]["pairs"] if k.startswith("gJ")]
    sJ = [k for k in recs[0]["pairs"] if k.startswith("sJ")]
    Lg = [k for k in recs[0]["pairs"] if k.startswith("Lg")]
    Ls = [k for k in recs[0]["pairs"] if k.startswith("Ls")]
    j5 = [k for k, (i, j) in enumerate(ALLP25) if i < 5 and j < 5]
    for comp, cn in ((0, "I"), (1, "I_trunk")):
        ex = per_win(recs, "J", comp, SUB50_IDX)
        out[f"J25_vs_gain_twins_{cn}"] = V10.two_way_T1(ex, np.stack([per_win(recs, t, comp) for t in gJ], 1), cl)
        out[f"J25_vs_sigma_twins_{cn}"] = V10.two_way_T1(ex, np.stack([per_win(recs, t, comp) for t in sJ], 1), cl) if len(sJ) > 0 else None
        exl = per_win(recs, "L", comp)
        out[f"L5_vs_gain_twins_{cn}"] = V10.two_way_T1(exl, np.stack([per_win(recs, t, comp) for t in Lg], 1), cl)
        out[f"L5_vs_sigma_twins_{cn}"] = V10.two_way_T1(exl, np.stack([per_win(recs, t, comp) for t in Ls], 1), cl)
        lj = np.log((exl + 1e-12) / (per_win(recs, "J", comp, j5) + 1e-12))
        out[f"log_L5_over_J5_{cn}"] = {"mean": float(lj.mean()), "ratio": float(math.exp(lj.mean())), "ci95": V10.cboot(lambda ix: lj[ix].mean(), cl)}
        lh = np.log((per_win(recs, "H5", comp) + 1e-12) / (per_win(recs, "J", comp, j5) + 1e-12))
        out[f"log_H5_over_J5_{cn}"] = {"mean": float(lh.mean()), "ratio": float(math.exp(lh.mean()))}
    # per unit main effect (T3 style): log((ex/mex)/(tw/mtw)) averaged over twins
    def t3j(twins):
        ex, mex = per_win(recs, "J", 0, SUB50_IDX), per_win_main(recs, "J", SUB50_IDX)
        v = np.mean([np.log((ex / mex) / (per_win(recs, t, 0) / per_win_main(recs, t))) for t in twins], 0)
        return {"mean": float(v.mean()), "ratio": float(math.exp(v.mean())), "ci95": V10.cboot(lambda ix: v[ix].mean(), cl),
                "I_per_main_arm": float(np.mean(ex / mex)), "I_per_main_twins": float(np.mean([np.mean(per_win(recs, t, 0) / per_win_main(recs, t)) for t in twins]))}

    def t3(arm, twins, pairs=None):
        ex, mex = per_win(recs, arm, 0, pairs), per_win_main(recs, arm, pairs)
        v = np.mean([np.log((ex / mex) / (per_win(recs, t, 0, pairs) / per_win_main(recs, t, pairs))) for t in twins], 0)
        return {"mean": float(v.mean()), "ratio": float(math.exp(v.mean())), "ci95": V10.cboot(lambda ix: v[ix].mean(), cl),
                "I_per_main_arm": float(np.mean(ex / mex)), "I_per_main_twins": float(np.mean([np.mean(per_win(recs, t, 0, pairs) / per_win_main(recs, t, pairs)) for t in twins]))}
    out["per_main_J25_vs_gain_twins"] = t3j(gJ)
    out["per_main_L5_vs_gain_twins"] = t3("L", Lg)
    vj = np.log(per_win(recs, "L") / per_win_main(recs, "L")) - np.log(per_win(recs, "J", 0, j5) / per_win_main(recs, "J", j5))
    out["per_main_log_L5_over_J5"] = {"mean": float(vj.mean()), "ratio": float(math.exp(vj.mean()))}
    # pair ranking
    PI = np.stack([np.abs(r["pairs"]["J"][:, 0]) for r in recs])          # [n_win, 300]
    PT = np.stack([np.abs(r["pairs"]["J"][:, 1]) for r in recs])
    PM = np.stack([np.abs(r["pairs"]["J"][:, 3]) + np.abs(r["pairs"]["J"][:, 4]) for r in recs])
    DI = np.concatenate([np.stack([np.abs(r["pairs"][t][:, 0]) for r in recs]) for t in gJ], 1)   # decoy pairs
    dmean = DI.mean(0)
    mI = PI.mean(0)
    order = np.argsort(-mI)
    out["decoy_pair_meanI"] = {"median": float(np.median(dmean)), "q95": float(np.quantile(dmean, 0.95)), "max": float(dmean.max()), "n": int(len(dmean))}
    out["J_pair_meanI"] = {"median": float(np.median(mI)), "median_on_SUB50": float(np.median(mI[SUB50_IDX])), "n_above_decoy_max": int((mI > dmean.max()).sum())}
    out["top_pairs"] = [{"pair": [int(ALLP25[k][0]), int(ALLP25[k][1])], "labels": [labels[ALLP25[k][0]], labels[ALLP25[k][1]]],
                         "mean_absI": float(mI[k]), "ci95": V10.cboot(lambda ix, k=k: PI[ix, k].mean(), cl, B=1000),
                         "mean_absI_trunk": float(PT[:, k].mean()), "I_per_main": float(PI[:, k].mean() / PM[:, k].mean()),
                         "mean_signedI": float(np.mean([r["pairs"]["J"][k, 0] for r in recs])),
                         "vs_decoy_max": float(mI[k] / dmean.max())} for k in order[:15]]
    out["all_pairs_meanI"] = mI.tolist()
    return out


def analyze_b(brec, labels, nb=1000):
    out = {"n": len(brec)}
    if not brec:
        return out
    gJ = [k for k in brec[0]["pairs"] if k.startswith("gJ")]
    by = {}
    for r in brec:
        by.setdefault(r["bank"], []).append(r)
    if NULLB not in by:
        return out

    def mat(rs, arm):
        return np.stack([np.abs(r["pairs"][arm][:, 0]) for r in rs])
    nul = by[NULLB]
    NJ, ND = mat(nul, "J"), np.concatenate([mat(nul, t) for t in gJ], 1)
    ncl = np.array([r["cluster"] for r in nul])
    rb = np.random.default_rng(3)
    out["null_meanI_J"] = float(NJ.mean()); out["null_meanI_decoy"] = float(ND.mean())
    for b in COMP:
        if b not in by:
            continue
        rs = by[b]
        BJ, BD = mat(rs, "J"), np.concatenate([mat(rs, t) for t in gJ], 1)
        bcl = np.array([r["cluster"] for r in rs])
        sJ = np.log(BJ.mean(0) + 1e-12) - np.log(NJ.mean(0) + 1e-12)
        sD = np.log(BD.mean(0) + 1e-12) - np.log(ND.mean(0) + 1e-12)
        ub, un = np.unique(bcl), np.unique(ncl)
        ib = {u: np.where(bcl == u)[0] for u in ub}; inn = {u: np.where(ncl == u)[0] for u in un}
        bsJ, bsD = [], []
        for _ in range(nb):
            pb = np.concatenate([ib[u] for u in rb.choice(ub, len(ub))]); pn = np.concatenate([inn[u] for u in rb.choice(un, len(un))])
            bsJ.append(np.log(BJ[pb].mean(0) + 1e-12) - np.log(NJ[pn].mean(0) + 1e-12))
            bsD.append(np.log(BD[pb].mean(0) + 1e-12) - np.log(ND[pn].mean(0) + 1e-12))
        loJ = np.quantile(np.stack(bsJ), 0.025, axis=0); loD = np.quantile(np.stack(bsD), 0.025, axis=0)
        dmax, dq95 = sD.max(), np.quantile(sD, 0.95)
        selJ = (sJ > dmax) & (loJ > dq95)
        selD = np.array([(sD[k] > np.delete(sD, k).max()) and (loD[k] > np.quantile(np.delete(sD, k), 0.95)) for k in range(len(sD))])
        rJ, rD = selJ.mean(), selD.mean()
        order = np.argsort(-sJ)
        out[b] = {"n": len(rs), "meanI_J": float(BJ.mean()), "meanI_decoy": float(BD.mean()),
                  "J_mean_s": float(sJ.mean()), "decoy_mean_s": float(sD.mean()), "decoy_s_q95": float(dq95), "decoy_s_max": float(dmax),
                  "n_selective_J": int(selJ.sum()), "n_selective_decoy": int(selD.sum()), "rate_J": float(rJ), "rate_decoy": float(rD),
                  "selective_present": bool(selJ.sum() >= 3 and rJ >= 3 * (rD + 1 / len(sD))), "n_decoy_pairs": int(len(sD)),
                  "top_pairs_by_s": [{"pair": list(ALLP25[k]), "labels": [labels[ALLP25[k][0]], labels[ALLP25[k][1]]], "s": float(sJ[k]),
                                      "ci_low": float(loJ[k]), "meanI_bank": float(BJ[:, k].mean()), "meanI_null": float(NJ[:, k].mean()),
                                      "selective": bool(selJ[k])} for k in order[:8]],
                  "top_pairs_by_meanI": [{"pair": list(ALLP25[k]), "labels": [labels[ALLP25[k][0]], labels[ALLP25[k][1]]],
                                          "meanI": float(BJ[:, k].mean())} for k in np.argsort(-BJ.mean(0))[:8]]}
    return out


def verdict(L):
    E, W, Lf = L.get("energy", {}), L.get("wikitext", {}), L.get("leftover", {})
    flags = {}
    if W.get("n", 0) < MIN_WIN:
        return {"combined": "INCONCLUSIVE (too few windows)", "core": "INCONCLUSIVE", "flags": {"n_windows": W.get("n", 0)}}
    g = W["J25_vs_gain_twins_I"]
    flags["E_major"] = E["E_major"]; flags["E_specific"] = E["E_specific"]
    flags["IJ_specific"] = bool(g["T1"] > LOG2 and g["ci95"][0] > 0 and g["rank"] == 1)
    lg = W["L5_vs_gain_twins_I"]
    flags["LR"] = Lf["LR"]
    flags["LI"] = bool(lg["T1"] > LOG2 and lg["ci95"][0] > 0 and lg["rank"] == 1 and lg["p_PI"] < 0.05)
    flags["LC"] = bool(W["log_L5_over_J5_I"]["mean"] >= math.log(0.5))
    if flags["LR"] and flags["LI"] and flags["LC"]:
        core = "SUBSTANTIAL LEFTOVER"; comb = core
    elif flags["E_major"]:
        core = "LIVES IN J-SPACE"
        comb = core + (" (J-specific)" if flags["E_specific"] and flags["IJ_specific"] else " (not J-specific: matched by gain twins)")
    else:
        core = "NEITHER"; comb = core
    if core != "SUBSTANTIAL LEFTOVER" and flags["LR"] and flags["LI"] and not flags["LC"]:
        comb += "; weak leftover"
    return {"combined": comb, "core": core, "flags": flags}


def summary(res):
    s = {"script": "hlens/hs_dev_jxj.py", "sha256": res["sha256"], "model": "Qwen/Qwen3.5-0.8B", "runtime_min": res.get("runtime_min"),
         "verdict": res.get("verdict"), "layers": {}}
    for l, L in res["layers"].items():
        E, W, Lf, Bk = L.get("energy", {}), L.get("wikitext", {}), L.get("leftover", {}), L.get("banks", {})
        s["layers"][l] = {
            "energy_split": {k: E.get("split", {}).get(k) for k in ("J25", "J10", "J5", "PCA25", "rand25")},
            "touch_ci95": {k: E.get("touch_ci95", {}).get(k) for k in ("J25", "PCA25", "rand25")},
            "gain_twins_touch": [E["split"][f"gt{t}"]["touch"] for t in range(N_GT)] if E else None,
            "sigma_twins_touch": [E["split"][f"st{t}"]["touch"] for t in range(N_ST)] if E else None,
            "log_touch_J25_over_gain_twin": E.get("log_touch_J25_over_median_gain_twin"),
            "log_touch_J25_over_sigma_twin": E.get("log_touch_J25_over_median_sigma_twin"),
            "raw_split": E.get("raw_split"), "trim5_J25": E.get("split_trim5", {}).get("J25"), "heavy_tail": E.get("heavy_tail"),
            "decoys": L.get("decoys"), "mask": L.get("mask"),
            "wikitext": {k: v for k, v in W.items() if k != "all_pairs_meanI"},
            "leftover": {k: v for k, v in Lf.items() if k not in ("decodes_L5",)},
            "banks": Bk, "verdict": L.get("verdict"), "J_labels": L.get("J", {}).get("labels")}
    return s


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        tb = traceback.format_exc()
        log("CRASH\n" + tb)
        prog(status="failed", note=f"crash: {type(e).__name__}: {str(e)[:150]}")
        raise
