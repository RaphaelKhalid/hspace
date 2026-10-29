"""Cross-position H-lens, $0 dev run on Qwen3.5-0.8B (Oct 26 2026). Laptop only, no paid compute.

Question. Every 27B test since v8 measured second-order interaction of two moves at the SAME position. Attention's
query x key (and DeltaNet's q.k v) is bilinear ACROSS positions. Is there a low-dimensional cross-position H-space:
a q-side subspace at an earlier token and a p-side subspace at a later token whose 1-sigma moves interact?

Estimator (project H-lens convention, theory-v2 section 0).
  Readout: F_c = sum_{t in V} c . h_L*[t] (pre-final-norm linear readout, V = 16..T-2, c ~ N(0, Sigma_U)), at layer l.
  Bank prompts are short, so there F_c = c . h_L*[ans] (answer position only).
  Cross block (Lemma 1 block of the residual Hessian): C_qp = d^2 F_c / dx_l[q] dx_l[p], q < p, whitened
  C~ = Sigma^1/2 C Sigma^1/2 (Sigma, mu, J25 from the run-1 dev frame out/hspace_small.pt).
  q-side rows: tangent Sigma^1/2 xi at p (xi ~ N(0, I)), one exact double-backward HVP, row y_q = Sigma^1/2 (Hv)_q.
    E[y y^T] = C~ C~^T. p-side rows: tangent at q, row at p: E[z z^T] = C~^T C~.
  Wikitext (validation, offsets k >= 2, T = 96): q-side probes draw p uniform in [17, 79] and keep rows at every
  q in [16, p-1]; p-side probes draw q uniform in [16, 78] and keep rows at every p in [q+1, 79]. Both are then
  uniform over pairs q < p <= 79. Banks (entity_attribute, binding_backup, negation_truth, relational_composition):
  q = last token of the bank's natural_site, p = answer position (last prompt token), all 4 cells, FIT clusters only.
  Massive axes are masked the way the project does since v5 (hs_v8.drop_mask: drop PCA axes 1-5 plus the axes that
  carry 90% of the massive/radial span; axis-aligned, so it commutes with the sign flips): rows are projected onto the
  kept PCA axes before the operators are formed. Positions < 16 (incl. the position-0 sink) are never q.
  Operators M_q = mean y y^T, M_p = mean z z^T (raw energy, masked). Candidate = top-5 eigenvectors (Uq, Up).
  The diagonal block (q = p) is never used.
Prediction written before any data (single-head decomposition; literature kill-check: Kamath et al. 2025 QK feature
interactions; Lee et al. 2026 arXiv 2602.04752 TODO-verify; J-lens broadcast heads). For one softmax head the mixed block
is about a scalar times W_Q^T W_K (query side = p, key side = q) plus a rank-1 (W_V^T g) x (W_Q^T dK) term; DeltaNet's
q_p . k_q gives the same shape. So the expected (explained) result is: Up aligns with the top query-side singular
directions of the pooled downstream W_QK, Uq aligns with the key side and/or with J-space via OV transport. The
candidate is only interesting if a large part of it lies OUTSIDE span(QK25 of its side, softmax and DeltaNet) + J25.
Reported (descriptive, no threshold): overlaps with those spans and the leftover 1 - overlap after projecting them out.
QK25 = top-25 eigenvectors of sum_h A_h A_h^T (p side) or A_h^T A_h (q side), A_h = Sh G_in Wq_h^T diag(gq gk) Wk_h G_in Sh
/ ||.||_F over heads of downstream blocks (> l), ignoring RoPE, the q/k RMS scale, conv1d and l2norm (a static proxy:
the residual is also transported through blocks before reaching each head).
Tests use the natural log-likelihood instead: wikitext f = sum log p(next 16 tokens after p); banks f = lp[X] - lp[Y]
at the answer. Moves are 1 sigma: Sigma^1/2 u for whitened unit u. Pairs: all 25 (u_i at q, v_j at p).
  I_ij = f(x + u_i@q + v_j@p) - f(x + u_i@q) - f(x + v_j@p) + f(x)   (Lemma 3 four-point difference)
  per record: absI = mean_ij |I_ij|; main = mean_ij (|f(u_i)-f0| + |f(v_j)-f0|)/2; pu = absI / main.
  Decoys: 8 Sigma-orbit twins per side (hs_v10.twins_w on the global PCA basis: exact 1-sigma norms and Gram), J5
  (top-5 of J25 at both positions), PC5 (descriptive).

DECISION RULE (frozen before the first run on real data; sample sizes are set from the timing run and logged in
out/devlog_2026-10-26.md, they are not part of the rule). Layers 12 (primary) and 6. alpha = 0.05 two-sided CIs.
  V0 instrument valid  := FD: at every checked pair (6 per layer), the best over eps/|x| in {3e-2,1e-2,3e-3,1e-3}
                          of ||FD(C_qp v) - HVP(C_qp v)|| / ||HVP(C_qp v)|| < 1e-2 (central FD of the exact gradient);
                          AND planted control: a bilinear gate G (z_{t-1}.a)(z_t.b) w added to block l+1's output,
                          G calibrated so the planted energy equals lambda_5 of that side's clean operator; recovery
                          ||U5^T a||^2 (q-side) and ||U5^T b||^2 (p-side) >= 0.8 each.
  R  reproducible      := for BOTH sides of the wikitext operator: top-5 split-half overlap (halves = disjoint article
                          sets) >= 0.7 AND > the maximum of 200 per-probe PCA sign-flip replicates of the same statistic
                          (theory-v2 Cor. 8.1; exact null for Sigma-commuting curvature).
  D  beats decoys      := wikitext held-out pairs, absI: hs_v10.two_way_T1 of H vs the 8 twins has T1 > log 2,
                          CI95 lower > 0, rank 1 of 9, p_PI < 0.05 (the v10 "strong" rule).
  U  per unit main eff := two_way_T1 on pu (H vs 8 twins): T1 > 0 and CI95 lower > 0.
  J  beats J5          := mean log(absI_H / absI_J5) over held-out wikitext records has cluster-bootstrap CI95 lower > 0.
  Jd J-distinct (flag) := overlap(Uq, J25) < 0.5 and overlap(Up, J25) < 0.5 (mean squared cosine; random 25/1024).
  S  composition-selective := bank candidate (Hb): s = log absI_Hb - mean_j log absI_twin_j per record;
                          mean_s(composition TEST items) - mean_s(null_unrelated) has CI95 lower > 0 (clusters resampled
                          within each group) AND Hb beats its twins on composition TEST items (two_way_T1 CI95 lower > 0).
  Verdict per layer: "INSTRUMENT INVALID" if not V0; else "cross-position H-space candidate" iff R and D and U;
  "... beyond J5" iff also J; S and Jd are reported as separate flags. Overall: the per-layer verdicts as they stand.
Secondary, descriptive only: hs_flip5.run_layer variant d (norm estimator, 100 flips, 20 planted) on the wikitext rows;
bank-operator split-half and flip null; distance profile of the cross energy; overlaps with PCA-25 and run-1 H25; the
unmasked operator's top-5; lag strata (lag <= 8 vs > 8: separate operators, and the wikitext interaction test per
stratum); QK/J explained-subspace overlaps and leftover; Lemma 1 anatomy of the cross rows (hs_anatomy.anatomy_batch:
share of the cross-position curvature generated in softmax-attention, DeltaNet and MLP sublayers, on a few windows).
Layer types between l and the readout: 0.8B has full attention at blocks 3, 7, 11, 15, 19, 23, DeltaNet elsewhere.

    python hs_dev_xpos.py timing     # small sizes, out/dev_xpos/timing.json
    python hs_dev_xpos.py full       # out/dev_xpos/full.{json,pt} + ../results/dev/hs_dev_xpos.json
"""
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
import traceback

ARGS = sys.argv[1:]
sys.argv = sys.argv[:1]
MODE = ARGS[0] if ARGS else "timing"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.cuda.set_per_process_memory_fraction(0.42)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "flagship", "src"))  # common.py ROOT is the private layout
import bf16w  # noqa: E402
import hspace as H  # noqa: E402
import hs_v10 as V10  # noqa: E402
import hs_v8  # noqa: E402
import hs_flip5 as F5  # noqa: E402
from hs_v9 import boot_ci  # noqa: E402

T0 = time.time()
T_W, T_B, NEXT, SKIP = 96, 32, 16, 16
PMIN, PMAX = SKIP + 1, T_W - 1 - NEXT          # p in [17, 79]
K5 = 5
LOG2 = math.log(2)
NTW = 8
COMP = ["entity_attribute", "binding_backup", "negation_truth", "relational_composition"]
NULLB = "null_unrelated"
CFG = {
    "timing": dict(layers=[12], n_fit=8, nprobe=2, bank_fit_items=6, n_xi=1, n_test=3, npt=1, bank_test=3, null_test=3,
                   nflip=10, f5_n=4, f5_plant=2, plant_wins=4, n_fd=2, bh=2, bb=4, n_anat=1),
    "full": dict(layers=[12], n_fit=64, nprobe=3, bank_fit_items=999, n_xi=1, n_test=48, npt=1, bank_test=64,
                 null_test=999, nflip=200, f5_n=50, f5_plant=10, plant_wins=32, n_fd=6, bh=2, bb=4, n_anat=4),
}[MODE]
OUTD = H.OUT / "dev_xpos"
OUTD.mkdir(exist_ok=True)
PROG = H.OUT / "progress" / "xpos.json"
SUMF = H.HERE.parent / "results" / "dev" / "hs_dev_xpos.json"


def log(msg):
    print(f"[xpos {time.strftime('%H:%M:%S')} +{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


# ------------------------------------------------------------------ live progress (dashboard)
class Prog:
    lock = threading.Lock()
    st = {}

    @classmethod
    def init(cls, n_stages):
        started = T0
        try:
            old = json.load(open(PROG))
            if old.get("status") == "running" and old.get("started"):
                started = old["started"]
        except Exception:  # noqa: BLE001
            pass
        cls.st = {"job": "xpos", "title": "Cross-position H-lens (0.8B)", "status": "running", "stage": "load",
                  "stage_index": 1, "stage_count": n_stages, "done": 0, "total": 1, "unit": "steps",
                  "started": started, "updated": time.time(), "note": f"mode {MODE}"}
        cls.write()
        threading.Thread(target=cls._beat, daemon=True).start()

    @classmethod
    def write(cls):
        with cls.lock:
            cls.st["updated"] = time.time()
            PROG.parent.mkdir(parents=True, exist_ok=True)
            tmp = str(PROG) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(cls.st, f)
            os.replace(tmp, PROG)

    @classmethod
    def _beat(cls):
        while cls.st.get("status") == "running":
            time.sleep(5)
            try:
                cls.write()
            except Exception:  # noqa: BLE001
                pass

    @classmethod
    def stage(cls, name, idx, total, unit):
        cls.st.update(stage=name, stage_index=idx, done=0, total=int(total), unit=unit)
        cls.write()

    @classmethod
    def tick(cls, done=None, note=None, inc=1):
        cls.st["done"] = int(cls.st["done"] + inc if done is None else done)
        if note is not None:
            cls.st["note"] = str(note)[:160]
        cls.write()

    @classmethod
    def end(cls, status, note):
        cls.st.update(status=status, note=str(note)[:300])
        cls.write()


# ------------------------------------------------------------------ model with an optional cross-position plant
class XModel(H.Model):
    plant = None

    def run(self, h, start, stop=None, plant=None):
        stop = self.n_layers if stop is None else stop
        B = h.shape[0]
        for i in range(start, stop):
            out = self.layers[i](h, **self.kwargs(i, B))
            hn = out[0] if isinstance(out, tuple) else out
            P = self.plant
            if P is not None and i == P["block"]:
                z = (h - P["mu"]) @ P["Sinvh"]
                gate = (z[:, :-1] @ P["a"]) * (z[:, 1:] @ P["b"])          # gate at t uses z[t-1].a and z[t].b
                hn = hn + P["G"] * F.pad(gate, (1, 0))[..., None] * P["w"]
            h = hn
        return h


def use_T(m, T):
    if m.curT != T:
        m._capture_kwargs(T)
        m.curT = T


def sample_c(m, n, gen):
    eta = torch.randn(n, m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
    return (eta.to(torch.bfloat16) @ m.W_U).float() * m.gain[None]


def hvp(m, X, l, tpos, V, C, rmask):
    """X [B,T,d] layer-l residual; tangent V[b] (raw coords) at tpos[b]; F = sum_b sum_t rmask[b,t] C[b].hL[b,t].
    Returns (H v) [B,T,d] at every position (exact double backward)."""
    ar = torch.arange(X.shape[0], device="cuda")
    x = X.clone().requires_grad_(True)
    with torch.enable_grad():
        hL = m.run(x, l + 1)
        Fv = (hL * rmask[..., None] * C[:, None, :]).sum()
        (g,) = torch.autograd.grad(Fv, x, create_graph=True)
        (hv,) = torch.autograd.grad((g[ar, tpos] * V).sum(), x)
    return hv.detach()


def grad_at(m, X, l, C, rmask):
    x = X.clone().requires_grad_(True)
    with torch.enable_grad():
        hL = m.run(x, l + 1)
        Fv = (hL * rmask[..., None] * C[:, None, :]).sum()
        (g,) = torch.autograd.grad(Fv, x)
    return g.detach()


def overlap(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def top_k(M, k=K5):
    ev, U = torch.linalg.eigh(M.double().cuda())
    o = torch.argsort(ev, descending=True)
    return ev[o].cpu(), U[:, o[:k]].cpu()


def top5_f32(M):
    ev, U = torch.linalg.eigh(M)
    return U[:, torch.argsort(ev, descending=True)[:K5]]


def qk_spaces(m, l, Sh, k=25):
    """Static explained subspaces: pooled whitened W_QK of downstream softmax heads and DeltaNet heads (see docstring)."""
    acc = {kind: {"p": torch.zeros(m.d, m.d, device="cuda", dtype=torch.float64), "q": torch.zeros(m.d, m.d, device="cuda", dtype=torch.float64),
                  "n": 0} for kind in ("attn", "dn")}
    Shd = Sh.double()
    for i in range(l + 1, m.n_layers):
        blk = m.layers[i]
        gin = 1.0 + blk.input_layernorm.weight.double()
        if blk.layer_type == "full_attention":
            at = blk.self_attn
            hd = at.head_dim
            Wq = at.q_proj.weight.double().view(-1, 2 * hd, m.d)[:, :hd]          # [nh, hd, d] query half
            Wk = at.k_proj.weight.double().view(-1, hd, m.d)                       # [nkv, hd, d]
            gg = (1.0 + at.q_norm.weight.double()) * (1.0 + at.k_norm.weight.double())
            grp = Wq.shape[0] // Wk.shape[0]
            pairs, kind = [(Wq[h], Wk[h // grp], gg) for h in range(Wq.shape[0])], "attn"
        else:
            la = blk.linear_attn
            W = la.in_proj_qkv.weight.double()
            kd, hk = la.key_dim, la.head_k_dim
            Wq = W[:kd].view(-1, hk, m.d); Wk = W[kd:2 * kd].view(-1, hk, m.d)
            pairs, kind = [(Wq[h], Wk[h], None) for h in range(Wq.shape[0])], "dn"
        for Wq_, Wk_, g_ in pairs:
            Qm = (Wq_ * gin[None]) @ Shd                                           # [hd, d]: whitened p-side map
            Km = (Wk_ * gin[None]) @ Shd
            if g_ is not None:
                Qm = Qm * g_[:, None]
            A = Qm.T @ Km                                                          # [d, d], x_p^T A x_q
            A = A / A.norm()
            acc[kind]["p"] += A @ A.T; acc[kind]["q"] += A.T @ A; acc[kind]["n"] += 1
    out = {}
    for kind in acc:
        for side in ("p", "q"):
            if acc[kind]["n"]:
                out[f"{kind}_{side}"] = top_k(acc[kind][side], k)[1]
    return out


def anatomy_fd(HA, m, ctxs, l, pos, C, Vv):
    """hs_anatomy.anatomy_batch with the forward tangents from a central difference instead of forward-mode AD
    (bf16w's autograd Functions define no jvp). The recon error vs the exact HVP checks the substitution."""
    B = len(ctxs); c0 = ctxs[0]
    subs = HA.subs_after(m, l)
    x0 = torch.cat([c.h_all[l] for c in ctxs], 0)
    V = torch.zeros_like(x0); V[torch.arange(B), pos] = Vv
    valid = c0.valid
    eps = 1e-3 * float(x0[:, SKIP:T_W - 1].norm(dim=-1).mean()) / float(Vv.norm(dim=-1).mean())
    with torch.no_grad():
        yp, _ = HA.run_subs(m, c0, x0 + eps * V, subs, B)
        ym, _ = HA.run_subs(m, c0, x0 - eps * V, subs, B)
        tangents = [(a - b) / (2 * eps) for a, b in zip(yp, ym)]
        del yp, ym
    x = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        _, yL = HA.run_subs(m, c0, x, subs, B)
        Fv = (yL[:, valid] * C[:, None, :]).sum()
        (g,) = torch.autograd.grad(Fv, x, create_graph=True)
        (hv,) = torch.autograd.grad((g * V).sum(), x)
    hv = hv.detach(); del g, yL, Fv
    x = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        ys, yL = HA.run_subs(m, c0, x, subs, B)
        outs = ys[1:] + [yL]
        Fv = (yL[:, valid] * C[:, None, :]).sum()
        adj = torch.autograd.grad(Fv, outs, retain_graph=True)
        groups = {}
        for si, (i, k) in enumerate(subs):
            kind = k if k == "mlp" else ("attn" if m.layers[i].layer_type == "full_attention" else "deltanet")
            groups.setdefault(kind, []).append(si)
        pulls, keys = [], []
        for kind, sis in groups.items():
            ws = []
            for si in sis:
                i, k = subs[si]
                y = ys[si].detach().requires_grad_(True)
                inner = (adj[si].detach() * HA.branch(m, c0.kwargs_for(i, B), i, k, y)).sum()
                (gy,) = torch.autograd.grad(inner, y, create_graph=True)
                (w_,) = torch.autograd.grad((gy * tangents[si]).sum(), y)
                ws.append(w_.detach())
            pb = torch.autograd.grad([ys[si] for si in sis], x, grad_outputs=ws, retain_graph=True)[0]
            pulls.append(pb.detach()); keys.append((kind, "all"))
            del ws
    return hv, torch.stack(pulls), keys


def anatomy_shares(m, fit, l, Sh, gen, nb, label):
    """Lemma 1 decomposition of q-side cross rows (tangent at p, rows at q < p) by generating sublayer type."""
    import hs_anatomy as HA

    class Shim:
        pass
    rmask = torch.zeros(T_W, dtype=torch.bool, device="cuda"); rmask[SKIP:T_W - 1] = True
    tot, recon = {}, []
    for k in range(nb):
        ctxs = []
        for b in range(2):
            ids = fit[(2 * k + b) % len(fit)][0][None].cuda()
            with torch.no_grad():
                X = m.run(m.embed(ids), 0, l + 1)
            c = Shim(); c.h_all = {l: X}; c.valid = rmask; c.kwargs_for = (lambda i, B: m.kwargs(i, B)); ctxs.append(c)
        pos = torch.randint(PMIN, PMAX + 1, (2,), generator=gen, device="cuda")
        C = sample_c(m, 2, gen)
        Vv = torch.randn(2, m.d, device="cuda", generator=gen) @ Sh
        hv, pulls, keys = anatomy_fd(HA, m, ctxs, l, pos, C, Vv)
        recon.append(float((pulls.sum(0) - hv).norm() / hv.norm()))
        for b in range(2):
            rr = list(range(SKIP, int(pos[b])))
            den = float(hv[b, rr].pow(2).sum())
            for gi, (kind, depth) in enumerate(keys):
                sh = float((pulls[gi, b, rr] * hv[b, rr]).sum()) / den / (2 * nb)
                tot[kind] = tot.get(kind, 0.0) + sh
        Prog.tick(note=f"{label}: batch {k+1}/{nb}, recon err {recon[-1]:.1e}")
        del pulls, hv
        torch.cuda.empty_cache()
    return {"share_by_type": tot, "recon_rel_err_max": max(recon), "n_rows_batches": nb}


# ------------------------------------------------------------------ data
def wiki_split(tok):
    wins, art = V10.windows_val(tok, 400, T_W, seed=2026, kmin=2)
    ua = sorted(set(art))
    random.Random(26).shuffle(ua)
    fit_a, test_a = set(ua[: len(ua) // 2]), set(ua[len(ua) // 2:])
    fa = sorted(fit_a)
    halfA = {a: (i % 2 == 0) for i, a in enumerate(fa)}
    fit = [(torch.tensor(w), a, halfA[a]) for w, a in zip(wins, art) if a in fit_a]
    test = [(torch.tensor(w), a) for w, a in zip(wins, art) if a in test_a]
    random.Random(27).shuffle(fit)
    random.Random(28).shuffle(test)
    return fit, test


def bank_prep(tok, item, cell):
    pr = item["prompts"][cell]
    ids = tok(pr, add_special_tokens=False)["input_ids"]
    if len(ids) > T_B:
        return None
    ns = item["natural_site"][cell] if isinstance(item["natural_site"], dict) else item["natural_site"]
    if ns not in pr:
        return None
    pre = tok(pr[: pr.index(ns) + len(ns)], add_special_tokens=False)["input_ids"]
    if ids[: len(pre)] != pre:
        return None
    q, p = len(pre) - 1, len(ids) - 1
    if not q < p:
        return None
    X_ = tok(item["X"], add_special_tokens=False)["input_ids"][0]
    Y_ = tok(item["Y"], add_special_tokens=False)["input_ids"][0]
    if X_ == Y_:
        return None
    return {"ids": ids + [ids[-1]] * (T_B - len(ids)), "q": q, "p": p, "X": X_, "Y": Y_, "cluster": f"{item['context']}:{item['cluster']}",
            "item": item["item"], "cell": cell}


def bank_split(tok):
    MAN = json.load(open(H.HERE / "v10_banks" / "MANIFEST.json"))
    fit, test, null, skipped = [], [], [], 0
    for b in COMP + [NULLB]:
        raw = open(H.HERE / "v10_banks" / f"{b}.jsonl", "rb").read()
        assert hashlib.sha256(raw).hexdigest() == MAN["banks"][b]["sha256"], f"bank {b} hash mismatch"
        items = [json.loads(x) for x in raw.decode("utf-8").splitlines() if x.strip()]
        cl = sorted({it["cluster"] for it in items})
        random.Random(29).shuffle(cl)
        fit_c = set(cl[: len(cl) // 2])
        for it in items:
            if b == NULLB:
                r = bank_prep(tok, it, "11")
                if r is None:
                    skipped += 1
                else:
                    null.append(r)
            elif it["cluster"] in fit_c:
                for cell in ("00", "01", "10", "11"):
                    r = bank_prep(tok, it, cell)
                    if r is None:
                        skipped += 1
                    else:
                        r["half"] = (sorted(fit_c).index(it["cluster"]) % 2 == 0)
                        fit.append(r)
            else:
                r = bank_prep(tok, it, "11")
                if r is None:
                    skipped += 1
                else:
                    test.append(r)
    return fit, test, null, skipped


# ------------------------------------------------------------------ operator rows
def wiki_rows(m, fit, l, side, nprobe, gen, Sh, bh, label):
    jobs = []
    for wi in range(len(fit)):
        for _ in range(nprobe):
            pos = int(torch.randint(PMIN, PMAX + 1, (1,), generator=gen, device="cuda")) if side == "q" else \
                int(torch.randint(SKIP, PMAX, (1,), generator=gen, device="cuda"))
            jobs.append((wi, pos))
    rmask = torch.zeros(1, T_W, device="cuda"); rmask[0, SKIP:T_W - 1] = 1
    Ys, pid, half, dist = [], [], [], []
    t1 = time.time()
    for s in range(0, len(jobs), bh):
        ch = jobs[s:s + bh]
        ids = torch.stack([fit[wi][0] for wi, _ in ch]).cuda()
        with torch.no_grad():
            X = m.run(m.embed(ids), 0, l + 1)
        C = sample_c(m, len(ch), gen)
        V = torch.randn(len(ch), m.d, device="cuda", generator=gen) @ Sh
        tpos = torch.tensor([pos for _, pos in ch], device="cuda")
        hv = hvp(m, X, l, tpos, V, C, rmask.expand(len(ch), -1))
        for b, (wi, pos) in enumerate(ch):
            rr = list(range(SKIP, pos)) if side == "q" else list(range(pos + 1, PMAX + 1))
            Ys.append(hv[b, rr] @ Sh)
            j = s + b
            pid += [2 * j + (0 if fit[wi][2] else 1)] * len(rr)
            half += [bool(fit[wi][2])] * len(rr)
            dist += [abs(pos - r) for r in rr]
        Prog.tick(inc=len(ch), note=f"{label}: {min(len(jobs), s+bh)}/{len(jobs)} HVPs, {(time.time()-t1)/(s//bh+1):.2f}s/batch")
    return torch.cat(Ys), torch.tensor(pid), torch.tensor(half), torch.tensor(dist), (time.time() - t1) / len(jobs)


def bank_rows(m, items, l, side, n_xi, gen, Sh, bb, label):
    jobs = [(i, r) for i in range(len(items)) for r in range(n_xi)]
    Ys, pid, half = [], [], []
    t1 = time.time()
    for s in range(0, len(jobs), bb):
        ch = jobs[s:s + bb]
        its = [items[i] for i, _ in ch]
        ids = torch.tensor([it["ids"] for it in its], device="cuda")
        with torch.no_grad():
            X = m.run(m.embed(ids), 0, l + 1)
        rmask = torch.zeros(len(ch), T_B, device="cuda")
        for b, it in enumerate(its):
            rmask[b, it["p"]] = 1
        C = sample_c(m, len(ch), gen)
        V = torch.randn(len(ch), m.d, device="cuda", generator=gen) @ Sh
        tpos = torch.tensor([it["p"] if side == "q" else it["q"] for it in its], device="cuda")
        rpos = torch.tensor([it["q"] if side == "q" else it["p"] for it in its], device="cuda")
        hv = hvp(m, X, l, tpos, V, C, rmask)
        Ys.append(hv[torch.arange(len(ch), device="cuda"), rpos] @ Sh)
        for b, it in enumerate(its):
            pid.append(2 * (s + b) + (0 if it["half"] else 1)); half.append(bool(it["half"]))
        Prog.tick(inc=len(ch), note=f"{label}: {min(len(jobs), s+bb)}/{len(jobs)} HVPs")
    return torch.cat(Ys), torch.tensor(pid), torch.tensor(half), (time.time() - t1) / len(jobs)


def split_flip(Y, half, pid, P, nflip, seed, label):
    """Raw top-5 split-half overlap and its per-probe PCA sign-flip null (Cor. 8.1)."""
    Yp = (Y.cuda() @ P.float().cuda())
    hA = half.cuda()
    pr = pid.cuda()
    npb = int(pid.max()) + 1

    def stat(Z):
        A, B = Z[hA], Z[~hA]
        return overlap(top5_f32(A.T @ A / len(A)), top5_f32(B.T @ B / len(B)))
    obs = stat(Yp)
    g = torch.Generator(device="cuda").manual_seed(seed)
    null = []
    for r in range(nflip):
        Fl = (torch.randint(0, 2, (npb, Yp.shape[1]), generator=g, device="cuda").float() * 2 - 1)[pr]
        null.append(stat(Yp * Fl))
        if r % 10 == 0 or r == nflip - 1:
            Prog.tick(inc=0, note=f"{label}: flip {r+1}/{nflip}, obs split5 {obs:.3f}, null max {max(null):.3f}")
    Prog.tick()
    return {"obs_split5": obs, "null_max": float(max(null)), "null_q95": float(np.quantile(null, 0.95)),
            "null_mean": float(np.mean(null)), "p": (1 + sum(x >= obs for x in null)) / (nflip + 1), "nflip": nflip}


# ------------------------------------------------------------------ interaction (Lemma 3 four-point differences)
def forward_rows(m, l, Xs, rowpos, chunk):
    outs = []
    for c0 in range(0, len(Xs), chunk):
        with torch.no_grad():
            hL = m.run(torch.stack(Xs[c0:c0 + chunk]), l + 1)
        outs.append(m.text.norm(hL[:, rowpos]))
    return torch.cat(outs)                                     # [n, len(rowpos), d] normed


def wiki_scores(m, Xn, tgt):
    """Xn [n, R, d] normed rows; tgt [R]; returns sum_r log p(tgt_r) per sequence (fp32 logits, fp64 log-softmax)."""
    n, R, d = Xn.shape
    if n * R > 512:
        cs = max(1, 512 // R)
        return np.concatenate([wiki_scores(m, Xn[i:i + cs], tgt) for i in range(0, n, cs)])
    x = Xn.reshape(-1, d).float()
    lse = torch.full((x.shape[0],), -float("inf"), dtype=torch.float64, device="cuda")
    tl = torch.zeros(x.shape[0], dtype=torch.float64, device="cuda")
    tg = tgt.repeat(n)
    for s in range(0, m.W_U.shape[0], 16384):
        Wc = m.W_U[s:s + 16384].float()
        lg = F.linear(x, Wc).double()
        lse = torch.logaddexp(lse, torch.logsumexp(lg, -1))
        inr = (tg >= s) & (tg < s + Wc.shape[0])
        if inr.any():
            ix = torch.nonzero(inr).squeeze(1)
            tl[ix] = lg[ix, tg[ix] - s]
    return (tl - lse).reshape(n, R).sum(1).cpu().numpy()


def arms_seqs(X, q, p, arms):
    seqs, key = [X], {}
    for name, (Aq, Ap) in arms.items():
        for i in range(K5):
            xx = X.clone(); xx[q] += Aq[:, i]; key[(name, "u", i)] = len(seqs); seqs.append(xx)
        for j in range(K5):
            xx = X.clone(); xx[p] += Ap[:, j]; key[(name, "v", j)] = len(seqs); seqs.append(xx)
        for i in range(K5):
            for j in range(K5):
                xx = X.clone(); xx[q] += Aq[:, i]; xx[p] += Ap[:, j]; key[(name, "uv", i, j)] = len(seqs); seqs.append(xx)
    return seqs, key


def arms_out(fs, key, arms):
    f0 = fs[0]
    out = {"f0": float(f0)}
    for name in arms:
        Mu = np.array([fs[key[(name, "u", i)]] - f0 for i in range(K5)])
        Mv = np.array([fs[key[(name, "v", j)]] - f0 for j in range(K5)])
        I = np.array([[fs[key[(name, "uv", i, j)]] - fs[key[(name, "u", i)]] - fs[key[(name, "v", j)]] + f0 for j in range(K5)]
                      for i in range(K5)])
        out[name] = {"I": I.tolist(), "Mu": Mu.tolist(), "Mv": Mv.tolist()}
    return out


def rec_stats(r, name):
    I, Mu, Mv = np.array(r[name]["I"]), np.array(r[name]["Mu"]), np.array(r[name]["Mv"])
    absI = float(np.abs(I).mean())
    main = float(((np.abs(Mu)[:, None] + np.abs(Mv)[None, :]) / 2).mean())
    return absI, main, absI / max(main, 1e-15)


def compare(recs, arm, twins, other=None, key="cl"):
    cl = [r[key] for r in recs]
    ex = np.array([rec_stats(r, arm)[0] for r in recs])
    exu = np.array([rec_stats(r, arm)[2] for r in recs])
    tw = np.array([[rec_stats(r, t)[0] for t in twins] for r in recs])
    twu = np.array([[rec_stats(r, t)[2] for t in twins] for r in recs])
    out = {"n": len(recs), "n_clusters": len(set(cl)), "absI": V10.two_way_T1(ex, tw, cl), "per_unit": V10.two_way_T1(exu, twu, cl),
           "mean_absI": float(ex.mean()), "mean_absI_twins": float(tw.mean()),
           "mean_main": float(np.mean([rec_stats(r, arm)[1] for r in recs])),
           "mean_main_twins": float(np.mean([[rec_stats(r, t)[1] for t in twins] for r in recs]))}
    if other is not None:
        eo = np.array([rec_stats(r, other)[0] for r in recs])
        eou = np.array([rec_stats(r, other)[2] for r in recs])
        v = np.log((ex + 1e-15) / (eo + 1e-15)); vu = np.log((exu + 1e-15) / (eou + 1e-15))
        out[f"vs_{other}"] = {"mean_log_ratio": float(v.mean()), "ci95": boot_ci(v, cl), "mean_absI_other": float(eo.mean()),
                              "per_unit_mean_log_ratio": float(vu.mean()), "per_unit_ci95": boot_ci(vu, cl),
                              "frac_records_H_gt": float((ex > eo).mean())}
    return out


def selectivity(comp, null, arm, twins, B=2000, seed=0):
    def s_of(r):
        return math.log(rec_stats(r, arm)[0] + 1e-15) - float(np.mean([math.log(rec_stats(r, t)[0] + 1e-15) for t in twins]))
    sc, sn = np.array([s_of(r) for r in comp]), np.array([s_of(r) for r in null])
    cc, cn = np.array([r["cl"] for r in comp]), np.array([r["cl"] for r in null])
    rb = np.random.default_rng(seed)
    uc, un = np.unique(cc), np.unique(cn)
    ic, inn = {u: np.where(cc == u)[0] for u in uc}, {u: np.where(cn == u)[0] for u in un}
    bs = []
    for _ in range(B):
        a = np.concatenate([ic[u] for u in rb.choice(uc, len(uc))])
        b = np.concatenate([inn[u] for u in rb.choice(un, len(un))])
        bs.append(sc[a].mean() - sn[b].mean())
    return {"S": float(sc.mean() - sn.mean()), "ci95": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))],
            "mean_s_comp": float(sc.mean()), "mean_s_null": float(sn.mean()), "n_comp": len(sc), "n_null": len(sn),
            "mean_absI_comp": float(np.mean([rec_stats(r, arm)[0] for r in comp])),
            "mean_absI_null": float(np.mean([rec_stats(r, arm)[0] for r in null]))}


# ------------------------------------------------------------------ main
def main():
    cfg = CFG
    layers = cfg["layers"]
    per_layer = ["FD check", "wiki operator", "bank operator", "split-half + flip null", "secondary flip5", "anatomy", "planted control",
                 "wiki interaction", "bank interaction"]
    Prog.init(1 + len(per_layer) * len(layers) + 1)
    res = {"mode": MODE, "cfg": cfg, "layers": {}, "timing": {}, "errors": {}}
    raw = {}
    outj, outp = OUTD / f"{MODE}.json", OUTD / f"{MODE}.pt"

    def save():
        tmp = str(outj) + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1); os.replace(tmp, outj)
        torch.save(raw, str(outp) + ".tmp"); os.replace(str(outp) + ".tmp", outp)

    m = XModel("Qwen/Qwen3.5-0.8B", T_W)
    m.curT = T_W
    bf16w.MODE = "fp32"
    tok, d = m.tok, m.d
    log(f"loaded model; mem {torch.cuda.memory_allocated()/2**30:.2f} GB; mode {MODE}; layers {layers}")
    Prog.tick(note="model loaded")
    TS = torch.load(H.OUT / "hspace_small.pt", map_location="cpu", weights_only=False)
    fit, test = wiki_split(tok)
    bfit, btest, bnull, bskip = bank_split(tok)
    log(f"wiki: {len(fit)} fit windows ({len(set(a for _, a, _ in fit))} articles), {len(test)} test windows "
        f"({len(set(a for _, a in test))} articles); banks: fit rows {len(bfit)}, test {len(btest)}, null {len(bnull)}, skipped {bskip}")
    fit = fit[: cfg["n_fit"]]
    test = test[: cfg["n_test"]]
    rngb = random.Random(31)
    bfit = rngb.sample(bfit, min(len(bfit), cfg["bank_fit_items"] * 4)) if cfg["bank_fit_items"] < 999 else bfit
    btest = rngb.sample(btest, min(len(btest), cfg["bank_test"])) if cfg["bank_test"] < 999 else btest
    bnull = bnull[: cfg["null_test"]]
    res["data"] = {"wiki_fit": len(fit), "wiki_test": len(test), "bank_fit_rows": len(bfit), "bank_test": len(btest),
                   "bank_null": len(bnull), "bank_skipped": bskip,
                   "bank_test_by_bank": {b: sum(r["cluster"].startswith(b + ":") for r in btest) for b in COMP}}
    from huggingface_hub import hf_hub_download
    lens = torch.load(hf_hub_download("neuronpedia/jacobian-lens", "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt",
                                      revision="main", local_files_only=True), map_location="cpu", weights_only=True)
    U2 = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    for s in range(0, m.W_U.shape[0], 16384):
        Wg = m.W_U[s:s + 16384].float() * m.gain[None]
        U2 += (Wg.T @ Wg).double()
    U2 = (U2 / m.W_U.shape[0]).float()
    si = 1
    for l in layers:
        L = {}
        res["layers"][str(l)] = L
        rl = {}
        raw[l] = rl
        gen = torch.Generator(device="cuda").manual_seed(1000 + l)
        Sh = TS[f"Sh{l}"].float().cuda(); mu = TS[f"mu{l}"].float().cuda()
        Shd = TS[f"Sh{l}"].double()
        Sih = torch.linalg.inv(Shd).float().cuda()
        J25 = TS[f"J25_{l}"].double()
        H25run1 = TS[f"H25_{l}"].double()
        P, keep, Kd = hs_v8.drop_mask(Shd, TS[f"mu{l}"].double())   # PCA basis (whitened frame, descending) + v5 mask
        Pk = ((P * keep[None]) @ P.T).float().cuda()                     # projector onto the kept PCA axes
        L["mask"] = {"n_dropped_axes": int((keep == 0).sum()), "dropped_axes_first20": torch.nonzero(keep == 0).squeeze(1)[:20].tolist()}
        use_T(m, T_W)
        # ---------------- FD check
        si += 1; Prog.stage(f"L{l} FD check", si, cfg["n_fd"], "pairs")
        fd = []
        rmask = torch.zeros(1, T_W, device="cuda"); rmask[0, SKIP:T_W - 1] = 1
        for k in range(cfg["n_fd"]):
            ids = fit[k % len(fit)][0][None].cuda()
            with torch.no_grad():
                X = m.run(m.embed(ids), 0, l + 1)
            p = int(torch.randint(PMIN, PMAX + 1, (1,), generator=gen, device="cuda"))
            q = int(torch.randint(SKIP, p, (1,), generator=gen, device="cuda"))
            c = sample_c(m, 1, gen)
            v = torch.randn(1, d, device="cuda", generator=gen) @ Sh; v = v / v.norm()
            hq = hvp(m, X, l, torch.tensor([p], device="cuda"), v, c, rmask)[0, q]
            scale = X[0, SKIP:T_W - 1].norm(dim=-1).mean().item()
            errs = {}
            for rel in (3e-2, 1e-2, 3e-3, 1e-3):
                eps = rel * scale
                Xp, Xm = X.clone(), X.clone(); Xp[0, p] += eps * v[0]; Xm[0, p] -= eps * v[0]
                fdv = (grad_at(m, Xp, l, c, rmask)[0, q] - grad_at(m, Xm, l, c, rmask)[0, q]) / (2 * eps)
                errs[str(rel)] = float((fdv - hq).norm() / hq.norm())
            fd.append({"q": q, "p": p, "norm_Cv": float(hq.norm()), "rel_err": errs, "best": min(errs.values())})
            Prog.tick(note=f"FD pair {k+1}: best rel err {min(errs.values()):.2e}")
        L["fd"] = {"pairs": fd, "max_best_rel_err": max(x["best"] for x in fd), "pass": all(x["best"] < 1e-2 for x in fd)}
        log(f"L{l} FD: best rel errs {[round(x['best'], 6) for x in fd]} -> pass {L['fd']['pass']}")
        save()
        # ---------------- wiki operator
        si += 1; Prog.stage(f"L{l} wiki operator", si, 2 * len(fit) * cfg["nprobe"], "HVPs")
        W, Wun = {}, {}
        for side in ("q", "p"):
            Y, pid, half, dist, spr = wiki_rows(m, fit, l, side, cfg["nprobe"], gen, Sh, cfg["bh"], f"L{l} wiki {side}-side")
            Wun[side] = Y
            W[side] = (Y @ Pk, pid, half, dist)
            res["timing"][f"L{l}_wiki_hvp_s_per_row"] = spr
            log(f"L{l} wiki {side}-side: {len(Y)} rows from {len(fit)*cfg['nprobe']} HVPs, {spr:.3f} s/HVP, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GB")
        ops = {}
        for side in ("q", "p"):
            Y, pid, half, dist = W[side]
            M = (Y.double().T @ Y.double()) / len(Y)
            ev, U = top_k(M)
            ops[side] = (M, ev, U)
            dd = dist.numpy(); e = Y.pow(2).sum(1).cpu().numpy()
            bins = [(1, 1), (2, 4), (5, 16), (17, 64)]
            L[f"wiki_{side}"] = {"n_rows": len(Y), "eigs_top10": ev[:10].tolist(), "top5_energy_frac": float(ev[:5].sum() / ev.sum()),
                                 "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum()),
                                 "energy_share_by_distance": {f"{a}-{b}": float(e[(dd >= a) & (dd <= b)].sum() / e.sum()) for a, b in bins},
                                 "rows_share_by_distance": {f"{a}-{b}": float(((dd >= a) & (dd <= b)).mean()) for a, b in bins}}
        Uq, Up = ops["q"][2], ops["p"][2]
        # secondary: unmasked operator and lag strata
        L["secondary"] = {}
        for side in ("q", "p"):
            Yu = Wun[side].double()
            evu, Uu = top_k(Yu.T @ Yu / len(Yu))
            U_ = ops[side][2]
            L["secondary"][f"unmasked_{side}"] = {"top5_energy_frac": float(evu[:5].sum() / evu.sum()), "overlap_with_masked_top5": overlap(Uu, U_),
                                                  "vs_PCA5": overlap(Uu, P[:, :5]), "vs_J25": overlap(Uu, TS[f"J25_{l}"].double())}
            Ym, dist = W[side][0], W[side][3].cuda()
            for nm_, msk in (("lag_le8", dist <= 8), ("lag_gt8", dist > 8)):
                Ys_ = Ym[msk].double()
                if len(Ys_) > 50:
                    evl, Ul = top_k(Ys_.T @ Ys_ / len(Ys_))
                    L["secondary"][f"{nm_}_{side}"] = {"n_rows": len(Ys_), "overlap_with_full_top5": overlap(Ul, U_),
                                                       "mean_row_energy": float(Ys_.pow(2).sum(1).mean()),
                                                       "vs_J25": overlap(Ul, TS[f"J25_{l}"].double())}
                    rl[f"U_{nm_}_{side}"] = Ul
        L["geometry"] = {"Uq_vs_J25": overlap(Uq, J25), "Up_vs_J25": overlap(Up, J25), "random_vs_J25": 25 / d,
                         "Uq_vs_Up": overlap(Uq, Up), "Uq_vs_PCA25": overlap(Uq, P[:, :25]), "Up_vs_PCA25": overlap(Up, P[:, :25]),
                         "Uq_vs_run1_H25": overlap(Uq, H25run1), "Up_vs_run1_H25": overlap(Up, H25run1),
                         "J25_energy_frac_Mq": float(torch.trace(J25.T @ ops["q"][0].cpu() @ J25) / torch.trace(ops["q"][0].cpu())),
                         "J25_energy_frac_Mp": float(torch.trace(J25.T @ ops["p"][0].cpu() @ J25) / torch.trace(ops["p"][0].cpu()))}
        L["geometry"]["Jd_flag"] = bool(L["geometry"]["Uq_vs_J25"] < 0.5 and L["geometry"]["Up_vs_J25"] < 0.5)
        # explained subspaces (literature kill-check): pooled downstream W_QK (softmax, DeltaNet) and J25
        try:
            QK = qk_spaces(m, l, Sh)
            ex = {}
            for side, U_ in (("q", Uq), ("p", Up)):
                for kind in ("attn", "dn"):
                    if f"{kind}_{side}" in QK:
                        ex[f"U{side}_vs_QK25_{kind}_{side}side"] = overlap(U_, QK[f"{kind}_{side}"])
                        oth = "p" if side == "q" else "q"
                        ex[f"U{side}_vs_QK25_{kind}_{oth}side"] = overlap(U_, QK[f"{kind}_{oth}"])
                span = torch.linalg.qr(torch.cat([QK[k_] for k_ in QK if k_.endswith("_" + side)] + [J25], 1))[0]
                ex[f"U{side}_leftover_after_QKside+J25"] = 1 - overlap(U_, span)
                ex[f"random_leftover_{side}"] = 1 - span.shape[1] / d
            L["explained"] = ex
            rl["QK"] = QK
            log(f"L{l} explained: " + " ".join(f"{k_} {v:.3f}" for k_, v in ex.items()))
        except Exception:  # noqa: BLE001
            res["errors"][f"qk_L{l}"] = traceback.format_exc(); log("QK ERROR\n" + res["errors"][f"qk_L{l}"])
        rl["Uq"], rl["Up"] = Uq, Up
        rl["Mq"], rl["Mp"] = ops["q"][0].cpu().float(), ops["p"][0].cpu().float()
        rl["wiki_rows"] = {s: (W[s][0].half().cpu(), W[s][1], W[s][2], W[s][3]) for s in W}
        log(f"L{l} wiki operator: q eigs {[float(f"{x:.3g}") for x in ops["q"][1][:6].tolist()]} p eigs {[float(f"{x:.3g}") for x in ops["p"][1][:6].tolist()]} | "
            f"J25 overlap q {L['geometry']['Uq_vs_J25']:.3f} p {L['geometry']['Up_vs_J25']:.3f} | q-vs-p {L['geometry']['Uq_vs_Up']:.3f}")
        save()
        # ---------------- bank operator
        use_T(m, T_B)
        si += 1; Prog.stage(f"L{l} bank operator", si, 2 * len(bfit) * cfg["n_xi"], "HVPs")
        Bo = {}
        for side in ("q", "p"):
            Y, pid, half, spr = bank_rows(m, bfit, l, side, cfg["n_xi"], gen, Sh, cfg["bb"], f"L{l} bank {side}-side")
            Y = Y @ Pk
            M =(Y.double().T @ Y.double()) / len(Y)
            ev, U = top_k(M)
            Bo[side] = (Y, pid, half, M, ev, U)
            res["timing"][f"L{l}_bank_hvp_s_per_row"] = spr
            L[f"bank_{side}"] = {"n_rows": len(Y), "eigs_top10": ev[:10].tolist(), "top5_energy_frac": float(ev[:5].sum() / ev.sum()),
                                 "participation_ratio": float(ev.sum() ** 2 / (ev ** 2).sum())}
        Bq, Bp = Bo["q"][5], Bo["p"][5]
        L["geometry"].update({"Bq_vs_Uq": overlap(Bq, Uq), "Bp_vs_Up": overlap(Bp, Up), "Bq_vs_J25": overlap(Bq, J25),
                              "Bp_vs_J25": overlap(Bp, J25), "Bq_vs_PCA25": overlap(Bq, P[:, :25]), "Bp_vs_PCA25": overlap(Bp, P[:, :25])})
        rl["Bq"], rl["Bp"] = Bq, Bp
        log(f"L{l} bank operator: q eigs {[float(f"{x:.3g}") for x in Bo["q"][4][:6].tolist()]} | Bq vs Uq {L['geometry']['Bq_vs_Uq']:.3f} "
            f"Bp vs Up {L['geometry']['Bp_vs_Up']:.3f}")
        save()
        # ---------------- split-half + flip null (primary R)
        si += 1; Prog.stage(f"L{l} split-half + flip null", si, 4, "flip tests")
        L["R"] = {}
        for nm, (Y, pid, half) in (("wiki_q", W["q"][:3]), ("wiki_p", W["p"][:3]), ("bank_q", Bo["q"][:3]), ("bank_p", Bo["p"][:3])):
            nf = cfg["nflip"]
            L["R"][nm] = split_flip(Y, half, pid, P, nf, 5000 + l, f"L{l} {nm}")
            log(f"L{l} R {nm}: split5 {L['R'][nm]['obs_split5']:.3f} vs flip null max {L['R'][nm]['null_max']:.3f} "
                f"(mean {L['R'][nm]['null_mean']:.3f}, p {L['R'][nm]['p']:.4f})")
        L["R"]["pass"] = bool(all(L["R"][s]["obs_split5"] >= 0.7 and L["R"][s]["obs_split5"] > L["R"][s]["null_max"] for s in ("wiki_q", "wiki_p")))
        save()
        # ---------------- secondary: hs_flip5 variant d (descriptive)
        si += 1; Prog.stage(f"L{l} secondary flip5", si, 2, "sides")
        try:
            Pd, keep, Kd = hs_v8.drop_mask(Shd, TS[f"mu{l}"].double())
            for side in ("q", "p"):
                Y, pid = Wun[side], W[side][1]
                Yp = (Y.cuda() @ Pd.float().cuda())
                rr = F5.run_layer(Yp, Yp, pid, {"d": keep.cuda()}, Kd, cfg["f5_n"], cfg["f5_plant"], "cuda", 7000 + l, f"L{l} f5 {side}")
                L[f"flip5_{side}"] = {"norm": {k: rr["d"]["norm"][k] for k in ("obs_lam", "null_lam1_max", "obs_split", "null_split3_max",
                                                                               "null_split5_max", "p_lam1", "p_split3")},
                                      "planted": rr["planted"]["norm"]}
                Prog.tick(note=f"f5 {side}: split3 {rr['d']['norm']['obs_split']['k3']:.2f} vs null max {rr['d']['norm']['null_split3_max']:.2f}")
                del Yp
        except Exception:  # noqa: BLE001
            res["errors"][f"flip5_L{l}"] = traceback.format_exc(); log("flip5 ERROR\n" + res["errors"][f"flip5_L{l}"])
        save()
        # ---------------- secondary: Lemma 1 anatomy of the cross rows by sublayer type
        use_T(m, T_W)
        si += 1; Prog.stage(f"L{l} anatomy", si, cfg["n_anat"], "batches")
        try:
            L["anatomy"] = anatomy_shares(m, fit, l, Sh, gen, cfg["n_anat"], f"L{l} anatomy")
            log(f"L{l} anatomy of q-side cross rows: {L['anatomy']}")
        except Exception:  # noqa: BLE001
            res["errors"][f"anatomy_L{l}"] = traceback.format_exc(); log("ANATOMY ERROR\n" + res["errors"][f"anatomy_L{l}"])
        torch.cuda.empty_cache()
        save()
        # ---------------- planted control
        use_T(m, T_W)
        si += 1; Prog.stage(f"L{l} planted control", si, 2 * cfg["plant_wins"] * cfg["nprobe"] + 1, "HVPs")
        try:
            gq = torch.Generator(device="cuda").manual_seed(123 + l)
            a = Pk @ torch.randn(d, device="cuda", generator=gq); a /= a.norm()          # planted dirs inside the kept axes
            b = Pk @ torch.randn(d, device="cuda", generator=gq); b -= (b @ a) * a; b /= b.norm()
            J1 = lens["J"][l + 1].float().cuda()
            _, Uw = H.top_eig(J1.T @ U2 @ J1, 64)
            w = Uw[:, :64] @ torch.randn(64, device="cuda", generator=gq); w /= w.norm()
            del J1
            # calibration: E[(w . dF/dh_{l+1}[t])^2] over t in [17, 79] (planted rows are 1 per probe; ~32 rows per probe)
            ee = []
            for k in range(min(8, len(fit))):
                ids = fit[k][0][None].cuda()
                with torch.no_grad():
                    h1 = m.run(m.embed(ids), 0, l + 2)
                c = sample_c(m, 1, gen)
                g1 = grad_at(m, h1, l + 1, c, rmask)
                ee.append(float((g1[0, PMIN:PMAX + 1] @ w).pow(2).mean()))
            e_wg = float(np.mean(ee))
            rows_per_probe = {s: L[f"wiki_{s}"]["n_rows"] / (len(fit) * cfg["nprobe"]) for s in ("q", "p")}
            Prog.tick(note=f"plant calibration E[(w.g)^2] {e_wg:.3g}")
            PC = {}
            pfit = fit[: cfg["plant_wins"]]
            for side, vec in (("q", a), ("p", b)):
                lam5 = float(ops[side][1][4])
                G = math.sqrt(lam5 * rows_per_probe[side] / e_wg)
                m.plant = {"block": l + 1, "mu": mu, "Sinvh": Sih, "a": a, "b": b, "w": w, "G": G}
                try:
                    Yp_, _, _, _, _ = wiki_rows(m, pfit, l, side, cfg["nprobe"], gen, Sh, cfg["bh"], f"L{l} plant {side}-side")
                finally:
                    m.plant = None
                Yp_ = Yp_ @ Pk
                Mpl =(Yp_.double().T @ Yp_.double()) / len(Yp_)
                ev_, U_ = top_k(Mpl)
                rec_ = float((U_.T @ vec.double().cpu()).pow(2).sum())
                PC[side] = {"G": G, "lambda5_target": lam5, "recovery_top5": rec_,
                            "clean_recovery_top5": float((ops[side][2].T @ vec.double().cpu()).pow(2).sum()),
                            "planted_energy_measured": float(vec.double().cpu() @ Mpl.cpu() @ vec.double().cpu()),
                            "eigs_top6": ev_[:6].tolist()}
                log(f"L{l} planted {side}-side: G {G:.3g}, recovery {rec_:.3f} (clean {PC[side]['clean_recovery_top5']:.4f}), "
                    f"planted-dir energy {PC[side]['planted_energy_measured']:.3g} vs lambda5 {lam5:.3g}")
            PC["e_wg"] = e_wg
            PC["pass"] = bool(PC["q"]["recovery_top5"] >= 0.8 and PC["p"]["recovery_top5"] >= 0.8)
            L["planted"] = PC
        except Exception:  # noqa: BLE001
            m.plant = None
            res["errors"][f"plant_L{l}"] = traceback.format_exc(); log("PLANT ERROR\n" + res["errors"][f"plant_L{l}"])
            L["planted"] = {"pass": False}
        L["V0"] = bool(L["fd"]["pass"] and L["planted"].get("pass", False))
        save()
        # ---------------- wiki interaction
        TWq = V10.twins_w(Uq, P, NTW, 9100 + l); TWp = V10.twins_w(Up, P, NTW, 9200 + l)
        J5 = J25[:, :K5]; PC5 = P[:, :K5]
        Shc = Sh.double()

        def mv(U):
            return (Shc @ U.cuda().double()).float()
        arms = {"H": (mv(Uq), mv(Up)), "J5": (mv(J5), mv(J5)), **{f"tw{j}": (mv(TWq[j]), mv(TWp[j])) for j in range(NTW)}}
        twn = [f"tw{j}" for j in range(NTW)]
        si += 1; Prog.stage(f"L{l} wiki interaction", si, len(test) * cfg["npt"], "pairs")
        recs = []
        gp = random.Random(300 + l)
        t1 = time.time()
        for wi, (w_, art) in enumerate(test):
            ids = w_[None].cuda()
            with torch.no_grad():
                X = m.run(m.embed(ids), 0, l + 1)[0]
            for _ in range(cfg["npt"]):
                p = gp.choices(range(PMIN, PMAX + 1), weights=[pp - SKIP for pp in range(PMIN, PMAX + 1)])[0]
                q = gp.randrange(SKIP, p)
                seqs, key = arms_seqs(X, q, p, arms)
                Xn = forward_rows(m, l, seqs, list(range(p, p + NEXT)), 32)
                fs = wiki_scores(m, Xn, ids[0, p + 1:p + 1 + NEXT])
                r = arms_out(fs, key, arms); r.update({"cl": int(art), "q": q, "p": p})
                recs.append(r)
                Prog.tick(note=f"L{l} wiki pair {len(recs)}: |I| H {rec_stats(r,'H')[0]:.2e} J5 {rec_stats(r,'J5')[0]:.2e} "
                               f"tw0 {rec_stats(r,'tw0')[0]:.2e} ({(time.time()-t1)/len(recs):.1f}s/pair)")
        res["timing"][f"L{l}_wiki_s_per_pair"] = (time.time() - t1) / max(1, len(recs))
        rl["wiki_recs"] = recs
        if len(recs) >= 3:
            Wt = compare(recs, "H", twn, other="J5")
            Wt["J5_vs_twins"] = compare(recs, "J5", twn)["absI"]
            Wt["J5_vs_twins_per_unit"] = compare(recs, "J5", twn)["per_unit"]
            s_ = Wt["absI"]
            Wt["D"] = bool(s_["T1"] > LOG2 and s_["ci95"][0] > 0 and s_["rank"] == 1 and s_["p_PI"] < 0.05)
            Wt["U"] = bool(Wt["per_unit"]["T1"] > 0 and Wt["per_unit"]["ci95"][0] > 0)
            Wt["J"] = bool(Wt["vs_J5"]["ci95"][0] > 0)
            for nm_, sel in (("lag_le8", lambda r: r["p"] - r["q"] <= 8), ("lag_gt8", lambda r: r["p"] - r["q"] > 8)):
                sub = [r for r in recs if sel(r)]
                if len(sub) >= 6 and len({r["cl"] for r in sub}) >= 3:
                    cs = compare(sub, "H", twn, other="J5")
                    Wt[nm_] = {"n": len(sub), "absI": cs["absI"], "per_unit": cs["per_unit"], "vs_J5": cs["vs_J5"]}
            L["wiki_test"] = Wt
            log(f"L{l} wiki test (n={len(recs)}): H vs twins T1 {s_['T1']:+.2f} {[round(x, 2) for x in s_['ci95']]} rank {s_['rank']} "
                f"p_PI {s_['p_PI']:.3f} | per-unit T1 {Wt['per_unit']['T1']:+.2f} {[round(x, 2) for x in Wt['per_unit']['ci95']]} | "
                f"vs J5 {Wt['vs_J5']['mean_log_ratio']:+.2f} {[round(x, 2) for x in Wt['vs_J5']['ci95']]} | D {Wt['D']} U {Wt['U']} J {Wt['J']}")
        save()
        # ---------------- bank interaction
        use_T(m, T_B)
        TBq = V10.twins_w(Bq, P, NTW, 9300 + l); TBp = V10.twins_w(Bp, P, NTW, 9400 + l)
        barms = {"Hb": (mv(Bq), mv(Bp)), "J5": (mv(J5), mv(J5)), **{f"tb{j}": (mv(TBq[j]), mv(TBp[j])) for j in range(NTW)}}
        tbn = [f"tb{j}" for j in range(NTW)]
        si += 1; Prog.stage(f"L{l} bank interaction", si, len(btest) + len(bnull), "items")
        brec = {"comp": [], "null": []}
        t1 = time.time()
        for grp, items in (("comp", btest), ("null", bnull)):
            for it in items:
                ids = torch.tensor([it["ids"]], device="cuda")
                with torch.no_grad():
                    X = m.run(m.embed(ids), 0, l + 1)[0]
                seqs, key = arms_seqs(X, it["q"], it["p"], barms)
                Xn = forward_rows(m, l, seqs, [it["p"]], 96)[:, 0]
                dvec = (m.W_U[it["X"]].float() - m.W_U[it["Y"]].float())
                fs = (Xn.double() @ dvec.double()).cpu().numpy()
                r = arms_out(fs, key, barms); r.update({"cl": it["cluster"], "item": it["item"], "q": it["q"], "p": it["p"]})
                brec[grp].append(r)
                Prog.tick(note=f"L{l} bank {grp} {len(brec[grp])}: |I| Hb {rec_stats(r,'Hb')[0]:.2e} tb0 {rec_stats(r,'tb0')[0]:.2e}")
        res["timing"][f"L{l}_bank_s_per_item"] = (time.time() - t1) / max(1, len(btest) + len(bnull))
        rl["bank_recs"] = brec
        if len(brec["comp"]) >= 3 and len(brec["null"]) >= 3:
            Bt = {"comp_Hb": compare(brec["comp"], "Hb", tbn, other="J5"), "null_Hb": compare(brec["null"], "Hb", tbn, other="J5"),
                  "sel_Hb": selectivity(brec["comp"], brec["null"], "Hb", tbn)}
            Bt["by_bank_Hb"] = {}
            for bn in COMP:
                sub = [r for r in brec["comp"] if r["cl"].startswith(bn + ":")]
                if len(sub) >= 3:
                    Bt["by_bank_Hb"][bn] = compare(sub, "Hb", tbn, other="J5")["absI"]
            Bt["S"] = bool(Bt["sel_Hb"]["ci95"][0] > 0 and Bt["comp_Hb"]["absI"]["ci95"][0] > 0)
            L["bank_test"] = Bt
            log(f"L{l} banks: comp Hb vs twins T1 {Bt['comp_Hb']['absI']['T1']:+.2f} {[round(x, 2) for x in Bt['comp_Hb']['absI']['ci95']]} | "
                f"null {Bt['null_Hb']['absI']['T1']:+.2f} | selectivity S {Bt['sel_Hb']['S']:+.2f} {[round(x, 2) for x in Bt['sel_Hb']['ci95']]} -> {Bt['S']}")
        # ---------------- verdict for the layer (frozen rule)
        Wt = L.get("wiki_test", {})
        if not L["V0"]:
            vd = "INSTRUMENT INVALID"
        elif L["R"]["pass"] and Wt.get("D") and Wt.get("U"):
            vd = "cross-position H-space candidate" + (" beyond J5" if Wt.get("J") else "")
        else:
            vd = "no cross-position H-space candidate"
        L["verdict"] = {"layer": vd, "V0": L["V0"], "R": L["R"]["pass"], "D": Wt.get("D"), "U": Wt.get("U"), "J": Wt.get("J"),
                        "Jd": L["geometry"]["Jd_flag"], "S": L.get("bank_test", {}).get("S")}
        log(f"L{l} VERDICT: {L['verdict']}")
        save()
        torch.cuda.empty_cache()
    res["runtime_min"] = (time.time() - T0) / 60
    res["verdict"] = {l: res["layers"][str(l)]["verdict"] for l in layers}
    save()
    if MODE == "full":
        SUMF.parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(SUMF, "w"), indent=1)
    log(f"ALL DONE {res['runtime_min']:.1f} min; verdict {res['verdict']}")
    Prog.st["stage_index"] = Prog.st["stage_count"]
    Prog.end("done", f"{MODE} done in {res['runtime_min']:.1f} min: " + "; ".join(f"L{l} {v['layer']}" for l, v in res["verdict"].items()))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        try:
            Prog.end("failed", f"{type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001
            pass
        raise
