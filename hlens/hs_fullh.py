"""Full per-token mean Hessian lens: EXACT d x d Hessians, no Hutchinson, no random readout. Local GPU, $0. Exploratory.

The J-lens one derivative up. For output token t (u_t = (1 + w_final) * W_U[t], readout F_t = u_t . h_L*[q], pre-final-norm
last-block output) and a move delta at layer l, position q (the last position of a truncated window):
  g_t(x) = Sigma^1/2 dF_t/dh_l[q]                 (whitened gradient: the J-lens row of t)
  H_t(x) = Sigma^1/2 d2F_t/dh_l[q]^2 Sigma^1/2     (whitened exact Hessian; d columns from d batched exact HVPs)
Per token, over n contexts:
  phi_H,t = between-context <H_a, H_b>_F / mean |H_a|_F^2  (exact unbiased share of the context-invariant part); phi_J,t same
  spectrum of Hbar_t (signed), top |lambda| vs the context sign-flip null (H_a -> eps_a H_a keeps every second moment and
    kills the mean: an exact null for "there is a mean")
  split-half (even / odd contexts) top-k eigenspace overlap vs the same flips
  causal, held out: v = top-|lambda| eigvec of Hbar_t(even); on odd contexts the symmetric second difference
    c2 = F(x + m) + F(x - m) - 2 F(x),  m = Sigma^1/2 v rescaled to f * sqrt(tr Sigma)
    cancels every odd order (pure curvature). Report sign agreement with lambda and c2 vs random whitened directions,
    and decode the mean symmetric change of ALL logits (which tokens does +-v boost?).
    python hs_fullh.py [--layer 18] [--n 128] [--C 16] [--f 0.25] [--check]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hs_phi_cpu import SKIP, load, windows

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "fullh"
PROG = HERE / "out" / "progress"
TOKENS = [" the", " .", " ,", " of", " and", " in", " to", " a", " was", " is", " he", " she", "1", " year", " city", " not"]


def _patch_linear_cache():
    """transformers' LinearAttentionLayer updates conv / recurrent states in place (copy_, for cudagraphs), which breaks
    double backward through a cached single-token step. Same values, out of place."""
    from transformers import cache_utils as CU

    def conv(self, conv_states, **kw):
        if not self.is_conv_states_initialized:
            self.lazy_initialization(conv_states=conv_states)
        if not self.has_previous_state:
            self.conv_states = conv_states
            self.has_previous_state = True
        else:
            n, k = conv_states.shape[-1], self.conv_kernel_size
            self.conv_states = conv_states[..., -k:] if n >= k else torch.cat([self.conv_states[..., n:], conv_states], -1)
        return self.conv_states

    def rec(self, recurrent_states, **kw):
        if not self.is_recurrent_states_initialized:
            self.lazy_initialization(recurrent_states=recurrent_states)
        self.recurrent_states = recurrent_states
        return self.recurrent_states
    CU.LinearAttentionLayer.update_conv_state = conv
    CU.LinearAttentionLayer.update_recurrent_state = rec


def progress(done, total, note, t0, stage="exact Hessians", idx=1, unit="exact HVP columns"):
    PROG.mkdir(parents=True, exist_ok=True)
    tmp = PROG / "fullh.json.tmp"
    json.dump({"job": "fullh", "title": "full per-token mean Hessian lens (0.8B L18, local GPU)", "status": "running",
               "stage": stage, "stage_index": idx, "stage_count": 2, "done": done, "total": total, "unit": unit,
               "started": t0, "updated": time.time(), "note": note}, open(tmp, "w"))
    for _ in range(5):                       # the monitor may hold the target open (Windows): never let this kill a job
        try:
            tmp.replace(PROG / "fullh.json")
            return
        except PermissionError:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--C", type=int, default=16, help="column slots per token per second backward")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--center", action="store_true", help="readout u_t - mean_vocab(u): no all-logit shift (fix 1)")
    ap.add_argument("--prof", action="store_true")
    ap.add_argument("--V", type=int, default=1, help="chunks per vmapped second backward")
    ap.add_argument("--G", type=int, default=4, help="tokens per batched forward")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--start", type=int, default=0, help="contexts already done (resume without a state file)")
    ap.add_argument("--f", type=float, default=0.25)
    ap.add_argument("--flips", type=int, default=200)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=len(TOKENS))
    ap.add_argument("--toklist", type=str, default="", help="'|'-separated tokens to use instead of TOKENS, e.g. ' simulated| fake'")
    ap.add_argument("--tag", default="08b")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.enabled = False   # a transient CUDNN_STATUS_BAD_PARAM_STREAM_MISMATCH killed a run; only the tiny DeltaNet conv used it
    _patch_linear_cache()
    m, tok = load(a.model)
    m = m.to(dev)
    d, T, l = m.config.get_text_config().hidden_size, a.T, a.layer
    toks = a.toklist.split("|") if a.toklist else TOKENS[: a.tokens]
    tid = [[0] if t == "@ubar" else tok(t, add_special_tokens=False)["input_ids"] for t in toks]
    assert all(len(x) == 1 for x in tid), tid
    tid = [x[0] for x in tid]
    W_U = m.lm_head.weight.float()
    gain = 1.0 + m.model.norm.weight.float()
    Ut = (W_U[tid] * gain).double()                                      # [nt, d]
    for i, t in enumerate(toks):                    # "@ubar": readout = mean over the vocab of (1+w) * W_U (uncentered), used
        if t == "@ubar":                            # to center uncentered Hessians exactly (the Hessian is linear in the readout)
            assert not a.center
            Ut[i] = (W_U * gain).mean(0).double()
    if a.center:                                   # remove the all-logit shift direction (softmax-invariant), fix 1
        Ut = Ut - (W_U * gain).mean(0).double()[None]
    cov = windows(tok, "train", 24, T, seed=5)
    ctx = windows(tok, "train", a.n, T, seed=12)
    assert ctx.shape[0] == a.n
    rng = random.Random(21 + l)
    qs = [rng.randrange(SKIP + 8, T) for _ in range(a.n)]                # q = last position of the truncated window

    layers = m.model.layers
    st = {"delta": None, "q": None, "hL": None}

    def add(mod, inp, out):
        if st["delta"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype); oh[st["q"]] = 1
        h = h + st["delta"][:, None, :] * oh[None, :, None]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    layers[l].register_forward_hook(add)
    layers[-1].register_forward_hook(lambda mod, inp, o: st.__setitem__("hL", o[0] if isinstance(o, tuple) else o))

    hs = []
    with torch.no_grad():
        for w in cov:
            o = m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False)
            hs.append(o.hidden_states[l + 1][0, SKIP:T - 1].double())
    S = torch.cov(torch.cat(hs).T)
    S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device=dev, dtype=S.dtype)
    ev, V = torch.linalg.eigh(S)
    Sh = (V * ev.clamp_min(1e-12).sqrt()) @ V.T
    Shf = Sh.float()
    trS = float(torch.trace(S))

    def F(ids, q, delta, u):
        st["delta"], st["q"] = delta, q
        m.model(input_ids=ids, use_cache=False)
        st["delta"] = None
        return st["hL"][:, q].float() @ u

    NT, C = len(toks), a.C
    assert d % C == 0
    PR = {"done": 0, "total": a.n * NT * d, "t0": time.time(), "last": 0.0, "note": ""}

    def tick(cols):
        PR["done"] += cols
        now = time.time()
        if now - PR["last"] > 0.5:
            rate = (PR["done"] - PR.get("base", 0)) / max(now - PR["t0"], 1e-9)
            PR["last"] = now
            progress(PR["done"], PR["total"], f"{rate:.0f} HVP/s | {PR['done'] / d:.1f} of {a.n * NT} Hessians | {PR['note']}", PR["t0"])

    def hess_fast(ids1, q):
        Hs_, Gs_ = [], []
        for g0 in range(0, NT, a.G):
            H_, G_ = hess_group(ids1, q, list(range(g0, min(NT, g0 + a.G))))
            Hs_.append(H_); Gs_.append(G_)
        return torch.cat(Hs_), torch.cat(Gs_)

    def grads_only(ids1, q):
        out_ = []
        for g0 in range(0, NT, a.G):
            tsel = list(range(g0, min(NT, g0 + a.G)))
            with torch.no_grad():
                pkv = m.model(input_ids=ids1[:, :q], use_cache=True).past_key_values
            delta = torch.zeros(len(tsel), d, device=dev, requires_grad=True)
            pkv.reorder_cache(torch.zeros(len(tsel), dtype=torch.long, device=dev))
            st["delta"], st["q"] = delta, 0
            m.model(input_ids=ids1[:, q:q + 1].expand(len(tsel), -1), past_key_values=pkv, use_cache=True)
            st["delta"] = None
            (g,) = torch.autograd.grad((st["hL"][:, 0].float() * Ut[tsel].float()).sum(), delta)
            out_.append(g.double() @ Sh)
        return torch.cat(out_)

    def hess_group(ids1, q, tsel):
        nt = len(tsel)
        Ug = Ut[tsel].float()
        tok_of_row = torch.arange(nt * C, device=dev) // C                # batch row b -> token b // C, column slot b % C
        """Exact whitened Hessians and gradients for ALL tokens at context (ids1[:q+1]): cache the prefix, then one
        single-token step from layer l up, batched over (token, column slot); d / C exact second backwards."""
        tp = [time.time()]
        with torch.no_grad():
            pkv = m.model(input_ids=ids1[:, :q], use_cache=True).past_key_values
        pkv.reorder_cache(torch.zeros(nt * C, dtype=torch.long, device=dev))
        delta = torch.zeros(nt * C, d, device=dev, requires_grad=True)
        st["delta"], st["q"] = delta, 0
        m.model(input_ids=ids1[:, q:q + 1].expand(nt * C, -1), past_key_values=pkv, use_cache=True)
        st["delta"] = None
        Fv = (st["hL"][:, 0].float() * Ug[tok_of_row]).sum()
        (g,) = torch.autograd.grad(Fv, delta, create_graph=True)
        torch.cuda.synchronize(); tp.append(time.time())
        HS = torch.zeros(nt, d, d, device=dev, dtype=torch.float64)
        ks = list(range(0, d, C))
        for j0 in range(0, len(ks), a.V):                                   # a.V chunks per vmapped (batched) backward
            kk = ks[j0:j0 + a.V]
            go = torch.stack([Shf[:, k0:k0 + C].T.repeat(nt, 1) for k0 in kk]).contiguous()   # [V, rows, d]
            (hv,) = torch.autograd.grad(g, delta, grad_outputs=go if a.V > 1 else go[0], retain_graph=j0 + a.V < len(ks),
                                        is_grads_batched=a.V > 1)
            hv = hv.reshape(len(kk), nt, C, d).detach().double()
            for jj, k0 in enumerate(kk):
                HS[:, k0:k0 + C] = hv[jj]                                    # HS[t, i] = (H_t Sh[:, i])^T
            tick(len(kk) * C * nt)
        torch.cuda.synchronize(); tp.append(time.time())
        if a.prof:
            print(f"prof: prefix+forward+grad {tp[1]-tp[0]:.2f}s | {d // C} second backwards {tp[2]-tp[1]:.2f}s", flush=True)
        Ht = HS @ Sh                                                        # Sh H_t Sh
        Ht = (Ht + Ht.transpose(1, 2)) / 2
        G = (g.detach().double().view(nt, C, d)[:, 0] @ Sh)
        return Ht, G

    if a.check and not a.resume:                                                             # fast path vs full-sequence exact HVPs
        q = qs[0]; ids1 = ctx[0, : q + 1][None].to(dev)
        Hf_, _ = hess_fast(ids1, q)
        delta = torch.zeros(C, d, device=dev, requires_grad=True)
        (g,) = torch.autograd.grad(F(ids1.expand(C, -1), q, delta, Ut[0].float()).sum(), delta, create_graph=True)
        (hv,) = torch.autograd.grad(g, delta, grad_outputs=Shf[:, :C].T.contiguous())
        ref = (Sh @ (hv.detach().double().T))                               # Sh H Sh[:, :C]
        err = float((Hf_[0][:, :C] - ref).norm() / ref.norm())
        print(f"check: fast vs full-sequence exact Hessian columns, rel err {err:.2e}", flush=True)
        assert err < 1e-3, err

    PR["done"], PR["t0"] = 0, time.time()
    Hpath = HERE / "out" / f"fullh_{a.tag}_L{l}_H.f16"
    Spath = HERE / "out" / f"fullh_{a.tag}_L{l}_state.json"
    mode = "r+" if a.resume else "w+"
    Hs = np.memmap(Hpath, dtype=np.float16, mode=mode, shape=(NT, a.n, d, d))    # whitened, on disk
    Gpath = HERE / "out" / f"fullh_{a.tag}_L{l}_G.f64"
    Gm = np.memmap(Gpath, dtype=np.float64, mode="r+" if a.resume and Gpath.exists() else "w+", shape=(NT, a.n, d))
    start = json.load(open(Spath))["done"] if a.resume and Spath.exists() else a.start
    if a.resume and not (Gpath.exists() and Spath.exists()):              # crash before the state files existed
        for ci in range(start):
            Gm[:, ci] = grads_only(ctx[ci, : qs[ci] + 1][None].to(dev), qs[ci]).cpu().numpy()
        print(f"resume: recomputed gradients for contexts 0..{start - 1}", flush=True)
    PR["done"] = PR["base"] = start * NT * d
    PR["t0"] = time.time()
    t0 = time.time()
    for ci in range(start, a.n):
        q = qs[ci]
        PR["note"] = f"context {ci + 1}/{a.n}"
        Ht, G = hess_fast(ctx[ci, : q + 1][None].to(dev), q)
        assert torch.isfinite(Ht).all()
        Hs[:, ci] = Ht.half().cpu().numpy()
        Gm[:, ci] = G.cpu().numpy()
        Hs.flush(); Gm.flush()
        json.dump({"done": ci + 1}, open(Spath, "w"))
        if (ci + 1) % 8 == 0:
            el = (time.time() - t0) / 60
            print(f"L{l}: context {ci+1}/{a.n} ({el:.1f} min, ETA {el / (ci + 1) * (a.n - ci - 1):.0f} min)", flush=True)
            progress(ci + 1, a.n, f"L{l} exact Hessians", t0)
    Hs.flush()
    Gs = torch.from_numpy(np.asarray(Gm)).clone()

    # ---------------- statistics per token
    gen = torch.Generator(device=dev).manual_seed(7)
    even = (torch.arange(a.n) % 2 == 0).to(dev)
    out = {"layer": l, "n": a.n, "model": a.model, "f": a.f, "tokens": {}}

    def between(X):                                                       # X [n, ...]: unbiased |E X|^2 / E|X|^2
        Xf = X.reshape(X.shape[0], -1)
        s = Xf.sum(0); n = Xf.shape[0]
        return float((s @ s - (Xf * Xf).sum()) / (n * (n - 1)) / (Xf * Xf).sum(1).mean())

    def topk_vecs(M, k):
        ev_, V_ = torch.linalg.eigh(M)
        o = torch.argsort(ev_.abs(), descending=True)[:k]
        return ev_[o], V_[:, o]

    def ovl(A, Bm):
        return float((A.T @ Bm).pow(2).sum() / A.shape[1])

    def logits_of(hvec):
        lg = (hvec.to(gain.device).float() * gain) @ W_U.T
        return lg - lg.mean()
    for ti, t in enumerate(toks):
        progress(ti, NT, f"token {t!r}: 200 context sign flips + held-out causal test", PR["t0"], stage="flip null + causal", idx=2, unit="tokens")
        H = torch.from_numpy(np.asarray(Hs[ti])).to(dev).float()           # [n, d, d]
        R = {"phi_H": between(H), "phi_J": between(Gs[ti].to(dev))}
        Hbar = H.mean(0)
        ev_all = torch.linalg.eigvalsh(Hbar)
        lam, _ = topk_vecs(Hbar, 5)
        R["top_lambda"] = [float(x) for x in lam]
        top20 = ev_all[torch.argsort(ev_all.abs(), descending=True)[:20]]
        R["top20_signs_pos"] = int((top20 > 0).sum())
        R["fro_mean_over_rms"] = float(Hbar.norm() / H.reshape(a.n, -1).norm(dim=1).pow(2).mean().sqrt())
        _, Ve = topk_vecs(H[even].mean(0), 5); _, Vo = topk_vecs(H[~even].mean(0), 5)
        R["split"] = {k: ovl(Ve[:, :k], Vo[:, :k]) for k in (1, 3, 5)}
        nl, ns = [], {1: [], 3: [], 5: []}
        for _ in range(a.flips):
            eps = (torch.randint(0, 2, (a.n,), device=dev, generator=gen).float() * 2 - 1)
            ee, eo = eps * even.to(dev), eps * (~even).to(dev)
            nl.append(float(torch.linalg.eigvalsh(torch.einsum("n,nij->ij", eps, H) / a.n).abs().max()))
            _, Fe = topk_vecs(torch.einsum("n,nij->ij", ee, H), 5); _, Fo = topk_vecs(torch.einsum("n,nij->ij", eo, H), 5)
            for k in ns:
                ns[k].append(ovl(Fe[:, :k], Fo[:, :k]))
        R["lam1_null_q95"] = float(np.quantile(nl, 0.95)); R["lam1_p"] = float((np.sum(np.array(nl) >= abs(R["top_lambda"][0])) + 1) / (a.flips + 1))
        R["split_null_q95"] = {k: float(np.quantile(v, 0.95)) for k, v in ns.items()}
        R["split_p"] = {k: float((np.sum(np.array(v) >= R["split"][k]) + 1) / (a.flips + 1)) for k, v in ns.items()}
        # ---- causal, held out: symmetric second difference along the even-half top eigvec, on odd contexts
        lam_e, Ve1 = topk_vecs(H[even].mean(0), 1)
        gbar = Gs[ti][even.cpu()].mean(0).to(dev).float()
        R["cos_v_gbar"] = float((Ve1[:, 0] @ gbar).abs() / gbar.norm())   # is v just the J-lens direction of t?
        R["cos_v_pc1"] = float(torch.linalg.eigh(S.float())[1][:, -1] @ (Sh.float() @ Ve1[:, 0]) / (Sh.float() @ Ve1[:, 0]).norm())
        gr = torch.Generator(device=dev).manual_seed(100 + ti)
        Vr = torch.randn(d, 8, device=dev, generator=gr, dtype=torch.float64); Vr /= Vr.norm(dim=0, keepdim=True)
        dirs = torch.cat([Ve1.double(), Vr], 1)                           # whitened unit directions
        M = (Sh @ dirs); M = M / M.norm(dim=0, keepdim=True) * a.f * trS ** 0.5
        c2 = []; sym_all = []; lp2 = []
        with torch.no_grad():
            for ci in torch.nonzero(~even).flatten().tolist():
                q = qs[ci]
                ids = ctx[ci, : q + 1][None].to(dev)
                nd = M.shape[1]
                idsb = ids.expand(2 * nd + 1, -1)
                dl = torch.cat([M.T, -M.T, torch.zeros(1, d, device=dev, dtype=M.dtype)]).float()
                st["delta"], st["q"] = dl, q
                m.model(input_ids=idsb, use_cache=False)
                st["delta"] = None
                hq = st["hL"][:, q].double()
                sym = hq[:nd] + hq[nd:2 * nd] - 2 * hq[-1:]                # [nd, d]
                c2.append((sym @ Ut[ti]).cpu())
                sym_all.append(sym[0].cpu())
                lp = torch.log_softmax(m.lm_head(m.model.norm(st["hL"][:, q])).float(), -1)[:, tid].double()   # true log p, all set tokens
                lp2.append((lp[:nd] + lp[nd:2 * nd] - 2 * lp[-1:]).cpu())      # [nd, NT] symmetric second difference
        c2 = torch.stack(c2)                                              # [n_odd, 1 + 8]
        sgn = float(np.sign(float(lam_e[0])))
        R["causal"] = {"lambda_even": float(lam_e[0]),
                       "sign_agree_top": float((torch.sign(c2[:, 0]) == sgn).float().mean()),
                       "mean_c2_top": float(c2[:, 0].mean()), "mean_c2_rand": [float(x) for x in c2[:, 1:].mean(0)],
                       "z_top_vs_rand": float((c2[:, 0].mean() * sgn) / (c2[:, 1:].mean(0).std() + 1e-30)),
                       "rms_c2_top": float(c2[:, 0].pow(2).mean().sqrt()), "rms_c2_rand": float(c2[:, 1:].pow(2).mean().sqrt())}
        lp2 = torch.stack(lp2)                                            # [n_odd, nd, NT]
        own = lp2[:, :, ti]
        R["causal_logprob"] = {"sign_agree_top": float((torch.sign(own[:, 0]) == sgn).float().mean()),
                               "z_top_vs_rand": float((own[:, 0].mean() * sgn) / (own[:, 1:].mean(0).std() + 1e-30)),
                               "mean_top": float(own[:, 0].mean()), "rms_top": float(own[:, 0].pow(2).mean().sqrt()),
                               "rms_rand": float(own[:, 1:].pow(2).mean().sqrt()),
                               "cross_tokens_mean_top": {tt: float(lp2[:, 0, j].mean()) for j, tt in enumerate(toks)},
                               "cross_tokens_mean_rand": {tt: float(lp2[:, 1:, j].mean()) for j, tt in enumerate(toks)}}
        torch.save(Ve1.cpu(), HERE / "out" / f"fullh_{a.tag}_L{l}_v_{ti}.pt")
        lg = logits_of(torch.stack(sym_all).mean(0))
        R["decode_sym_top"] = {"up": [tok.decode([i]) for i in torch.topk(lg, 12).indices.tolist()],
                               "down": [tok.decode([i]) for i in torch.topk(-lg, 12).indices.tolist()],
                               "rank_of_t": int((lg > lg[tid[ti]]).sum()) if sgn > 0 else int((lg < lg[tid[ti]]).sum())}
        out["tokens"][t] = R
        print(f"{t!r:8s} phi_J {R['phi_J']:.3f} phi_H {R['phi_H']:.4f} | lam1 {R['top_lambda'][0]:+.3g} null95 {R['lam1_null_q95']:.3g} p {R['lam1_p']:.3f} "
              f"| split1/3/5 {R['split'][1]:.2f}/{R['split'][3]:.2f}/{R['split'][5]:.2f} p {R['split_p'][1]:.3f}/{R['split_p'][3]:.3f}/{R['split_p'][5]:.3f} "
              f"| causal sign {R['causal']['sign_agree_top']:.2f} z {R['causal']['z_top_vs_rand']:+.1f} rms top/rand {R['causal']['rms_c2_top']:.3g}/{R['causal']['rms_c2_rand']:.3g} "
              f"| logp sign {R['causal_logprob']['sign_agree_top']:.2f} z {R['causal_logprob']['z_top_vs_rand']:+.1f} "
              f"| t rank {R['decode_sym_top']['rank_of_t']} up {R['decode_sym_top']['up'][:5]} | cos(v, gbar) {R['cos_v_gbar']:.2f} cos(raw v, PC1) {R['cos_v_pc1']:+.2f}", flush=True)
        RES.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(RES / f"fullh_{a.tag}_L{l}.json", "w"), indent=1)
        torch.save(Hbar.half().cpu(), HERE / "out" / f"fullh_{a.tag}_L{l}_Hbar_{ti}.pt")
        del H
    json.dump({"job": "fullh", "title": "full per-token mean Hessian lens", "status": "done", "stage": "done", "stage_index": 2,
               "stage_count": 2, "done": a.n, "total": a.n, "unit": "contexts", "started": t0, "updated": time.time(), "note": ""},
              open(PROG / "fullh.json", "w"))


if __name__ == "__main__":
    main()
