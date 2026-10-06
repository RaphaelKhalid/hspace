"""H-space anatomy (Oct 5 2026): exact per-position H-lens and its exact decomposition into the
local curvature of every downstream sublayer (Theorem 1, the second-order adjoint identity):

    H_l[c] v  =  sum_s  J_{s<-l}^T  ( d^2/dy^2 [ g_{s+1} . phi_s(y) ] at y_s )  J_{s<-l} v

s runs over the sublayers (mixer, MLP) of blocks l+1..L, y_s is the residual entering sublayer s,
phi_s its branch (y_{s+1} = y_s + phi_s(y_s)), g_{s+1} = dF/dy_{s+1} the ordinary adjoint.

Objective (J-lens convention): F = sum_{p' valid} c . h_L[p'] (L = last block output, pre-final-norm).
Per-position H-lens: the tangent v is placed at ONE source position p and we read (Hv)[p], i.e. the
diagonal block d^2 F / dh_l[p]^2 (the J-lens analogue: source p, all later targets, mean over p).
Cross-position blocks (Hv)[q != p] are reported separately.

Usage: python hs_anatomy.py [layers...]   (default 6 12 18), writes out/anatomy_08b.json
"""

from __future__ import annotations

import json
import sys
import time

import torch
import torch.autograd.forward_ad as fwAD

from common import OUT, Ctx, gpu_mem, load_model, log, patch_deltanet
from hl2 import windows

T = 96
SKIP = 16


def branch(m, kw, i, kind, y):
    blk = m.layers[i]
    if kind == "mix":
        x = blk.input_layernorm(y)
        if blk.layer_type == "linear_attention":
            return blk.linear_attn(hidden_states=x, cache_params=None, attention_mask=kw.get("attention_mask"))
        out, _ = blk.self_attn(hidden_states=x, attention_mask=kw.get("attention_mask"),
                               position_ids=kw.get("position_ids"), past_key_values=None,
                               position_embeddings=kw.get("position_embeddings"))
        return out
    return blk.mlp(blk.post_attention_layernorm(y))


def subs_after(m, l):
    return [(i, k) for i in range(l + 1, m.n_layers) for k in ("mix", "mlp")]


def run_subs(m, c0, x, subs, B):
    """Returns list ys of residuals entering each sublayer (len = len(subs)) and the final residual."""
    ys = []
    y = x
    for (i, k) in subs:
        ys.append(y)
        y = y + branch(m, c0.kwargs_for(i, B), i, k, y)
    return ys, y


def natural_cov(m, ctxs, l):
    H = torch.cat([c.h_all[l][0, SKIP:T - 1] for c in ctxs], 0).double()
    mu = H.mean(0)
    S = torch.cov(H.T)
    S = 0.95 * S + 0.05 * torch.trace(S) / S.shape[0] * torch.eye(S.shape[0], device=S.device, dtype=S.dtype)
    ev, U = torch.linalg.eigh(S)
    return mu.float(), S.float(), ((U * ev.clamp_min(0).sqrt()) @ U.T).float()


def vocab_cov_sample(m, n, gen):
    eta = torch.randn(n, m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
    return (eta.to(torch.bfloat16) @ m.W_U).float() * m.gain[None]


def anatomy_batch(m, ctxs, l, pos, C, Vv):
    """ctxs: B contexts (equal T); pos[b] source position; C [B,d] cotangent rows; Vv [B,d] tangents.
    Returns exact Hv [B,T,d], per-sublayer pullbacks [S,B,T,d] (float32), subs."""
    B = len(ctxs)
    c0 = ctxs[0]
    subs = subs_after(m, l)
    x0 = torch.cat([c.h_all[l] for c in ctxs], 0)
    V = torch.zeros_like(x0)
    V[torch.arange(B), pos] = Vv
    valid = c0.valid
    # 1) forward-mode tangents at every sublayer input
    with torch.no_grad(), fwAD.dual_level():
        xd = fwAD.make_dual(x0, V)
        ys_d, _ = run_subs(m, c0, xd, subs, B)
        tangents = [fwAD.unpack_dual(y).tangent.clone() for y in ys_d]
        del ys_d
    # 2a) exact HVP (its own graph, freed before the attribution pass)
    x = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        _, yL = run_subs(m, c0, x, subs, B)
        F = (yL[:, valid] * C[:, None, :]).sum()
        (g,) = torch.autograd.grad(F, x, create_graph=True)
        (hv,) = torch.autograd.grad((g * V).sum(), x)
    hv = hv.detach(); del g, yL, F
    # 2b) graph forward for adjoints g_{s+1} (= dF/dy_{s+1}) and the pullbacks
    x = x0.clone().requires_grad_(True)
    with torch.enable_grad():
        ys, yL = run_subs(m, c0, x, subs, B)
        outs = ys[1:] + [yL]
        F = (yL[:, valid] * C[:, None, :]).sum()
        adj = torch.autograd.grad(F, outs, retain_graph=True)
        # 3) local curvature of each sublayer applied to its tangent; pulled back to layer l in
        #    groups (one backward pass per group: VJPs are linear, so a group's pullback is the sum)
        ws = []
        for si, (i, k) in enumerate(subs):
            y = ys[si].detach().requires_grad_(True)
            inner = (adj[si].detach() * branch(m, c0.kwargs_for(i, B), i, k, y)).sum()
            (gy,) = torch.autograd.grad(inner, y, create_graph=True)
            (w,) = torch.autograd.grad((gy * tangents[si]).sum(), y)
            ws.append(w.detach())
        groups = {}
        for si, (i, k) in enumerate(subs):
            kind = k if k == "mlp" else ("attn" if m.layers[i].layer_type == "full_attention" else "deltanet")
            rel = (i - l - 1) / max(1, m.n_layers - l - 1)
            groups.setdefault((kind, "near" if rel < 1 / 3 else ("mid" if rel < 2 / 3 else "far")), []).append(si)
        pulls, keys = [], []
        for key, sis in groups.items():
            pb = torch.autograd.grad([ys[si] for si in sis], x, grad_outputs=[ws[si] for si in sis], retain_graph=True)[0]
            pulls.append(pb.detach()); keys.append(key)
    return hv, torch.stack(pulls), keys


def main():
    layers = [int(a) for a in sys.argv[1:]] or [6, 12, 18]
    m = load_model()
    patch_deltanet(m)
    gen = torch.Generator(device="cuda").manual_seed(0)
    wins = windows(m.tok, 48, T=T, seed=1)
    res = {"T": T, "layers": {}}
    t0 = time.time()
    for l in layers:
        ctxs = [Ctx(m, w, l) for w in wins]
        for c in ctxs:  # keep only what we need
            c.h_all = {l: c.h_all[l]}
        mu, S, Shalf = natural_cov(m, ctxs[:32], l)
        nb, B = 24, 2
        sh_type, sh_depth, recon, diag_share, nrm = {}, {}, [], [], []
        for bi in range(nb):
            cs = ctxs[32 + (bi * B) % 16: 32 + (bi * B) % 16 + B]
            pos = torch.randint(SKIP, T - 1, (B,), device="cuda", generator=gen)
            C = vocab_cov_sample(m, B, gen)
            Vv = torch.randn(B, m.d, device="cuda", generator=gen) @ Shalf
            hv, pulls, keys = anatomy_batch(m, cs, l, pos, C, Vv)
            ar = torch.arange(B)
            tot = pulls.sum(0)
            recon.append(((tot - hv).norm() / hv.norm()).item())
            hp = hv[ar, pos]                     # per-position diagonal block applied to v
            pp = pulls[:, ar, pos]               # [G,B,d]
            diag_share.append((hp.pow(2).sum() / hv.pow(2).sum()).item())
            den = hp.pow(2).sum()
            for gi, (kind, depth) in enumerate(keys):
                sh = ((pp[gi] * hp).sum() / den).item()
                sh_type[kind] = sh_type.get(kind, 0) + sh / nb
                sh_depth[f"{kind}:{depth}"] = sh_depth.get(f"{kind}:{depth}", 0) + sh / nb
            nrm.append(hp.norm().item())
        res["layers"][l] = {"recon_rel_err": recon, "share_by_type": sh_type, "share_by_sublayer": sh_depth,
                            "diag_block_energy_frac_of_all_positions": diag_share, "|Hv|": nrm}
        log(f"[anatomy L{l}] recon err max {max(recon):.1e} | per-position shares {json.dumps({k: round(v, 3) for k, v in sh_type.items()})}"
            f" | diag-block energy frac {sum(diag_share)/len(diag_share):.2f} | {time.time()-t0:.0f}s {gpu_mem()}")
        json.dump(res, open(OUT / "anatomy_08b.json", "w"), indent=1)


if __name__ == "__main__":
    main()
