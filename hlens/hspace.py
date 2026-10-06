"""H-space one-shot run (Oct 2026). Prereg: paper/SCOPE-hspace.md (frozen before the 27B run).

    python hspace.py small     # Qwen3.5-0.8B on the laptop (dev + smoke)
    python hspace.py full      # Qwen3.6-27B on one 96 GB GPU (bf16 weights, exact fp32 compute)

Stages (each writes out/hspace_<tag>.json + .pt incrementally; every stage logs real progress counts):
  S2 stats      per-layer activation mean mu and covariance Sigma (whitening, shrinkage 0.05)
  S3 jspace     J25: top-25 directions of the per-token ACTIVE J-lens vectors (paper's J-space), whitened
  S4 hspace     H-space: top eigenvectors of M = E_{x,c,xi} mean_p y_p y_p^T, y_p = Sigma^1/2 s_p (H (s (x) Sigma^1/2 xi))_p
                (Hutchinson random signs s over positions; c ~ vocab-weighted covector; whitened probes)
                two disjoint halves -> split-half reliability; also M_lens from position-averaged y
  S5 control    planted bilinear gate G (a.z)(b.z) w at the calibrated strength of the 25th H-eigenvalue; recovery
  S6 natural    1-sigma finite interactions of top H pairs vs random pairs (LM log-lik of later tokens), and
                the decoded interaction vectors (rectangle theorem: exact natural-scale second order)
  S7 causal     ablate H25 / J25 / H25-perp-J / PCA25 / rand25 x3 at each layer; KL and the double dissociation
                on interaction vs additive tokens (labels from 2x2 span removal, forward only)
  S8 decode     J-lens decode of the top H directions
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bf16w  # noqa: E402

OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
SKIP = 16

CFG = {
    "small": dict(model="Qwen/Qwen3.5-0.8B", lens=("neuronpedia/jacobian-lens", "main",
                  "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"),
                  layers=[6, 12, 18], pc_layer=12, T=96, n_cov=96, n_probe=64, B=2, n_pc=32,
                  n_jact=512, n_nat_ctx=6, n_top=6, n_eval=96, n_lab_targets=2, k=25, split="validation"),
    "smoke": dict(model="Qwen/Qwen3.5-0.8B", lens=("neuronpedia/jacobian-lens", "main",
                  "qwen3.5-0.8b/jlens/Salesforce-wikitext/Qwen3.5-0.8B_jacobian_lens.pt"),
                  layers=[12], pc_layer=12, T=64, n_cov=8, n_probe=4, B=2, n_pc=4,
                  n_jact=64, n_nat_ctx=1, n_top=3, n_eval=8, n_lab_targets=1, k=4, split="validation"),
    "full": dict(model="Qwen/Qwen3.6-27B", lens=("neuronpedia/jacobian-lens", "main",
                 "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"),
                 layers=[16, 28, 40, 52], pc_layer=40, T=128, n_cov=256, n_probe=256, B=4, n_pc=128,
                 n_jact=2048, n_nat_ctx=12, n_top=6, n_eval=512, n_lab_targets=2, k=25, split="train"),
}


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] [hspace] {msg}"
    print(line, flush=True)
    with open(HERE / "runlog.md", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------ model
class Model:
    def __init__(self, name, T):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(name)
        hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16, device_map="cuda",
                                                  low_cpu_mem_usage=True, attn_implementation="eager")
        hf.eval()
        for p in hf.parameters():
            p.requires_grad_(False)
        self.hf, self.text = hf, hf.model
        self.layers = self.text.layers
        bf16w.patch_bf16_compute(self.layers)
        self.text.norm.float()
        if hasattr(self.text, "rotary_emb"):
            self.text.rotary_emb.float()
        from common import patch_deltanet
        patch_deltanet(self)
        cfg = hf.config.get_text_config()
        self.n_layers, self.d = cfg.num_hidden_layers, cfg.hidden_size
        self.W_U = hf.lm_head.weight  # [V,d] bf16
        self.gain = (1.0 + self.text.norm.weight.float())
        self.T = T
        self._capture_kwargs(T)

    def _capture_kwargs(self, T):
        self.kw = {}
        hooks = []
        for i, blk in enumerate(self.layers):
            def pre(mod, args, kwargs, i=i):
                self.kw[i] = {k: v for k, v in kwargs.items() if k not in ("past_key_values", "use_cache")}
            hooks.append(blk.register_forward_pre_hook(pre, with_kwargs=True))
        ids = torch.full((1, T), 11, device="cuda")
        with torch.no_grad():
            self.text(inputs_embeds=self.embed(ids), use_cache=False)
        for h in hooks:
            h.remove()

    def kwargs(self, i, B):
        kw = dict(self.kw[i])
        pe = kw.get("position_embeddings")
        if pe is not None and B > 1:
            kw["position_embeddings"] = tuple(t.expand(B, *t.shape[1:]) for t in pe)
        return kw

    def embed(self, ids):
        return self.text.embed_tokens(ids).float()

    def run(self, h, start, stop=None, plant=None):
        stop = self.n_layers if stop is None else stop
        B = h.shape[0]
        for i in range(start, stop):
            out = self.layers[i](h, **self.kwargs(i, B))
            hn = out[0] if isinstance(out, tuple) else out
            if plant is not None and i == plant["block"]:
                z = (h - plant["mu"]) @ plant["Sinvh"]
                hn = hn + plant["G"] * (z @ plant["a"])[..., None] * (z @ plant["b"])[..., None] * plant["w"]
            h = hn
        return h

    @torch.no_grad()
    def states(self, ids, layer):
        return self.run(self.embed(ids), 0, layer + 1)

    def logits(self, x):
        """fp32 (or bf16 in bf16 mode) logits for normed rows x [N,d], vocab-chunked (no fp32 copy of W_U)."""
        if bf16w.MODE == "bf16":
            return F.linear(x.to(torch.bfloat16), self.W_U).float()
        return torch.cat([F.linear(x.float(), self.W_U[s:s + 16384].float()) for s in range(0, self.W_U.shape[0], 16384)], -1)

    def logprobs_next(self, hL, ids):
        """log p(ids[t+1] | <= t) at every t < T-1. hL [B,T,d] last-block output."""
        x = self.text.norm(hL)
        out = []
        for b in range(x.shape[0]):
            lp = torch.log_softmax(self.logits(x[b, :-1]), -1)
            out.append(lp.gather(-1, ids[b, 1:, None])[:, 0])
        return torch.stack(out)  # [B, T-1]

    def u(self, token_ids):
        ids = torch.as_tensor(token_ids, device="cuda")
        return self.W_U[ids].float() * self.gain


# ------------------------------------------------------------------ data
def wikitext_windows(tok, split, n, T, seed, skip_articles=0):
    import glob

    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=[f"wikitext-103-raw-v1/{split}-*"])
    fs = sorted(glob.glob(os.path.join(root, "wikitext-103-raw-v1", f"{split}-*.parquet")))
    assert fs, f"no wikitext {split} parquet under {root}"
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
    rng = random.Random(seed)
    rng.shuffle(arts)
    out = []
    for k in range(8):
        for a in arts[skip_articles:]:
            if len(a) < 2000:
                continue
            ids = tok(a[: 20000], add_special_tokens=False)["input_ids"]
            s = 100 + k * (T + 50)
            if len(ids) >= s + T:
                out.append(ids[s:s + T])
                if len(out) == n:
                    return torch.tensor(out)
    assert len(out) >= n // 2, f"only {len(out)} windows"
    return torch.tensor(out)


# ------------------------------------------------------------------ linear algebra helpers
def sym_sqrt(S, inv=False):
    ev, U = torch.linalg.eigh(S.double())
    ev = ev.clamp_min(ev.max() * 1e-8)
    p = -0.5 if inv else 0.5
    return ((U * ev.pow(p)) @ U.T).float()


def top_eig(M, k):
    ev, U = torch.linalg.eigh(M.double())
    o = torch.argsort(ev, descending=True)
    return ev[o].float(), U[:, o].float()


def overlap(Q1, Q2):
    """mean squared cosine between two k-dim subspaces (columns orthonormal); random ~ k/d."""
    return float((Q1.T @ Q2).pow(2).sum() / Q1.shape[1])


# ------------------------------------------------------------------ main
def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "small"
    cfg = CFG[tag]
    torch.manual_seed(0)
    gen = torch.Generator(device="cuda").manual_seed(0)
    t0 = time.time()
    res = {"tag": tag, "cfg": {k: v for k, v in cfg.items()}, "layers": {}}
    res_path, tens_path = OUT / f"hspace_{tag}.json", OUT / f"hspace_{tag}.pt"
    tens = {}

    def save():
        json.dump(res, open(res_path, "w"), indent=1)
        torch.save(tens, tens_path)

    m = Model(cfg["model"], cfg["T"])
    T, d, k = cfg["T"], m.d, cfg["k"]
    log(f"{tag}: loaded {cfg['model']} L={m.n_layers} d={d} in {time.time()-t0:.0f}s; mem {torch.cuda.memory_allocated()/2**30:.1f} GB")
    from huggingface_hub import hf_hub_download
    lens_path = hf_hub_download(cfg["lens"][0], cfg["lens"][2], revision=cfg["lens"][1])
    lens = torch.load(lens_path, map_location="cpu", weights_only=True)
    Jall = lens["J"]
    target = max(Jall) + 1
    assert target == m.n_layers - 1, (target, m.n_layers)
    n_est = cfg["n_probe"] * cfg["B"] + cfg["n_cov"]
    est = wikitext_windows(m.tok, cfg["split"], n_est, T, seed=0)
    ev_ids = wikitext_windows(m.tok, "test" if tag == "full" else "validation", cfg["n_eval"] + cfg["n_jact"] // (T - SKIP - 1) + 8,
                              T, seed=1, skip_articles=0 if tag == "full" else 30)
    cov_ids, probe_ids = est[: cfg["n_cov"]].cuda(), est[cfg["n_cov"]:].cuda()
    ev_ids = ev_ids.cuda()
    log(f"data: {len(est)} estimation windows, {len(ev_ids)} eval windows (T={T})")
    valid = torch.zeros(T, dtype=torch.bool, device="cuda"); valid[SKIP:T - 1] = True
    # output metric for c: Sigma_c = (1/V) sum_t u_t u_t^T
    Ug = None

    def sample_c(n):
        eta = torch.randn(n, m.W_U.shape[0], device="cuda", generator=gen) / m.W_U.shape[0] ** 0.5
        return (eta.to(torch.bfloat16) @ m.W_U).float() * m.gain[None]

    U2 = torch.zeros(d, d, device="cuda", dtype=torch.float64)
    for s in range(0, m.W_U.shape[0], 16384):
        Wg = m.W_U[s:s + 16384].float() * m.gain[None]
        U2 += (Wg.T @ Wg).double()
    U2 = (U2 / m.W_U.shape[0]).float()

    for l in cfg["layers"]:
        L = {}
        res["layers"][str(l)] = L
        # ---------------- S2 stats
        S = torch.zeros(d, d, device="cuda", dtype=torch.float64); mu = torch.zeros(d, device="cuda", dtype=torch.float64); n = 0
        for s in range(0, len(cov_ids), 4):
            h = m.states(cov_ids[s:s + 4], l)[:, valid].reshape(-1, d).double()
            mu += h.sum(0); S += h.T @ h; n += h.shape[0]
        mu /= n; S = S / n - torch.outer(mu, mu)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device="cuda", dtype=S.dtype)
        mu, S = mu.float(), S.float()
        Sh, Sih = sym_sqrt(S), sym_sqrt(S, inv=True)
        tens[f"mu{l}"], tens[f"Sh{l}"] = mu.cpu(), Sh.cpu()
        log(f"L{l} S2 stats from {n} tokens; tr Sigma {float(torch.trace(S)):.3g} ({time.time()-t0:.0f}s)")

        # ---------------- S3 J-space (per-token active J-lens vectors, whitened)
        J = Jall[l].float().cuda()
        Gj = Sh @ J.T @ U2 @ J @ Sh
        _, Ujg = top_eig(Gj, k)
        Jglob = Ujg[:, :k]
        acc = torch.zeros(d, d, device="cuda", dtype=torch.float64)
        cnt = 0
        JS = J @ Sh  # whitened J-lens map
        for s in range(0, len(ev_ids), 2):
            if cnt >= cfg["n_jact"]:
                break
            h = m.states(ev_ids[s:s + 2], l)[:, valid].reshape(-1, d)
            jh = (h @ J.T) * m.gain[None]                         # [N,d]
            lg = jh.to(torch.bfloat16) @ m.W_U.T                   # [N,V]
            top = lg.float().topk(25, dim=-1).indices              # [N,25]
            uniq, counts = torch.unique(top, return_counts=True)
            vt = (m.u(uniq) @ JS)                                  # whitened J-lens vectors [n_u,d]
            vt = vt / vt.norm(dim=-1, keepdim=True)
            acc += (vt.T.double() * counts[None].double()) @ vt.double()
            cnt += h.shape[0]
        _, Uja = top_eig(acc, k)
        J25 = Uja[:, :k]
        tens[f"J25_{l}"], tens[f"Jglob_{l}"] = J25.cpu(), Jglob.cpu()
        L["J_active_vs_global_overlap"] = overlap(J25, Jglob)
        log(f"L{l} S3 J-space from {cnt} positions; active-vs-global overlap {L['J_active_vs_global_overlap']:.2f} ({time.time()-t0:.0f}s)")
        del J

        # ---------------- S4 H-space
        def run_probes(n_probe, ids_pool, plant=None, label="H"):
            Ms = [torch.zeros(d, d, device="cuda", dtype=torch.float64) for _ in range(2)]
            Ml = [torch.zeros(d, d, device="cuda", dtype=torch.float64) for _ in range(2)]
            ny = [0, 0]
            B = cfg["B"]
            for i in range(n_probe):
                half = i % 2
                ix = torch.arange(i * B, (i + 1) * B) % len(ids_pool)
                with torch.no_grad():
                    X = m.run(m.embed(ids_pool[ix]), 0, l + 1, plant=plant)
                c = sample_c(1)[0]
                v = torch.randn(d, device="cuda", generator=gen) @ Sh
                s = (torch.randint(0, 2, (B, T), device="cuda", generator=gen).float() * 2 - 1) * valid[None]
                x = X.clone().requires_grad_(True)
                with torch.enable_grad():
                    hL = m.run(x, l + 1, plant=plant)
                    Fv = (hL[:, valid] @ c).sum()
                    (g,) = torch.autograd.grad(Fv, x, create_graph=True)
                    (hv,) = torch.autograd.grad((g * (s[..., None] * v)).sum(), x)
                y = (s[..., None] * hv)[:, valid].reshape(-1, d) @ Sh   # whitened per-position samples
                Ms[half] += y.double().T @ y.double()
                yl = y.mean(0, keepdim=True).double()
                Ml[half] += yl.T @ yl
                ny[half] += y.shape[0]
                if i % max(1, n_probe // 8) == 0 or i == n_probe - 1:
                    log(f"L{l} {label} probe {i+1}/{n_probe} ({time.time()-t0:.0f}s, peak {torch.cuda.max_memory_allocated()/2**30:.1f} GB)")
            return Ms, Ml, ny

        Ms, Ml, ny = run_probes(cfg["n_probe"], probe_ids)
        M = (Ms[0] + Ms[1]) / sum(ny)
        evM, UM = top_eig(M, k)
        H25 = UM[:, :k]
        _, UA = top_eig(Ms[0] / ny[0], k); _, UB = top_eig(Ms[1] / ny[1], k)
        _, UL = top_eig(Ml[0] + Ml[1], k)
        trM = float(torch.trace(M))
        L["H_eigs_top64"] = [float(x) for x in evM[:64]]
        L["H25_energy_frac"] = float(evM[:k].sum() / trM)
        L["H64_energy_frac"] = float(evM[:64].sum() / trM)
        L["split_half_overlap"] = overlap(UA[:, :k], UB[:, :k])
        L["H25_vs_J25_overlap"] = overlap(H25, J25)
        L["H25_vs_Jglob_overlap"] = overlap(H25, Jglob)
        L["H25_vs_Hlens25_overlap"] = overlap(H25, UL[:, :k])
        L["random_overlap"] = k / d
        Mf = M.float()
        L["J25_energy_frac_of_M"] = float(torch.trace(J25.T @ Mf @ J25) / trM)       # descriptive (not a criterion)
        L["Jglob_energy_frac_of_M"] = float(torch.trace(Jglob.T @ Mf @ Jglob) / trM)
        Rr = torch.linalg.qr(torch.randn(d, k, device="cuda", generator=gen))[0]
        L["rand25_energy_frac_of_M"] = float(torch.trace(Rr.T @ Mf @ Rr) / trM)
        L["H25_energy_frac_of_Jlens_Gram"] = float(torch.trace(H25.T @ Gj @ H25) / torch.trace(Gj))
        L["J25_energy_frac_of_Jlens_Gram"] = float(torch.trace(J25.T @ Gj @ J25) / torch.trace(Gj))
        evals_full = torch.linalg.eigvalsh(M.double()).flip(0)
        L["participation_ratio"] = float(evals_full.sum() ** 2 / (evals_full ** 2).sum())
        tens[f"H25_{l}"], tens[f"Hevec64_{l}"] = H25.cpu(), UM[:, :64].cpu()
        log(f"L{l} S4 H-space: top{k} energy {L['H25_energy_frac']:.3f} | split-half {L['split_half_overlap']:.2f} | "
            f"vs J25 {L['H25_vs_J25_overlap']:.2f} (random {k/d:.4f}) | PR {L['participation_ratio']:.1f} | M-energy in J25 "
            f"{L['J25_energy_frac_of_M']:.2f} / rand25 {L['rand25_energy_frac_of_M']:.3f} | J-Gram energy in H25 {L['H25_energy_frac_of_Jlens_Gram']:.2f} "
            f"vs J25 {L['J25_energy_frac_of_Jlens_Gram']:.2f} ({time.time()-t0:.0f}s)")
        save()

        # ---------------- S5 positive control (one layer)
        if l == cfg["pc_layer"]:
            gq = torch.Generator(device="cuda").manual_seed(123)
            a = torch.randn(d, device="cuda", generator=gq); a /= a.norm()
            b = torch.randn(d, device="cuda", generator=gq); b -= (b @ a) * a; b /= b.norm()
            # write direction: high output sensitivity at block l+1 (J-lens of record at layer l+1)
            J1 = Jall[l + 1].float().cuda()
            _, Uw = top_eig(J1.T @ U2 @ J1, 64)
            w = Uw[:, :64] @ torch.randn(64, device="cuda", generator=gq); w /= w.norm()
            kappa = float((J1 @ w) @ U2 @ (J1 @ w))
            G = math.sqrt(float(evM[k - 1]) / kappa)
            plant = {"block": l + 1, "mu": mu, "Sinvh": Sih, "a": a, "b": b, "w": w, "G": G}
            Msp, _, nyp = run_probes(cfg["n_pc"], probe_ids, plant=plant, label="PC")
            _, Up = top_eig((Msp[0] + Msp[1]) / sum(nyp), k)
            AB = torch.stack([a, b], 1)
            L["pos_control"] = {"G": G, "kappa": kappa, "target_eig": float(evM[k - 1]),
                                "recovery_top25_of_planted": overlap(AB, Up[:, :k])}
            log(f"L{l} S5 positive control: planted span recovered {L['pos_control']['recovery_top25_of_planted']:.2f} (G={G:.3g}) ({time.time()-t0:.0f}s)")
            del J1
            save()

        # ---------------- S6 natural-scale interactions (1 sigma, exact finite differences, fp32)
        nt = cfg["n_top"]
        Qh = Sh @ UM[:, :nt]                                  # 1-sigma activation moves along top H dirs
        pairs_h = [(i, j) for i in range(nt) for j in range(i + 1, nt)]
        R = torch.randn(d, 2 * len(pairs_h), device="cuda", generator=gen); R = torch.linalg.qr(R)[0]
        Qr = Sh @ R
        Ih, Ir, dec = [], [], []
        for ci in range(cfg["n_nat_ctx"]):
            ids = ev_ids[ci:ci + 1]
            with torch.no_grad():
                X = m.states(ids, l)
                p = int(torch.randint(SKIP + 8, T - 24, (1,), device="cuda", generator=gen))
                base_lp = None
                for (Q, pairs, store) in ((Qh, pairs_h, Ih), (Qr, [(2 * i, 2 * i + 1) for i in range(len(pairs_h))], Ir)):
                    for (i, j) in pairs:
                        Xs = X.repeat(4, 1, 1)
                        Xs[1, p] += Q[:, i]; Xs[2, p] += Q[:, j]; Xs[3, p] += Q[:, i] + Q[:, j]
                        hL = m.run(Xs, l + 1)
                        lp = m.logprobs_next(hL, ids.repeat(4, 1))[:, p:].sum(-1)
                        store.append(float(lp[3] - lp[1] - lp[2] + lp[0]))
                        if store is Ih and ci < 2:
                            dv = (hL[3] - hL[1] - hL[2] + hL[0])[p:].sum(0)
                            lg = F.linear((dv * m.gain).to(torch.bfloat16)[None], m.W_U)[0].float()
                            dec.append({"pair": [i, j], "ctx": ci,
                                        "up": m.tok.convert_ids_to_tokens(lg.topk(8).indices.tolist()),
                                        "down": m.tok.convert_ids_to_tokens((-lg).topk(8).indices.tolist())})
        L["natural"] = {"median_absI_Hpairs": float(np.median(np.abs(Ih))), "median_absI_random": float(np.median(np.abs(Ir))),
                        "ratio": float(np.median(np.abs(Ih)) / max(1e-12, np.median(np.abs(Ir)))), "n": len(Ih),
                        "decoded_interactions": dec}
        log(f"L{l} S6 natural 1-sigma |I|: H-pairs {L['natural']['median_absI_Hpairs']:.4f} vs random {L['natural']['median_absI_random']:.4f} "
            f"(ratio {L['natural']['ratio']:.1f}) ({time.time()-t0:.0f}s)")

        # ---------------- S8 decode of top H directions through the J-lens
        J = Jall[l].float().cuda()
        decs = []
        for i in range(nt):
            for sgn in (1, -1):
                vec = sgn * (Sh @ UM[:, i])
                lg = F.linear(((J @ vec) * m.gain).to(torch.bfloat16)[None], m.W_U)[0].float()
                decs.append({"dir": i, "sign": sgn, "top": m.tok.convert_ids_to_tokens(lg.topk(10).indices.tolist())})
        L["H_decodes"] = decs
        del J
        save()

    # ---------------- S7 causal: ablations + double dissociation (forward only, bf16 compute)
    bf16w.MODE = "bf16"
    n_ev = cfg["n_eval"]
    E = ev_ids[:n_ev]
    filler = m.tok(" the", add_special_tokens=False)["input_ids"][0]
    rng = random.Random(7)
    labs = []   # (window, target position t, I, M1, M2)
    with torch.no_grad():
        for wi in range(n_ev):
            ids = E[wi:wi + 1]
            base = m.logprobs_next(m.run(m.embed(ids), 0), ids)[0]
            for _ in range(cfg["n_lab_targets"]):
                t = rng.randrange(max(SKIP + 44, T // 2), T - 2)
                s1 = rng.randrange(SKIP, t - 40); s2 = rng.randrange(s1 + 10, t - 8)
                vs = []
                for rm in ((1, 0), (0, 1), (1, 1)):
                    mod = ids.clone()
                    if rm[0]: mod[0, s1:s1 + 8] = filler
                    if rm[1]: mod[0, s2:s2 + 8] = filler
                    vs.append(float(m.logprobs_next(m.run(m.embed(mod), 0), mod)[0, t - 1]))
                f0 = float(base[t - 1])
                labs.append((wi, t, f0 - vs[0] - vs[1] + vs[2], f0 - vs[0], f0 - vs[1]))
            if wi % max(1, n_ev // 8) == 0:
                log(f"S7 labels {wi+1}/{n_ev} ({time.time()-t0:.0f}s)")
    labs_np = np.array([[x[2], x[3], x[4]] for x in labs])
    tot = np.abs(labs_np).sum(1)
    big = tot >= np.median(tot)
    ratio = np.abs(labs_np[:, 0]) / (np.abs(labs_np[:, 1]) + np.abs(labs_np[:, 2]) + 1e-6)
    qi, qa = np.quantile(ratio[big], 0.8), np.quantile(ratio[big], 0.2)
    inter = big & (ratio >= qi); addv = big & (ratio <= qa)
    res["labels"] = {"n": len(labs), "n_inter": int(inter.sum()), "n_add": int(addv.sum()),
                     "median_absI": float(np.median(np.abs(labs_np[:, 0])))}
    log(f"S7 labels done: {len(labs)} targets, {int(inter.sum())} interaction / {int(addv.sum())} additive ({time.time()-t0:.0f}s)")
    gr = torch.Generator(device="cuda").manual_seed(99)
    for l in cfg["layers"]:
        L = res["layers"][str(l)]
        mu, Sh = tens[f"mu{l}"].cuda(), tens[f"Sh{l}"].cuda()
        Sih = sym_sqrt(Sh @ Sh, inv=True)
        H25, J25 = tens[f"H25_{l}"].cuda(), tens[f"J25_{l}"].cuda()
        Hperp = torch.linalg.qr(H25 - J25 @ (J25.T @ H25))[0]
        conds = {"H25": H25, "J25": J25, "H25perpJ": Hperp}
        for r in range(3):
            conds[f"rand25_{r}"] = torch.linalg.qr(torch.randn(d, k, device="cuda", generator=gr))[0]
        # raw-space PCA25 (oblique in whitened coords): ablate raw top PCs
        Sraw = Sh @ Sh
        _, Upca = top_eig(Sraw, k)
        names = list(conds) + ["PCA25raw"]
        kls = {n_: [] for n_ in names}
        dmg = {n_: np.zeros(len(labs)) for n_ in names}
        by_w = {}
        for li, (w2, t, *_r) in enumerate(labs):
            by_w.setdefault(w2, []).append((li, t))
        for wi in range(n_ev):
            ids = E[wi:wi + 1]
            with torch.no_grad():
                X = m.run(m.embed(ids), 0, l + 1)
                lb = torch.log_softmax(m.logits(m.text.norm(m.run(X, l + 1))[0, valid]), -1)
                for name in names:
                    if name == "PCA25raw":
                        P = Upca[:, :k]
                        Xa = X - ((X - mu) @ P) @ P.T
                    else:
                        Q = conds[name]
                        z = (X - mu) @ Sih
                        Xa = mu + (z - (z @ Q) @ Q.T) @ Sh
                    la = torch.log_softmax(m.logits(m.text.norm(m.run(Xa, l + 1))[0, valid]), -1)
                    kls[name].append(float((lb.exp() * (lb - la)).sum(-1).mean()))
                    for li, t in by_w.get(wi, []):
                        pi, tok = t - 1 - SKIP, ids[0, t]
                        dmg[name][li] = float(lb[pi, tok] - la[pi, tok])
            if wi % max(1, n_ev // 4) == 0:
                log(f"L{l} S7 ablations window {wi+1}/{n_ev} ({time.time()-t0:.0f}s)")
        out = {}
        rb = np.random.default_rng(0)
        for name in names:
            di, da = dmg[name][inter], dmg[name][addv]
            boots = []
            for _ in range(1000):
                bi, ba = rb.choice(di, len(di)), rb.choice(da, len(da))
                boots.append(bi.mean() / max(1e-9, abs(ba.mean())))
            out[name] = {"KL": float(np.mean(kls[name])), "dmg_inter": float(di.mean()), "dmg_add": float(da.mean()),
                         "ratio": float(di.mean() / max(1e-9, abs(da.mean()))),
                         "ratio_ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}
        L["ablation"] = out
        log(f"L{l} S7 ablation KL: " + " ".join(f"{n_}={v['KL']:.4f}" for n_, v in out.items())
            + " | dissoc ratio: " + " ".join(f"{n_}={v['ratio']:.2f}" for n_, v in out.items()) + f" ({time.time()-t0:.0f}s)")
        save()
    bf16w.MODE = "fp32"
    log(f"ALL DONE {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
