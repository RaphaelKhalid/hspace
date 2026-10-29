"""phi_J vs phi_H on Qwen3.5-0.8B, CPU only. Exploratory (not preregistered).

How much of the readout's first-order (Jacobian) and second-order (Hessian) energy at layer l is context-invariant?
Same estimator for both, in Sigma^{1/2}-whitened coordinates, readout F_c = sum_{p'} c . h_L*[p'] (pre-final-norm, the
J-lens proxy), c ~ N(0, Sigma_U):
  first order:  rows g_p = Sigma^{1/2} dF_c/dh_l[p]                       (one VJP per (probe, window); exact per p)
  second order: rows y_p = s_p Sigma^{1/2} (H (s (x) Sigma^{1/2} z))_p        (one HVP per sign vector; two sign vectors)
  phi = E[<mean_w, mean_w'>] (distinct windows w != w', same probe) / E[per-row energy]
  per-row energy: ||g_p||^2 (first order); <y1_p, y2_p> (second order, unbiased for ||H~_pp z||^2, Prop. 6b)
phi_J near 1: the averaged lens captures the map (the J-lens premise). phi_H near 0: curvature is context-switched.
    python hs_phi_cpu.py [--layers 6 12 18] [--probes 24] [--windows 4] [--T 64] [--model M] [--dtype bf16] [--tag t]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "mlens"
SKIP = 16


def load(name, dt=torch.float32):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, dtype=dt, attn_implementation="eager")
    m = m.to(dt)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, tok


def windows(tok, split, n, T, seed):
    import glob, os, random
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=[f"wikitext-103-raw-v1/{split}-*"])
    lines = []
    for f in sorted(glob.glob(os.path.join(root, "wikitext-103-raw-v1", f"{split}-*.parquet"))):
        lines += pq.read_table(f).column("text").to_pylist()
        if len(lines) > 200000:
            break
    arts, cur = [], []
    for ln in lines:
        if ln.startswith(" = ") and not ln.startswith(" = = ") and cur:
            arts.append("".join(cur)); cur = []
        cur.append(ln)
    random.Random(seed).shuffle(arts)
    out = []
    for a in arts:
        if len(a) < 2000:
            continue
        ids = tok(a[:8000], add_special_tokens=False)["input_ids"]
        if len(ids) >= 100 + T:
            out.append(ids[100:100 + T])
        if len(out) == n:
            break
    return torch.tensor(out)


class Readout:
    """F_c(delta) = sum_{p in valid} c . h_L*[p] with h_l <- h_l + delta (hooks; pre-final-norm target = last block output)."""

    def __init__(self, m, l):
        self.m, self.l = m, l
        self.layers = m.model.layers
        self.delta, self.hL = None, None
        self.layers[l].register_forward_hook(self._add)
        self.layers[-1].register_forward_hook(self._cap)

    def _add(self, mod, inp, out):
        if self.delta is None:
            return out
        if isinstance(out, tuple):
            return (out[0] + self.delta,) + tuple(out[1:])
        return out + self.delta

    def _cap(self, mod, inp, out):
        self.hL = out[0] if isinstance(out, tuple) else out

    def F(self, ids, c, delta):
        self.delta = delta
        self.m.model(input_ids=ids[None], use_cache=False)
        self.delta = None
        T = ids.shape[0]
        return (self.hL[0, SKIP:T - 1].float() @ c).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, nargs="*", default=[6, 12, 18])
    ap.add_argument("--probes", type=int, default=24)
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "bf16"])
    ap.add_argument("--tag", default="08b_cpu")
    ap.add_argument("--dump", action="store_true", help="save second-order rows (y1, y2, probe, win, tok) per layer")
    a = ap.parse_args()
    torch.set_num_threads(12)
    RES.mkdir(parents=True, exist_ok=True)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float32
    m, tok = load(a.model, dt)
    d = m.config.get_text_config().hidden_size
    W_U = m.lm_head.weight.float()
    gain = 1.0 + m.model.norm.weight.float()
    Ug = W_U * gain
    SigU = (Ug.T @ Ug) / Ug.shape[0]
    LU = torch.linalg.cholesky(SigU + 1e-8 * torch.eye(d) * SigU.diagonal().mean())
    T = a.T
    cov = windows(tok, "train", 24, T, seed=5)
    ev_ids = windows(tok, "train", a.probes * a.windows, T, seed=6)        # validation has ~60 long articles: too few
    assert ev_ids.shape[0] == a.probes * a.windows, ev_ids.shape
    out = {}
    for l in a.layers:
        t0 = time.time()
        R = Readout(m, l)
        # Sigma at layer l
        hs = []
        with torch.no_grad():
            for w in cov:
                o = m.model(input_ids=w[None], output_hidden_states=True, use_cache=False)
                hs.append(o.hidden_states[l + 1][0, SKIP:T - 1].float())
        X = torch.cat(hs).double()
        S = torch.cov(X.T)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, dtype=S.dtype)
        ev, V = torch.linalg.eigh(S)
        Sh = (V * ev.clamp_min(1e-12).sqrt()) @ V.T
        Shf = Sh.float()
        g = torch.Generator().manual_seed(100 + l)
        J_means, J_rows, H_means, H_rows, Js_means, Hs_means = [], [], [], [], [], []
        dump = {"y1": [], "y2": [], "probe": [], "win": [], "tok": []}
        for k in range(a.probes):
            c = LU @ torch.randn(d, generator=g)
            z = torch.randn(d, generator=g)
            v = Shf @ z
            gw, hw, gws, hws = [], [], [], []
            for w_i in range(a.windows):
                ids = ev_ids[k * a.windows + w_i]
                delta = torch.zeros(1, T, d, dtype=dt, requires_grad=True)
                Fv = R.F(ids, c, delta)
                (gr,) = torch.autograd.grad(Fv, delta, create_graph=True)                 # [1,T,d] per-position rows
                rows_g = (gr[0, SKIP:T - 1].detach().float() @ Shf)                                # whitened first-order rows
                J_rows.append(float(rows_g.pow(2).sum(1).mean())); gw.append(rows_g.mean(0))
                gws.append((rows_g / rows_g.norm(dim=1, keepdim=True).clamp_min(1e-30)).mean(0))
                ys = []
                for rep in range(2):
                    s = (torch.randint(0, 2, (T, 1), generator=g).float() * 2 - 1)
                    s[:SKIP] = 0; s[T - 1:] = 0
                    (hv,) = torch.autograd.grad(gr, delta, grad_outputs=(s * v)[None].to(dt), retain_graph=(rep == 0))
                    ys.append((s * hv[0].float())[SKIP:T - 1].detach() @ Shf)                      # y_p = s_p Sigma^1/2 (H(s x v))_p
                H_rows.append(float((ys[0] * ys[1]).sum(1).mean())); hw.append(((ys[0] + ys[1]) / 2).mean(0))
                if a.dump:
                    nr = ys[0].shape[0]
                    dump["y1"].append(ys[0].clone()); dump["y2"].append(ys[1].clone())
                    dump["probe"].append(torch.full((nr,), k)); dump["win"].append(torch.full((nr,), w_i)); dump["tok"].append(ids[SKIP:T - 1].clone())
                ym = (ys[0] + ys[1]) / 2
                hws.append((ym / ym.norm(dim=1, keepdim=True).clamp_min(1e-30)).mean(0))
            J_means.append(torch.stack(gw)); H_means.append(torch.stack(hw))
            Js_means.append(torch.stack(gws)); Hs_means.append(torch.stack(hws))
            if (k + 1) % 4 == 0:
                print(f"L{l}: probe {k+1}/{a.probes} ({(time.time()-t0)/60:.1f} min)", flush=True)

        def between(G):
            vals = []
            for gk in G:
                s_ = gk.sum(0); n = gk.shape[0]
                vals.append(float((s_ @ s_ - (gk * gk).sum()) / (n * (n - 1))))
            return np.array(vals)
        bJ, bH = between(J_means), between(H_means)
        dJ, dH = np.array(J_rows), np.array(H_rows)
        rng = np.random.default_rng(l)

        def ratio_ci(b, dd):
            bs = []
            for _ in range(2000):
                i = rng.integers(0, len(b), len(b)); j = rng.integers(0, len(dd), len(dd))
                bs.append(b[i].mean() / dd[j].mean())
            return float(b.mean() / dd.mean()), [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
        phiJ, ciJ = ratio_ci(bJ, dJ)
        phiH, ciH = ratio_ci(bH, dH)
        # directional (heavy-tail robust): window means of unit rows; per-row energy is 1 by construction
        sJ, sH = between(Js_means), between(Hs_means)
        out[str(l)] = {"phi_J": phiJ, "phi_J_ci95": ciJ, "phi_H": phiH, "phi_H_ci95": ciH,
                       "phi_dir_J": float(sJ.mean()), "phi_dir_H": float(sH.mean()), "probes": a.probes,
                       "windows": a.windows, "T": T, "model": a.model, "dtype": a.dtype, "minutes": (time.time() - t0) / 60}
        print(f"L{l}: phi_J = {phiJ:.3f} CI {[round(x, 3) for x in ciJ]} | phi_H = {phiH:.4f} CI {[round(x, 4) for x in ciH]} "
              f"| directional J {sJ.mean():.3f} H {sH.mean():.4f} | {(time.time()-t0)/60:.1f} min", flush=True)
        json.dump(out, open(RES / f"phi_{a.tag}.json", "w"), indent=1)
        if a.dump:
            torch.save({k_: torch.cat(v_) for k_, v_ in dump.items()}, HERE / "out" / f"phi_{a.tag}_rows_L{l}.pt")
        for h_ in list(R.layers[l]._forward_hooks) + list(R.layers[-1]._forward_hooks):
            pass
        R.layers[l]._forward_hooks.clear(); R.layers[-1]._forward_hooks.clear()


if __name__ == "__main__":
    main()
