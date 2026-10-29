"""Lanczos on the context-averaged Hessian (the H-lens operator) without ever forming it. Exploratory.

Operator (whitened, single readout position = last token of each context, as hs_fullh):
    v -> Hbar v = mean_x  Sigma^1/2 H_x Sigma^1/2 v,   H_x = d^2 (u . h_L*[q]) / d h_l[q]^2
Each matvec = one exact HVP per context (double backward through a cached-prefix single-token step). The first backward
graph of every context is built once and retained, so a matvec costs n second backwards. scipy eigsh (ARPACK Lanczos)
returns the top-k eigenpairs by magnitude: the dials the operator itself picks, unsupervised.

Validation mode (--validate TOKEN_INDEX): uses the hs_fullh contexts/positions/readout and compares with the exact mean of
the saved per-context Hessians (out/fullh_08b_L18_H.f16) over the same contexts.
    python hs_lanczos.py --validate 6 --n 16 --k 6 --device cpu        (token " to", 0.8B L18)
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hs_phi_cpu import SKIP, windows
from hs_fullh import TOKENS, _patch_linear_cache

HERE = Path(__file__).resolve().parent


class HbarOperator:
    def __init__(self, m, layer, ctx, qs, u, Sh, dev):
        self.m, self.l, self.dev, self.Sh = m, layer, dev, Sh
        self.st = {"delta": None, "hL": None}
        self.handles = [m.model.layers[layer].register_forward_hook(self._add),
                        m.model.layers[-1].register_forward_hook(lambda mod, i, o: self.st.__setitem__("hL", o[0] if isinstance(o, tuple) else o))]
        self.graphs, self.nmv = [], 0
        for ids, q in zip(ctx, qs):
            ids1 = ids[: q + 1][None].to(dev)
            with torch.no_grad():
                pkv = m.model(input_ids=ids1[:, :q], use_cache=True).past_key_values
            delta = torch.zeros(1, Sh.shape[0], device=dev, dtype=Sh.dtype, requires_grad=True)
            self.st["delta"] = delta
            m.model(input_ids=ids1[:, q:q + 1], past_key_values=pkv, use_cache=True)
            hL = self.st["hL"][:, -1]                                           # pre-final-norm last-block output (J-lens proxy)
            self.st["delta"] = None
            (g,) = torch.autograd.grad((hL.to(u.dtype) @ u).sum(), delta, create_graph=True)
            self.graphs.append((delta, g))

    def close(self):
        for h in self.handles:
            h.remove()
        self.graphs = []

    def _add(self, mod, inp, out):
        if self.st["delta"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        h = h + self.st["delta"][:, None, :].to(h.dtype)
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def matvec(self, v):
        self.nmv += 1
        w = self.Sh @ torch.as_tensor(np.asarray(v).reshape(-1), device=self.dev, dtype=self.Sh.dtype)
        acc = torch.zeros_like(w)
        for delta, g in self.graphs:
            (hv,) = torch.autograd.grad(g, delta, grad_outputs=w[None].to(g.dtype), retain_graph=True)
            acc += hv[0].to(acc.dtype)
        return (self.Sh @ (acc / len(self.graphs))).detach().cpu().double().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--validate", type=int, default=6, help="token index into hs_fullh.TOKENS (6 = ' to')")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from scipy.sparse.linalg import LinearOperator, eigsh
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(12)
    _patch_linear_cache()
    dev, l, d = a.device, a.layer, None
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32, attn_implementation="eager").to(dev).float().eval()
    for p in m.parameters():
        p.requires_grad_(False)
    d = m.config.get_text_config().hidden_size
    t = TOKENS[a.validate]
    tid = tok(t, add_special_tokens=False)["input_ids"][0]
    u = (m.lm_head.weight.float()[tid] * (1.0 + m.model.norm.weight.float())).to(dev)       # uncentered, as fullh_08b
    cov = windows(tok, "train", 24, 64, seed=5)
    ctx = windows(tok, "train", 64, 64, seed=12)
    rng = random.Random(21 + l)
    qs = [rng.randrange(SKIP + 8, 64) for _ in range(64)]
    with torch.no_grad():
        hs = [m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False).hidden_states[l + 1][0, SKIP:63].double() for w in cov]
    S = torch.cov(torch.cat(hs).T)
    S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device=dev, dtype=S.dtype)
    ev, V = torch.linalg.eigh(S)
    Sh = ((V * ev.clamp_min(1e-12).sqrt()) @ V.T).float()
    t0 = time.time()
    op = HbarOperator(m, l, [ctx[i] for i in range(a.n)], qs[: a.n], u, Sh, dev)
    t1 = time.time()
    vals, vecs = eigsh(LinearOperator((d, d), matvec=op.matvec, dtype=np.float64), k=a.k, which="LM", tol=1e-6)
    t2 = time.time()
    o = np.argsort(-np.abs(vals)); vals, vecs = vals[o], vecs[:, o]
    # exact reference: mean of the saved whitened per-context Hessians over the same contexts
    H = np.memmap(HERE / "out" / "fullh_08b_L18_H.f16", dtype=np.float16, mode="r", shape=(8, 64, d, d))
    Hbar = np.asarray(H[a.validate, : a.n], dtype=np.float64).mean(0)
    Hbar = (Hbar + Hbar.T) / 2
    ev2, V2 = np.linalg.eigh(Hbar)
    o2 = np.argsort(-np.abs(ev2))[: a.k]; ev2, V2 = ev2[o2], V2[:, o2]
    ov = [float(np.linalg.norm(V2[:, :j].T @ vecs[:, :j]) ** 2 / j) for j in range(1, a.k + 1)]
    res = {"token": t, "layer": l, "n_contexts": a.n, "k": a.k, "lanczos_eigs": vals.tolist(), "exact_eigs": ev2.tolist(),
           "eig_rel_err": (np.abs(vals - ev2) / np.abs(ev2)).tolist(), "subspace_overlap_top_j": ov,
           "matvecs": op.nmv, "hvps": op.nmv * a.n, "build_s": t1 - t0, "lanczos_s": t2 - t1}
    print(f"{t!r} L{l} n={a.n}: Lanczos {np.round(vals, 4).tolist()} vs exact {np.round(ev2, 4).tolist()} | "
          f"overlap top-1..k {np.round(ov, 3).tolist()} | {op.nmv} matvecs = {op.nmv * a.n} HVPs in {t2 - t1:.0f}s", flush=True)
    (HERE.parent / "results" / "fullh").mkdir(parents=True, exist_ok=True)
    json.dump(res, open(HERE.parent / "results" / "fullh" / f"lanczos_validate_{a.validate}_n{a.n}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
