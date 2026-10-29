"""J-deflated full Hessian lens: is there second-order structure OFF the J-lens directions? POST HOC, exploratory. $0.

hs_fullh.py found, per token t, a stable, causal mean-Hessian top direction v, but cos(v, gbar_t) = 0.59-0.82: mostly the
token's own (whitened) J-lens direction. Here every saved whitened Hessian H_t(x) is projected off the J directions,
H_perp = P H P, P = I - Q Q^T, and the hs_fullh tests are rerun on H_perp:
  own : Q = gbar_t / |gbar_t|                    (the token's own J-lens direction)
  all8: Q = orthonormal basis of all 8 tokens' gbar (every J-lens direction in the token set)
Leakage control: for the causal test, P and v both come from the even half; the test runs on the odd half, and the
random baseline directions are projected by the same P. Reads out/fullh_<tag>_L<l>_H.f16 and _G.f64 (no recompute).
    python hs_fullh_defl.py [--layer 18] [--tag 08b]
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from hs_phi_cpu import SKIP, load, windows
from hs_fullh import TOKENS

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "fullh"
PROG = HERE / "out" / "progress"


def progress(done, total, note, t0, status="running"):
    PROG.mkdir(parents=True, exist_ok=True)
    tmp = PROG / "fullh_defl.json.tmp"
    json.dump({"job": "fullh_defl", "title": "J-deflated full Hessian lens (0.8B L18)", "status": status, "stage": "deflated tests",
               "stage_index": 1, "stage_count": 1, "done": done, "total": total, "unit": "token x deflation",
               "started": t0, "updated": time.time(), "note": note}, open(tmp, "w"))
    for _ in range(5):
        try:
            tmp.replace(PROG / "fullh_defl.json")
            return
        except PermissionError:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--f", type=float, default=0.25)
    ap.add_argument("--flips", type=int, default=200)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--tag", default="08b")
    ap.add_argument("--toklist", type=str, default="")
    ap.add_argument("--center", action="store_true")
    ap.add_argument("--kinds", type=str, default="own,all8")
    ap.add_argument("--skip", type=int, default=0, help="tokens already done (merged from the existing JSON)")
    ap.add_argument("--only", type=str, default="", help="comma list of token indices to run (written to a separate _only JSON)")
    a = ap.parse_args()
    dev = "cuda"
    torch.backends.cudnn.enabled = False
    m, tok = load(a.model)
    m = m.to(dev)
    d, T, l, NT = m.config.get_text_config().hidden_size, a.T, a.layer, a.tokens
    toks = a.toklist.split("|") if a.toklist else TOKENS[:NT]
    NT = len(toks)
    tid = [tok(t, add_special_tokens=False)["input_ids"][0] for t in toks]
    W_U = m.lm_head.weight.float()
    gain = 1.0 + m.model.norm.weight.float()
    ubar = (W_U * gain).mean(0).double() if a.center else torch.zeros(W_U.shape[1], device=W_U.device, dtype=torch.float64)
    Ut = (W_U[tid] * gain).double() - ubar[None]                          # must match the hs_fullh readout
    cov = windows(tok, "train", 24, T, seed=5)                            # identical to hs_fullh
    ctx = windows(tok, "train", a.n, T, seed=12)
    rng = random.Random(21 + l)
    qs = [rng.randrange(SKIP + 8, T) for _ in range(a.n)]
    st = {"delta": None, "q": None, "hL": None}

    def add(mod, inp, out):
        if st["delta"] is None:
            return out
        h = out[0] if isinstance(out, tuple) else out
        oh = torch.zeros(h.shape[1], device=h.device, dtype=h.dtype); oh[st["q"]] = 1
        h = h + st["delta"][:, None, :] * oh[None, :, None]
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h
    m.model.layers[l].register_forward_hook(add)
    m.model.layers[-1].register_forward_hook(lambda mod, inp, o: st.__setitem__("hL", o[0] if isinstance(o, tuple) else o))
    hs = []
    with torch.no_grad():
        for w in cov:
            o = m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False)
            hs.append(o.hidden_states[l + 1][0, SKIP:T - 1].double())
    S = torch.cov(torch.cat(hs).T)
    S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device=dev, dtype=S.dtype)
    ev, V = torch.linalg.eigh(S)
    Sh = (V * ev.clamp_min(1e-12).sqrt()) @ V.T
    trS = float(torch.trace(S))

    Hs = np.memmap(HERE / "out" / f"fullh_{a.tag}_L{l}_H.f16", dtype=np.float16, mode="r", shape=(NT, a.n, d, d))
    Gs = torch.from_numpy(np.array(np.memmap(HERE / "out" / f"fullh_{a.tag}_L{l}_G.f64", dtype=np.float64, mode="r",
                                             shape=(NT, a.n, d)))).to(dev)
    even = (torch.arange(a.n) % 2 == 0).to(dev)
    gen = torch.Generator(device=dev).manual_seed(7)

    def basis(G):                                                          # orthonormal basis of the rows of G
        Q, _ = torch.linalg.qr(G.T.float())
        return Q

    def proj(Q):
        return torch.eye(d, device=dev) - Q @ Q.T

    def between(X):
        Xf = X.reshape(X.shape[0], -1)
        s = Xf.sum(0); n = Xf.shape[0]
        return float((s @ s - (Xf * Xf).sum()) / (n * (n - 1)) / (Xf * Xf).sum(1).mean())

    def topk_vecs(M, k):
        e_, V_ = torch.linalg.eigh(M)
        o = torch.argsort(e_.abs(), descending=True)[:k]
        return e_[o], V_[:, o]

    def ovl(A, B_):
        return float((A.T @ B_).pow(2).sum() / A.shape[1])

    gb_all, gb_even = Gs.mean(1).float(), Gs[:, even].mean(1).float()     # [NT, d]
    out = {"layer": l, "n": a.n, "model": a.model, "post_hoc": True, "tokens": {}}
    prev = RES / f"fullh_defl_{a.tag}_L{l}.json"
    if a.skip and prev.exists():
        out["tokens"] = {k: v for k, v in json.load(open(prev))["tokens"].items() if k in toks[: a.skip]}
        assert len(out["tokens"]) == a.skip, list(out["tokens"])
    kinds = a.kinds.split(",")
    t0 = time.time(); jobs = NT * len(kinds)

    def var_grads(ids_var):
        """mean whitened gradient over EVEN contexts of the readouts of tokens ids_var (their J-lens directions)"""
        Uv = (W_U[ids_var] * gain).double() - ubar[None]
        acc = torch.zeros(len(ids_var), d, device=dev, dtype=torch.float64)
        ev_idx = torch.nonzero(even).flatten().tolist()
        for ci in ev_idx:
            q = qs[ci]; ids = ctx[ci, : q + 1][None].to(dev).expand(len(ids_var), -1)
            delta = torch.zeros(len(ids_var), d, device=dev, requires_grad=True)
            st["delta"], st["q"] = delta, q
            m.model(input_ids=ids, use_cache=False)
            st["delta"] = None
            (g,) = torch.autograd.grad((st["hL"][:, q].double() * Uv).sum(), delta)
            acc += g.double() @ Sh
        return (acc / len(ev_idx)).float()
    for ti, t in enumerate(toks):
        if ti < a.skip or (a.only and ti not in [int(x) for x in a.only.split(",")]):
            continue
        H = torch.from_numpy(np.asarray(Hs[ti])).to(dev).float()
        R = {}
        var_ids = None
        for kind in kinds:
            progress(ti * len(kinds) + kinds.index(kind), jobs, f"token {t!r}, deflation {kind}", t0)
            if kind == "own":
                Qa, Qe = basis(gb_all[ti:ti + 1]), basis(gb_even[ti:ti + 1])
            elif kind == "all8":
                Qa, Qe = basis(gb_all), basis(gb_even)
            else:                                   # allvar: also the J directions of the 10 tokens the all8 dial boosts most
                assert var_ids is not None, "allvar needs all8 first"
                gv = var_grads(var_ids)
                Qa, Qe = basis(torch.cat([gb_all, gv])), basis(torch.cat([gb_even, gv]))
            Pa, Pe = proj(Qa), proj(Qe)
            Hp = Pa @ H @ Pa                                                # [n, d, d]
            r = {"phi_H_perp": between(Hp), "phi_H_full": between(H),
                 "mean_energy_kept": float((Pa @ H.mean(0) @ Pa).pow(2).sum() / H.mean(0).pow(2).sum())}
            lam, _ = topk_vecs(Hp.mean(0), 5)
            r["top_lambda"] = [float(x) for x in lam]
            _, Ve_ = topk_vecs(Hp[even].mean(0), 5); _, Vo_ = topk_vecs(Hp[~even].mean(0), 5)
            r["split"] = {k: ovl(Ve_[:, :k], Vo_[:, :k]) for k in (1, 3, 5)}
            nl, ns = [], {1: [], 3: [], 5: []}
            for _ in range(a.flips):
                eps = (torch.randint(0, 2, (a.n,), device=dev, generator=gen).float() * 2 - 1)
                nl.append(float(torch.linalg.eigvalsh(torch.einsum("n,nij->ij", eps, Hp) / a.n).abs().max()))
                _, Fe = topk_vecs(torch.einsum("n,nij->ij", eps * even, Hp), 5)
                _, Fo = topk_vecs(torch.einsum("n,nij->ij", eps * (~even), Hp), 5)
                for k in ns:
                    ns[k].append(ovl(Fe[:, :k], Fo[:, :k]))
            r["lam1_null_q95"] = float(np.quantile(nl, 0.95))
            r["lam1_p"] = float((np.sum(np.array(nl) >= abs(r["top_lambda"][0])) + 1) / (a.flips + 1))
            r["split_null_q95"] = {k: float(np.quantile(v, 0.95)) for k, v in ns.items()}
            r["split_p"] = {k: float((np.sum(np.array(v) >= r["split"][k]) + 1) / (a.flips + 1)) for k, v in ns.items()}
            # causal, held out: P and v from the even half only
            He = Pe @ H[even].mean(0) @ Pe
            lam_e, Ve1 = topk_vecs(He, 1)
            gr = torch.Generator(device=dev).manual_seed(100 + ti)
            Vr = Pe @ torch.randn(d, 8, device=dev, generator=gr); Vr /= Vr.norm(dim=0, keepdim=True)
            dirs = torch.cat([Ve1, Vr], 1).double()
            r["cos_v_own_gbar"] = float((Ve1[:, 0] @ gb_all[ti]).abs() / gb_all[ti].norm())
            r["cos_v_other_gbar_max"] = float(max((Ve1[:, 0] @ gb_all[j]).abs() / gb_all[j].norm() for j in range(NT) if j != ti))
            M = Sh @ dirs; M = M / M.norm(dim=0, keepdim=True) * a.f * trS ** 0.5
            c2, sym_all, asym_all, c1 = [], [], [], []
            with torch.no_grad():
                for ci in torch.nonzero(~even).flatten().tolist():
                    q = qs[ci]; ids = ctx[ci, : q + 1][None].to(dev); nd = M.shape[1]
                    st["delta"], st["q"] = torch.cat([M.T, -M.T, torch.zeros(1, d, device=dev, dtype=M.dtype)]).float(), q
                    m.model(input_ids=ids.expand(2 * nd + 1, -1), use_cache=False)
                    st["delta"] = None
                    hq = st["hL"][:, q].double()
                    sym = hq[:nd] + hq[nd:2 * nd] - 2 * hq[-1:]
                    asym = (hq[:nd] - hq[nd:2 * nd]) / 2                         # odd orders: the first-order (J) effect of +-m
                    c2.append((sym @ Ut[ti]).cpu()); sym_all.append(sym[0].cpu())
                    c1.append(float(asym[0] @ Ut[ti])); asym_all.append(asym[0].cpu())
            c2 = torch.stack(c2); sgn = float(np.sign(float(lam_e[0])))
            r["causal"] = {"lambda_even": float(lam_e[0]), "sign_agree_top": float((torch.sign(c2[:, 0]) == sgn).float().mean()),
                           "z_top_vs_rand": float((c2[:, 0].mean() * sgn) / (c2[:, 1:].mean(0).std() + 1e-30)),
                           "rms_c2_top": float(c2[:, 0].pow(2).mean().sqrt()), "rms_c2_rand": float(c2[:, 1:].pow(2).mean().sqrt())}
            lg = (torch.stack(sym_all).mean(0).to(dev).float() * gain) @ W_U.T; lg = lg - lg.mean()
            r["decode_up"] = [tok.decode([i]) for i in torch.topk(lg * sgn, 10).indices.tolist()]
            if kind == "all8":
                var_ids = torch.topk(lg * sgn, 10).indices.tolist()
            if kind == "allvar":
                r["deflated_variants"] = [tok.decode([i]) for i in var_ids]
            r["rank_of_t"] = int(((lg * sgn) > (lg * sgn)[tid[ti]]).sum())
            la = (torch.stack(asym_all).mean(0).to(dev).float() * gain) @ W_U.T; la = la - la.mean()
            r["decode_first_order_up"] = [tok.decode([i]) for i in torch.topk(la, 10).indices.tolist()]
            r["decode_first_order_down"] = [tok.decode([i]) for i in torch.topk(-la, 10).indices.tolist()]
            r["first_order_on_t_rms"] = float(np.sqrt(np.mean(np.square(c1))))       # |odd-order effect on logit t|
            r["second_order_on_t_rms"] = float(c2[:, 0].pow(2).mean().sqrt()) / 2     # sym = 2 x even-order part
            R[kind] = r
            c = r["causal"]
            print(f"{t!r:7s} {kind:4s} | kept {r['mean_energy_kept']:.2f} phi_perp {r['phi_H_perp']:.3f} (full {r['phi_H_full']:.3f}) "
                  f"| lam1 {r['top_lambda'][0]:+.3g} null95 {r['lam1_null_q95']:.3g} p {r['lam1_p']:.3f} "
                  f"| split1/3/5 {r['split'][1]:.2f}/{r['split'][3]:.2f}/{r['split'][5]:.2f} p {r['split_p'][1]:.3f}/{r['split_p'][3]:.3f}/{r['split_p'][5]:.3f} "
                  f"| causal sign {c['sign_agree_top']:.2f} z {c['z_top_vs_rand']:+.1f} rms {c['rms_c2_top']:.3g}/{c['rms_c2_rand']:.3g} "
                  f"| 1st-order up {r['decode_first_order_up'][:5]} down {r['decode_first_order_down'][:3]} | on t: 1st {r['first_order_on_t_rms']:.3g} 2nd {r['second_order_on_t_rms']:.3g} "
                  f"| t rank {r['rank_of_t']} | cos own/other gbar {r['cos_v_own_gbar']:.2f}/{r['cos_v_other_gbar_max']:.2f} | up {r['decode_up'][:6]}", flush=True)
        out["tokens"][t] = R
        RES.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(RES / (f"fullh_defl_{a.tag}_L{l}" + ("_only" if a.only else "") + ".json"), "w"), indent=1)
        del H, Hp
    progress(jobs, jobs, "done", t0, status="done")


if __name__ == "__main__":
    main()
