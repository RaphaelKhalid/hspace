"""Does the H-dial carry a first-order feature that the J-lens AVERAGES AWAY? Offline, $0, POST HOC (bar set before the run).

J-lens = mean over contexts of the gradient g_x (whitened). A direction v whose per-context first-order effect g_x . v is
large but flips sign across contexts is invisible to the mean (J) yet shows up in curvature. For each token, with the
dial v taken from the EVEN contexts only (top-|lambda| eigvec of P Hbar_even P, P = projector off the J directions):
  rms_v   = sqrt(mean_x (g_x . v)^2) over the held-out ODD contexts      (per-context first-order size along v)
  null    = the same for 500 random unit directions in the same deflated subspace (P r / |P r|)
  flip    = |mean_x g_x . v| / rms_v                                       (0 = pure sign flipping, 1 = constant sign)
  ceiling = sqrt(top eigenvalue of P Cov_odd(g) P)                          (the largest rms any deflated direction can have)
Preset bar (2026-10-28, before running): J-blind feature iff rms_v >= 3 x median(null) AND flip <= 0.2, on ODD contexts.
CAVEAT (found before running): g_x ~ gbar + H (h_x - mu) makes a big zero-mean, sign-flipping g_x . v a mechanical
consequence of curvature along v, so that bar is near-tautological and is reported as DESCRIPTIVE only.
DECISIVE test (predictive, the H-lens as a context-free law for how the J-lens moves): with z_x the whitened layer-l state
at the readout position, predict each held-out ODD context's gradient from EVEN-context means:
  F_H    = 1 - E|g_x - gbar - Hbar z~_x|^2 / E|g_x - gbar|^2      (share of the context variation of the J row explained)
  F_v    = same, restricted to the dial coordinate g_x . v (deflation "all")
  null   = Hbar replaced by the mean of sign-flipped even Hessians (200 flips), which keeps the noise and kills the mean
  z~_x   = z_x - mean_even(z)
Decisive bar (preset): F_H >= 0.10 and above the flip-null 95th percentile, for >= half of the tokens.
Deflations: raw (P = I), own (token's own gbar), all (all tokens' gbar in the run).
    python hs_fullh_jblind.py --tag 08b
    python hs_fullh_jblind.py --tag concept --dir out/pod_concept/concept --toklist " simulated| fake| virtual| artificial| real| genuine| river| music"
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from hs_fullh import TOKENS

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "fullh"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="08b")
    ap.add_argument("--dir", default="out")
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--d", type=int, default=1024)
    ap.add_argument("--toklist", default="")
    ap.add_argument("--nrand", type=int, default=500)
    ap.add_argument("--flips", type=int, default=200)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--T", type=int, default=64)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    toks = a.toklist.split("|") if a.toklist else TOKENS[:8]
    NT, n, d = len(toks), a.n, a.d
    base = HERE / a.dir / f"fullh_{a.tag}_L{a.layer}"
    Hs = np.memmap(f"{base}_H.f16", dtype=np.float16, mode="r", shape=(NT, n, d, d))
    G = torch.from_numpy(np.array(np.memmap(f"{base}_G.f64", dtype=np.float64, mode="r", shape=(NT, n, d)))).to(dev)
    even = torch.arange(n, device=dev) % 2 == 0
    odd = ~even
    gb_even = G[:, even].mean(1)                                              # [NT, d] J directions from EVEN only
    gen = torch.Generator(device=dev).manual_seed(3)
    R = torch.randn(d, a.nrand, device=dev, dtype=torch.float64, generator=gen)

    def proj(rows):
        if rows is None:
            return torch.eye(d, device=dev, dtype=torch.float64)
        Q, _ = torch.linalg.qr(rows.T)
        return torch.eye(d, device=dev, dtype=torch.float64) - Q @ Q.T

    # ---- whitened states z_x at the readout position (same windows, positions and Sigma as hs_fullh)
    import random
    from hs_phi_cpu import SKIP, load, windows
    m, tok = load(a.model); m = m.to(dev)
    T, l = a.T, a.layer
    cov = windows(tok, "train", 24, T, seed=5)
    ctx = windows(tok, "train", n, T, seed=12)
    rng = random.Random(21 + l)
    qs = [rng.randrange(SKIP + 8, T) for _ in range(n)]
    with torch.no_grad():
        hs = [m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False).hidden_states[l + 1][0, SKIP:T - 1].double() for w in cov]
        S = torch.cov(torch.cat(hs).T)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device=dev, dtype=S.dtype)
        ev, V = torch.linalg.eigh(S)
        Sih = (V / ev.clamp_min(1e-12).sqrt()) @ V.T                          # Sigma^-1/2
        hq = torch.stack([m.model(input_ids=ctx[ci, : qs[ci] + 1][None].to(dev), output_hidden_states=True,
                                  use_cache=False).hidden_states[l + 1][0, qs[ci]].double() for ci in range(n)])
    Z = hq @ Sih                                                               # whitened states (mean removed below)
    Zt = Z - Z[even].mean(0)
    del m

    out = {"tag": a.tag, "layer": a.layer, "bar": "rms_v >= 3 x median(null) and flip <= 0.2 on odd contexts", "tokens": {}}
    for ti, t in enumerate(toks):
        He = torch.from_numpy(np.asarray(Hs[ti])).to(dev)[even].double().mean(0)
        go = G[ti, odd]                                                      # [n_odd, d] held-out per-context gradients
        ghat = gb_even[ti] / gb_even[ti].norm()
        Rt = {"J_direction": {"flip": float((go @ ghat).mean().abs() / (go @ ghat).pow(2).mean().sqrt()),
                              "rms": float((go @ ghat).pow(2).mean().sqrt())}}
        for kind, rows in (("raw", None), ("own", gb_even[ti:ti + 1]), ("all", gb_even)):
            P = proj(rows)
            e_, V_ = torch.linalg.eigh(P @ He @ P)
            v = V_[:, int(torch.argmax(e_.abs()))]
            s = go @ v
            rms_v, flip = float(s.pow(2).mean().sqrt()), float(s.mean().abs() / s.pow(2).mean().sqrt())
            Rp = P @ R; Rp = Rp / Rp.norm(dim=0, keepdim=True)
            null = (go @ Rp).pow(2).mean(0).sqrt()
            C = P @ torch.cov(go.T) @ P
            ce, cV = torch.linalg.eigh(C)
            r = {"rms_v": rms_v, "flip": flip, "null_median": float(null.median()), "ratio": rms_v / float(null.median()),
                 "null_pct": float((null < rms_v).double().mean()), "ceiling": float(ce[-1].clamp_min(0).sqrt()),
                 "frac_of_ceiling": rms_v / float(ce[-1].clamp_min(1e-30).sqrt()),
                 "cos_top_switch_modes": [float((v @ cV[:, -k]).abs()) for k in (1, 2, 3)],
                 "sign_pos_frac": float((s > 0).double().mean())}
            r["pass"] = bool(r["ratio"] >= 3 and flip <= 0.2)
            Rt[kind] = r
            print(f"{t!r:12s} {kind:3s} | rms(g.v) {rms_v:.3g} = {r['ratio']:.1f}x random (>{100*r['null_pct']:.0f}% of 500), "
                  f"{100*r['frac_of_ceiling']:.0f}% of ceiling | flip {flip:.2f} (+frac {r['sign_pos_frac']:.2f}) "
                  f"| cos top switch modes {[round(x, 2) for x in r['cos_top_switch_modes']]} | {'PASS' if r['pass'] else 'fail'}", flush=True)
        # ---- decisive: does Hbar_even predict the held-out per-context gradient variation?
        Hfull = torch.from_numpy(np.asarray(Hs[ti])).to(dev).double()
        ge = G[ti, even].mean(0)
        res0 = go - ge                                                      # J-only residual on odd contexts
        pred = Zt[odd] @ He                                                 # Hbar_even z~_x (He symmetric)
        FH = float(1 - (res0 - pred).pow(2).sum() / res0.pow(2).sum())
        vall = None
        P = proj(gb_even); e_, V_ = torch.linalg.eigh(P @ He @ P); vall = V_[:, int(torch.argmax(e_.abs()))]
        Fv = float(1 - ((res0 - pred) @ vall).pow(2).sum() / (res0 @ vall).pow(2).sum())
        gq = torch.Generator(device=dev).manual_seed(40 + ti); nullF = []
        ne = int(even.sum())
        for _ in range(a.flips):
            eps = (torch.randint(0, 2, (ne,), device=dev, generator=gq).double() * 2 - 1)
            Hf = torch.einsum("x,xij->ij", eps, Hfull[even]) / ne
            nullF.append(float(1 - (res0 - Zt[odd] @ Hf).pow(2).sum() / res0.pow(2).sum()))
        Rt["predictive"] = {"F_H": FH, "F_v_all": Fv, "null_q95": float(np.quantile(nullF, 0.95)), "null_max": float(np.max(nullF)),
                            "J_only_share_of_g_energy": float(res0.pow(2).sum() / go.pow(2).sum()),
                            "pass": bool(FH >= 0.10 and FH > np.quantile(nullF, 0.95))}
        del Hfull
        p_ = Rt["predictive"]
        print(f"{t!r:12s} PREDICTIVE | Hbar z explains {100*FH:.1f}% of held-out gradient variation (flip null q95 {100*p_['null_q95']:.1f}%, max {100*p_['null_max']:.1f}%) "
              f"| along the deflated dial {100*Fv:.1f}% | context variation = {100*p_['J_only_share_of_g_energy']:.0f}% of gradient energy | {'PASS' if p_['pass'] else 'fail'}", flush=True)
        print(f"{t!r:12s} J-direction itself: flip {Rt['J_direction']['flip']:.2f} (constant-sign reference)", flush=True)
        out["tokens"][t] = Rt
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / f"fullh_jblind_{a.tag}_L{a.layer}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
