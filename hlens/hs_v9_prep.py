"""v9 prep (laptop, $0): the replicated punctuation-conditional subspace EX from the FRESH v8 data, its Sigma-orbit twins,
and its geometry. EX = top-k (k=5) patterns of the spatial-sign estimator on punctuation rows, after the v5 axis-aligned
drop (PCA 1-5 + massive-span axes), D-normalized (R = D^-1/2 M D^-1/2), patterns QR(D^1/2 V), mapped to the run-2
whitened frame. Twins: Q_j = P (s_j * P'EX), s_j iid +-1 (seed 9090 + l): identical raw variance and per-PCA-axis mass.
    python hs_v9_prep.py <hspace2_full.pt> <v8 dump dir> 16 40
"""
import json
import sys

import torch
from transformers import AutoTokenizer

sys.argv, _a = sys.argv[:1], sys.argv[1:]
import hs_flip5 as F5  # noqa: E402

K_EX, N_TWIN = 5, 8


def klass(s):
    t = s.strip()
    if t == "":
        return "whitespace"
    if all(not ch.isalnum() for ch in t):
        return "punctuation"
    if t.isdigit():
        return "digit"
    return "word"


def ov(A, B):
    return float((A.T @ B).pow(2).sum() / A.shape[1])


def main(t2p, ddir, layers):
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")
    T2 = torch.load(t2p, map_location="cpu", weights_only=False)
    out, geo = {}, {}
    punct_ids = None
    for l in layers:
        Sh, mu = T2[f"Sh{l}"].double(), T2[f"mu{l}"].double()
        ev, P = torch.linalg.eigh(Sh)
        P = P[:, torch.argsort(ev, descending=True)]
        d, K = P.shape[0], 1000
        C = F5.conf_basis(mu, Sh, 4)
        wC = (P.T @ C).pow(2).sum(1)
        drop = torch.zeros(d, dtype=torch.bool); drop[:5] = True
        tot = float(wC[:K].sum())
        for i_ in torch.argsort(wC[:K], descending=True):
            if float(wC[:K][drop[:K]].sum()) >= 0.9 * tot:
                break
            drop[int(i_)] = True
        keep = (~drop).double()
        cols = torch.nonzero(keep[:K] > 0).squeeze(1)
        D = torch.load(f"{ddir}/hs_v8_full_L{l}.pt", map_location="cpu", weights_only=False)
        ut = {int(t): klass(tok.decode([int(t)])) for t in torch.unique(D["loc_tok"])}
        if punct_ids is None:
            punct_ids = sorted(t for t, c in ut.items() if c == "punctuation")
        idx = torch.tensor([i for i, t in enumerate(D["loc_tok"].tolist()) if ut[t] == "punctuation"])
        Y = D["loc_y1"][idx].double() @ P
        U = (Y * keep / (Y * keep).norm(dim=1, keepdim=True).clamp_min(1e-30))[:, cols]
        M = U.T @ U / len(U)
        dg = torch.diagonal(M).clamp_min(1e-30)
        R = dg.rsqrt()[:, None] * M * dg.rsqrt()[None, :]
        e, V = torch.linalg.eigh(R)
        V = V[:, torch.argsort(e, descending=True)[:K_EX]]
        pat = torch.linalg.qr(dg.sqrt()[:, None] * V)[0]
        EX = P[:, cols] @ pat                                      # whitened frame, orthonormal [d, 5]
        g = torch.Generator().manual_seed(9090 + l)
        tw = [P @ ((torch.randint(0, 2, (d, 1), generator=g).double() * 2 - 1) * (P.T @ EX)) for _ in range(N_TWIN)]
        J = torch.linalg.qr(T2[f"J25w_{l}"].double())[0]
        raw_var = lambda Q: float((Sh @ Q).pow(2).sum() / (Sh @ Sh).trace())
        geo[str(l)] = {"n_punct_rows": len(idx), "EX_vs_twins": [ov(EX, t) for t in tw],
                       "twin_twin_mean": sum(ov(tw[i], tw[j]) for i in range(N_TWIN) for j in range(i + 1, N_TWIN)) / 28,
                       "EX_in_J25": ov(EX, J), "twins_in_J25": [ov(t, J) for t in tw], "EX_in_PCA25": ov(EX, P[:, :25]),
                       "EX_in_C": ov(EX, C), "raw_var_EX": raw_var(EX), "raw_var_twins": [raw_var(t) for t in tw],
                       "top_eigs_R": [float(x) for x in e.sort(descending=True).values[:8]]}
        print(f"L{l}: punct rows {len(idx)} | EX vs twins {sum(geo[str(l)]['EX_vs_twins'])/N_TWIN:.3f} | EX in J25 {geo[str(l)]['EX_in_J25']:.3f} "
              f"(twins {sum(geo[str(l)]['twins_in_J25'])/N_TWIN:.3f}) | EX in PCA25 {geo[str(l)]['EX_in_PCA25']:.3f} | in C {geo[str(l)]['EX_in_C']:.3f} | "
              f"raw var EX {geo[str(l)]['raw_var_EX']:.5f} twins {min(geo[str(l)]['raw_var_twins']):.5f}-{max(geo[str(l)]['raw_var_twins']):.5f}", flush=True)
        out[f"EX_{l}"] = EX.float()
        out[f"tw_{l}"] = torch.stack(tw).float()
    out["punct_token_ids"] = torch.tensor(punct_ids)
    torch.save(out, "out/v9_subspaces.pt")
    json.dump(geo, open("out/v9_geometry.json", "w"), indent=1)
    print(f"saved out/v9_subspaces.pt ({len(punct_ids)} punctuation token ids)", flush=True)


if __name__ == "__main__":
    main(_a[0], _a[1], [int(x) for x in _a[2:]] or [16, 40])
