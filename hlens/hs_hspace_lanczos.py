"""Is there an H-SPACE? (paper/PREREG-hspace-lanczos-2026-10-28.md). Unsupervised Lanczos on the context-averaged Hessian of
a class contrast readout, fitted on GENERIC WikiText contexts, then tested against the AND battery's class directions.
    python hs_hspace_lanczos.py --model Qwen/Qwen3.5-0.8B --battery ../results/and_battery/and_battery_08b.json --tag 08b
    python hs_hspace_lanczos.py --model Qwen/Qwen3.6-27B --bf16 --battery ../results/and_battery/and_battery_27b.json --tag 27b
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

import hs_and_items as ITEMS
from hs_fullh import _patch_linear_cache
from hs_lanczos import HbarOperator
from hs_phi_cpu import SKIP, windows

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results" / "and_battery"
READOUTS = {"negation_polarity": (" positive", " negative"), "quantifier_truth": (" True", " False"),
            "comparison_order": (" first", " second"), "number_comparison": (" Yes", " No"), "truth_frame": (" Yes", " No"),
            "xor_bits": (" 0", " 1"), "agreement_tense": (" are", " is"), "pronoun_case": (" her", " him"), "refusal": (" I", " To")}
XOR = {"negation_polarity", "quantifier_truth", "comparison_order", "number_comparison", "truth_frame", "xor_bits", "binding"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--battery", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--hutch", type=int, default=30)
    ap.add_argument("--tag", default="08b")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    from scipy.sparse.linalg import LinearOperator, eigsh
    from transformers import AutoModelForCausalLM, AutoTokenizer
    _patch_linear_cache()
    dev = a.device; dt = torch.bfloat16 if a.bf16 else torch.float32
    tok = AutoTokenizer.from_pretrained(a.model)
    m = AutoModelForCausalLM.from_pretrained(a.model, dtype=dt, attn_implementation="eager", device_map=dev).to(dt).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    d = m.config.get_text_config().hidden_size
    B = json.load(open(a.battery))["classes"]
    U = (m.lm_head.weight.float() * (1.0 + m.model.norm.weight.float()))
    cov = windows(tok, "train", 24, 64, seed=5)
    ctx = windows(tok, "train", a.n, 64, seed=12)
    out = {"model": a.model, "prereg": "PREREG-hspace-lanczos-2026-10-28.md", "classes": {}}
    for cname, (ta, tb) in READOUTS.items():
        if cname not in B:
            continue
        cells = {k: v for k, v in B[cname]["cells"].items() if k.endswith(("lensB", "lensAB"))}
        if not cells:
            continue
        best = max(cells.items(), key=lambda kv: kv[1]["z"])[0]                 # layer of the best lens cell (prereg)
        l = int(best.split("_")[0][1:])
        t0 = time.time()
        with torch.no_grad():
            hs = [m.model(input_ids=w[None].to(dev), output_hidden_states=True, use_cache=False).hidden_states[l + 1][0, SKIP:63].double() for w in cov]
        S = torch.cov(torch.cat(hs).T)
        S = 0.95 * S + 0.05 * torch.trace(S) / d * torch.eye(d, device=dev, dtype=S.dtype)
        ev, V = torch.linalg.eigh(S)
        Sh = ((V * ev.clamp_min(1e-12).sqrt()) @ V.T).float(); Sih = ((V / ev.clamp_min(1e-12).sqrt()) @ V.T).float()
        ia, ib = (tok(x, add_special_tokens=False)["input_ids"][0] for x in (ta, tb))
        c = (U[ia] - U[ib]).to(dev)
        rng = random.Random(21 + l)
        qs = [rng.randrange(SKIP + 8, 64) for _ in range(a.n)]
        op = HbarOperator(m, l, [ctx[i] for i in range(a.n)], qs, c, Sh, dev)
        L = LinearOperator((d, d), matvec=op.matvec, dtype=np.float64)
        vals, vecs = eigsh(L, k=a.k, which="LM", tol=1e-4)
        o = np.argsort(-np.abs(vals)); vals, vecs = vals[o], vecs[:, o]
        g = np.random.default_rng(3)
        fro = float(np.mean([np.sum(op.matvec(g.choice([-1.0, 1.0], d)) ** 2) for _ in range(a.hutch)]))
        # battery class-mean directions at layer l (first phrasing), whitened
        sA, sB, n_it = torch.zeros(d, device=dev), torch.zeros(d, device=dev), 0
        for it in ITEMS.CLASSES[cname]():
            ids = [[tok(it["p"][i][j], add_special_tokens=False)["input_ids"] for j in range(2)] for i in range(2)]
            if len({len(ids[i][j]) for i in range(2) for j in range(2)}) != 1:
                continue
            batch = torch.tensor([ids[0][0], ids[1][0], ids[0][1]], device=dev)
            with torch.no_grad():
                Hh = m.model(input_ids=batch, output_hidden_states=True, use_cache=False).hidden_states[l + 1][:, -1].float()
            sA += Hh[1] - Hh[0]; sB += Hh[2] - Hh[0]; n_it += 1
        At, Bt = (Sih @ (sA / n_it)).cpu().double().numpy(), (Sih @ (sB / n_it)).cpu().double().numpy()
        ah, bh = At / np.linalg.norm(At), Bt / np.linalg.norm(Bt)
        eA, eB = float(np.sum((vecs.T @ ah) ** 2)), float(np.sum((vecs.T @ bh) ** 2))
        R = g.standard_normal((d, 200)); R /= np.linalg.norm(R, axis=0)
        null = np.sum((vecs.T @ R) ** 2, axis=0); q99 = float(np.quantile(null, 0.99))
        full = float(ah @ op.matvec(bh)); rank = float((vecs.T @ ah) @ (vals * (vecs.T @ bh)))
        top_mass = float(np.sum(vals ** 2) / fro) if fro > 0 else float("nan")
        S1 = bool(eA > q99 and eB > q99); S2 = bool(np.sign(rank) == np.sign(full) and abs(rank) >= 0.5 * abs(full)); S3 = bool(top_mass >= 0.5)
        R_ = {"layer": l, "battery_best_lens_cell": best, "readout": [ta, tb], "eigs": vals.tolist(), "fro2_hutch": fro,
              "S1": {"energy_A": eA, "energy_B": eB, "random_q99": q99, "random_mean": float(null.mean()), "pass": S1},
              "S2": {"full": full, "rank_k": rank, "ratio": rank / full if full else None, "pass": S2},
              "S3": {"top_k_mass": top_mass, "pass": S3}, "all_pass": bool(S1 and S2 and S3),
              "matvecs": op.nmv, "minutes": (time.time() - t0) / 60}
        out["classes"][cname] = R_
        print(f"{cname:20s} L{l}: S1 A {eA:.3f} B {eB:.3f} vs random q99 {q99:.3f} -> {S1} | S2 rank-{a.k}/full {rank:+.3g}/{full:+.3g} -> {S2} "
              f"| S3 top-{a.k} mass {top_mass:.2f} -> {S3} | {'H-SPACE' if R_['all_pass'] else '-'} ({R_['minutes']:.1f} min, {op.nmv} matvecs)", flush=True)
        op.close(); del op
        torch.cuda.empty_cache() if dev == "cuda" else None
        json.dump(out, open(RES / f"hspace_lanczos_{a.tag}.json", "w"), indent=1)
    passing = [k for k, v in out["classes"].items() if v["all_pass"]]
    out["VERDICT"] = ("THERE IS AN H-SPACE (J-space construction)" if len(passing) >= 2 and any(k in XOR for k in passing)
                      else "SUPERVISED H-LENS ENTRIES ONLY / NO H-SPACE")
    json.dump(out, open(RES / f"hspace_lanczos_{a.tag}.json", "w"), indent=1)
    print(f"VERDICT {a.tag}: {out['VERDICT']} | passing {passing}", flush=True)


if __name__ == "__main__":
    main()
