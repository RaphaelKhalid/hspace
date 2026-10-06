# H-space: looking for the second-order workspace of a language model

Anthropic's J-lens (arXiv 2607.15495) averages the **Jacobian** of a model's final residual with respect to an earlier layer, and finds a low-dimensional "J-space" that carries verbalisable content. This repo asks the next question up the Taylor series:

> Is there a low-dimensional subspace where the model's **second-order** computation lives, i.e. the directions that matter only *in combination*?

We call that subspace **H-space**, after the Hessian.

**Status (Oct 6 2026): work in progress, honest negatives included.** This is research code plus a paper draft, built in one push on a $17 GPU budget.

## Results so far

### Run 1: the preregistered one-shot on Qwen3.6-27B (`hlens/paper/SCOPE-hspace.md`). Verdict: **FAIL**

| layer | C0 control | C1 split-half ≥ 0.7 | C2 top-25 energy ≥ 0.5 | C3 J-overlap < 0.5 | C4 1σ interaction ≥ 10× | C5 KL ≥ 3× random | C6 dissociation ≥ 2 |
|---|---|---|---|---|---|---|---|
| 16 | **0.997 ✅** | 0.33 ❌ | 0.97 ✅ | 0.15 ✅ | 80× ✅ | 143× ✅ | 0.79 ❌ |
| 28 | | 0.37 ❌ | 0.85 ✅ | 0.23 ✅ | 103× ✅ | 84× ✅ | 0.35 ❌ |
| 40 | | 0.33 ❌ | 0.86 ✅ | 0.21 ✅ | 79× ✅ | 144× ✅ | 1.21 ❌ |
| 52 | | 0.41 ❌ | 0.99 ✅ | 0.29 ✅ | 97× ✅ | 197× ✅ | 1.86 ❌ |

**What this shows.**
- **What holds:** there is a subspace, distinct from J-space, whose directions interact about 100× more than random ones at natural activation scale. Deleting it damages next-token predictions 84–197× more than deleting random subspaces; at layers 40 and 52 that is *more* than deleting J-space.
- **What fails:** the raw estimator is not reproducible (C1), and its deletion is not interaction-specific (C6). A planted-gate positive control (C0) shows the instrument itself works, so these are findings, not a broken instrument.

### Why C1 and C6 failed: diagnosis, written up before any follow-up data
- **C1 was measured at k = 25.** But the participation ratio is 1.3–8.5, so only about 2–8 directions carry signal and the rest of the top 25 is noise.
- **C6 is a ratio whose denominator is ≈ 0.** Its 95% CIs span [−1, 10] and [0.4, 18].
- **The run-1 operator includes cross-position blocks.**

### Run 2 and v3 (in progress)
- **Run 2 (exploratory).** Unbiased cross-moment estimator (two HVPs with independent sign vectors), spatial-sign (robust) estimator, and a rectangle-averaged Hessian.
- **v3 preregistration** (frozen before any run-2 data): `hlens/paper/SCOPE-hspace-v3.md`. It adds C1′ (reliability at the noise-floor dimension) and C6′ (ablate H-space and measure the 2×2 interaction itself).

## Theory (`hlens/paper/theory-v2.md`, red-teamed by 48 adversarial agents)
- **Lemma 1, the second-order adjoint (standard):** the exact block decomposition of the residual Hessian, verified to ≤ 1.5e-6 on Qwen3.5-0.8B.
- **Lemma 3, the mixed-difference identity (standard):** a finite 2×2 interaction equals the Hessian averaged over the rectangle the two moves span.
  - On real 0.8B activations at feature scale, the Hessian at the data point predicts interactions at r = −0.09. The full rectangle average predicts them at r = 0.98 (a check of the code and quadrature).
- **Proposition 0:** three energy operators, M^row ⪰ M^diag ⪰ M^lens, and what each estimator converges to (§6A).
- **Construction 6.1, the J-adjoint local H-lens:** truncate to the next k blocks and use the J-lens adjoint of record.
  - On 0.8B, k = 1 (6–9% of the exact cost) captures 83–84% of the exact H-space energy, and k = 4 captures 90–94%.
  - This is the H-lens at roughly J-lens cost.
- **Engineering:** `bf16w.BFLinear` stores weights in bf16 and computes in exact fp32, differentiable to any order. Its HVP matches all-fp32 to 7e-7, which puts exact Hessian-vector products of a 27B model on one 96 GB GPU.

## Goal 3: white-box signals in AI control (in progress)
- Qwen3.6-27B reads 1,500 BashArena attack/honest agent trajectories, after **sanitizing out label leaks**: attacker prose confessions and formatting tells.
- We then test training-free multi-token J-lens matched filters, label-free pre-emption rules and Neyman–Pearson fusion with a black-box monitor, at fixed audit budgets (`hlens/ctrl_score_v2.py`, spec in `hlens/paper/ctrl-score-v2-spec.md`).

## Layout
```
hlens/                 code (run from this folder)
  bf16w.py             bf16-stored / fp32-computed linears (exact HVPs at 27B)
  hspace.py            run 1 (S2-S8); hs_s7.py = S7 re-run verbatim on 455 windows
  hspace2.py           run 2 (cross-moment / spatial-sign / rectangle estimators)
  hs_c6prime.py        v3 C6' interaction-ablation test
  hs_anatomy.py ...    0.8B development checks (anatomy, rectangle, integrated, local H-lens)
  ctrl_read.py         goal 3: sanitized surrogate white-box read of agent trajectories
  ctrl_score_v2.py     goal 3: offline scoring
  paper/               paper draft, theory, preregistrations (frozen + hashed), verdict scripts
  pod/                 one-shot pod scripts (watchdog, job chains, HF result relay)
  dash_hs/             OLED live monitor
  tests/               exactness tests (bf16w, SwiGLU closed form, HVP vs finite differences)
flagship/              lens registry + vendored anthropics/jacobian-lens (Apache-2.0, see its LICENSE/PROVENANCE)
results/               small result JSONs + runlog.md (every logged line, freeze hashes, deviations)
tools/sync.sh          sync from the research workspace (with a secret scan)
```

## Integrity notes
- **Preregistrations are hashed in `results/runlog.md` before the data they govern exist.** Search for `[freeze]`.
- **Every deviation is logged there**, for example: S7 re-run on the 455 windows that exist (the frozen loop would have crashed); control-read sanitization before any control data existed.
- **Raw tensors and sample dumps** live in a private Hugging Face dataset and are not in this repo.
- **The control dataset** is `adityaasinha28/control_arena_bash`. It is not redistributed here.

## License
MIT for this repo's code (`LICENSE`). `flagship/third_party/jlens` is Apache-2.0 (anthropics/jacobian-lens).
