# H-space: looking for the second-order workspace of a language model

Anthropic's J-lens (arXiv 2607.15495) averages the **Jacobian** of a model's final residual with respect to an earlier layer, and finds a low-dimensional "J-space" that carries verbalisable content. This repo asks the next question up the Taylor series:

> Is there a low-dimensional subspace where the model's **second-order** computation lives, i.e. the directions that matter only *in combination*?

I call that subspace **H-space**, after the Hessian.

**Status (Oct 6 2026): run 1, run 2 and the follow-up tests through v9 are complete (11 tests in the sequence, counting run 1 and run 2; v6 was designed but not run), including a preregistered replication (v8) and a preregistered interaction test (v9) of a punctuation-conditional H-subspace, both passed. Negatives, one retraction and all later corrections (`results/CORRECTIONS.md`) are included. Decoding, ablation/steering, class-specificity and a second model are untested.** This is research code plus a paper draft, built on about $26 of RunPod GPU time in total (Oct 5–8 account billing: about $17 through v9, about $9 for v10–v11).

## Status (Oct 8 2026)

- **What is established:** a reproducible second-order structure in Qwen3.6-27B.
  - v10 M3: the punctuation H-space has 7.3x (L16) and 3.4x (L40) more interaction than 16 matched twins.
  - R passes in v10 and v11.
  - The all-position version replicates across runs (v11a overlap 0.94) and still beats every one of 16 Sigma-orbit twins on fresh text at random positions, by 2.2x (L16) and 1.7x (L40). It also beats them after J-space is removed (v11).
- **Not established ("I found H-space" in the J-space sense):**
  - v10 and v11 MUST tiers not passed.
  - v11's all-position effect is below the preregistered 2x bar at L40.
  - "Used" (M4/M5) is still untestable. The tasks do not route through any single suffix position at L16 or L40, so only 1 of 3 contexts localized.
  - J-space directions carry far more interaction.
- **Details:** `results/runlog_laptop.md`, `results/v10/`, `results/v11/`, `results/hs_v11a.json`.
## Results so far

### Run 1: the preregistered one-shot on Qwen3.6-27B (`hlens/paper/SCOPE-hspace.md`). Verdict: **FAIL**

| layer | C0 control | C1 split-half ≥ 0.7 | C2 top-25 energy ≥ 0.5 | C3 J-overlap < 0.5 | C4 1σ interaction ≥ 10× | C5 KL ≥ 3× random | C6 dissociation ≥ 2 |
|---|---|---|---|---|---|---|---|
| 16 | **0.997 ✅** | 0.33 ❌ | 0.97 ✅ | 0.15 ✅ | 80× ✅ | 143× ✅ | 0.79 ❌ |
| 28 | | 0.37 ❌ | 0.85 ✅ | 0.23 ✅ | 103× ✅ | 84× ✅ | 0.35 ❌ |
| 40 | | 0.33 ❌ | 0.86 ✅ | 0.21 ✅ | 79× ✅ | 144× ✅ | 1.21 ❌ |
| 52 | | 0.41 ❌ | 0.99 ✅ | 0.29 ✅ | 97× ✅ | 197× ✅ | 1.86 ❌ |

**What this shows.**
- **What holds, with caveats:** the top-25 has low overlap with my whitened J-space (0.15–0.29), though two halves of the same estimate overlap only 0.33–0.41, so this does not show it is distinct. Its directions interact 79–103× more than random whitened pairs at 1σ, but C4 had no variance-matched control (at 0.8B the pair-norm product alone gives 23–29×; theory-v2 §6). Deleting it damages next-token predictions 84–197× more than deleting random subspaces; at layers 40 and 52 that is *more* than deleting J-space.
- **What fails:** the raw estimator is not reproducible (C1), and its deletion is not interaction-specific (C6). A planted-gate positive control (C0) shows the instrument itself works, so these are findings, not a broken instrument.

### Why C1 and C6 failed: diagnosis, written up before any follow-up data
- **C1 was measured at k = 25.** But the participation ratio is 1.3–8.5, so only about 2–8 directions carry signal and the rest of the top 25 is noise.
- **C6 is a ratio whose denominator is ≈ 0.** Its 95% CIs span [−1, 10] and [0.4, 18].
- **The run-1 operator includes cross-position blocks.**

### Run 2 and the v3 preregistration: verdict **not found (v3)**
- **The preregistered primary (`loc_x`, an unbiased cross-moment estimator) fails C1′ at 4/4 layers.** Split-half at k = 25 is 0.13–0.29, and at L16 no eigenvalue is above its noise floor.
- **Why (C1 and C1′):** heavy tails.
  - At L16 the heaviest 0.1% of (position, probe) rows carry **97.5%** of the raw Hessian energy.
  - Digit tokens are 7% of rows but 88% of that energy.
  - Dropping the heaviest 5% of rows gives split-half 0.95 at k = 5.
- **The Σ prior (theory §6B, Prop. 8).** My Σ^{1/2}-whitened curvature estimators return reproducible PCA-axis directions whenever curvature is a function of Σ (more generally, whenever its law is invariant under PCA sign flips). A similar prior plausibly affects J-space computed in whitened coordinates, as mine is; Anthropic's released J-lens does not whiten. So the robust estimators' 0.97–0.99 reproducibility is **not** by itself evidence of H-specific structure.

### Sign-flip randomization tests (exact null for Σ-commuting curvature; theory §6B, Cor. 8.1)
| test (rule frozen before computing) | layers | result |
|---|---|---|
| flip, massive span deflated by projection | 16, 28 | **FAIL**. λ1 beats all 200 flips everywhere, but split-half has no power under a non-axis-aligned projection |
| **flip v5**, axis-aligned drop (commutes with the flips) | **40, 52 (held out)** | **PASS**. λ1 3–6.6× the flip maximum; split@3 0.82–0.94 vs a null of 0.01–0.02; planted control detected |
| v5 vs matched-flatness basis-misspecification null | 40, 52 | **split-half robust**. Simulated split@3 0.01–0.07 vs data 0.82–0.94; simulated λ1 still beats the flip maximum (1.0–1.8× vs 3–6.6× for the data). A steep f = Σ misspecification *can* fake v5, but the data's spectrum (participation ratio 58–91 vs 2.6–4.0) rules that regime out. The calibration matches the participation ratio only. |
| **causal Σ-orbit twin ablation** of the v5 patterns (needs REAL at both layers) | 40, 52 | **FAIL**. At L40 the v5 patterns are not more interaction-specific than 8 exactly variance-matched twins: S = 0.014 vs 0.002, rank 2/9, CI includes 0 |

**Update (Oct 6, ~14:00 UTC): v9 PASS. Natural-scale pairwise interaction concentrates in the replicated H-subspace.**
- On 89 fresh wikitext-validation windows (a split never used at 27B; the 0.8B development runs used it), natural-scale 1σ moves along the 5-d punctuation-conditional H-subspace produce more pairwise interaction in next-16-token log-likelihood than 8 exactly variance- and PCA-profile-matched Σ-orbit twins:
  - **5.8× at L16** (e^1.766 = 5.85), log CI [1.58, 1.95];
  - **2.5× at L40**, log CI [0.78, 1.10].
- This holds in the trunk (not just the softmax) and per unit of single-move (main) effect, and EX ranks 1st of 9 at both layers. In absolute terms the effects are small (mean |I| 2.2e-3 / 1.0e-3 nats along EX pairs, against 3.3e-4 / 3.0e-4 for the twins).
- The rule was frozen before the pod, after an adversarial review and a $0 null/plant validation.
- **Combined with v8**, this gives a reproducible, cross-run-replicated H-subspace that beats global-covariance controls and predicts where natural-scale, same-position interaction lives. That is predictive validity in a narrow sense: it is a perturbation test, not evidence that the model relies on the subspace.
- **Caveats:** punctuation was chosen after a descriptive look at run-2 data, and v8/v9 are the 10th and 11th tests in the sequence; the twins match the *global* covariance, not the punctuation-row covariance.
- **Still untested:** decoding, ablation/steering, class-specificity, punctuation-conditional covariance and other models.
- Details: paper §4d and `results/hs_v9_full.json`.

**Update (Oct 6, ~12:40 UTC): two follow-up tests, both frozen before the data existed.**
- **v8: punctuation-conditional H-structure REPLICATED on fresh 27B data.**
  - At L16 and L40 the curvature at punctuation positions has a low-dimensional, non-Σ-commuting subspace. Split-half is 0.82–0.87, against a flip null of 0.02.
  - It reproduces across independent runs: top-5 overlap with run 2 is 0.865 / 0.777 after excluding the 15 windows v8 shared with run 2 (`results/CORRECTIONS.md`), against 0.006 by chance.
  - No causal test yet, so this is a replicated candidate, not proven function.
- **First-order screen (goal 2).**
  - At 27B a J-lens-cost gradient screen predicts where H-lens curvature lives: Spearman 0.82–0.85 over rows.
  - For the raw energy estimator, sampling windows by the screen cuts the HVPs needed by **80× / 10.6×** at equal Frobenius variance (computed retrospectively on the same 256 windows, no CI; not computed for the spatial-sign estimator behind EX).
  - End to end the saving is much smaller: if an HVP costs c screening passes, it is c / (1 + c/G) with G = 80 / 10.6, about 1.7–3.8× for c = 2–4, and never more than c.

**Bottom line (Oct 6, before the follow-ups).**
- At two held-out layers of Qwen3.6-27B, the H-lens contains a **reproducible second-order structure that is not a function of the activation covariance**. It is also not J-space (overlap at chance) and not the massive-activation geometry (largely by construction: those axes are masked), and its split-half reproducibility survives a flatness-matched (participation-ratio-matched) basis-misspecification null (its λ1 statistic does not).
- But under the frozen causal test it is **not interaction-specific**.
- So I have **not** found an H-space "as provably as J-space". What exists is a reproducible non-Σ-commuting curvature structure without demonstrated function.
- Every step, including one retraction (`results/CORRECTIONS.md`), is hashed in `results/runlog_pod.md` before its data.
- **C6′ (descriptive):** S is mostly dose. At L16 it tracks the raw variance each ablation removes (r = 0.999), and the robust subspace's interaction-per-damage ratio equals J25's. At matched damage (L28, L40) the robust subspace is about 1.5–1.7× more interaction-selective than J25, but it is mostly the Σ prior (correction in `results/CORRECTIONS.md`).
- **Compute:** one RTX PRO 6000 pod for 7.65 h, plus a 38-minute v8 pod and a 32-minute v9 pod: about $15.1 in total. All 78 result files were verified by sha256 on a private HF dataset before the pod was terminated.

## Theory (`hlens/paper/theory-v2.md`, red-teamed by 48 adversarial agents)
- **Lemma 1, the second-order adjoint (standard):** the exact block decomposition of the residual Hessian, verified to ≤ 1.5e-6 on Qwen3.5-0.8B.
- **Lemma 3, the mixed-difference identity (standard):** a finite 2×2 interaction equals the Hessian averaged over the rectangle the two moves span.
  - On real 0.8B activations at feature scale, the Hessian at the data point predicts interactions at r = −0.09. The full rectangle average predicts them at r = 0.98 (a check of the code and quadrature).
- **Proposition 0:** three energy operators, M^row ⪰ M^diag ⪰ M^lens, and what each estimator converges to (§6A).
- **Construction 6.1, the J-adjoint local H-lens:** differentiate only the next k blocks and use the J-lens adjoint of record. It is exact only at k = L*−ℓ; for smaller k it is a heuristic.
  - On 0.8B (one seed, in-sample), k = 1 differentiates 1/17–1/11 of the downstream blocks (roughly 13–26% of the exact estimator's cost, an estimate), and its top-25 captures 83% of the energy the exact top-25 captures; k = 4 captures 90–94%.
  - The global J-lens Gram top-25, which needs no HVPs, already captures 77–79%. So k = 1 is not clearly better than J-space; from k ≈ 4 it is (theory-v2 §6).
- **Engineering:** `bf16w.BFLinear` stores weights in bf16 and computes in exact fp32, differentiable to any order. Its HVP matches all-fp32 to 7e-7, which puts exact Hessian-vector products of a 27B model on one 96 GB GPU.

## Goal 3: white-box signals in AI control: mechanical prereg verdict UNCALIBRATED (descriptive only); NEGATIVE on the statistics
- **Verdict.** A calibration gate failed, so the frozen rule's mechanical verdict is "UNCALIBRATED: descriptive only". On the statistics alone it reads "calibrated NEGATIVE" (bootstrap upper bound of net < +5; `results/ctrl_score_v2_full.log`).
- Qwen3.6-27B reads 1,500 BashArena attack/honest agent trajectories written by other models (Claude Sonnet 4, Gemini 2.5 Pro, GPT-5), after **sanitizing out label leaks**: attacker prose confessions and formatting tells. Qwen is a surrogate reader, not the agent.
- I then test training-free multi-token J-lens matched filters, label-free pre-emption rules and Neyman–Pearson fusion with a black-box monitor, at fixed audit budgets (`hlens/ctrl_score_v2.py`, spec in `hlens/paper/ctrl-score-v2-spec.md`).
- **Result.** The training-free J-lens matched filter (W1) carries real signal: AUROC 0.71, against 0.48 for the rotated-lens control. In the OR slot at a 2% audit budget it pre-empts **net −6** attacks (95% CI [−18, +1]). It does not move attacks from "execute" to "audit" at any preregistered budget, and on the pre-attack prefix its AUROC is 0.40 (no early warning).
- **A supervised activation probe (W3)** gains +33 at a 5% budget, but collapses to chance when tested on an attacker model it wasn't trained on (`results/ctrl_score_v2_full.json`, paper §5).

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
