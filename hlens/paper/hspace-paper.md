# H-space: the second-order workspace of a language model

*Draft, Oct 5–6 2026. One-shot preregistered test on Qwen3.6-27B (prereg `SCOPE-hspace.md`, frozen and hashed before the pod existed), with theory and 0.8B development work. The 27B numbers are filled in from `out/pod/hspace_full.json`.*

## Abstract (to be finalised after the 27B verdict)
- **Background.** Anthropic's J-lens averages the Jacobian of a model's final residual with respect to an earlier layer. It revealed a low-dimensional "J-space" that carries verbalisable content.
- **What I ask.** I ask the next question up the Taylor series. Is there a low-dimensional subspace where the model's *second-order* computation lives, the place where two directions matter only together? I call it **H-space**.
- **Theory.** I give:
  - an exact block decomposition of the residual Hessian (the second-order adjoint), verified to 1e-6 on a real model;
  - closed forms showing every SwiGLU neuron and every gated-attention head contributes rank-2 "AND atoms";
  - a rectangle identity: any finite 2×2 interaction effect equals the Hessian averaged over the rectangle the two moves span;
  - complexity results: the top-k H-space costs O(k) Hessian-vector products independent of the width d, and is cheaper to find than fitting the J-lens itself.
- **Engineering.** A bf16-stored, fp32-computed linear layer that can be differentiated twice. Exact HVPs of a 27B model then fit on one 96 GB GPU.
- **Test.** I then test six preregistered criteria for H-space on Qwen3.6-27B: reproducibility, low rank, distinctness from J-space, natural-scale interactions, causal necessity, and a double dissociation on interaction-dependent tokens. A planted bilinear gate serves as the instrument-validity control.
- **Result.**
  - **Run 1 (preregistered): FAIL.** C1 (reproducibility) and C6 (dissociation) fail.
  - **Run 2 / v3 (preregistered): not found.** The unbiased estimator is pure heavy-tail noise at 27B: 0.1% of rows carry 97.5% of the energy.
  - **The Σ prior.** I prove that every Σ^{1/2}-whitened lens, the J-lens included, returns reproducible PCA-ordered directions whenever curvature is Σ-commuting. I give an exact sign-flip randomization test for this.
  - **v5 sign-flip test (frozen rule, held-out layers): PASS.** It finds a reproducible non-Σ-commuting second-order structure that is not J-space or massive-activation geometry, and is robust to basis-estimation error.
  - **Frozen causal test: FAIL.** Deleting that structure is not more interaction-specific than deleting exact variance-matched twins.
  - **Follow-ups (v8, v9; frozen rules, fresh data).** A punctuation-conditional H-subspace replicates across runs (top-5 overlap 0.865 / 0.777). On fresh text, its directions produce 2.5–6× more natural-scale pairwise interaction than exactly variance-matched twins. That establishes predictive second-order validity; decoding and steering are untested.
  - **Goal 3.** A training-free multi-token J-lens add-on to a control monitor is net-negative at fixed audit budgets.

## 1. Introduction
J-space (arXiv 2607.15495) is a first-order object. The J-lens transports a residual direction linearly to the output and decodes it as tokens.

Much of what a transformer computes is *not* first-order:
- **gating:** SwiGLU multiplies a gate read by an up read;
- **attention:** it multiplies "does this key match my query" by "what value does it carry";
- **conditional behaviour:** a backdoor or a scheming policy is literally an AND.

The Hessian is the natural object for all of these. Earlier pilots in this project (v1, v2) found two problems:
- local Hessians at data points mispredict interactions at effective scales;
- saturated trained triggers are invisible to every local derivative.

This paper makes three moves:
1. **Exact decompositions** of where curvature comes from (anatomy).
2. **A coordinate-free definition** of H-space as the dominant subspace of per-position Hessian row energy, in whitened units.
3. **A one-shot preregistered test at 27B** whose criteria were fixed before any 27B data existed, with a positive control that makes a negative interpretable.

## 2. Theory (full statements and proofs: `theory-v2.md`, adversarially reviewed by 48 agents; the summary below is pre-review and is superseded by theory-v2 wherever they differ)

**Corrections adopted from the review** (theory-v2 changelog):
- **Standard results are now labelled as such and cited:**
  - Lemma 1, the second-order adjoint (Christianson 1992; Pearlmutter 1994; Griewank & Walther 2008);
  - Lemma 3, the mixed-difference identity;
  - Lemma 7, Hutchinson;
  - G1, the Neyman–Pearson lemma.
- **"Exactly three atom types" is withdrawn.** Only per-site rank and inertia are basis-invariant.
- **The Pearce reduction is 2Q,** in post-norm coordinates only.
- **The attention later-query atom** is corrected for k_norm, plus a softmax threshold term.
- **Energy operators.** Three are defined: M^row ⪰ M^diag ⪰ M^lens (Proposition 0). Run 1 estimated M^row; run 2's cross-moment estimates M^diag.
- **G2's workspace gain** is ≤ 204.8ρ² (spike test) and ≤ 14.3ρ (mean test), where ρ is the signal-capture fraction. The earlier "~200×" claim is withdrawn.
- **Anatomy numbers are corrected:** the nearest third of blocks generates 36–62%, and the off-position share is 19–38%.

**New verified constructions:**
- **The local J-adjoint H-lens** (Construction 6.1; `hs_local.py`, 0.8B). Truncate to the k blocks after the source layer and use the J-lens adjoint of record.
  - With **k = 1 (6–9% of the exact cost) it captures 83–84% of the exact top-25 H-energy**, with subspace overlap 0.56–0.60.
  - With k = 4 it captures 90–94%; with k = 8, 93–98%.
  - This is the H-lens at roughly J-lens cost (goal 2).
- **The rectangle check** (`hs_rectangle.py`, 0.8B, layer 12, feature-scale moves). The local Hessian predicts finite interactions at r = −0.09. The full-rectangle average (5×5 Gauss–Legendre) predicts them at r = 0.98.
  - This is a check of the code and the quadrature, not of the lemma.
  - It locates v1's effective-scale failure in *where* the Hessian is evaluated.
- **Thm 1, block decomposition.** ∇²F = Σ_s J_{s←ℓ}ᵀ ∇²[g_{s+1}·φ_s] J_{s←ℓ}, verified to ≤ 1.5e-6 at 5 layers of Qwen3.5-0.8B.
- **Thm 2, anatomy.** Local curvature is a finite sum of three kinds of atom:
  - AND atoms (products of two reads);
  - threshold atoms (σ'' of one read);
  - normalisation atoms.

  It has closed forms for SwiGLU (verified 4e-16 on a toy and 1e-6 on real blocks) and for gated softmax attention (key×value at later queries, gate×value, query×query).
- **Thm 3, rectangle identity.** f(h) − f(h−a) − f(h−b) + f(h−a−b) = ∫∫ aᵀ∇²f(h−a−b+sa+tb) b ds dt.
- **Prop 4.** Local, integrated and potential H-lenses share one atom geometry and differ only in per-atom coefficients. The potential lens sees dormant gates.
- **Cor 5.** Under the J-lens's own decoupling assumption, the averaged H-lens factorises into atoms decoded through the J-lens: "IF [read a] AND [read b] THEN [write]".
- **Thm 6, complexity.** Exact H-space costs (k + oversampling) × B × T HVP-token passes, independent of d. Hutchinson over positions (Lemma 7) gives the per-position diagonal in one pass. Streaming through Thm 1 needs one block resident at a time.
- **§7, control.**
  - NP-optimal fusion of white-box and black-box scores (G1).
  - A spiked-covariance (BBP) bound on how many agent steps a label-free hidden-goal detector needs: K > p/(fθ²)². This makes the low-dimensional workspace the natural place to monitor (G2).
  - Matched filters for known threats (G3).

## 3. Development evidence on Qwen3.5-0.8B (laptop, $0; not part of the verdict)

**3.1 Anatomy** (`hs_anatomy.py`, 48 samples per layer). Shares of the per-position H-lens:

| layer | MLP | attention | DeltaNet | MLP in nearest third | diagonal share |
|---|---|---|---|---|---|
| 4 | 0.58 | 0.14 | 0.29 | 0.35 | 0.62 |
| 8 | 0.52 | 0.20 | 0.27 | 0.21 | 0.67 |
| 12 | 0.52 | 0.16 | 0.32 | 0.36 | 0.68 |
| 16 | 0.51 | 0.24 | 0.26 | 0.34 | 0.77 |
| 20 | 0.50 | 0.19 | 0.31 | 0.24 | 0.75 |

**3.2 Where to evaluate the Hessian** (`hs_integrated.py`, layer 12; 32 natural moves × 4 output covectors). Each move removes half of a token's whitened deviation from the mean, scaled by λ. Correlation of predicted with true finite interactions:

| λ | median \|I\| | local, per context | integrated (rectangle diagonal), per context | local lens (corpus-avg) | integrated lens (corpus-avg) |
|---|---|---|---|---|---|
| 0.1 | 0.002 | 1.00 | 1.00 | 0.26 | 0.31 |
| 0.3 | 0.028 | 0.78 | 0.94 | 0.27 | 0.37 |
| 1.0 | 0.44 | 0.29 | 0.57 | 0.25 | 0.16 |

- **Reading 1.** The rectangle theorem fixes much of v1's effective-scale failure *per context* (r 0.29 → 0.57).
- **Reading 2.** A corpus-averaged *signed* H-lens transfers poorly across contexts (r ≈ 0.3). Second-order structure is context-specific in sign and coefficient. That motivates defining H-space by interaction *energy* (which directions carry interactions), not by an averaged signed tensor.

**3.3 The full pipeline at 0.8B** (`hspace.py small`, layers 6/12/18; the numbers are in SCOPE §A):
- low rank: top-25 energy ~0.6, participation ratio 19–28 of 1024;
- reproducible: split-half 0.69–0.74;
- 40–69× natural-scale interaction ratio;
- deleting H25 costs 60–170× the KL of random-25 and ~1.7× that of J25;
- J overlap ≈ 0.53;
- no interaction-specific damage (dissociation ratio ≈ 1.3, 20/20 tokens).

## 4. The 27B one-shot (prereg `SCOPE-hspace.md`): verdict FAIL

**Setup.**
- Qwen3.6-27B on one RTX PRO 6000 (96 GB), community pod.
- Weights stored in bf16, with exact fp32 compute via `bf16w.BFLinear` (HVP equal to all-fp32 within 7e-7). The peak HVP graph is 90.0 GB at layer 16.
- 256 probes × 4 windows per layer, at layers 16 / 28 / 40 / 52.
- The prereg was frozen and hashed before the pod existed.

**Deviations, all logged with hashes in `runlog.md`; none changes a threshold or the analysis code:**
- a pip flag;
- a wikitext path bug, fixed before any data;
- S7 re-run verbatim on 455 windows (only 455 exist; the original loop would have crashed at window 456);
- the watchdog was moved after the user extended the budget.

| layer | C0 control | C1 split-half ≥ 0.7 | C2 top-25 energy ≥ 0.5 | C3 J-overlap < 0.5 | C4 1σ interaction ≥ 10× | C5 KL ≥ 3× random | C6 dissociation ≥ 2 | PR |
|---|---|---|---|---|---|---|---|---|
| 16 | **0.997 ✅** (at L40) | 0.33 ❌ | 0.97 ✅ | 0.15 ✅ | 80× ✅ | 143× ✅ | 0.79 (95% CI [−1.0, 10.1]) ❌ | 2.3 |
| 28 | | 0.37 ❌ | 0.85 ✅ | 0.23 ✅ | 103× ✅ | 84× ✅ | 0.35 ❌ | 4.8 |
| 40 | | 0.33 ❌ | 0.86 ✅ | 0.21 ✅ | 79× ✅ | 144× ✅ | 1.21 ❌ | 8.5 |
| 52 | | 0.41 ❌ | 0.99 ✅ | 0.29 ✅ | 97× ✅ | 197× ✅ | 1.86 (CI [0.43, 18.4]) ❌ | 1.3 |

**Verdict (mechanical, `paper/verdict.py`): FAIL.** C1 fails at 4/4 layers and C6 at 4/4.

**What the numbers do show (descriptive, not a claim):**
- There is a subspace, distinct from J-space (overlap 0.15–0.29), whose top directions interact 79–103× more than random pairs at 1σ natural scale.
- Deleting its top 25 directions changes next-token distributions 84–197× more than deleting random 25-dimensional subspaces. At layers 40 and 52 that is *more* than deleting J25 (0.34 vs 0.30, and 0.56 vs 0.50 KL).
- But the raw estimator's top-25 is not reproducible, and its deletion is not interaction-specific.

**Diagnosis (exploratory).**
- **Heavy tails.** The spectrum is heavy-tailed: the participation ratio is 1.3–8.5, and each top eigenvalue is 2–13× the next. The top direction puts 42% / 37% / 17% / 14% of its mass on the massive-activation coordinates (dim 3994, mean ≈ 50) at L16 / L28 / L40 / L52 (`hs_diag_top.py`).
- **The run-1 operator is the per-position *row* energy,** so it includes cross-position blocks (theory-v2 §6A).
- **C1 at k = 25 cannot exceed roughly n_signal/25 when PR ≈ 2–8.** The criterion conflates "unstable" with "smaller than 25".
- **C6's ratio statistic is unstable,** because its denominator ≈ 0 (CIs span 10–20× the estimate). It also tests damage, not the interaction itself.
- These lead to the v3 preregistration (C1′ at the noise-floor dimension; C6′ interaction-ablation via Lemma 3). It was frozen **before** run-2 data existed.

## 4b. Run 2, the v3 verdict, and the sign-flip tests (Oct 6)

**Run 2.**
- **Estimators.** Four estimators of the per-position H-lens energy at layers 16, 28, 40 and 52:
  - raw;
  - cross-moment (`x`), unbiased, using two independent sign vectors;
  - spatial sign (`norm`);
  - spatial-sign cross-moment (`xnorm`).
- **Budget.** 256 probes × 4 windows per layer, on the same pod.
- **Samples.** 64 rows per probe were kept for offline tests.

**v3 preregistered verdict (`SCOPE-hspace-v3.md`): H-space NOT found (v3).**
- The primary estimator `loc_x` fails C1′ at all four layers:
  - split-half at k = 25 is 0.13 / 0.15 / 0.17 / 0.29;
  - at L16, no eigenvalue is above the noise floor (k_floor = 1).
- The verdict needed C1′ at ≥ 2 layers, so it was fixed before C6′ ran. C6′ was then run as a descriptive test.

**Why C1 and C1′ fail: heavy tails (`hs_dump_analysis.py`, theory §6B Obs. 8.2).**
- At L16, the heaviest 0.1% of (position, probe) rows carry 97.5% of the raw second moment.
- Digit tokens are 7.4% of rows but 88% of that energy.
- Both the raw and the unbiased cross-moment estimators are controlled by about 10 rows. Dropping the heaviest 5% of rows gives raw split-half 0.95 at k = 5.

**The Σ prior (theory §6B Prop. 8).**
- Any whitened lens returns PCA-ordered directions whenever the Hessian law is invariant under sign flips along PCA axes. Examples: H = cI, any f(Σ), random GOE-like curvature.
- The reproducible spatial-sign estimators reproduce at 0.92–0.99, but their top-25 overlaps PCA-25 by 0.78 (`loc_norm` at L16; 0.78–0.83 for `loc_norm` and `loc_xnorm` across the four layers, against 0.005 at random; `hs_pca_overlap.py`, `results/hs_pca_overlap_run2.json`). A reweighted-PCA axis set (Csur25) recovers 0.78–0.83 of `loc_norm`.
- **Reproducibility alone is therefore not evidence of H-specific structure.** I retracted one attempted fix, the relative-curvature lens (`results/CORRECTIONS.md`).

**Exact test: per-probe PCA sign-flip randomization (theory §6B Cor. 8.1, `hs_flip.py`).** Under the null "the curvature is Σ-commuting", flipping the PCA-coordinate signs of every row of a probe leaves the law of the data unchanged, so 200 flips give an exact null.

| test | rule frozen before computation | layers | outcome |
|---|---|---|---|
| flip, variant (b): massive/radial span deflated by projection | yes | 16, 28 | **FAIL**. λ1 beats all 200 flips for every estimator (1.4–2.1× the flip maximum), but split-half has no power: a non-axis-aligned projection makes the null itself reproduce (null split3 max 0.95–0.98) |
| flip v5, variant (d): axis-aligned drop of PCA 1–5 + massive-span axes (commutes with flips) | yes, before any L40/L52 flip computation | **40, 52 (held out)** | **PASS** |

**v5 details (spatial sign `norm` / spatial-sign cross-moment `xnorm`):**

| layer | λ1 (flip max) | split k = 1/3/5/10 (flip max at k3) | planted control |
|---|---|---|---|
| 40 | 5.12 (1.56) / 3.47 (1.16) | 0.87/0.90/0.85/0.74 and 0.87/0.82/0.77/0.64 (0.01–0.02) | detected, recovered 0.59 / 0.81 |
| 52 | 12.5 (1.90) / 8.54 (1.66) | 0.95/0.94/0.93/0.85 and 0.92/0.88/0.86/0.74 (0.02) | detected; recovery 0.01 / 0.24, because the real structure dominates |

**What the v5 structure is not:**
- J-space: its top-10 patterns sit 0.02 inside J25, which is chance level.
- The massive/radial span: 0.002 of their mass.
- The top PCs: 0 in PCA-25 and 0.07–0.11 in PCA-100.
- An axis-aligned artifact: its participation ratio in PCA coordinates is 59–204, so it is diffuse.

**C6′ (preregistered, run after the v3 verdict was already fixed, so descriptive).** S = r_M − r_I on the first 228 eval windows:

| layer | loc_x | loc_norm | run-1 H25 | J25 | random |
|---|---|---|---|---|---|
| 16 | 0.17 [0.07, 0.28] | **0.52** [0.32, 0.71] | 0.36 | 0.28 | 0.02–0.03 |
| 28 | 0.33 [0.13, 0.55] | **0.59** [0.40, 0.76] | 0.45 | 0.27 | 0.01–0.02 |
| 40 | 0.12 | 0.20 | 0.20 | 0.12 | 0.01–0.02 |
| 52 | 0.03 | 0.15 | 0.05 | 0.01 | 0.01 |

- `loc_x` fails C6′ at every layer: the CI of S − max(J25, rand) includes 0.
- **Correction (Oct 6, after the v6 design review): most of S is dose.** Across the four learned subspaces at L16, S is almost linear in the raw variance each ablation removes (r = 0.999):
  - `loc_norm` removes 22% of raw variance; J25 removes 13%.
  - Their interaction-per-main-effect damage ratios are equal: φ = (1 − r_I)/(1 − r_M) = 4.7 vs 4.6.
  - So the larger S of `loc_norm` at L16 is dose, not specificity.
- **At L28 and L40, main-effect damage is matched** (1 − r_M = 0.129 vs 0.126, and 0.051 vs 0.051). There `loc_norm` removes interactions about 1.5–1.7× more selectively than J25 (φ 5.5 / 5.0 vs 3.2 / 3.3). This is descriptive, with no CIs, and it does not separate H from the Σ prior, which `loc_norm` largely is (Prop. 8).
- An earlier draft of this paragraph said "about twice the specificity of J25". That was wrong; see `results/CORRECTIONS.md`.

**Robustness to estimation error in the PCA basis (`hs_flip5_misspec*.py`).**
- **The threat.** Rows that are Σ-commuting with respect to the *true* Σ look non-commuting in the estimated basis. I used run-1's independent Σ estimate as a stand-in for the truth.
- **A steep f = Σ null can fake the v5 pass.** Simulated λ1 is 122–221 and split@3 is 0.88–0.93.
- **The data rule that regime out.** That null has spectrum PR 3–4; the data's is 58–91.
- **Matched null.** I calibrated f = Σ^a so the simulated spectrum flatness matches the data at each layer. The simulated split@3 is 0.01–0.07, against 0.90–0.94 in the data. **v5 survives** (rule frozen before running).

**Causal test (frozen rule, `hs_c6twin.py --v5`): FAIL.**
- **Design.** Ablate the top-5 v5 patterns versus 8 Σ-orbit twins, which match variance and PCA profile exactly. I used the 227 held-out eval windows and ablated positions ≥ 16 only.

| layer | S(EX) | twin mean S | rank of EX among 9 | CI of ΔS | KL(EX) vs twin range | REAL? |
|---|---|---|---|---|---|---|
| 40 | +0.014 | +0.002 | 2 | [−0.003, +0.029] | 0.0015 vs [0.0009, 0.0011] | no |
| 52 | +0.013 | +0.007 | 3 | [−0.011, +0.026] | 0.0018 vs [0.0012, 0.0013] | no |

- At both layers, deleting the v5 patterns does 1.4–1.6× more general damage (KL) than deleting the exact twins. It is **not** detectably more interaction-specific.
- For reference, deleting PCA-25 gives S = +0.39 at L40 and −0.08 at L52.

**Claim, scoped honestly.**
- At two held-out layers of Qwen3.6-27B, the H-lens contains a reproducible second-order structure that is:
  - not a function of the activation covariance;
  - not J-space;
  - not the massive-activation geometry;
  - not an artifact of basis estimation.
- Under the frozen causal test it is **not interaction-specific**. Deleting it is slightly more consequential than deleting matched directions, but it does not selectively remove AND-type computation.
- **So this is not an H-space "as provable as J-space":** it is a reproducible, non-Σ-commuting curvature structure without a demonstrated function.
- **Open caveat.** Position-dependent covariance could produce non-Σ-commuting structure. Several patterns carry 1.2–1.6× more energy early in the window. The dumps cannot rule this out.

## 4c. Token-conditional H-structure and a first-order screen (Oct 6, after run 2)

**v7 (laptop, dumps, frozen rule; digits): not found (0/4 layers).** Digit-row curvature has large λ1 (2.7–5.3× the flip maximum; 3.0–5.3× for norm), but no stable 3-D subspace (split@3 0.40–0.68). Punctuation rows were the descriptive exception: split@3 0.60–0.80.

**v8 (fresh 27B data, prereg `SCOPE-hspace-v8.md` pushed before the pod existed): punctuation-conditional H-structure REPLICATED.**
- **Data.** 256 fresh wikitext-train windows. The script dropped 33 windows that duplicated **run 1** (seed 0); it should have deduplicated against run 2 (seed 10), and 15 of the 256 windows turned out to be shared with run 2. Excluding them leaves the replication intact (`hs_v8_dedup.py`, `results/CORRECTIONS.md`).
- **Method.** The v5 sign-flip machinery restricted to punctuation rows (3,775 per layer), in run-2's whitening frame.

| layer | norm λ1 (flip max) | norm split@3 | xnorm λ1 (flip max) | xnorm split@3 | cross-run top-5 overlap with run 2 (chance) |
|---|---|---|---|---|---|
| 16 | 16.4 (2.4) | 0.82 | 8.3 (1.5) | 0.82 | **0.865** (0.0055) |
| 40 | 10.9 (2.4) | 0.87 | 8.4 (1.9) | 0.82 | **0.777** (0.0061) |

Cross-run overlaps exclude the 15 windows v8 shared with run 2 (as first logged: 0.865 / 0.792; see `results/CORRECTIONS.md`).

- The planted rank-3 control was "detected" at both layers, but its recovery was low (0.01–0.24) because the real structure dominates, so it is a weak power check.

**What this licenses.** At two layers of Qwen3.6-27B, the curvature measured at punctuation positions contains a low-dimensional, non-Σ-commuting subspace. It reproduces within a run and **across independent runs and fresh data** (top-5 overlap 0.777–0.865), and it is specific to that token class.

**What it does not license.** Function. v8 has no causal test, and the earlier causal test of the all-rows v5 patterns failed. So this is a replicated *candidate* H-subspace, not an H-space "as provable as J-space".

**First-order screen (goal 2; Prop. 9, Obs. 8.3).**
- At 27B the H-lens rows are extremely heavy-tailed: the Kish effective sample size is 0.005% / 0.015% of rows.
- The J-lens-cost screen |Σ^{1/2} ∂F/∂x_p|² predicts each row's curvature energy: Spearman 0.82 / 0.85 over rows and 0.71 / 0.78 over windows.
- Sampling windows by screen mass (Prop. 9) cuts the HVPs needed at equal Frobenius variance by **80× (L16) and 10.6× (L40)**. The oracle bound is 12,272× / 368×.
- End to end, including one backward pass per screened window, the saving is about 2.5–3×.
- This makes the H-lens's dominant cost, HVPs, nearly negligible relative to the J-lens-cost screen.

## 4d. v9: the replicated punctuation-conditional H-subspace carries excess second-order interaction (preregistered PASS)

**Prereg.** `SCOPE-hspace-v9.md` was frozen and pushed before the pod existed. Before that, an adversarial review found that a bf16 trunk would have measured rounding noise, so the design was revised. A $0 laptop validation on the final code then passed: a random EX was correctly "not shown", and a planted bilinear gate was detected on both I and I_trunk.

**Object.** EX is the top-5 punctuation-conditional patterns from fresh v8 data. The cross-run top-5 overlap is 0.865 / 0.777, after excluding the windows v8 shared with run 2.

**Controls.** 8 Σ-orbit twins with identical raw variance and per-PCA-axis profile; their overlap with EX is 0.01–0.03.

**Data.** 89 fresh wikitext-103 validation windows from 60 articles. This split was never used at 27B.

**Measurement.** Exact fp32 trunk, float64 log-softmax, 1σ moves at a punctuation position. I is the 2×2 interaction in the summed log-likelihood of the next 16 tokens. I_trunk removes the final-norm/softmax part.

| layer | EX / twins (I) | log CI | EX / twins (I_trunk) | log CI | per-main-effect T3 CI | rank of EX among 9 | noise floor / twin signal |
|---|---|---|---|---|---|---|---|
| 16 | **5.8×** | [1.58, 1.95] | **6.0×** | [1.60, 1.99] | [0.82, 1.14] | 1 / 1 | 0.20 |
| 40 | **2.5×** | [0.78, 1.10] | **2.5×** | [0.75, 1.09] | [0.24, 0.57] | 1 / 1 | 0.10 |

**Verdict (mechanical): "EX carries excess within-span trunk pairwise interaction vs Σ-orbit twins (v9)", REAL9 at both layers.**

**Robustness (descriptive).**
- At L16, EX beats every twin in 88% of windows (69% at L40), and beats the mean twin in 98% (82%).
- The median per-window ratio is 4.95× (2.26×), and without the top-5 windows the geometric-mean ratio is 5.2× (2.3×) (`hs_v9_robust.py`, `results/hs_v9_robust.json`; an earlier "4.3× (2.4×)" did not reproduce, see `results/CORRECTIONS.md`).
- In absolute terms the effects are small: mean |I| along EX pairs is 2.2e-3 (1.0e-3) nats, against 3.3e-4 (3.0e-4) for the twins.
- All 10 EX pairs agree with the 3 matched pairs.
- The effect is mostly in the trunk: |I_trunk| is 2.4e-3, against 6.6e-4 for |I_soft|.

**What the chain of preregistered tests now establishes, at L16 and L40 of Qwen3.6-27B:**
1. **Reproducible structure.** At punctuation positions the H-lens has a 5-dimensional structure that reproduces within a run (split@3 0.82–0.87 against a flip null of 0.02) and across independent runs on fresh data (0.865 / 0.777). That is v8.
2. **Not trivial.** It is not explained by activation covariance (sign-flip test), nor by J-space (overlap 0.06–0.12), nor by the massive-activation geometry.
3. **Predictive of second-order function.** On fresh text, moving along its directions produces 2.5–6× more natural-scale pairwise interaction than exactly variance-matched directions, per unit of first-order effect as well. That is v9.

**Not established:**
- decoded or verbalisable content;
- steering or ablation function;
- class-specificity (no matched arm at non-punctuation positions);
- generality across models, corpora, or the other token classes.

Note also that 15–23% of the "punctuation" positions are wikitext markup. So this is a **validated H-subspace candidate**: a second-order analogue of J-space's *predictive* validity. It is not yet J-space's full evidential package.

## 5. White-box control read (goal 3): mechanical prereg verdict UNCALIBRATED (descriptive only); NEGATIVE on the statistics

**Setup.**
- Qwen3.6-27B reads 1,500 BashArena trajectories (750 attack, 750 honest; attackers Claude Sonnet 4, Gemini 2.5 Pro, GPT-5). It reads a sanitized render: the first user message, the agent's tool calls and the tool outputs. Attacker prose and formatting tells are removed (`ctrl_read.py`).
- 624 attacks are eligible (558 have the malicious call k\* inside the 8k-token window; 66 are censored). k\* is the first tool call carrying the IOC in any decoded form (amendment A1).
- Scoring is `ctrl_score_v2.py` against `ctrl-score-v2-spec.md` (amendments A1–A9). The freeze hashes were logged before scoring: words 173f589ac56de6af, spec 79d1b26eaa90c4ef.

**Primary test (F1, label-free pre-emption).**
- Arm A alarms on the information-matched prefix-text monitor T_sup.
- Arm B gives a quarter of the audit budget to W1, the training-free multi-token J-lens matched filter, in an OR slot.
- The endpoint is an alarm *before* the malicious call exists.

| audit budget α | Arm A pre-empts | Arm B pre-empts | b / c / net | McNemar p (Holm) |
|---|---|---|---|---|
| 1% | 0.30 | 0.26 | 1 / 23 / **−22** | 1.0 (1.0) |
| 2% | 0.31 | 0.30 | 1 / 7 / **−6** | 0.996 (1.0) |
| 5% | 0.33 | 0.33 | 1 / 1 / **0** | 0.75 (1.0) |

- At 2% the task-cluster bootstrap 95% CI of net is [−18, +1]. The 200 random-lexicon W1s give a 95th-percentile net of −6. Across 20 fold seeds the median net is −7 and none is positive.
- **Mechanical verdict: "UNCALIBRATED: descriptive only".** The G-calibration gate failed: W1's empirical threshold is 10.3 against 4.4 for the iid reference, and its power at planted shifts is roughly half the reference. On the statistics alone the verdict would read **calibrated NEGATIVE** (the bootstrap upper bound of net is below +5). All other gates passed: placebo AUROC 0.50 and 0.50; content at k\* 0.94.

**Descriptive (trajectory AUROC, full path):**
- W1 scores 0.714 [0.692, 0.737], against 0.478 for the rotated-lens control and 0.49–0.51 for length and position nuisances. So the J-lens read carries real signal.
- But a lexical twin over the same words does better (T_lex 0.769), and so do the black-box score (0.852) and the supervised text monitor (T_sup 0.756). A single regex for the in-window pastebin call reaches 0.929.
- On the pre-k\* prefix, W1's AUROC is 0.398: there is no early-warning signal.
- G2 (workspace-dimension GLR): every subspace, including J25 and PCA25, scores below 0.5 on the prefix. The predicted ordering does not hold.

**Exploratory positive, with a caveat that kills generalisation.**
- W3 is a supervised probe on Qwen's activations (cross-fitted, task-grouped folds). In the OR slot it pre-empts **+33 net attacks at α = 5%** (b/c = 33/0, Holm p < 0.001) and +9 at 2% (Holm-adjusted p = 0.078), but −12 at 1%.
- **Leave-one-attacker-out, it collapses to chance:** AUROC 0.53 (Sonnet 4), 0.57 (Gemini 2.5 Pro) and 0.45 (GPT-5).
- As preregistered, a gain from W3 means "a better reader of the same text", not privileged access to the agent's intent: the reader is a surrogate, so I(Y; W) ≤ I(Y; text).

**Take-away.** On this data, a training-free J-lens add-on does not move any attack from "execute" to "audit" at a fixed audit budget, at any of the three preregistered budgets. This replicates the Oct 5 interpcontrol negative at 27B, with a sanitized render and a much stricter test.

## 6. Limitations
- One definition of H-space (per-position row energy, whitened, vocabulary-weighted outputs).
- One model family at 27B.
- The control read is a surrogate reader: Qwen reads other models' trajectories.
- Deviations from the prereg (all logged in `runlog.md`): a pip flag; a wikitext path bug fixed before any 27B result; 455 rather than 512 eval windows (data availability); the control read capped at 8k tokens after a 16k-token OOM.

## 7. Related work (from `lit-2026-10-05.md`)
- **Second-order and bilinear interpretability:**
  - bilinear MLPs (Pearce et al. 2024): my SwiGLU atom with SiLU replaced by the identity;
  - quadratic approximations of gated MLPs (Belrose & Rigg 2025);
  - HETA, HEAT, and HVP-corrected attribution patching (2606.09899).
- **Workspace readers:** the J-lens and R-lens; Concept Lens (Gong et al. 2608.31084, the best training-free multi-token reader at 40.3% Rank@10 on Qwen3.6-27B).
- **Control:** Ctrl-Z; SHADE-Arena protocols; BashArena; LinuxArena; NP combination of probes and black-box monitors (Hua et al. 2025); deception probes (Goldowsky-Dill et al.).
- **Novelty claim:** an averaged second-order residual lens and its subspace, gated-transformer atom anatomy, the rectangle identity as the bridge from local curvature to finite interactions, and a preregistered causal test at 27B.
