# SCOPE v12: is H-space's joint content context-invariant, and readable? (a second-order lens)

Status: DRAFT. Frozen when the sha256 of this file and of `hlens/hs_v12.py` are logged as a `[freeze]` line in `results/runlog_laptop.md` and committed, before the pod exists.

## Why this test
J-space is convincing because a J direction's first-order effect on the output is context-invariant and readable. v8 to v11 showed H-space is reproducible and carries more pairwise interaction than covariance-matched decoys. They never showed that its joint effects *mean* anything. v12 asks the J-space question one order up, in two stages.
- **Stage 1 (pod):** is the context-invariant part of H pairs' joint effect larger, as a fraction of the whole, than for decoys matched in interaction magnitude and mean alignment? It must also hold with normalization curvature switched off, and be distinct from J-space.
- **Stage 2 (offline, after the pod):** do H pairs' decoded joint contents read as coherent concepts more often than matched decoys', to a blind judge that is not the model under test?

Text is fresh wikitext-validation windows at one random position each. No punctuation conditioning.

Kill-checks: `paper/killcheck-2026-10-26.md` and `paper/killcheck-v12-2026-10-26.md`. The measurement is unpublished (OPEN). The adversarial review (2026-10-26) found that a pair-identification statistic saturates for any fixed direction pair: random twins hit ID 0.95 on 6 windows. v12 therefore uses a continuous invariant fraction.

## Objects (frozen earlier)
- **H:** v11a all-position H-space, `out/v11a_subspaces.pt` (sha256 prefix `29e0b7cb95991011`), at L16 and L40 of Qwen3.6-27B.
- **HJ:** H minus J25.
- **J5:** the top-5 J directions.
- **tw0-7:** Σ-orbit twins of H, seeds 12100+l.
- **tx0-7:** Σ-orbit twins of HJ, seeds 12200+l.
- **Pool:** 48 further Σ-orbit twins of H, seeds 12300+l.
- All moves are raw 1σ, in the run-2 whitening frame.

## Measurement
For window x, layer l, token q, and each of an arm's 10 direction pairs (i, j):

    r_ij(x) = [h_L(x+u_i+u_j) − h_L(x+u_i) − h_L(x+u_j) + h_L(x)]_q / rms(h_L(x)_q)
    m_i(x)  = [h_L(x+u_i) − h_L(x)]_q / rms(h_L(x)_q)

- h_L is the last block's output before the final norm.
- Each window's vectors are projected off n = h_L(x)_q / |h_L(x)_q|, which the final norm removes.
- Vectors are read in the centred logit metric M = G W_Uᵀ(I − 11ᵀ/V) W_U G, with G the final-norm gain.

**Primary statistic:**

    F_det(arm) = mean over pairs of [mean_{a≠b} <r̃_a, r̃_b>_M] / [mean_a ||r̃_a||²_M]

- r̃ is r projected off span{template, the arm's full-sample mean m_i, mean m_j}.
- F_det is the unbiased fraction of the joint effect that is the same in every context. It is continuous, with no ceiling.

**Template:** the top right-singular vector (k = 1) of the pool's unit-normalized pair means, from stage K. Stage K uses separate windows and separate arms, so the template is not fitted on any compared arm.

## Stages
- **S0:** whitening frame, objects, and the whitened mean energy of every arm (mean-alignment diagnostic).
- **K (4 windows, articles disjoint from W):**
  - Run H and the 48-twin pool. Pick **dm0-7**: the 8 pool twins closest to H on a combined distance |log magnitude − log magnitude_H|/sd + |mean energy − mean energy_H|/sd. Magnitude is the median ||r||_M.
  - Fit the template.
  - If K fails, W is skipped and the verdict is INCONCLUSIVE.
- **P (8 windows, articles disjoint from W), planted positive control:**
  - Plant a move-only bilinear gate at block l+1. The plant's mu is the clean state, so its z is the whitened move.
  - The gate writes G(z·a)(z·b)w, with w = lstsq(J_l, gain⊙W_U[" banana"]). a and b are the first two of 5 random whitened directions (arm P).
  - G sets the planted per-window term to 0.5× the median de-templated null cross-term norm, on window 0. The realized ratio is logged.
  - P runs planted and unplanted, plus 4 Σ-orbit twins of P (pt0-3).
- **W (40 windows):** plain arms H, HJ, J5, tw0-7, dm0-7, tx0-7; frozen arms H, J5, tw0-7.
  - Frozen means every RMSNorm, gated norm and l2norm scale is fixed at the clean pass, so normalization curvature is exactly zero. No template is removed in the frozen condition.
  - Script assertions: every DeltaNet norm is the patchable torch class, and the frozen clean pass equals the plain one (relative error ≤ 1e-4, else freeze_valid = false).
  - A window with any non-finite value at a layer is dropped from all conditions at that layer, and the count is logged.

## Decision rules
**compare_F(ex, decoys, min_LR):**
- LR = log F(ex) − log median F(decoys). F is floored at 1e-4 before the log.
- 95% CI: article-cluster bootstrap, 2000 resamples. Pairs of copies of one window are excluded inside a resample.
- Passes iff rank = 1 of 9 (ties count against ex), LR ≥ min_LR, and CI lower bound > 0.

**Content (per layer):** passes iff all four hold.
- C1: H vs tw0-7, plain, min_LR = log 2.
- C2: H vs dm0-7, plain, min_LR = log 1.5.
- C3: H vs tw0-7, frozen, min_LR = 0.
- freeze_valid.

**Distinct (per layer):** passes iff both hold.
- HJ vs tx0-7, plain, min_LR = log 2.
- H vs tw0-7, plain, min_LR = log 2, after also projecting every r off the full-sample mean single effects of J5 (J5's output span).

**Planted control (per layer):** passes iff all four hold.
- F(P pair 01, planted) > the max F over all pairs of pt0-3.
- F(P pair 01, planted) > the max F over P's other 9 pairs.
- F(P pair 01, unplanted) ≤ that null max.
- " banana" ranks in the top 10 of the decoded mean of (planted − unplanted) r_01.

**Stage-1 verdict:**
| Outcome | Verdict |
|---|---|
| K, P or W raised an error | INCONCLUSIVE (stage error) |
| Planted control fails at either layer | INSTRUMENT FAILED |
| Fewer than 24 W windows at a layer | INCONCLUSIVE |
| Content and Distinct pass at both layers | **STAGE-1 PASS** |
| Content passes at both layers, Distinct does not | PARTIAL (not distinct) |
| Content passes at one layer | PARTIAL (one layer) |
| Otherwise | FAIL |

**Descriptive only, not in the verdict:**
- Template depth k = 3 and 10.
- F with the heaviest window dropped.
- J5's F, plain and frozen, and per pair.
- Per-pair F for H.
- Every arm's mean energy.
- Decoded token lists for the de-templated pair means of H, HJ, J5 and dm0-7, and the single effects of H, HJ and J5.

## Stage 2: blind readability (offline, after the pod; preregistered here)
- **Inputs:** the decoded top-15 token lists of the de-templated pair means: 10 H pairs and 80 dm pairs (dm0-7) per layer, from `hs_v12_full.json`.
- **Judge:** one model that is not Qwen. It will be named, with its OpenRouter id and the exact prompt, in a second `[freeze]` line *before any judge call*. Temperature 0. Budget ≤ $1.
- **Task:**
  - Two-alternative forced choice: "Which list is more about a single coherent concept?"
  - Each H pair is paired with each of the 8 dm decoys' same-index pair: 80 trials per layer.
  - Presentation order is randomized with seed 12, and the judge sees no labels.
- **Pass (per layer):** H win rate ≥ 0.65 and one-sided binomial p < 0.01, against 0.5, over the 80 trials.
- **"Readable" claim:** requires a stage-2 pass at both layers. It is only made if stage 1 also passed.

## Sizes, budget, runtime
- 40 W windows, 8 P and 4 K windows. All are 64-token windows from wikitext-103 validation (`windows_val`, k ≥ 2).
- W uses seed 1202; P and K use seeds 1201 and 1203, with W's articles excluded. q is in [24, 62].
- Validation text was never used to estimate H.
- Stage budgets: K 10, P 8, W 70 min. Run cap 115 min. Watchdog 150 min.
- One RTX PRO 6000 (96 GB): about $1.7-2.1/h, so about $3-5.

## What each outcome means
- **STAGE-1 PASS:** H pairs' joint effects have a larger shared-across-contexts core than equally strong, equally mean-aligned decoys. This holds with norm curvature off and outside J-space's span. It is necessary for, but not the same as, meaning. Stage 2 tests meaning.
- **FAIL:** H-space's extra interaction is not more stable than matched decoys'. Read with J5's numbers, second-order content lives in J-space or nowhere special.
- **C1 passes, C2 fails:** the earlier "beats twins" results are explained by magnitude or mean alignment.
- **Plain passes, frozen fails:** the invariant core rides on normalization. Freezing also changes the first-order function, so this is suggestive, not proof.

## Deviations
Log any deviation in `results/runlog_laptop.md` (append only) before viewing affected outcomes. Technical relaunches rerun the identical frozen code and need the user's explicit OK.
