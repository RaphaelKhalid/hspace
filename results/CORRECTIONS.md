# Corrections and retractions

I keep every correction here, with the time it was made and the reason. Earlier commits are not rewritten.

## Oct 6 2026, 07:35 UTC: relative-curvature "H-space" result RETRACTED

**What I claimed (commits "Relative-curvature H-lens ..." and "Relative-curvature results L16 + L28").** After dividing out the isotropic-Hessian null, the spatial-sign H-lens had a reproducible "excess-curvature" subspace:
- split-half 0.65–0.94, against 0.00–0.01 for a simulated null;
- zero overlap with the top-25 PCA directions;
- ~1% mass in the massive-activation span.

**Why it is wrong.** An adversarial review (4 independent skeptics plus a synthesis) found three problems, and I verified each against the code:
1. **Filters, not patterns.** `geig()` in `hs_relcurv.py` returns M0^{-1/2} V. For an excess along w, that is proportional to M0^{-1} w, a detector *filter* tilted toward low-variance PCA directions. A filter has zero overlap with PCA-25 even on pure nulls, so "zero PCA overlap" carries no information.
2. **A null rigged to lose.** The simulated null used one row per probe repeated (rank ≤ 256), so its split-half is ≈ 0 whatever the truth. "Data beats null" was guaranteed.
3. **The wrong null.** M0 is the H = cI point null, which the data rejects (participation ratio 3 vs 58). Under misspecification, the ridge ε decides which PCA band the "excess" lands in. The data's mean PCA rank tracked the null's at every ε.

**Consequences.**
- `results/relcurv2_full.json` and `results/hs_relcurv_gain_full.json` should not be cited.
- The post-hoc causal test that consumed these subspaces (`hlens/hs_c6defl.py`, `hlens/pod/chain4.sh`) was **gated before it ran**. Its contrasts would mostly have measured how much variance each ablation removes.

## Oct 6 2026, 07:45 UTC: run-2 confound numbers computed in the wrong whitening frame

- `results/hs_confound_dumps_L16.json` was produced with run-1 μ/Σ^{1/2} (`hspace_full.pt`) applied to run-2 samples. The two whitening estimates differ by about 30% (relative Frobenius norm of Σ^{1/2}) at L16 and L28.
- The "16–31% of estimator energy in the massive-activation span" figure for run 2 is therefore withdrawn.
- The run-1 eigenvector check (`results/hs_confound_run1.json`, 11–37% of top-25 H-energy) used run-1 data in the run-1 frame and stands.

## What replaces them

A per-probe PCA **sign-flip randomization test** (`hlens/hs_flip.py`). It is an exact null for "the reproducible structure is a function of Σ" (whitening prior, reweighted PCA, structureless curvature).
- It has a planted power control.
- Its decision rule was written into the pod run log before any real-data computation.
- Results: see `results/hs_flip_full.json` when available.

## Oct 6 2026, ~11:00 UTC: C6′ "about 2× J25 specificity" corrected to dose

**What I claimed.** Deleting the robust spatial-sign subspace (`loc_norm`) is about twice as interaction-specific as deleting J25 (S = 0.52 / 0.59 vs 0.28 / 0.27 at L16 / L28). This was in the paper §4b and the README.

**Why it is wrong.** The v6 design review (3 skeptics plus a synthesis) found it, and I re-checked it against `hspace2_full.pt`, `hspace_full.pt` and `c6prime_full.json`.
- **At L16, S is nearly linear in the raw variance removed** across the four learned subspaces (r = 0.999).
- `loc_norm` removes 22% of raw variance, J25 13%.
- The interaction-per-main-effect damage ratio φ = (1 − r_I)/(1 − r_M) is equal: 4.7 vs 4.6.
- So the larger S is dose, not specificity.

**What survives, descriptively (no CIs).** At L28 and L40, where main-effect damage is matched, φ is 5.5 / 5.0 for `loc_norm` against 3.2 / 3.3 for J25.

## Oct 6 2026, ~13:30 UTC: v8 "33 duplicates of run-2 windows" was wrong

**What I claimed.** `hs_v8.py` deduplicated its fresh windows against **seed 0**, which is run 1, but run 2's estimation windows used **seed 10**. "Dropped 33 duplicates of run-2 windows" was therefore false. The v9 design review found this.

**The recount (`hs_v8_dedup.py`).**
- 15 of the 256 v8 windows (9 of them run-2 probe windows) were identical to run-2 estimation windows.
- Excluding their 190 punctuation rows, the cross-run top-5 overlap is:
  - L16: 0.865 → **0.865**;
  - L40: 0.792 → **0.777**.
- Chance is 0.006, so the v8 replication verdict is unchanged (bar ≥ 0.5).

## Oct 6 2026, ~18:25 UTC: v9 L16 headline is 5.8×, not 5.9×

- The L16 statistic is T1 = 1.766 (`results/hs_v9_full.json`), so e^T1 = 5.85, which rounds to **5.8×**. The v9 result commit message, the README and the paper said 5.9×. The CI and the verdict are unchanged.
- 5.8× (and 2.5× at L40) is a geometric-mean ratio over windows. The ratio of mean |I| is 6.6× / 3.3×.

## Oct 6 2026, ~18:27 UTC: v9 "4.3× / 2.4× without the top-5 windows" does not reproduce

- `runlog_laptop.md` and paper §4d said that dropping the 5 strongest windows still leaves 4.3× (L16) / 2.4× (L40).
- Recomputed from `hs_v9_full_raw.pt` (88 saved windows) by the new `hlens/hs_v9_robust.py` (`results/hs_v9_robust.json`), dropping the 5 windows with the largest per-window T1 gives **5.2× / 2.3×** (geometric mean) or 4.9× / 2.6× (ratio of means). No definition I tried gives 4.3×.
- The other robustness figures reproduce: EX beats every twin in 88% / 69% of windows, and the median per-window ratio is 4.95× / 2.26×. The conclusion (a few extreme windows do not drive the result) is unchanged.

## Oct 6 2026, ~18:28 UTC: source added for "the spatial-sign top-25 overlaps PCA-25 by 0.78"

- Paper §4b quoted this figure, but no committed file contained it. `hlens/hs_pca_overlap.py` now computes it from `hspace2_full.pt` and writes `results/hs_pca_overlap_run2.json`: `loc_norm` 0.776 at L16 (0.78–0.83 across layers), `loc_xnorm` 0.78–0.82, and my whitened J25 0.30–0.61 (random 0.005). The number stands; it is now reproducible.

## Oct 6 2026, ~18:28 UTC: v7 digit λ1 range is 2.7–5.3×, not 3.3–5.3×

- From `results/hs_flip7.log`, digit-row λ1 divided by the flip maximum is 5.32 / 2.83 (L16, norm / xnorm), 3.00 / 2.68 (L28), 3.78 / 2.98 (L40) and 3.68 / 3.30 (L52). So the range is **2.7–5.3×** (3.0–5.3× for norm). `runlog_laptop.md` and paper §4c said 3.3–5.3×. The v7 verdict ("not found", decided by split@3 0.40–0.68 < 0.7) is unchanged.

## Oct 6 2026, ~18:29 UTC: theory-v2 Observation 8.2 overstated two facts

- "Run 2 `loc_x`: no eigenvalue above its noise floor" holds at L16 only. Per `results/hspace2_full.json` (now committed; it is the run-2 summary that `verdict_v23_run2.txt` was computed from), the counts are 0 / 2 / 5 / 2 of 5,120 at L16 / L28 / L40 / L52, and k_floor = 1 at every layer, so the v3 verdict is unchanged.
- "L28 is similar" (after the L16 digit-energy figure) holds for concentration (top 0.1% of rows: 88%) but not for digits, which carry 4.4% of L28 energy (`results/hs_dump_analysis_full.json`).
