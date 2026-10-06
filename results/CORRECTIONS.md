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
