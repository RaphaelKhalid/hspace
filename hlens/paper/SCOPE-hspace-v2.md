# Run 2 addendum: robust and cross-moment H-space on Qwen3.6-27B (EXPLORATORY; frozen Oct 6 2026 ~04:20 UTC before run 2 starts)

**Status.** Run 1 is the preregistered test. Its verdict stands as computed by `paper/verdict.py`: **FAIL**, because C1 (split-half ≥ 0.7) fails at all 4 layers, at 0.33 / 0.37 / 0.33 / 0.41. C0, C2, C3 and C4 pass at all 4 layers; C5 and C6 are pending. Run 2 was designed *after* seeing that, so it is **exploratory** and is never merged into the run-1 verdict.

The code is `hspace2.py` (sha256 prefix recorded in runlog at freeze). It was revised after two adversarial pre-flight reviews (14 + 10 agents).

## Hypotheses
- **H-R1 (heavy tails).** Run 1's C1 failure comes from heavy-tailed per-position magnitudes. Test it with the spatial-sign estimator `norm` = E[unit(y₁) unit(y₁)ᵀ].
- **H-R2 (cross-position contamination).** Run 1's operator E[y₁y₁ᵀ] is the per-position *row* energy Σ_q H̃_pq H̃_qp, which includes cross-position blocks. The unbiased cross-moment `x` = E[sym(y₁y₂ᵀ)] isolates the exact per-position diagonal-block energy, using two HVPs with independent Rademacher sign vectors s₁ and s₂.
- **H-R3 (rectangle average).** `int_x` takes the cross-moment at two independent random points of the rectangle. Each point removes random fractions (s, t ~ U[0,1]) of two random whitened halves of the deviation from the mean, at a sparse set P of positions: every 8th valid position, with a random offset per window. It estimates the energy of the diagonal blocks of the rectangle-averaged Hessian (Thm 3) under a joint move of P.
  - `int_raw` and `int_norm` are rectangle averages of the *squared* curvature. They are descriptive only.
  - `locP` reads the local HVPs only at P. It is the sample-size-matched control for `int`.
  - `xnorm` is a robust heuristic that still depends on cross-position blocks. It is descriptive only.

## Estimators and diagnostics
- **Variants:** for each kind (loc, int, locP), raw / norm / x / xnorm.
- **Probes:** 256 probes × 4 windows per layer, at layers 16, 28, 40 and 52, on fresh train windows (seed 10).
- **Split-half** at k = 25 (the criterion), plus k = 5, 10, 50, and k_floor = the number of eigenvalues above the noise floor.
- **Signed cross-moment matrices.** They are PSD in expectation. We report the noise floor |λ_min|, n_above_floor, and the energy three ways: raw, positive-part, and debiased (top-25 minus the floor).

## Criteria (thresholds as in run 1, applied per variant)
- **C1:** split-half at k = 25 ≥ 0.7. Split-half at k_floor is reported alongside.
- **C2:** top-25 **debiased** energy share ≥ 0.5 (the positive-part share is reported too).
- **C3:** overlap with run-1 J25 < 0.5. J25 is mapped into run-2 whitened coordinates with the covector map Σ₂^{1/2} Σ₁^{-1/2}.
- **C5:** KL ≥ 3× the mean of three random 25-dimensional subspaces (the same seed-99 subspaces as run 1).
- **C6:** dissociation ratio ≥ 2, with the lower bound of the bootstrap 95% CI above both J25's ratio and the random mean. Ablation conditions: loc_raw, loc_norm, loc_x, int_x, J25, and rand×3. The labels are identical to run 1's (`hs_s7.py`): the same 455 test windows and Random(7).
- **C4:** not re-measured.

**Exploratory claim** "robust H-space found": `loc_x` or `loc_norm` passes C1, C2, C3, C5 and C6 at ≥ 2 of the 4 layers. We report which of the two passes. Either one counts, because they answer different questions (H-R2 vs H-R1).

**Also measured.** The J-adjoint local H-lens (k = 1) at 15 layers (4–60), with both `x` and `xnorm`, using 128 probes per layer.

**Spend.** The same pod continues after run 1's S7 and the control read. The watchdog deletes the pod at 11:00 UTC; stages B and C stop cleanly at 10:30 UTC. That puts the total RunPod spend at ≤ $13.8 of the user's $16.99. All artifacts are pulled to the laptop with sha256 checks before termination.
