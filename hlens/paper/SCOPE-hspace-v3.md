# Preregistration v3: corrected reliability (C1′) and interaction-ablation (C6′) criteria

Written Oct 6 2026, ~04:45 UTC. This is **before** any run-2 result exists (run 2 has not started; it is queued behind the control read). It is also before the C6′ measurement exists. File hashes go into runlog.md at freeze.

## Why v3 (the reasoning, stated before seeing new data)
Run 1's verdict stays **FAIL** (C1 at all 4 layers, C6 at layers 16 and 28). That verdict is final for the v1 definitions. v3 does not reinterpret run 1. It defines better-targeted criteria for a *new* test, run on data that do not exist yet.

- **C1 conflates "unstable" with "smaller than 25".** Run 1's participation ratio was 1.3–8.5, so ~2–8 directions carry the energy. The remaining top-25 eigenvectors are sampling noise and cannot replicate. Even a perfectly stable 5-dimensional H-space would score about 5/25. Reliability has to be measured at the signal dimension.
- **C6 measures the wrong effect.**
  - It is a ratio of mean log-prob damage. The denominator (damage on additive tokens) can be ≈ 0; the random-subspace ratios ran from −206 to +3.8.
  - Deleting a subspace removes main effects as well as interactions, so "hurts interaction tokens more" is not the signature of an AND.
  - By the mixed-difference lemma (theory-v2 Lemma 3), the interaction term I of two context spans *is* a rectangle-averaged second derivative. The direct test of "H-space carries the AND" is to measure how much of I itself survives the ablation, compared with how much of the main effects survives.

## C1′ (reliability at the signal dimension)
Computed by `hspace2.py` (frozen 04:20 UTC, sha b655f09e…). Primary variant: `loc_x`, the cross-moment estimate of per-position diagonal-block energy.

- **noise floor = |λ_min|** of the signed cross-moment matrix. The population matrix is PSD, so negative eigenvalues are pure sampling noise.
- **k_floor = min(25, n_above_floor of each half).**

**C1′ passes at a layer** if:
1. k_floor ≥ 3;
2. split-half overlap of the top-k_floor eigenspaces is ≥ 0.7;
3. secondary: split-half at k = 5 is ≥ 0.7.

## C6′ (interaction ablation; `hs_c6prime.py`, queued after run 2 on the same pod)
**Setup.**
- 228 test windows (the first 228 of run 1's 455) × 2 targets: 456 targets.
- Labels are the same as run 1's (`hs_s7`): Random(7), two 8-token spans S1 and S2 replaced by " the".
- Layers 16, 28, 40, 52.

**Measurement.** For each ablation A, applied at layer ℓ at every position as whitened mean-ablation (the same code as S7):
- I_A = f(S1,S2) − f(−S1) − f(−S2) + f(−S1−S2), on the target log-prob, with the ablation applied in all four forwards.
- M_A = (f(S1,S2) − f(−S1), f(S1,S2) − f(−S2)).

**Retention scores** (energy-weighted, so near-zero denominators cannot dominate):
- r_I(A) = Σ_t I_A,t · I_0,t / Σ_t I_0,t²
- r_M(A) = Σ_t M_A,t · M_0,t / Σ_t ‖M_0,t‖²

**AND-specificity:** S(A) = r_M(A) − r_I(A). It is positive when A removes interactions more than main effects.

**Conditions:**
- `loc_x` (run 2, primary);
- `loc_norm` (run 2);
- run-1 H25;
- J25 (run 1, covector-mapped);
- rand ×2 (whitened, seed 99).

**C6′ passes at a layer** if, for `loc_x`, all three hold:
1. S(loc_x) ≥ 0.2;
2. the bootstrap 95% CI of S(loc_x) excludes 0 (1000 resamples over windows);
3. the bootstrap 95% CI of S(loc_x) − max(S(J25), mean S(rand)) excludes 0. This is a paired bootstrap over the same windows.

## Verdict v3 (exploratory → confirmatory for the v3 definitions only)
**"H-space found (v3)"** requires:
- C1′ AND C6′ at ≥ 2 of the 4 layers for `loc_x`;
- C0 already passed (run 1, 0.997);
- C3 (J-overlap < 0.5) for `loc_x` at those layers.

**Reported regardless of the verdict:** the same statistics for `loc_norm` and run-1 H25 (descriptive).

**Spend:** within the existing pod. C6′ stops cleanly at 10:30 UTC (deadline guard). Partial layers are reported as partial. The total stays under the user's $16.99.
