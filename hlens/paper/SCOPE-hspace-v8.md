# SCOPE v8: fresh-data replication of punctuation-conditional H-structure, and the first-order screen at 27B

*Frozen and pushed publicly before the pod exists (Oct 6 2026). This is the 10th test in the sequence: run 1, v3, flip L16/L28, v5, twin, C6′, v6 (not run), v7, the importance analysis, and this one.*

## Why
- **v7** (laptop, frozen rule: digits) was not found, 0 of 4 layers.
- But punctuation-conditional curvature reproduced descriptively, with split@3 (norm/xnorm) of 0.79/0.70 at L16, 0.79/0.68 at L28, 0.80/0.72 at L40 and 0.60/0.69 at L52. It was class-specific (overlap with the all-rows patterns 0.44–0.63).
- That cannot be claimed from the same data. v8 tests it on **fresh windows**.

## Data
- Qwen3.6-27B at layers 16 and 40.
- 64 probes × 4 windows, on fresh wikitext-103 **train** windows (seed 5). Any window identical to one of run 2's 1,280 estimation windows is dropped.
- Run-2 frame: μ and Σ^{1/2} from `hspace2_full.pt`.
- Two HVPs per probe with independent sign vectors. All valid positions are kept.

## (B) Decision rule (v8)
At **both** L16 and L40, using punctuation rows only and the v5 machinery (`hs_flip5.run_layer`: variant d axis-aligned drop, 200 per-probe sign flips, planted rank-3 control), **all** of the following must hold:
1. **norm and xnorm:** λ1 > every flip replicate.
2. **norm and xnorm:** split@3 > every flip replicate.
3. **norm and xnorm:** split@3 ≥ 0.7.
4. **norm and xnorm:** the planted control is detected.
5. **Cross-run replication:** the overlap of the fresh top-5 punctuation patterns with the run-2 top-5 punctuation patterns (same frame, same retained coordinates, norm) is ≥ 0.5. Chance is about 5/820.

- If every condition holds: **"punctuation-conditional H-structure REPLICATED (v8)"**.
- Otherwise: "NOT replicated (v8)".

**What a pass would and would not license.** It would show reproducible, class-specific, non-Σ-commuting curvature on fresh data. It would **not** show function: there is no causal test in v8. So it is **not** "H-space found as provably as J-space".

## (A) Screen (goal 2, descriptive with fixed statistics)
Statistics:
- Spearman(|y1_p|², |Σ^{1/2} g_p|²) over rows and over windows;
- the Kish effective sample size of the rows;
- the exact per-draw Frobenius-variance reduction of window-level importance sampling (Prop. 9), with q_w ∝ Σ_p proxy_p, and the oracle.

Interpretation, fixed in advance:
- The screen is "useful at 27B" if the window-level reduction is ≥ 5×.
- It is "no help" if the reduction is < 1.5×.

## Budget
- One RTX PRO 6000 pod at about $1.71/h.
- A self-terminate watchdog fires at 50 min.
- Expected cost is about $1.0–1.2 of the remaining ≈ $3.9.
