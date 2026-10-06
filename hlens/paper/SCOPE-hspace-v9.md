# SCOPE v9: does the replicated punctuation-conditional H-subspace carry excess second-order (trunk) interaction?

*Frozen and pushed publicly before any v9 pod exists (Oct 6 2026). This is the 11th test in the sequence. The design was revised after an adversarial review (wf_695de7ae), which found that a bf16 trunk would have measured rounding noise.*

## Object
- EX is the top-5 punctuation-conditional H-lens patterns from the fresh v8 data at L16 and L40 (`hs_v9_prep.py`, `out/v9_subspaces.pt`).
- It replicated across runs. The top-5 overlap with run 2 is 0.865 / 0.792, or **0.865 / 0.777 after excluding the 15 v8 windows that were identical to run-2 estimation windows** (correction, `hs_v8_dedup.py`). Chance is 0.006.

## Controls
- 8 Σ-orbit twins, Q_j = P(s_j ⊙ PᵀEX), seed 9090+l.
- They have identical raw variance (0.34%) and the same per-PCA-axis profile.
- Their overlap with EX is only 0.013 / 0.028.

## Data
- 96 windows from the wikitext-103 **validation** split, seed 9. This split was never used at 27B.
- Per window, one punctuation position p.

## Measurement
- All trunk forwards are exact fp32: `bf16w.MODE = "fp32"`, TF32 off. The log-softmax is computed in float64.
- 1σ moves are applied at p.
- **I** = f(u+w) − f(u) − f(w) + f(0), where f is the sum of log p over the next 16 tokens.
- **I_trunk** = f(u+w) − f_add, where f_add is f evaluated on h_u + h_w − h_0 (pre-norm, last layer).
- **I_soft** (descriptive) is the remainder.
- Matched pairs MP = (0,1), (2,3), (1,4), for EX and for every twin. All 10 EX pairs are reported descriptively.

## Statistics (per window, for X in {I, I_trunk})
- T1 = mean_j log(ex3 / tw3_j).
- T3 = mean_j log((ex3 / mEX3) / (tw3_j / mTWj3)): the interaction per unit of main effect.
- rank = EX's rank among the 9 by mean ex3.
- CIs come from an article-level bootstrap, B = 2000.

## Rule
- **REAL9 at a layer** iff, for **both** I and I_trunk: T1 > log 2 with CI low > 0, T3 CI low > 0, and rank = 1 of 9.
- **"EX carries excess within-span trunk pairwise interaction vs Σ-orbit twins (v9)"** iff REAL9 at **both** L16 and L40.
- **Validity outcomes** (never reported as "not shown"):
  - **INCOMPLETE:** fewer than 60 windows at a layer.
  - **ABORTED (throughput):** after 3 windows, the projection is under 60.
  - **UNINFORMATIVE (precision):** a layer fails and its median noise floor is ≥ 0.25 × the median twin |I|.

## Pre-pod $0 validation (laptop, Qwen3.5-0.8B L6, final code)
- **Null** (random EX): T1 +0.23, CI [−0.13, 0.59] → "NOT shown". Correct.
- **Plant** (bilinear gate on EX[:,0] × EX[:,1] at block l+1, G = 0.3): T1 +2.25, CI [1.97, 2.54], on I and on I_trunk → "REAL9". Correct.
- **Floor:** 3–5e-5 against a twin |I| of about 1.2e-3 (ratio ≈ 0.04).

## What a pass would and would not license
**A pass would license:**
- the H-lens has out-of-sample predictive validity for *where* natural-scale second-order sensitivity lives, at punctuation positions;
- under a null the covariance cannot fake (Σ-orbit twins);
- at two layers of Qwen3.6-27B, on fresh text.

**It would not license** "H-space as provable as J-space". Still missing:
- decoded content;
- steering or ablation function;
- class-specificity (the EX-at-word-position arm was dropped);
- other models and corpora.

Also, about 15–23% of the "punctuation" positions are wikitext markup.

**A fail refutes only** within-span, single-position pairwise interaction.

## Budget
- `--minutes 25` of compute after the model loads.
- A 45-min self-terminate watchdog.
- About $1.0–1.3 of the remaining ≈ $2.8.
