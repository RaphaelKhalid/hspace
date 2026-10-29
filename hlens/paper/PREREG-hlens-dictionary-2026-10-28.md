# PREREG: the H-lens dictionary (Phase 1, Qwen3.5-0.8B, layer 18)

Frozen 2026-10-28, before any table, correlation or causal number exists. Its sha256 is recorded in the run log. Exploratory programme; this file fixes the tests and bars for Phase 1 only.

## Question
Does the context-averaged Hessian of a token's logit encode **AND rules** between input features? That is, does it say "current word s together with previous word r raises next word t, beyond s and r separately", in agreement with the language itself? A first-order lens (J) cannot represent such an interaction: a mixed second derivative of any linear map is zero.

## Objects
- **Targets t (16).** Function words " the", " .", " ,", " of", " and", " in", " to", " a" (exact Hessians from `fullh_08b_L18`). Concept words " simulated", " fake", " virtual", " artificial", " real", " genuine", " river", " music" (exact Hessians from `fullh_concept_L18`).
- **Readout.** Centered: u_t − ū, with ū the mean over the vocabulary of (1 + w_final) ⊙ W_U. The function-word Hessians were taken with an uncentered readout, so they are centered exactly by subtracting the Hessian of the ū readout on the same 64 contexts. The Hessian is linear in the readout.
- **Whitening.** Σ^{1/2} at layer 18, from the same 24 covariance windows as `hs_fullh`. All directions and Hessians are in whitened coordinates.
- **Feature directions** (from a separate corpus pass of 3000 WikiText-103 train windows, seed 21, positions 16..62):
  - c_s = mean whitened state at positions whose current token is s, minus the mean over all positions;
  - p_r = the same for previous token r;
  - both unit-normalized;
  - candidates are the 300 most frequent tokens in that pass, each with ≥ 30 occurrences.
- **Model interaction.** I_t(s, r) = c_s^T H̄_t p_r, with H̄_t the mean over the 64 contexts. Primary uses the full H̄_t. A secondary analysis deflates the top eigenvector of the average of all 16 targets' H̄ (the generic axis).

## Ground truth: corpus trigram synergy
- **Corpus.** The first 20M tokens of the WikiText-103 train split.
- **Definition.** With prev = r, cur = s, next = t:
  Syn_t(r, s) = log P̂(t | r, s) − log P(t | s) − log P(t | r at distance 2) + log P(t),
  where P̂(t | r, s) = (N(r, s, t) + 10 · P(t | s)) / (N(r, s) + 10). This shrinks rare pairs toward no synergy.
- **Eligible pairs.** N(r, s) ≥ 30.

## Tests and bars
**T1 (dictionary agrees with the language).** Per target:
- the partial Spearman correlation between I_t(s, r) and Syn_t(r, s) over eligible pairs, controlling for log N(s), log N(r) and log N(r, s);
- null: the same statistic with H̄_t replaced by the mean of sign-flipped per-context Hessians (500 flips), which keeps the noise and kills the mean;
- a target passes if its partial ρ is above the null's 99th percentile and has ≥ 200 eligible pairs.

T1 PASSES if ≥ half of the targets with ≥ 200 eligible pairs pass.

**T2 (causal, held out).** Per target:
- take the 10 eligible pairs with the largest |I_t| and 20 random eligible pairs;
- on 32 fresh contexts (WikiText-103 train windows, seed 13, random position 24..63, never used for H̄), apply the 4-point mixed difference on the **true log-probability** of t:
  D = L(x + m_c + m_p) − L(x + m_c − m_p) − L(x − m_c + m_p) + L(x − m_c − m_p),
  with m = Σ^{1/2} · direction, rescaled to 0.25 · √tr Σ. D isolates the c × p interaction; first-order and pure self terms cancel to second order;
- a pair passes if sign(D) = sign(I_t) in ≥ 80% of contexts and |mean D| / sd(random pairs' mean D) ≥ 3;
- a target passes if ≥ 6 of its 10 pairs pass.

T2 PASSES if ≥ half of the targets pass.

**T3 (readability, secondary).** The corpus synergy of each target's top-10 pairs (signed by I) exceeds that of its random pairs (one-sided Mann-Whitney p < 0.01, pooled over targets). Also reported: the top pairs as words.

## Verdict
- **The "H-lens dictionary" claim** at 0.8B L18 holds iff T1 and T2 both pass.
- **T1 only:** statistical agreement without causal support.
- **T2 only:** causal interactions that don't match the corpus.
- **Neither:** no compositional H-space at this scale and layer. Phase 2 (27B) is not run.

## Not allowed after seeing results
Changing the candidate count, thresholds, f, shrinkage, contexts or eligibility. Any change is reported as post hoc.
