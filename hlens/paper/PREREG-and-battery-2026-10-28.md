# PREREG: the AND battery (Qwen3.5-0.8B dry run, then Qwen3.6-27B)

Frozen 2026-10-28, before any battery number exists. Its sha256 is in the run log.

## Question
Which computations does the model perform as **interactions** (A AND B → answer, including XOR), in a way that is invisible to any first-order (J-like) account? At which layers, and are they **within one position's state** or **across positions**?

## Classes (23)
Each class is a 2×2 factorial over prompts with single-substitution factors A and B and an answer per cell. The first token of each answer is the target. Items are generated in `hlens/hs_and_items.py`.

**Classes:** negation_polarity, quantifier_truth, comparison_order, number_comparison, facts_subject_relation, translation, entity_time, morph_irregular, morph_regular (control), plural_irregular, plural_regular (control), gender_role, comparative_superlative, agreement_tense, pronoun_case, bracket_matching, carry, xor_bits, code_intent, register, refusal, truth_frame, binding.

**Item validity.** Equal token length across the 4 cells. A and B change disjoint token positions, and cell (1,1) = cell (0,0) with both substitutions. ans11's first token differs from at least one other cell's.

## Measurements (per item, layer l, variant)
L = log-softmax probability of ans11's first token at the last position.
- **Within-position variant.** dA = h_l^{10}[last] − h_l^{00}[last] and dB likewise, added to the (0,0) prompt's last position at layer l.
- **Cross-position variant.** dA = h_l^{10} − h_l^{00} at A's changed positions, added there; dB likewise at B's positions.
- **Interaction (primary).** I = L(+A+B) − L(+A) − L(+B) + L(0). This is the operational AND: zero for any additive, J-like account.
- **Symmetric H-proxy (secondary).** L(+A+B) − L(+A−B) − L(−A+B) + L(−A−B).
- **Random null.** 3 pairs per item of Σ_l^{1/2}-shaped Gaussian vectors with the same per-position norms as dA and dB, put through the same formula.
- **Specificity.** I(ans11) − mean of I(each other distinct cell answer).
- **Completeness (descriptive).** L(+A+B) vs the true L at cell (1,1).
- **True interaction (descriptive).** L11 − L10 − L01 + L00 from the unpatched cells.

## Competence filter (per class)
In the unpatched cells, each cell's answer must have the highest log-probability among the item's distinct answers in ≥ 70% of all cells. Classes below this are reported as "not competent" and are not tested.

## Bar
A competent class with ≥ 12 valid items **passes** at (layer, variant) if:
- I > 0 for ≥ 80% of items;
- z = mean(I) / sd(random-pair I) ≥ 3;
- specificity > 0 for ≥ 70% of items.

A class is an **interaction class** if it passes at ≥ 1 of its (layer, variant) cells (6 at 0.8B, 8 at 27B; z ≥ 3 is the multiplicity guard).

## Layers
- 0.8B (24 blocks): 6, 12, 18.
- 27B (64 blocks): 16, 28, 40, 52.

## Runs and money
- 0.8B dry run on the local GPU validates items, competence and code.
- 27B runs on one A100 80 GB (or H100 if no A100), hard watchdog 120 min, cost check on 2 classes first with an abort if the projection exceeds $5, verified upload to the private HF dataset, then self-delete. Total spend cap $5.

## Not allowed after results
Changing items, thresholds, layers or the bar. Any addition is labelled post hoc.

## AMENDMENT 1 (2026-10-28, before any battery number; prompted by a logic review)
1. **Primary score becomes the contrast logit.**
   - Definition: I_c = I(ans11) − mean over the other distinct answers of I(answer).
   - In log-probabilities the softmax normalizer (logsumexp) cancels exactly, so I_c is free of the purely softmax-driven interaction that a log p score would carry.
   - New bar: pos_frac(I_c > 0) ≥ 0.8, and z_c = mean(I_c) / sd(random-pair I_c) ≥ 3, with ≥ 12 items and competence ≥ 0.7.
   - The separate specificity criterion is dropped, because I_c is the specificity contrast.
   - Log p I and the symmetric H-proxy stay as secondary and descriptive.
2. **Each class is labelled by logic before running.**
   - XOR-type classes cannot be solved additively, so competence implies an interaction somewhere. Prediction: they pass at some (layer, variant). A competent XOR class that passes nowhere means the interaction is computed outside the patched sites (e.g. at layers below the first tested one, or at positions not patched).
     XOR classes: negation_polarity, quantifier_truth, comparison_order, number_comparison, truth_frame, binding, xor_bits.
   - AND / lookup classes are linearly separable. Passing means the model uses a superadditive conjunction; failing means an additive account suffices at the tested sites.
     Classes: facts_subject_relation, translation, entity_time, morph_irregular, morph_regular, plural_irregular, plural_regular, gender_role, comparative_superlative, agreement_tense, pronoun_case, bracket_matching, carry, code_intent, register, refusal.
3. **Scope.** The battery measures interactions in the computation above layer l (all orders). It locates AND/XOR computation. It does not by itself establish a context-averaged H-lens.

## AMENDMENT 2 (2026-10-28, before any battery number; a smoke run was killed unseen and its output deleted)
1. **Realistic null** (within-position variant).
   - For every item and layer, 2 pairs combine this item's real dA (or dB) with a real dB (or dA) from a random item of a DIFFERENT class, at the same layer and position (last).
   - These pairs have realistic magnitudes and are on-manifold, but are logically unrelated.
   - The within-variant z_c uses the larger of the two null SDs: Σ-random or mismatched. The cross-position variant keeps the Σ-random null only, flagged as the weaker null in the report.
2. **Interactive fraction (descriptive).** |I_c| / (|ΔA_c| + |ΔB_c| + |I_c|), with ΔA_c = c(+A) − c(0) and ΔB_c = c(+B) − c(0).

## AMENDMENT 3 (2026-10-28, before any battery number; second smoke run killed unseen, outputs deleted): the LENS variants
This is the heart question: is there H-space in the J-space sense, a context-free second-order operator?
1. **Directions.** Per class and layer, the leave-one-out mean of the other items' last-position changes: Ā_i = mean over j ≠ i of dA_last(j), and B̄_i likewise.
2. **lensB.** dA_last(i) (item-specific) with B̄_i (context-free). Asks: does a fixed B operator (e.g. "past tense", "Spanish", "female", "not") interact the same way with any item's A?
3. **lensAB.** Ā_i and B̄_i (both context-free). Interpreted only where both factors carry the same meaning across items: negation_polarity, quantifier_truth, comparison_order, number_comparison, truth_frame, xor_bits, binding.
4. **Nulls.** Σ-random pairs with the same norms, plus realistic mismatched pairs: each lens direction paired with a real change from an unrelated class. Same bar as the within variant (pos_frac ≥ 0.8, z ≥ 3 against the larger null SD, competence ≥ 0.7, ≥ 12 items).
5. **Verdict on the heart question.**
   - A class has a context-free H-lens entry if it passes lensB, or lensAB where interpretable, at some layer.
   - "H-space in the J-space sense" is supported at a model scale if ≥ 1 XOR class and ≥ 1 AND class have such an entry.
   - It is refuted at that scale if no competent class passes any lens variant, while per-item variants pass for some. That would mean ANDs exist but are contextual.

## AMENDMENT 4 (2026-10-28, before any battery number; third smoke run killed unseen, outputs deleted): second order and verbalisability
1. **lens-small** (lensB and lensAB directions).
   - Symmetric 4-point contrast at small scale ε: S_ε = c(+εĀ+εB̄) − c(+εĀ−εB̄) − c(−εĀ+εB̄) + c(−εĀ−εB̄), at ε = 0.25 and ε = 0.5 of the lens vectors' norms.
   - S_ε ≈ 4ε²·Āᵀ H B̄ (the averaged-Hessian / H-lens entry), but only in the second-order regime.
   - **Scaling check:** mean S_0.5 / mean S_0.25 must lie in [2.5, 6] (4 = pure second order).
   - **Null:** Σ-random and mismatched pairs at the same ε, which also exposes the bf16 rounding floor.
   - **Pass:** pos_frac(S_0.25 > 0, signed by the class's one-sided lens I_c sign) ≥ 0.8, z ≥ 3 vs the larger null SD, and scaling OK.
2. **Vocabulary decode** (verbalisability).
   - For the real lens pair, the interaction is computed for every vocabulary token on centered logits (logit − vocab mean), for the one-sided natural-scale contrast and for S_0.25, then averaged over items.
   - **Reported:** the top-10 tokens and the rank of ans11's first token.
   - A mismatched-null decode (the same computation with one mismatched pair) is reported alongside.
   - **Verbalisable:** ans11 is in the top-10 of the class-averaged decode at the passing layer/variant, and not in the top-10 of the null decode.
3. **Context-free ratio** (descriptive). mean I_c(lensB) / mean I_c(within), at the same layer.
4. **Heart verdict tiers** (replaces amendment 3 §5).
   - **Tier 1, context-free interaction:** a lens variant passes (amendment 3 bar).
   - **Tier 2, H-lens entry:** Tier 1, and lens-small passes with the scaling check, for the same variant and layer.
   - **Tier 3, J-space sense:** Tier 2, and verbalisable.
   - **SUPPORTED** at a scale if ≥ 1 XOR class and ≥ 1 AND class reach Tier 3.
   - **PARTIAL** if any class reaches Tier 2.
   - **REFUTED (contextual ANDs)** if per-item variants pass but no class reaches Tier 1.
5. **Out of scope** (stated): unsupervised discovery of the directions from the operator, and low-dimensionality. That is a separate Lanczos run.

## AMENDMENT 5 (2026-10-28, before any second-phrasing or 27B number): template transfer and a fast path
1. **Template transfer** (step 2 of the claim ladder). Every class gets a second, independently worded phrasing (`hs_and_items_t2.py`) with the same factors and logic.
   - **xferB:** the second-phrasing item's own dA, with B̄ = the mean dB over ALL first-phrasing items of the class.
   - **xferAB:** Ā and B̄ both from the first phrasing.
   - Same bar, lens-small scaling check, decode and tiers as the lens variants. The tier computed on the xfer variants is the **transfer tier**.
   - **Claim ladder:** "context-free across phrasings" requires transfer tier ≥ the claimed tier.
   - Second-phrasing items also get the ordinary within / cross / lens variants as a reference.
2. **Fast path** (computation only, not a design change).
   - Variants that patch only the last position (within, lensB, lensAB, xferB, xferAB) reuse a cached prefix and run only the last token.
   - Exactness is asserted against the full-sequence path on the first item of every class (max |Δ contrast| < 1e-3 in fp32; < 5e-2 in bf16, reported).

## AMENDMENT 6 (2026-10-28, before any second-phrasing or 27B number): scope, for budget
Second-phrasing classes run only within, xferB and xferAB. The cross / lensB / lensAB "reference" runs on second phrasings (amendment 5 §1, last sentence) are dropped. They test nothing in the bar, and dropping them keeps the 27B run inside the $5 cap. Tests and bars are unchanged.
