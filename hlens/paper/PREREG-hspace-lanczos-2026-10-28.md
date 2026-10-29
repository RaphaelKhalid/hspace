# PREREG: is there an H-SPACE? Unsupervised Lanczos on the context-averaged Hessian (step 3 of the claim ladder)

Frozen 2026-10-28, before any 27B Lanczos number. The run waits for a budget OK.

## Question
J-space is found by fitting an averaged operator on generic text and reading its top structure. Can the same construction one derivative up do three things?
- **Find, unsupervised,** the directions the AND battery found by hand.
- Show they live in a **low-dimensional** subspace.
- Reproduce the battery's interactions from that subspace alone.

## Method (validated: Lanczos == exact on 0.8B L18 " to", eigenvalues to 4 decimals, top-6 overlap 1.00, 49× cheaper)
- **Readout per class.** Contrast c = u_a − u_b (centered), for classes whose answers are fixed tokens:
  - negation_polarity (" positive" − " negative"), quantifier_truth (" True" − " False"), comparison_order (" first" − " second"), number_comparison (" Yes" − " No"), truth_frame (" Yes" − " No"), xor_bits (" 0" − " 1");
  - agreement_tense (" are" − " is"), pronoun_case (" her" − " him");
  - refusal (" I" − " To").
- **Operator.** H̄_c = mean over 64 generic WikiText-103 train windows (seed 12, random last position 24..63) of the whitened exact Hessian of c · h_L* at the last position. This is the J-lens fitting distribution, not the battery prompts.
- **Lanczos.** Top-k = 25 eigenpairs by |λ|.
- **Layers.** The layer(s) where the battery's best lens cell for that class sits (27B: one of 16 / 28 / 40 / 52; 0.8B: one of 6 / 12 / 18).
- **Spectrum mass.** ‖H̄‖_F² estimated by Hutchinson with 30 Rademacher probes.

## Tests (per class)
- **S1, the space contains the concepts.** Projected energy of the battery's class-mean directions Ā and B̄ (whitened, unit) onto the top-25 subspace, vs 200 Σ-shaped random unit directions. Pass if Ā and B̄ both exceed the random 99th percentile.
- **S2, the space carries the interaction.** Ratio of the rank-25 interaction Āᵀ V Λ Vᵀ B̄ to the full Āᵀ H̄ B̄. Pass if ≥ 0.5 with the same sign. Also reported: sign agreement of Āᵀ H̄ B̄ with the battery's lens-small S_0.25.
- **S3, low-dimensional.** Pass if the top-25 eigenvalues hold ≥ 50% of the Hutchinson ‖H̄‖_F² (≥ 50% of the curvature energy in 25 of 5120 dimensions).

## Verdict
- **"There is an H-space (in the J-space construction)"** at a scale iff ≥ 2 classes pass S1, S2 and S3, at least one of them XOR.
- **"Supervised H-lens entries only"** if the battery passed but S1/S2 fail.

## Cost
About 21-40 matvecs × 64 HVPs per class. 27B on one A100/H100, auto-delete, cap to be agreed.
