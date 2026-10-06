# Laptop run log (free compute on the pod's dumps, relayed through HF)

[freeze 2026-10-06T11:10:48Z] v7 DECISION RULE, written and pushed publicly before any v7 computation. hs_flip7.py sha256 a7b32d5198e1b13c.
- Token-conditional H-space (digits): restrict the reviewed v5 machinery to rows whose token is a digit, at L16, L28, L40 and L52.
- **PASS** iff at >= 3 of the 4 layers, for BOTH norm and xnorm:
  - lambda_1 > all 200 flips;
  - split3 > all 200 flips;
  - split3 >= 0.7;
  - the planted control is detected;
  - AND the digit top-10 patterns overlap the all-rows top-10 patterns by <= 0.5 (class-specificity).
- Word and punctuation classes are descriptive only.
- Context: this is the 8th test in the sequence (run 1, v3, flip L16/L28, v5, twin, C6', and v6, which was designed but not run). Selected after seeing that digit rows carry 88% of the L16 energy; no class-conditional statistic had been computed.

[result] v7 frozen rule: **not found (0/4 layers)**.
- Digit-conditional curvature: lambda_1 is 3.3-5.3x the flip maximum at every layer, but split3 is only 0.40-0.68, below the 0.7 bar.
- Descriptive, not part of the rule:
  - punctuation-conditional split3 (norm/xnorm): 0.79/0.70 at L16, 0.79/0.68 at L28, 0.80/0.72 at L40, 0.60/0.69 at L52. Overlap with the all-rows patterns is 0.44-0.63.
  - Word rows (76% of rows) reproduce the all-rows structure (overlap 0.81-0.96).
- Punctuation is a lead for a fresh-data preregistration; it cannot be claimed from this data.
