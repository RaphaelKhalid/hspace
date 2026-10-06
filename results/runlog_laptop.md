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
