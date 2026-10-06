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

[freeze 2026-10-06T11:58:44Z] v8 frozen before the pod exists: SCOPE-hspace-v8.md a96729b3782e0c33, hs_v8.py 63c391646449f272, hs_flip5.py 4553eef77ee2e338, hs_run8.sh 72c99c05009078dd.
- Local 0.8B smoke test passed (the L6 screen and the punctuation pipeline ran end to end).
- Authorised by the user's "feel free to use up the full runpod budget" and "do whatever is necessary"; about .0–1.2 of the remaining ≈ .9, with a 50-min self-terminate.

[result, Oct 6 ~12:37 UTC] v8: **REPLICATED** (frozen rule, both layers). Pod terminated after sha-verifying 7/7 files on HF; about $1.10.
- **Punctuation rows on fresh data** (256 windows; 33 duplicates of run-2 windows dropped):
  - L16: norm λ1 16.4 vs flip max 2.4, split@3 0.82; xnorm 8.3 vs 1.5, split@3 0.82; cross-run top-5 overlap 0.86 (chance 0.0055).
  - L40: norm 10.9 vs 2.4, split@3 0.87; xnorm 8.4 vs 1.9, split@3 0.82; overlap 0.79 (chance 0.0061).
  - The planted control was "detected", but its recovery was low (0.01–0.24) because the real structure dominates, so it is a weak power check.
- **Screen (goal 2):**
  - Kish ESS of rows: 0.005% / 0.015%.
  - Spearman(curvature, first-order screen): 0.82 / 0.85 over rows, 0.71 / 0.78 over windows.
  - Window-level importance sampling cuts HVPs 80× / 10.6× (oracle 12,272× / 368×). That is "useful" by the frozen ≥ 5× bar.
  - End to end, including screening every candidate window with one backward pass, the saving is about 2.5–3×.
- **Deviation:** the first launch ran out of memory, so I switched to separate HVP graphs and 128×2 probes. The analysis is unchanged; it is logged in the pod run log.
- **Not licensed:** function. There was no causal test in v8.

[result, Oct 6 ~14:00 UTC] v9: **REAL9 at both layers**. Verdict: "EX carries excess within-span trunk pairwise interaction vs Σ-orbit twins (v9)". Pod 32 min, about $0.90; 5/5 files sha-verified on HF before termination.

| | L16 (n = 89) | L40 (n = 89) |
|---|---|---|
| T1 on I (log EX/twins) | +1.77, CI [1.58, 1.95] | +0.93, CI [0.78, 1.10] |
| T1 on I_trunk | +1.79 | +0.92 |
| T3 CI on I | [0.82, 1.14] | [0.24, 0.57] |
| rank | 1 / 9 | 1 / 9 |
| noise floor / twin | 0.20 | 0.10 |

Self-check from the raw per-window data: EX beats every twin in 88% / 69% of windows, the median per-window ratio is 4.95× / 2.26×, and without the top-5 windows the ratio is 4.3× / 2.4×.
