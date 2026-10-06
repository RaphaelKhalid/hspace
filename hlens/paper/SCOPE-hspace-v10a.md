# SCOPE v10a: token-class scan for H-space on the fresh v8 data (laptop, $0)

*Frozen and pushed before any non-punctuation row of the v8 dumps is analysed. Part of v10 (the full v10 registered report, `SCOPE-hspace-v10.md`, governs the pod run). Plan: `research/proposals/40-hspace-shopping-list.md`, item M8(b).*

**Naming.** H-space := the top-5 patterns of a class-conditional H-lens (spatial-sign estimator after the v5 axis-aligned drop, correlation-normalised, patterns QR(Δ^{1/2}V)), in run 2's whitened frame. The v8/v9 preregistrations and code call the punctuation instance "EX"; EX ≡ H-space_punct.

**Question.** Is the replicable second-order structure specific to punctuation, or does each token class carry its own replicable H-space (a "token-conditional family")?

**Data.** The fresh v8 dumps `hs_v8_full_L{16,40}.pt` (256 wikitext-train windows never used before v8; 28,416 rows per layer). Only the 3,775 punctuation rows have been analysed so far. The run-2 dumps `hspace2_full_dump_L{16,40}.pt` serve as the independent run for cross-run replication. The class of a row is decided by its token alone.

**Classes (fixed here).**
- `function`: a word token (v8 `klass` = word) whose stripped, lower-cased text is in the closed-class list in `hs_v10a_classes.py` (articles, determiners, pronouns, prepositions, conjunctions, auxiliaries and modals).
- `content`: every other word token.
- `digit`.
- `whitespace`.
- `punctuation` is reported as a reference row (already replicated in v8). It is not in the family.

**Rule (the v8 rule, verbatim, per class c and layer l).** Run `hs_flip5.run_layer` (variant d, the axis-aligned drop, 200 flips, 50 planted controls) on the class-c rows of the v8 dump. Then, for BOTH the norm and the xnorm estimators, require all of:
- λ1 above every flip;
- split@3 above every flip;
- split@3 ≥ 0.7;
- the planted control detected.

In addition, the cross-run top-5 overlap between the fresh and run-2 class-c patterns (norm estimator, same frame and coordinates) must be ≥ 0.5. Class c **replicates at l** iff all of these hold. Class c **passes v10a** iff it replicates at both L16 and L40. The family has 4 classes × 2 layers. Holm at 0.05 is automatically met by a pass (exact p = 1/201 < 0.05/8), and is stated for completeness.

**What a pass licenses.** Only that class c has a replicable, non-Σ-commuting second-order structure: the v8 claim, for class c. It does NOT show function. Every passing class's H-space_c (fresh top-5 patterns at each layer) goes, unchanged, into the v10 pod run. There it is tested by the v9-style rectangle against Σ_c-orbit twins on fresh text (the M8b confirmation; rule in `SCOPE-hspace-v10.md`).

**Reading.**
- No class other than punctuation passes: the replicable structure is punctuation-specific at this budget.
- ≥ 1 class passes: a candidate token-conditional family, pending the pod confirmation.

**Context.** This is the 12th test in the H-space sequence. Punctuation was chosen after a descriptive look at run-2 data (v7). The other classes have never been analysed on the v8 data. In v7, word rows on run-2 data reproduced the all-rows structure (overlap 0.81–0.96), and that structure failed the interaction test.
