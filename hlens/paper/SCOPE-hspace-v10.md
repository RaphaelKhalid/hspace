# SCOPE v10: is the frozen H-space real, used, and distinct from J-space? (registered report)

*Frozen and pushed before the pod exists and before any 27B data on these tests is computed. 13th test in the H-space sequence (v10a, the token-class scan, was the 12th). Code: `hs_v10.py`; banks: `v10_banks/` (sha256 in `MANIFEST.json`, asserted at run time); all hashes are in `results/runlog_laptop.md`. Plan: `research/proposals/40-hspace-shopping-list.md` (MUST M1–M8, addenda A/B). A three-lens adversarial review (stats/design, code against prereg, runtime) ran before freezing. All of its blocking and major findings are fixed below; see Deviations.*

**Naming.** **H-space** := the top-5 patterns of the punctuation-conditional H-lens from the fresh v8 data, at L16 and L40, in run 2's whitened frame. The v8/v9 preregistrations and code call it EX; EX ≡ H-space. It is **frozen**: never re-estimated here. Like J-space, the name is a definition, not a claim. The tests below decide what can be claimed.

**Model and compute.** Qwen3.6-27B on one RTX PRO 6000. The weights are bf16 with exact fp32 compute; TF32 is off; log-probabilities are float64. The script runs ≤ 170 min, inside a hard pod watchdog of 200 min. Stages run in priority order, with time caps; every stage saves incrementally. W, G and T are each guarded, so R and the verdict always run.

## Stages

**S0. Class moments, M2 gate, controls.**
- **Moments.** 768 fresh wikitext-103 *train* windows (T = 128, seed 11), with rows at positions 16..126, give per-class means μ_c and covariances Σ_c (5% shrinkage to scaled identity). The classes are punctuation, word, function, content, digit and whitespace.
- **M2 (kill check, per layer).** C̃ = Sh⁻¹Σ_punct Sh⁻¹, with V its eigenvectors.
  - ov25 = ‖V₂₅ᵀH‖²/5.
  - mean-energy = ‖Hᵀm̃‖², with m̃ = unit(Sh⁻¹(μ_punct − μ)).
  - The reference is 200 global Σ-orbit twins of H.
  - **KILL at l iff (ov25 ≥ 0.25 and ov25 > the max reference) or (mean-energy ≥ 0.25 and mean-energy > the max reference).**
  - If either value exceeds the reference 95th percentile without a kill, the forced qualifier **"punctuation-moment-aligned"** applies.
- **Controls.** All are raw 1σ moves Sh·q with q whitened.
  - **16 punctuation twins** q_j = V diag(s_j) Vᵀ H: sign flips in the eigenbasis of the *whitened* punctuation covariance. These match H exactly in global moments (1σ norms, Gram matrix) and in punctuation-conditional second moments.
    - 8 are plain.
    - 8 are J-matched: from 3,992 further candidates, the 8 with the smallest maximum relative deviation of J-lens response energy ‖J_l Sh q‖² from H's, over the 5 directions and the 3 matched pair sums. J_l is the raw neuronpedia J-lens of record.
  - **8 word twins**, built the same way from the whitened word covariance.
  - **8 global twins**: v9's exact twins, as a continuity arm.
  - **J5** = the top-5 columns of my whitened J25.
  - **PC5** = the top-5 eigenvectors of C̃.
- **Twin diagnostics.** The twins do not match two things: the punctuation-mean energy (qᵀm̃)² and the raw column norms ‖Sh q_i‖. Both are logged for H and every twin.
  - If H exceeds all 16 twins on either, the forced qualifier **"mean/norm-aligned"** applies.
  - T1 against the 8 twins closest to H on these two is reported.

**W. Wikitext (M3, M8a, sink, continuity).**
- **Data.** 40 wikitext-103 *validation* windows (T = 128, seed 10), at article offsets k ≥ 2. v9 used k = 0 and 1 and consumed all 60 long validation articles, so this text is disjoint from v9's.
  - Each window gets a random punctuation position p and word position q in 24..107.
  - Clusters are articles.
  - The run asserts it has 40 windows.
- **Measure.** The v9 rectangle, unchanged: 1σ moves, matched pairs MP, f = summed log-likelihood of the next 16 tokens. I and I_trunk are as in v9.
- **Arms at each layer:**
  - **main at p:** H, J5, PC5, the 16 punctuation twins and the 8 global twins;
  - **RMS-freeze at p:** H plus the 8 plain twins. Every RMSNorm, gated-RMSNorm and l2norm denominator in blocks > l is frozen at its unperturbed value. The final norm is not frozen. The run errors if the frozen scales are not all used;
  - **word at q:** H plus the 8 word twins;
  - **sink at p:** the attention weight on p from later queries, over full-attention layers > l.
- **Statistics.**
  - T1 = mean over windows and twins of log(mean_MP|I_H| / mean_MP|I_tw|).
  - 95% CI: a two-way bootstrap over articles and twins, 2,000 reps.
  - **p_PI**: a one-sided prediction-interval p of T1_H against the leave-one-out twin distribution T1_j (each twin against the other K−1), with t on K−1 df.
  - rank = H's position among H and the K twins by mean |I|.
- **M3 (per layer).**
  - **pass iff**, for both I and I_trunk against the 16 punctuation twins, T1 > log 2, CI low > 0, rank 1/17 and p_PI < 0.05; **and** the freeze arm is valid; **and** retention = T1_trunk(frozen) / T1_trunk(unfrozen) ≥ 0.5, on the same 8 plain twins.
  - The freeze arm is **valid** iff, in every window, |f₀(frozen) − f₀(unfrozen)| ≤ 1e-4·max(1, |f₀|).
  - Retention is defined only if the unfrozen T1 > log 2; if it is not, M3 fails on T1 anyway.
  - **inconclusive** if n < 24 windows or the freeze arm is invalid. Otherwise **fail**.
- **Sink scope.** If T1 in the lowest-sink tercile < 0.5 × T1 overall, the forced qualifier is "attention-sink curvature".
- **M8a.** H is "position-general" at l iff the word-arm T1 > log 2, CI > 0 and rank 1/9. Otherwise it is "punctuation-specific".

**G. H-blind task gates.** The banks are in `v10_banks/`, and their sha256 is asserted at run time.
- **Banks:**
  - primary: negation_truth, entity_attribute, relational_composition;
  - backup: binding_backup;
  - negative control: null_unrelated.
- **Sampling.** Up to 40 items per bank (24 for the null), shuffled with seed 42. Each bank has its own time cap, and the null runs before the backup.
- **Definitions.**
  - f = log p(X) − log p(Y) (first tokens) at the last prompt token. Prompts are right-padded to T = 32.
  - **site** = the last punctuation token before the final token.
  - **natural site** = the last token of the item's natural-site string.
  - DiD = f₁₁ − f₁₀ − f₀₁ + f₀₀.
- **Behavioural gate (M6 family):** the cluster-bootstrap 95% CI of mean DiD excludes 0. If fewer than 3 primaries pass, the backup is added if it passes.
- **Site gate at l (M4/M5).**
  - Patch the layer-l site states of the (1,0), (0,1) and (1,1) prompts into (0,0).
  - DiD_site = f(+δAB) − f(+δA) − f(+δB) + f₀₀.
  - The gate requires mean DiD_site / mean DiD ≥ 0.3.
  - For each main effect whose cluster CI excludes 0, the site patch must also recover ≥ 30% of it, as a ratio of means.

**T. Task arms** (behaviourally gated contexts plus the null; both layers). Clusters are the bank's cluster field; relational uses entity families.
- **Transfer (M6):** the rectangle at P₁₁'s site on f. H against the 16 punctuation twins, with T1, CI, p_PI and rank as in W.
- **Gold rank (M7).** Rank the cell-(1,1) gold token (X, or Y for binding) in the interaction logit vector L(u+w) − L(u) − L(w) + L(0), averaged over MP.
  - Per item, compare H's rank with the median twin rank.
  - Average per cluster, then run a one-sided Wilcoxon test over clusters (≥ 6 clusters; otherwise p = 1).
- **Routing (M4), on P₀₀ at its site.**
  - I = f(+δA+δB) − f(+δA) − f(+δB) + f₀₀.
  - Curvature share = mean I / mean DiD_site, with a cluster-bootstrap CI.
  - For S ∈ {H, J5, 16 twins}, deletion uses I_del = I(P⊥δA, P⊥δB), with projections in the whitened frame.
  - removed_frac(S) = mean(I − I_del)/mean I.
  - D(S) = removed_frac / mean((e_A + e_B)/2), where e is the share of the move's whitened energy in S.
  - robust z = (D_H − median D_twins)/(1.4826 · MAD of D_twins).
  - Secondary: sufficiency (kept_frac; A3), and the AtP (first-order) error removed (A1).
- **Ablation (M5).** For S ∈ {H, J5, 8 plain twins}, set S's whitened site coordinates to the item's 4-corner mean in all four prompts.
  - int_loss = 1 − mean DiD_abl / mean DiD.
  - main_loss = 1 − (Σ ablated sign-aligned main effects)/(Σ base).
  - **Δ = int_loss − main_loss.** φ = int_loss/main_loss is reported as descriptive only. The dose is reported.
- **Natural-site routing (M8c):** descriptive.
- **TOST (null).** At each layer, compute (H's removed interaction − the mean twin's) ÷ mean |DiD| over the gated contexts' items, with a cluster-bootstrap 90% CI.

**R. Random-weights null** (last). Shuffle the entries of every ≥2-D parameter in blocks > l: 41–63 first, then 17–40, cumulatively. Rerun the W main rectangle at p, for H plus the 8 plain twins, on the first 24 W windows. Compare with the trained T1 on the same windows and twins.
- **fail (KILL)** iff T1_random ≥ 0.5 × T1_trained, or T1_random or T1_trained is not finite.
- **inconclusive** if n < 12, or the median twin |I| under random weights is < 1% of the trained value.
- Otherwise **pass**.

## Pass rules and verdict (each check is pass / fail / inconclusive)

- **M2:** fail iff a layer is killed.
- **M3, R:** pass iff they pass at both layers; fail if either layer fails; otherwise inconclusive.
- **M6.**
  - Apply Holm over (gated contexts × both layers) using p = max(p_boot, p_PI) on I. Missing cells get p = 1.
  - A cell passes iff it is Holm-significant, T1 > log 2 and rank 1/17.
  - **pass iff ≥ 3 gated contexts pass at both layers.** Inconclusive if fewer than 3 contexts are gated; otherwise fail.
- **M4 at (context, layer).**
  - **evaluable** iff it is site-gated and the cluster CI of mean DiD_site excludes 0. Otherwise inconclusive.
  - **pass iff** curvature share ≥ 0.25 with CI low > 0, removed_frac(H) ≥ 0.10 with cluster-CI low > 0, robust z ≥ 3, and D(H) ranks first among H, J5 and the 16 twins.
  - **M4 pass iff ≥ 2 contexts pass at ≥ 1 layer.** Inconclusive if fewer than 2 contexts are evaluable; otherwise fail.
- **M5 at (context, layer).**
  - Inconclusive unless site-gated.
  - **pass iff** int_loss_H > 0, Δ_H > every plain twin's Δ, Δ_H > Δ_J5, and the cluster-bootstrap CI low of (int_loss_H − mean twin int_loss) > 0.
  - **M5 pass iff ≥ 2 contexts pass at ≥ 1 layer**, with the same inconclusive rule as M4. The H/J double dissociation is reported.
- **M7:** Holm over (gated contexts × both layers). **pass iff ≥ 1 cell is Holm-significant.** Inconclusive if no context is gated.
- **TOST:** pass iff the 90% CI lies within ±0.10 at both layers. Inconclusive if it is missing.
- **MUST verdict.**
  - **"I found H-space" iff M2, M3, R, M6, M4, M5, M7 and TOST all pass.**
  - Otherwise: "MUST tier not passed. fail: […]; inconclusive: […]", stated for this version only (frozen k = 5, L16/L40, punctuation-estimated), never as "there is no H-space".
- **Forced qualifiers** are always printed:
  - "attention-sink curvature";
  - "position-general" or "punctuation-specific" (per layer);
  - "mean/norm-aligned";
  - "punctuation-moment-aligned";
  - "subspace (k = 5 fixed)".

## Deviations from the plan (all pre-data, recorded here)

1. **Twin frame.** Twins are built in the *whitened* frame, as the eigenbasis of the whitened class covariance. A 0.8B null run with raw-frame reflections let a random 5-dim subspace beat its twins (T1 +1.03, rank 1/17); whitened twins give T1 +0.27 [−0.23, 0.72], rank 5/17. A two-layer null smoke of the final code gives +0.09 and +0.28 (ranks 9/17 and 3/17), and M3 fails, as it should.
2. **No class arm.** v10a passed no class (M8b).
3. **M5.** The reference is the per-item 4-corner (factor-erased) mean. The statistic is Δ = int_loss − main_loss, not φ: the review showed φ is decided by noise in a near-zero main_loss. Dose-matching and Fieller CIs are not used. The direct-effect variant (A2) is not run.
4. **Random-weights null.** H stays frozen; it is not rediscovered. The method-level null at 0.8B is deferred.
5. **Gates.**
   - A behavioural DiD gate decides the M6 family.
   - The site gate (M4/M5) uses DiD-site recovery, because main-effect recovery is ill-posed for AND-shaped banks whose main effects are ≈ 0 by design.
6. **Banks.** `fix_banks.py`, run pre-data:
   - moved the site after the factor words in entity_attribute and binding_backup;
   - gave per-cell natural sites, with the null's being the bare variant word;
   - set relational clusters to entity families.
   `MANIFEST.json` holds the post-fix hashes.
7. **W windows.** W uses validation offsets k ≥ 2 rather than "unused articles": v9 consumed all 60 long validation articles at k = 0 and 1. Without this change W would have had 0 windows (review, blocking).
8. **Review-driven statistics:**
   - p_PI is added to M3 and M6, because a twin-mean bootstrap alone is anti-conservative;
   - D is normalised by (e_A+e_B)/2, and robust z replaces "≥ 3× median";
   - M7 uses cluster-level Wilcoxon tests and Holm;
   - TOST is in units of task DiD;
   - R has NaN/n/floor rules;
   - verdicts are three-valued;
   - the M2 kill needs both ≥ 0.25 and the 200-twin maximum (the plan said "outside the twins' 95% range"; that range is used for the qualifier).
9. **Not run tonight:**
   - the GPT-2 known-answer battery, the k-curve, attention/DeltaNet freezes and the temperature split;
   - the blind judged decode (S1), the second model (S2) and agentic misalignment (N3);
   - the AtP head-output control and the out-of-sample check of twin moments.
   M6 power has not been simulated, so treat M6 as low-power.

## What a pass would and would not license

A pass licenses "I found H-space in Qwen3.6-27B", in this sense: a frozen, 5-dim, punctuation-estimated second-order subspace that has all of these properties.
- It is not punctuation-moment geometry, norm geometry or random-network geometry.
- It carries excess natural-scale interaction against covariance-matched, J-matched and global controls, as an outlier among its twins.
- It transfers to ≥ 3 externally defined task conjunctions.
- The model routes the interaction through it, and ablating it costs interaction more than main effects, unlike J5.
- Its interaction logits rank the conjunction's answer.

It does **not** license generality across models or a decoded meaning; those are the "general" and "workspace-like" tiers. A partial pass is reported check by check, with the forced wording.
