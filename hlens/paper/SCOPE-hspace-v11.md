# v11 registered report: is position-general H-space real, used, and distinct from J-space?

*Frozen and pushed before the pod exists and before any 27B data on these tests is computed. 15th test in the H-space sequence.*

*Code: `hs_v11.py`. It imports the frozen v10 machinery unchanged (`hs_v10.py` f8f550656772e1f7). Inputs:*
- *`out/v11a_subspaces.pt` (sha256 prefix 29e0b7cb95991011; shipped in the pod bundle and hash-checked on the pod);*
- *`hspace2_full.pt`;*
- *`v10_inputs_full.pt`;*
- *the v10 banks (sha256s asserted against `v10_banks/MANIFEST.json`).*

*Hashes are in `results/runlog_laptop.md`. A three-lens adversarial review ran before freezing; its findings and my fixes are under Deviations.*

## Why v11

v10 found the frozen punctuation H-space **real**: M3 passed (7.3× at L16, 3.4× at L40 against 16 matched twins), and R and TOST also passed. But the MUST tier did not pass:
- M4 and M5 (used) were **untestable**. The probe site, the last punctuation token, carried only 0–8% of each task's interaction at L16 and L40.
- M6's transfer showed up equally on the unrelated null bank (4× at L16), so it measures generic curvature, not task use.
- M7 failed.

v11a then found a top-5 all-position H-space subspace that replicates across independent runs (overlap 0.94 at both layers). The frozen v11a rule, however, called it "position-general at L40 only (punctuation-carried)".

v11 asks the questions that separate "a curved direction set" from "a J-space-like space": is H-space **used** where the model actually computes a two-factor interaction, and is it **distinct** from J-space?

## Objects (whitened run-2 frame; none is re-estimated)

| Name | Definition | Role |
|---|---|---|
| **H** | `H_all_{l}` from `v11a_subspaces.pt` | primary |
| **HJ** | `orth((I − Q_J Q_Jᵀ) H)`, Q_J the orthonormalized J25 (`J25w_{l}`) | geometric distinctness |
| **EX** | v10's punctuation H-space | descriptive |
| **J5** | the top-5 J25 columns, orthonormalized | functional comparison |

**Known before any data** (from the reviewers' CPU checks on frozen inputs):
- H's overlap with span(J25) is 0.060 at L16 and 0.019 at L40.
- HJ keeps 87–99% of each H column, so HJ ≈ H, and M3res is a geometric check rather than an independent test.
- Functional distinctness rests on M4's and M5's J5 contrasts.
- J5's raw 1σ norms are 3–4× H's, which inflates J5's effects and is conservative against H.

**Twins.** These are v9's global Σ-orbit twins in the whitened frame: `twins_w(·, Pg, ·)`, with Pg the eigenvectors of Sh.
- **What they match exactly:** each twin matches its object's Gram matrix, its support in the Sh eigenbasis, and its whitened and raw 1σ norms.
- **Sets:** 16 for H (tw0–15, seeds 11100+l) and 16 for HJ (tx0–15, seeds 11200+l). tw0–7 and tx0–7 are the 8-twin subsets used for the freeze arm, for M5 and for R, as in v10.
- **Moves:** every move is a raw 1σ move, `Sh @ v`.

## Stages (priority order; each guarded; R last)

**S0. Moments and diagnostics.** All-position and punctuation class moments from 768 fresh wikitext-train windows. Recorded: H∩J25, HJ retention, H∩EX, raw norms, and the twins' all-position moment deviation.
- **M2′ is descriptive only.** It cannot fire by construction, because H lies in drop_mask's kept span, which excludes Sh⁻¹μ.
- **Qualifier "mean/norm-aligned"** (v10's rule): fires if H's whitened mean-direction energy exceeds every tw's, or its raw norm exceeds every tw's.

**L. H-blind localization.** Done before any H statistic exists; H is never used here.
- **Banks:** v10's negation_truth, entity_attribute, relational_composition, with binding_backup as backup.
- **Items:** shuffled with seed 43. The first 16 preppable items per bank form split A; the next 32 form split B.
- **Candidate sites:** suffix offsets o = 1…6 counted from the last token, used only inside the item's shared suffix. The position in prompt c is n_c − o.
- **Per (layer, o), the 10, 01 and 11 deltas are patched into 00, giving:**
  - DiD_site = f(+δAB) − f(+δA) − f(+δB) + f₀₀;
  - the main-effect recoveries;
  - **I_site** = f(+δA+δB) − f(+δA) − f(+δB) + f₀₀, the curvature at that site.
- **Site thresholds** (v10's): DiD_site/DiD is at least 0.3, and every main effect whose cluster CI excludes 0 is recovered at least 0.3.
- **o\* selection.** Among the offsets that meet the thresholds (at least 75% item coverage), o\* = argmax of mean I_site / mean DiD on split A. This picks the place where the interaction itself lives, not merely where its result is carried.
- **Site gate.** A (bank, layer) cell is site-gated iff some offset meets the thresholds and the bank passes the behavioural gate (the cluster CI of mean DiD on A excludes 0).
- **Backup** as in v10.
- **Null offset.** At each layer, the null bank uses the most common o\* among the site-gated **primary** cells, clipped to the null bank's shared suffix (4 tokens). Layers with no site-gated primary cell get no null control.

**T. Used tests** on split B at o\*, using v10's formulas.
- **Routing (M4).** For S ∈ {H, HJ, J5, EX, tw0–15, tx0–15}, delete S's whitened component of δA and δB.
  - removed_frac(S) = (I − I_del)/I; e(S) is S's energy share of δ; D = removed_frac / e.
  - **Evaluable iff all of:**
    - the cell is site-gated;
    - the B-split CI of DiD_site excludes 0;
    - the curvature share I/DiD_site is at least 0.25, with CI low > 0.
  - A site with no curvature cannot route curvature, so that case is **inconclusive, not fail**.
  - **M4(H) passes iff** removed_frac(H) ≥ 0.10 with CI low > 0, the robust z of D(H) against tw0–15 is at least 3, and D(H) ranks first among {H, J5, tw0–15}.
- **Ablation (M5).** For S ∈ {H, HJ, J5, EX, tw0–7, tx0–7}, set S's whitened site coordinates to the item's 4-corner mean.
  - **M5(H) passes iff all of:**
    - int_loss_H > 0;
    - Δ_H exceeds every tw0–7 Δ;
    - Δ_H > Δ_J5;
    - the cluster CI low of (int_loss_H − mean twin int_loss) is above 0.
  - Inconclusive unless the cell is site-gated.
- **HJ** gets the same M4/M5 rules against tx (secondary).
- **TOST (negative control).** null_unrelated, 24 items, at the null offset.
  - diff = [(I − I_del,H) − mean over tw0–15 of (I − I_del,tw)] / mean |DiD| of the gated banks.
  - Equivalent iff the 90% cluster CI lies within ±0.10 (v10's rule).

**W. Real and distinct**, on fresh text.
- **Windows:** 40 wikitext-validation windows at offsets k ≥ 5, seed 11. v9 used k = 0, 1 and v10 used k = 2, 3. Windows identical to v10's 120 candidates are excluded and counted; 0 are expected.
- **Position:** one random position per window, from **all** tokens in 24…107.
- **Rectangles:** v10's rectangle interaction on the next 16 tokens, for the arms H, HJ, EX, J5, tw0–15 and tx0–15, plus an RMS-freeze arm for H and tw0–7.
- **M3 (H real), v10's rule:** both I and I_trunk are strong against tw0–15, the freeze is valid, and retention is at least 0.5.
  - "Strong" means T1 > log 2, CI low > 0, rank 1 of 17, and p_PI < 0.05.
  - Retention is the freeze T1 over the plain-8 T1, both on I_trunk.
- **M3res (geometric distinctness):** HJ against tx0–15, with I and I_trunk both strong.
- **Qualifier:** "excess at non-punctuation positions" if H against tw0–15 on the non-punctuation positions has T1 > log 2, CI low > 0 and rank 1. Otherwise "punctuation-carried in W".
- **Descriptive:** log(H/J5), log(HJ/J5), log(EX/H), and interaction per main effect.

**R. Random weights**, as in v10. Shuffle blocks > l, L40 first. Then H against tw0–7 on the first 24 W windows: pass iff the random T1 < 0.5 × the trained T1. Inconclusive if the twins' |I| collapses below 1%.

## Verdict (v10's aggregation)

- **MUST tier:** M3, R and M3res at both layers (as in v10: any fail is a fail, all pass is a pass); TOST at every layer with a site-gated primary cell (same aggregation; inconclusive if there is no such layer); plus M4 and M5.
- **M4 and M5** each pass iff at least 2 gated contexts pass at one or more layers. They are inconclusive if fewer than 2 contexts are evaluable, and fail otherwise.
- **"I found H-space"** iff every MUST check passes. The forced qualifiers are always reported:
  - the localized sites;
  - k = 5;
  - "v11a object: position-general at L40 only, punctuation-carried at L40";
  - "distinct = geometric + functional (J5 contrasts)";
  - mean/norm-aligned, if it fires;
  - the non-punctuation qualifier.
- **Otherwise**, the verdict lists every failed and inconclusive check.

## Budget and operations

- **Hardware:** one RTX PRO 6000 Server Edition (Secure), after the GEMM health check, which must exceed 45 TFLOPS.
- **Time caps (minutes):** S0 10, L 20, T 50, W 50, R 14; overall 160. The pod watchdog is 190 min.
- **Expected run time:** about 95 min, roughly $3.5.
- **Relaunches:** technical relaunches only, on identical code.

## Deviations (written before freezing)

**Differences from v10's bar, all decided before any v11 data:**
- **(a) M6 and M7 are dropped.**
  - M6 measures generic curvature: v10's null bank gave the same 4× transfer.
  - M7 tests steering toward the gold answer, which a curvature space is not expected to do.
- **(b) M2 becomes M2′, descriptive only.** It cannot fire by construction for H_all.
- **(c) M4 evaluability now includes the curvature-share condition**, which v10 had among M4's pass conditions. No curvature at the site now gives inconclusive, not fail.
- **(d) The site is chosen by an H-blind localization scan**, not fixed at the last punctuation token. Sites are selected on split A and tested on split B.
- **(e) M3res is new.**
- **(f) The H twins are global Σ-orbit twins** (v9's construction), not punctuation-class twins. The object is position-general, and the review showed that whitened all-position class twins do not match H's support or raw norm: about 2× smaller, which would have biased the test toward H.
- **(g) The object changes.** v11a's H_all replaces v10's punctuation EX as the primary object. EX is reported descriptively.

- **(h) TOST is required at every layer where routing is tested**, i.e. every layer with a site-gated primary cell. v10 required both layers, but there the site was fixed.

**Unchanged from v10:** 16 twins for M3 and M4 (rank of 17); M3 and R required at both layers; M4/M5 at ≥ 2 contexts at ≥ 1 layer; the M5 rule; R's rule; TOST's ±0.10 bound.

**Pre-freeze disclosure.** To check M2′, the reviewers computed from frozen inputs only (no model run):
- H's whitened mean-direction energy is 0.0084 at L16 and 0.0053 at L40. That is 2.9× and 6.8× above the maximum of 200 Σ-orbit refs, but far below the unreachable 0.25 floor.
- The "mean/norm-aligned" qualifier is therefore likely to fire.
- No 27B forward pass or outcome statistic of v11 has been computed.
