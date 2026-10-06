# Preregistration: H-space one-shot on Qwen3.6-27B, plus a white-box control read (Oct 5 2026)

This file is frozen before the pod is created. Its sha256, and the sha256s of `hspace.py`, `ctrl_read.py` and `bf16w.py`, are written to `runlog.md` at freeze time. Nothing below may be changed after the 27B data exist. Any deviation is reported as a deviation.

## A. H-space (goal 1)

**Model.** Qwen/Qwen3.6-27B (64 layers, d = 5120).
- Weights are stored in bf16, the checkpoint's native format. All second-order computation runs in exact fp32 through `bf16w.BFLinear`; its HVP matches the fully-fp32 model to 7e-7 (`tests/test_bf16w.py`).
- Attention is eager. DeltaNet uses the patched torch path (`common.patch_deltanet`).

**J-lens.** The lens of record: `neuronpedia/jacobian-lens` `qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt` (target = block 63 output; sources 0–62).

**Data.**
- Estimation: wikitext-103 **train** windows, T = 128, skipping the first 16 positions.
- Evaluation: wikitext-103 **test** windows.

**Layers.** 16, 28, 40, 52.

**Definitions** (as in `hspace.py`):
- **H-space.** The top-25 eigenvectors of the whitened operator M = E_{x,c,ξ} mean_p y_p y_pᵀ, where y_p = Σ^{1/2} s_p (H_x[c] (s ⊗ Σ^{1/2} ξ))_p.
  - s are Rademacher position signs.
  - c is a vocabulary-weighted output covector, with E[ccᵀ] = (1/V) Σ_t u_t u_tᵀ.
  - The objective is F = Σ_{valid p'} c · h₆₃[p'].
  - Σ is the layer covariance with shrinkage 0.05.
  - 256 probes × 4 windows per layer.
  - M is the per-position row energy of the Hessian, which includes cross-position blocks.
- **J25.** The top-25 eigenvectors of the count-weighted sum of whitened unit J-lens vectors of each position's top-25 lens tokens (2,048 test positions): the paper's "active J-space".

**Criteria** (per layer, on 27B):
- **C0, instrument validity.**
  - Plant G(a·z)(b·z)w at layer 40 (z is the whitened residual; a, b are random orthonormal; w lies in the top-64 output-sensitive directions of J₄₁).
  - G is calibrated so that the planted eigenvalue equals the 25th eigenvalue of the unplanted M.
  - Pass: span(a, b) recovery by the top-25 is **≥ 0.8**. If C0 fails, the run is **VOID**.
- **C1, reproducible.** Overlap of the top-25 subspaces from the two disjoint probe halves is **≥ 0.7** (mean squared cosine; random is 25/5120 ≈ 0.005).
- **C2, low-dimensional.** The top-25 hold **≥ 50%** of trace(M).
- **C3, not J-space.** The H25–J25 overlap is **< 0.5**.
- **C4, real at natural scale.**
  - Pass: the median |I| of 1σ finite interactions along the top-6 H-directions (15 pairs) is **≥ 10×** that of 15 random whitened pairs.
  - I is measured on the summed log-likelihood of later tokens, over 12 test windows, with one random position each.
- **C5, used by the model.**
  - Mean-ablate H25 at every position of the layer (in whitened coordinates).
  - Pass: KL to the clean next-token distribution, averaged over valid positions of 512 test windows, is **≥ 3×** the mean KL of three random 25-dimensional whitened subspaces.
- **C6, double dissociation.**
  - **Labels.** For 512 test windows × 2 targets, apply a 2×2 removal of two 8-token spans (replaced by " the") and compute I = f(both) − f(−S1) − f(−S2) + f(neither) on the target log-prob.
  - Among targets with total effect at or above the median: interaction tokens are the top quintile of |I|/(|M1|+|M2|), and additive tokens are the bottom quintile.
  - **Damage** is the drop in target log-prob under the ablation.
  - **Pass:**
    1. ratio(H25) = damage on interaction tokens / damage on additive tokens is **≥ 2**;
    2. ratio(H25⊥J) is also ≥ 2;
    3. the lower 95% bootstrap bound of ratio(H25) exceeds both ratio(J25) and the mean ratio of the random subspaces.

**Verdict.**
- **VOID** if C0 fails.
- **PASS: "H-space found"** if C1, C2, C3, C4, C5 and C6 all hold at **≥ 2 of the 4 layers**.
- **"H-space exists but coincides with J-space"** if everything except C3 holds.
- **FAIL** otherwise. A FAIL is final for this definition and this model.
- The 9B replication I proposed earlier is dropped at the user's instruction to stay on 27B. Replication is within-model: across layers and across the split halves.

**Reported regardless of the verdict.** Eigen-spectra, participation ratio, the M_lens (position-averaged) subspace and its overlap with H25, J-lens decodes of the top-6 H directions, decoded natural-scale interaction vectors, and the KL of raw-space PCA25.

**Development evidence that is *not* part of the verdict** (0.8B, laptop):
- Theorem 1 holds to ≤ 1.5e-6.
- Anatomy: ~50% MLP, ~30% DeltaNet, ~20% attention.
- Integrated-vs-local: per-context r 0.57 vs 0.29 at feature scale; the corpus-averaged signed lens has r ≈ 0.3.
- The `hspace.py small` dev run (0.8B, layers 6/12/18; seen before freezing):
  - top-25 energy 0.59 / 0.63 / 0.61;
  - split-half 0.74 / 0.73 / 0.69;
  - H25–J25 overlap 0.54 / 0.52 / 0.53 (borderline for C3);
  - 1σ interaction ratio 40 / 40 / 69×;
  - KL(H25) 2.28 / 2.44 / 1.83, vs J25 1.31 / 1.40 / 1.34 and random ≈ 0.01–0.05;
  - dissociation ratio H25 1.00 / 1.32 / 1.39, vs J25 1.54 / 1.50 / 1.33 (only 20/20 tokens; C6 would fail here);
  - positive control 0.96.

  **The thresholds were NOT changed after seeing these numbers.** They are the ones proposed to the user before any H-space result existed.

**Descriptive-only additions** (added before freezing; not criteria):
- the share of trace(M) captured by J25, by global J25 and by random-25;
- the share of the J-lens Gram energy captured by H25 vs by J25.

## B. White-box control read (goal 3)

**Data.** `adityaasinha28/control_arena_bash`: 750 attack and 750 honest BashArena trajectories from Claude Sonnet 4, Gemini 2.5 Pro and GPT-5, with an identical system prompt for both classes (checked: no label leak).

**Reader.**
- Qwen3.6-27B reads each trajectory as its own, under a generic system prompt, truncated at 16k tokens.
- Dumped per trajectory: residual states at layers 16, 28, 40 and 52 at every action start (the token before the action is generated) and the mean over each action, plus a same-model black-box verdict, logit(Yes) − logit(No), from a monitor question appended after the trajectory.
- This is a **surrogate** white-box reader (the trajectories come from other models), not the agent's own activations. It is reported as such.

**Analyses** (offline on the laptop; 5-fold cross-fitting stratified by source model):
- **T1, workspace dimension (theory §7 G2).** The zero-shot, honest-calibrated mean and spike statistics in J25 and in H25 vs the same statistics in full d and in random-25. Prediction: AUROC(J25 or H25) > AUROC(full d) > AUROC(random-25).
- **T2, fusion no-harm (G1).** At audit budgets of 1/2/5% trajectory FPR, count attacks lost vs BB-only for LR-fused (cross-fitted logistic on [BB, WB]) vs additive z-sum. Prediction: LR fusion loses ≤ additive fusion and never lowers TPR by more than sampling noise (one attack).
- **T3, added value.** A logistic probe on pooled J25 ⊕ H25 coordinates, LR-fused with BB. Report the attacks rescued at 2% audit (caught by the fusion, missed by BB).
- **T4, pre-emption.** Among attacks caught by WB at 2% FPR: the median number of actions between the WB alarm and the first attack-signature action (regex on action text). Report positive leads.

## C. Spend rules
- **One pod.** RTX PRO 6000 96 GB community ($1.69/h), or H100 NVL as fallback. Container disk only: no network volume, no stop.
- **Watchdog.** It terminates at 3.5 h. Credentials come from `/proc/1/environ`.
- **No pilot pod, no retries.** The ≤ 60 s local smoke run (`hspace.py smoke`, `ctrl_read.py small`) passed before creation.
- **Order.** `hspace.py full`, then `ctrl_read.py full`. Results are pulled and their sha256 checked, then the pod is terminated and termination is confirmed by `list-pods`.
- **Cap.** $7 for this pod ($16.99 total available).
