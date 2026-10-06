# H-space: theory (working draft, Oct 5 2026)

Status markers: **[proved]** = an exact identity with a proof, **[verified]** = checked numerically (the number is given), **[assumption]** = an approximation whose error we measure, **[open]** = not yet settled.

## 0. Objects

### Model
- Residual states h_k ∈ R^{T×d}, where "layer k" means the output of decoder block k (the jlens convention).
- Each block is two residual sublayers, y ↦ y + φ_s(y): a token mixer, then a SwiGLU MLP.
- The mixer is either gated softmax attention or Gated DeltaNet.
- Target layer: L* = the last block output, before the final norm.

### The J-lens (Anthropic 2026; the jlens fitting estimator)
  J_ℓ = E_x mean_{p∈V} ∂/∂h_ℓ[p] Σ_{p'∈V} h_{L*}[p']        (d×d)
- V is the set of valid positions (16 … T−2).
- The decode is logit_t(h) ≈ u_tᵀ J_ℓ h, where u_t = (1+w_final) ⊙ W_U[t].

### The H-lens (this work)
- **Definition.** It is the same reduction one derivative higher:
  T_ℓ = E_x mean_{p∈V} ∂²/∂h_ℓ[p]² Σ_{p'∈V} h_{L*}[p']        (output ⊗ Sym(input ⊗ input), d×d×d)
- **Token-resolved slice.** H_ℓ^{(t)} = u_tᵀ T_ℓ ∈ Sym(d) is the averaged Hessian of token t's logit with respect to one source position.
- **Per-position.** The source perturbation sits at a *single* position p, exactly as in the J-lens. Cross-position blocks ∂²/∂h[p]∂h[q] are a separate object.
- **Units.** Whitened: H̃ = Σ^{1/2} H Σ^{1/2}, where Σ is the activation covariance at layer ℓ. The v1 pilot showed that raw curvature-per-unit² is dominated by directions activations never use.

### H-space
- **Energy operator.** M_ℓ = E_{c} [ H̃_ℓ[c] H̃_ℓ[c] ], where c ~ N(0, Σ_U) and Σ_U = (1/V) Σ_t u_t u_tᵀ: an output covector weighted by how much the logits care.
- **Definition.** H-space_k = the top-k eigenspace of M_ℓ. By default k = 25, matching J-space's "≤ 25 active directions".

## 1. Theorem 1: block decomposition (second-order adjoint) [proved] [verified]

**Statement.**
- Let F = c-contraction of h_{L*}, and write the network after layer ℓ as a composition of residual sublayers y_{s+1} = y_s + φ_s(y_s).
- Let g_{s+1} = ∂F/∂y_{s+1} (the ordinary adjoint) and J_{s←ℓ} = ∂y_s/∂h_ℓ. Then, exactly:

  ∇²_{h_ℓ} F = Σ_s J_{s←ℓ}ᵀ · ∇²_y [ g_{s+1} · φ_s(y) ]|_{y_s} · J_{s←ℓ}.

**Proof.**
1. The second-order chain rule for a composition is D²(F∘G)[v,w] = DF·D²G[v,w] + D²F[DG v, DG w].
2. Apply it to F∘(y ↦ y + φ_s(y)) and induct over s.
3. The identity part of a residual sublayer has zero curvature, so each sublayer contributes exactly its branch's curvature, contracted with the adjoint at its output and pulled back by the Jacobian at its input. ∎

**Equivalent recursion.** In forward-over-reverse form:
- forward: carry ḣ_s = J_{s←ℓ} v;
- backward: carry the pair (g_s, ġ_s), with ġ_s = J_sᵀ ġ_{s+1} + C_s[g_{s+1}] ḣ_s.

This is Pearlmutter's (1994) R-operator, so the identity itself is not new. What it buys:
- **Attribution.** The exact H-lens splits into per-sublayer pieces.
- **Streaming.** Exact HVPs can be computed one block at a time.

**Verified on real Qwen3.5-0.8B** (`hs_anatomy.py`): the sum of the 2(L*−ℓ) pulled-back local terms equals the exact double-backward HVP to a relative error of **1.2e-6** (fp32, T = 96, layer 18).

## 2. Theorem 2: anatomy of local curvature in a gated transformer [proved for MLP and norm; structure for mixers]

Every operation in a block is linear, except for:
- elementwise maps: SiLU, sigmoid, exp, softplus;
- products of two linear reads: gate×up in the MLP, gate×attention-output, q·k inside softmax, the attention weight×value, and the k⊗v state write in DeltaNet;
- normalisations: RMSNorm, the per-head q/k RMSNorm, and the L2 norm in DeltaNet.

Linear maps contribute zero curvature. So:

**Corollary 2.1 (inventory).** The local curvature of a sublayer is a finite sum over its nonlinear sites. Each site's contribution is its scalar second derivative, sandwiched between the read covectors of its inputs. There are exactly three site types:
- **AND atoms:** a product of two different reads, p(x) = (a·x)(b·x). Their curvature is sym(a bᵀ), a 2-D "if a and b" interaction.
- **Threshold atoms:** a scalar nonlinearity of one read, σ(a·x). Their curvature is σ''·a aᵀ, a rank-1 self-curvature.
- **Normalisation atoms:** RMSNorm and L2 norm. Their curvature involves the token's own direction; it is the curvature of scale invariance.

### 2.1 SwiGLU MLP, exact closed form [proved] [verified 4e-16 on a toy; 1.1e-6 on real Qwen3.5-0.8B blocks via `tests/test_swiglu.py`]
- **Neuron.** Neuron i computes y_i = silu(z_i)·u_i, where z_i = w_gᵢ·x̃, u_i = w_uᵢ·x̃ and x̃ = N(y). It writes along w_downᵢ.
- **Its curvature.** With adjoint g and ρ_i = w_downᵢ·g:
  ∇²_x̃ (g·MLP) = Σ_i ρ_i [ silu''(z_i) u_i · w_g w_gᵀ + silu'(z_i) · (w_g w_uᵀ + w_u w_gᵀ) ]
- **Reading it.** Each neuron is one AND atom (gate read × up read) plus one threshold atom on its gate read.
- **Back to the residual.** Pulling through the RMSNorm Jacobian DN = r·diag(γ)(I − (r²/d) y yᵀ) adds an explicit norm-curvature term (`ladder.py` docstring).

**Relation to prior work.**
- When SiLU is replaced by the identity, this is exactly the bilinear-MLP interaction matrix of Pearce et al. 2024 (arXiv 2410.08417).
- Our extensions:
  1. the SiLU coefficient functions;
  2. the norm terms;
  3. averaging over a corpus;
  4. transport and decoding through the J/R-lens (§5);
  5. the mixers (§2.2).

### 2.2 Gated softmax attention (1 block in 4 in Qwen3.5/3.6/3.8)
- **Head.** The output at query position q is o_q = W_O[(Σ_j a_qj V_j) ⊙ σ(G x_q)], with a_qj = softmax_j(Q_q·K_j/√d_h).
- **For a source position p, there are three families of AND atoms:**
  1. **key×value at later queries q > p.** a_qp depends on x_p through K_p, and V_p is a linear read of x_p. So x_p's curvature contains sym((W_Kᵀ Q̂_q)(W_Vᵀ ω_q)ᵀ) · a_qp(1−a_qp)/√d_h, where ω_q = W_Oᵀ g_q ⊙ σ(G x_q). In words: *"if a later token's query matches me AND I carry content v, write."*
  2. **gate×value / gate×query at q = p.** These come from the output gate σ(G x_p) multiplying the attention output, which depends on x_p through Q_p and the self term.
  3. **query×query.** The softmax curvature in Q_p is a covariance of the attended keys weighted by values.

  Plus threshold atoms (σ'' of the gate) and normalisation atoms (q_norm, k_norm, RMSNorm).
- **Averaged form (§4).** Under decoupling, family 1 for head h reduces to a d_h×d_h statistic, M_h = E Σ_{q>p} a_qp(1−a_qp) σ(Gx_q) ⊗ Q̂_q^{(p)} (RoPE-rotated into p's frame). For Qwen3.6-27B that's 256² numbers per head.

### 2.3 Gated DeltaNet (3 blocks in 4)
- **Mechanism.** The state write is β_p (v_p − S_{p−1}k_p) k_pᵀ, decayed by exp(Σ g). It is read at q ≥ p as S_q q_q, and the output passes through the gated RMSNorm, ⊙ silu(z_q).
- **Atoms:**
  - AND atoms key×value, β×value, β×key and query×key, all with coefficients set by the later queries and decays;
  - the conv1d (kernel 4) spreads a source perturbation over positions p … p+3 before the SiLU, which gives cross-position threshold atoms;
  - normalisation atoms from the L2 norms on q and k.
- **How we compute it.** By exact local autograd per block (cheap: one block, no full-network graph), not by a closed form.

**Measured share of the per-position H-lens by sublayer type** (`hs_anatomy.py`, Qwen3.5-0.8B; the share is ⟨piece, total⟩/‖total‖² at the source position, so pieces sum to 1):

| source layer ℓ (of 24) | MLP | full attention | DeltaNet | MLP in nearest third of downstream blocks | position-diagonal share of ‖Hv‖² |
|---|---|---|---|---|---|
| 4 | 0.58 | 0.14 | 0.29 | 0.35 | 0.62 |
| 8 | 0.52 | 0.20 | 0.27 | 0.21 | 0.67 |
| 12 | 0.52 | 0.16 | 0.32 | 0.36 | 0.68 |
| 16 | 0.51 | 0.24 | 0.26 | 0.34 | 0.77 |
| 18 | 0.40 | 0.36 | 0.24 | — | 0.81 |
| 20 | 0.50 | 0.19 | 0.31 | 0.24 | 0.75 |

*(48 samples per layer; layer 18 used 32 samples from an earlier run.)* At every layer the pieces reconstruct the exact HVP to ≤ 1.5e-6.

**What the table shows:**
- **About half the curvature is generated by MLP AND-atoms, and half by the token mixers.** DeltaNet's share is roughly 2× attention's, matching their 3:1 layer count.
- **Most of it is generated close to the source layer:** the nearest third of downstream blocks produces 50–60%.
- **25–38% of the response to a single-position perturbation appears at *other* positions** (the cross-position blocks).

So an MLP-only H-lens cannot be complete. This matches v1's finding that the MLP-only approximation explained ~55% (on the broadcast definition).

## 3. Theorem 3: the rectangle identity (exact finite interactions) [proved]

**Statement.** For any C² function f and any moves a, b:
  I(a,b) := f(h) − f(h−a) − f(h−b) + f(h−a−b) = ∫₀¹∫₀¹ aᵀ ∇²f(h − a − b + s a + t b) b ds dt.

**Proof.** Let φ(s,t) = f(h − a − b + s a + t b). Then I = φ(1,1) − φ(0,1) − φ(1,0) + φ(0,0) = ∫₀¹∫₀¹ ∂_s∂_t φ ds dt, and ∂_s∂_t φ = aᵀ∇²f b. ∎

**Verified on real Qwen3.5-0.8B at feature scale** (`hs_rectangle.py`, layer 12).
- **Setup:** moves a and b each remove a random whitened half of a token's deviation from the mean (λ = 1); 8 samples × 4 output covectors.
- **Results**, correlation of each prediction with the true finite interaction:
  - Hessian at h: **r = −0.09**;
  - Hessian averaged along the rectangle's diagonal (5 nodes): r = 0.13;
  - **Hessian averaged over the full rectangle (5×5 Gauss–Legendre, 25 HVPs): r = 0.98**, with relative error 0.16, all of it quadrature error.

**Consequences.**
1. **What an interaction is.** A finite interaction effect is exactly a bilinear form of the Hessian, averaged over the rectangle spanned by the two moves. No Taylor approximation is involved. The 2×2 factorial "interaction term" is literally an averaged second derivative.
2. **Why v1 failed.** v1 found the Hessian *at h* stops predicting interactions at effective move sizes. That is the gap between H(h) and the rectangle average. It's a statement about *where* the Hessian is evaluated, not about whether second order matters.
3. **Integrated H-lens.** T_ℓ^{int} = E_x ∫₀¹ ∇²(h_x^{(s)}) w(s) ds, along the segment h^{(s)} = μ + s(h − μ) from the activation mean to the actual activation.
   - For moves that remove parts of a token's deviation from the mean, this segment is the diagonal of the relevant rectangle.
   - So T^{int} is the H-lens matched to natural, feature-scale interactions. *(Test: `hs_integrated.py`, pending.)*
4. **The dissociation test needs no Taylor caveat.** The output interaction of two context contributions a₁, a₂ at layer ℓ is exactly a₁ᵀ H̄_□ a₂. If H-space is the dominant subspace of the rectangle-averaged curvature, ablating it must remove interaction energy specifically.
   - Caveat that remains: contributions that interact *before* layer ℓ, or across positions.

## 4. Proposition 4: one geometry, three weightings [proved for MLP atoms; extends to every site]

For every atom the geometry (the read pair (a, b) and the write w) is fixed by the weights. Only the scalar coefficient depends on where the Hessian is evaluated. For a SwiGLU neuron:

| lens | AND coefficient | threshold coefficient | sees |
|---|---|---|---|
| local | E_x silu'(z_i(h_x)) · r² | E_x silu''(z_i) u_i · r² | what the model is sensitive to at data points |
| integrated | E_x ∫ w(s) silu'(z_i(h_x^{(s)})) ds · r² | likewise | natural-scale interactions (Thm 3) |
| potential | sup silu' ≈ 1.10 (or E[silu' \| z > 0]) | 0 | *dormant* gates, i.e. rules that never fire on the data |

**Why it matters for safety.**
- The v2 sleeper result failed because a trained trigger is *saturated*: at honest inputs its gate is closed, so silu'(z) ≈ 0 and every data-averaged derivative is tiny.
- The potential H-lens removes that dependence. It ranks atoms by what they *would* do if opened, decoded through the lens.
- *(Test on the pirate organism: pending.)*

## 5. Corollary 5: the averaged H-lens factorises through the J-lens [assumption: decoupling]

**Statement.** Replace each per-context Jacobian by its corpus average, as the J-lens does, and treat the per-site coefficients as independent of the transports. Then:

  T_ℓ ≈ Σ_{s>ℓ} Σ_{atoms i∈s} ω_{s,i} ⊗ γ_{s,i} · sym(ã_{s,i} ⊗ b̃_{s,i}),
  ω_{s,i} = J̄_{L*←s} w_{s,i}  (the J-lens applied to the atom's write),  ã = J̄_{s←ℓ}ᵀ a, b̃ = J̄_{s←ℓ}ᵀ b.

**Reading.** Every atom is a verbalisable rule: "IF [J-decode of read a] AND [J-decode of read b] THEN [J-decode of write]".
- Reads are covectors. They decode through Σ (whitening), as u_tᵀ J̄ Σ a.
- The error is a sum of covariance terms, Cov(J, γ) and Var(J): E[JᵀCJ] − J̄ᵀC̄J̄.
- v1 measured this for MLP-only, broadcast: subspace overlap 0.55. Most of the gap was mixer curvature, not decoupling (L1 ≈ L2).

## 6. Theorem 6: complexity [proved]

Let P = FLOPs per token per forward pass, n = number of prompts and k = target rank.

| object | cost | Qwen3.6-27B (d = 5120) |
|---|---|---|
| J-lens (full d×d per layer) | n · d backward passes · T · P | n = 25: 1.3×10⁵ backward passes |
| exact H-space (top-k, all layers in one pass if desired) | (k + o) HVP probes × B contexts × T · P · c_HVP; **independent of d** | k = 64: 256 probes × 4 contexts ≈ 4×10³ backward-equivalents per layer, about **30× cheaper than fitting the J-lens** |
| full averaged H-lens in atom form (Cor. 5) | one forward pass over n prompts (per-site statistics) + J-lens | ≈ J-lens cost; storage = weights + O(#sites) scalars (MLP: 2 × 1.1M) instead of d³ = 1.3×10¹¹ |
| exact per-position diagonal over all p | one HVP per probe via Hutchinson signs over positions (Lemma 7) | — |

**Lemma 7 (Hutchinson over positions) [proved].**
- Perturb every source position by s_p v, with independent Rademacher signs s_p.
- Multiply the output at p by s_p and average over p.
- The expectation is mean_p H_pp v, because cross terms carry E[s_p s_q] = 0. Their variance falls as 1/(#positions × #contexts).

### Corollary 6.1: the J-adjoint local H-lens [construction proved; fidelity measured in `hs_local.py`]
- **The truncation.** Truncate the Thm 1 sum to the k blocks after ℓ, and replace the exact adjoint at layer ℓ + k by the J-lens adjoint of record, J̄_{ℓ+k}ᵀ c.
- **The objective.** It becomes F_k = Σ_p (J̄_{ℓ+k}ᵀ c) · h_{ℓ+k}[p], whose Hessian needs double backward through only k blocks.
- **Cost.** About k / (L* − ℓ) of the exact H-lens, so O(k) blocks rather than O(L).
- **Exactness.** It is exact when k = L* − ℓ.
- **Error.** The error has two parts:
  1. curvature generated beyond block ℓ + k;
  2. the J-lens adjoint error, i.e. the context-variance of ∂F/∂h_{ℓ+k}.

  The anatomy (§2) says (1) is about 40–50% of the *signed* share. Whether the *subspace* survives is the empirical question.
- **Why it matters.** If a small k preserves H-space, the H-lens costs the same order as the J-lens it is built on, and all 64 layers of a 27B model become affordable.

**Streaming (Thm 1 recursion).** Only one block's weights and local graph need to be resident at a time. That makes exact fp32 HVPs of a 27B model feasible on any GPU that holds one block (~1.7 GB in fp32), with weights streamed from disk.

## 7. White-box signals in control protocols (goal 3)

**Setup.**
- At step k, the untrusted agent emits an action a_k. The protocol chooses execute, replace (defer to or edit by the trusted model), or audit, under an audit budget α and a replace budget β.
- The black-box score B_k sees the action text.
- The white-box score W_k sees the agent's residual states at positions *before* the first token of a_k. So it can act before the action exists.

### G1. Likelihood-ratio fusion never hurts; other fusions can [proved; NP lemma]
- At a fixed honest false-positive rate α, the most powerful test thresholds Λ(B,W) = p_attack(B,W)/p_honest(B,W).
- Hence TPR_{Λ(B,W)}(α) ≥ max(TPR_{Λ(B)}(α), TPR_{Λ(W)}(α)).
- **Counterexample for additive fusion.** If W is pure noise, threshold(B + W) at α raises the bar on B and loses attacks. Our Oct 5 interpcontrol run measured exactly this: additive fusion rescued 2 attacks and lost 41.
- **Prior work.** Hua et al. 2025 (arXiv 2507.15886) already use Neyman–Pearson combination of a probe with a black-box monitor. In practice Λ is estimated by a cross-fitted calibrated classifier.

### G2. Hidden goals as spikes, and the workspace as dimension reduction [proved for the Gaussian model; empirical test pending]
**Model.**
- Let r_k = P z_k − E_honest[P z_k | context] be the task-residualised state at the start of action k, projected by P onto a p-dimensional subspace and whitened.
- **Honest (H0):** r_k are i.i.d. N(0, I_p).
- **Attack (H1):** r_k = θ ε_k u + noise. Here u is an unknown unit "hidden goal" direction, persistent across steps, and ε_k ∈ {0, 1} marks the steps (a fraction f of all K) where the goal is active.
- The sample second-moment matrix S = (1/K) Σ_k r_k r_kᵀ is then a spiked Wishart matrix with spike strength f θ².

**Result.**
- By the Baik–Ben Arous–Péché transition, the top eigenvalue of S separates from the Marchenko–Pastur bulk edge (1 + √(p/K))², and the GLRT max_u uᵀSu detects the goal, iff
  **f θ² > √(p / K)**, i.e. **K > p / (f θ²)²**.
- So the number of steps needed for *zero-shot, label-free* detection of an unknown persistent goal scales **linearly with the ambient dimension p** of the subspace being monitored.

**If the goal always pushes in the same direction (a mean shift):**
- The better statistic is the norm of the mean, ‖(1/√K) Σ_k r_k‖² − p, which is approximately N(0, 2p) under H0.
- It detects iff K f² θ² ≳ √(2p), i.e. **K ≳ √(2p) / (f θ)²**.
- So the dependence on dimension is weaker (√p rather than p), but it is still there.

**In both regimes,** the trajectory length a label-free detector needs grows with the dimension of the space it watches.

**Consequence.**
- Monitoring the 25-dimensional workspace (J25 or H25) instead of the full residual stream (p = 5120) lowers the required trajectory length, for the same signal strength, provided the intent lives in the workspace:
  - **~200×** for the spike/GLRT test;
  - **~14×** for the mean test.
- The global-workspace hypothesis says intent must pass through the workspace if the model is to act on it or verbalise it.
- So this is where **workspace theory meets random-matrix theory**: the workspace is the dimension reduction that makes unknown-goal detection feasible on agent-length trajectories (tens of steps).

### G3. Known threats: matched filters on Concept-Lens directions [standard]
- Some side tasks are enumerable. BashArena has 4: exfiltration, malware download, privilege escalation, disabling the firewall.
- For those, the optimal statistic is a matched filter: max_i ⟨φ(τ_i), r⟩, where φ(τ_i) is the Concept-Lens direction of the multi-token description of side task i. Taking the max over m candidates costs only a √(2 log m) shift in threshold.
- Concept Lens reaches 80–91% scoring accuracy on Qwen3.6-27B when the candidate is supplied (Gong et al. 2608.31084). That is exactly the known-threat case.

### G4. Conditional intent is an AND, so H-space [hypothesis]
- A scheming policy "act IF unobserved AND opportunity" is a conjunction.
- At honest-looking steps its gate is closed, so it is invisible to first-order reads (the v2 saturation result).
- The *potential* H-lens (Prop. 4) ranks such gates by what they would do if opened. A WB score from H-space projections may therefore flag conditional intent at steps where no black-box signal exists yet.

**Tests (planned, on `adityaasinha28/control_arena_bash`: 750 attack / 750 honest BashArena trajectories, read by Qwen3.6-27B as a surrogate white-box reader):**
1. Does the zero-shot GLRT in J25/H25 beat the same GLRT in full d, as G2 predicts?
2. Does the known-threat matched filter work?
3. Does LR-fused B + W beat B at 1/2/5% audit budgets, with no lost attacks, where additive fusion loses them?
4. How many steps ahead of the black-box alarm does the white-box alarm fire (pre-emption lead)?

## 8. Open questions [open]
1. Does H-space exist at 27B (the preregistered one-shot test)?
2. Does the integrated H-lens fix v1's effective-scale failure (`hs_integrated.py`)?
3. Does the potential H-lens find the planted pirate × animal sleeper from honest data?
4. Is H-space distinct from J-space, or is it J-space seen at second order?
