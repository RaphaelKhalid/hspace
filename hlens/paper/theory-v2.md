# H-space: theory (v2, Oct 5 2026)

This file supersedes `theory.md`, which is kept unchanged for the record. It was rebuilt claim by claim from the adjudicated adversarial review in `redteam-theory.json` (48 agents: math, numerics and novelty lenses, then an adjudicator). Every retained number was re-checked against `out/`, `out/pod/`, `runlog.md` and the scripts. Where a review replacement text disagreed with the raw files, the raw files win.

**Status markers**
- **[standard; proved]**: a known result, stated with proof and citation. No novelty is claimed.
- **[proved]**: an exact statement proved here (elementary unless noted).
- **[verified: x]**: a numerical check of size x, with its source.
- **[assumption]**: an approximation with no proved error bound. Any measurement of it is stated.
- **[measured]**: an empirical number from a logged run (file named).
- **[open]**: not settled, or a number whose script or raw output is not in the repo.

**Provenance.**
- `out/…`, `runlog HH:MM` and script names refer to `interpcontrol/hlens/`.
- **"Toy (v2)"** marks fp64 CPU checks run while preparing this file. Their scripts are not in the repo; they only back identities that are proved anyway.
- **"Review scratch"** marks numbers computed during the review whose scripts are not in the repo. They are [open] until committed.
- Citations added from the review have not been re-checked against the papers.

## Changelog vs theory.md

- **Conventions (T2a, T2b, T2c, C5).** sym(M) := M + Mᵀ throughout, so ∇²[(a·x)(b·x)] = sym(abᵀ). The draft never defined sym. Under the usual (M+Mᵀ)/2 its Cor. 2.1 was off by 2, and its Pearce reduction was off by 2 under either convention.
- **§0 (T1, T6, EMP).**
  - u_tᵀT_ℓ is the Hessian of the pre-norm proxy u_t·h_{L*} = rms(h_{L*})·logit_t, not of the logit.
  - Three energy operators are now distinguished, with a proved ordering: M^lens (the draft's §0 definition), M^diag, and M^row (what run 1 actually estimated).
- **§1 (T1).**
  - "Theorem 1" becomes Lemma 1, the standard discrete second-order adjoint, with citations, hypotheses, notation, boundary conditions and the terminal term for final-norm readouts.
  - Exactness holds per input, not after averaging.
  - The L18 "1.2e-6" (its raw output was overwritten) is replaced by the per-layer maxima in `out/anatomy_08b.json`.
  - Streaming is demoted to an unimplemented possibility.
- **§2 (T2a, T2b, T2c, EMP).**
  - "Theorem 2" becomes Lemma 2 (curvature bookkeeping; standard AD).
  - "Exactly three atom types" is withdrawn. Only per-site rank and inertia are basis-invariant; the product/threshold split is a graph convention.
  - The RMSNorm and softmax blocks are given in closed form.
  - SwiGLU: the residual-space Hessian is written out with its norm curvature. The Pearce reduction is 2Q and holds only in post-norm coordinates. The unsourced "4e-16 toy" tag is removed, and Belrose & Rigg are cited.
  - Attention: the q > p block is corrected (k_norm-dependent key read, a missing softmax threshold term, k_norm curvature) as Lemma 2.2. The q = p and cross-position blocks are inventoried. "query×query" is relabelled, and the averaged form is corrected.
  - The DeltaNet inventory is completed.
  - Anatomy table corrections:
    - L4 MLP is 0.57, not 0.58.
    - The nearest third of downstream blocks carries 36–62%, not 50–60%.
    - The off-position share is 19–38%, not 25–38%.
    - "DeltaNet ≈ 2× attention, matching the 3:1 count" is withdrawn.
    - "MLP AND-atoms" becomes "MLP sublayers".
- **§3 (T3, EMP).**
  - "Theorem 3" becomes Lemma 3, the standard mixed-difference identity, with citations.
  - The r = 0.98 check is relabelled a code + quadrature check, with bootstrap CIs. A corner-vs-centre expansion is added.
  - Consequence 2 is a restatement, not an explanation.
  - Consequence 3 (integrated lens) is not implied by the lemma, and it was measured negative at λ = 1; the draft still said "pending".
  - Consequence 4 is withdrawn and marked [open].
  - A λ² bookkeeping error in `out/integrated_08b_L12.json` is flagged.
- **§4 (P4).**
  - [proved] now covers only per-block linearity. The lens-level single-geometry form is an assumption.
  - "Extends to every site" is withdrawn. The coefficients are corrected (ρ_i; r² inside the path integral).
  - The potential lens is a definition (∝ Pearce's weight-only matrix within each block).
  - The dormant-gate and natural-scale claims become hypotheses: H1 [open], and H2, which the data do not support.
- **§5 (C5).**
  - "Corollary 5" becomes Approximation 5, a mean-field (K-FAC-style) form for SwiGLU sites only.
  - Lag-resolved transports, the full error expansion and the correct write index are added.
  - The evidence is for the broadcast object only, and it failed its preregistered 0.7 bar. "L1 ≈ L2" is replaced by what was measured.
- **§6 (T6, C61).**
  - "Theorem 6" becomes cost accounting (arithmetic; c_HVP assumed).
  - "30× cheaper" is scoped: 17–31× for one layer, about break-even for all 63 layers, at n = 25.
  - "(k + o) probes, independent of d" is withdrawn. The 27B probe budget did not determine the subspace.
  - Lemma 7 is the standard Hutchinson estimator: unbiased, not exact, and first moment only.
  - "Corollary 6.1" becomes Construction 6.1, with an exact error identity, the corrected anatomy range (38–64%), both fidelity metrics and the J-lens baselines.
- **§6A (new).** What each estimator in `hspace.py` and `hspace2.py` converges to (Proposition 6), and what the 27B run-1 numbers do and do not show.
- **§7 (G1, G2).**
  - G1 becomes a standard result (Neyman–Pearson) with randomisation. It guarantees a net gain, not an item-wise one, and only for the true Λ under a fixed attack law.
  - The Oct 5 interpcontrol illustration is corrected: add-only fusion; vulnerable code, not attacks; placebo p_perm 0.984; verdict Inconclusive.
  - G2 is rewritten:
    - The fixed-sign model is a mean shift; the spike test needs symmetric signs.
    - Workspace gains scale with a signal-capture fraction ρ: at most 204.8ρ² and 14.3ρ, with break-even at ρ ≈ 0.07.
    - The H0 assumptions are explicit, and textbook sources are cited.
  - Test 3's criterion becomes net > 0.
- **§8.**
  - Q1 is settled for run 1: FAIL, on C1, at all four layers.
  - Q2 is answered negatively at 0.8B, ℓ = 12.
  - The remaining questions stay open, and new ones are added.

## 0. Objects

### Model
- Residual states h_k ∈ R^{T×d}. "Layer k" means the output of decoder block k (the jlens convention).
- Each block is two residual sublayers, y ↦ y + φ_s(y): a token mixer, then a SwiGLU MLP. Each branch starts with an RMSNorm. The mixer is gated softmax attention in 1 block of 4 and Gated DeltaNet otherwise.
- Target: L* = the last block output, before the final norm.
  - Qwen3.5-0.8B: 24 blocks, d = 1024, L* = 23 (attention at blocks 3, 7, …, 23).
  - Qwen3.6-27B: 64 blocks, d = 5120, L* = 63.
- V = valid positions {16, …, T−2}.

### The J-lens (Anthropic 2026; the jlens fitting estimator)
  J_ℓ = E_x mean_{p∈V} ∂/∂h_ℓ[p] Σ_{p'∈V} h_{L*}[p']        (d×d)
- It decodes as u_tᵀ J_ℓ h, with u_t = (1+w_final) ⊙ W_U[t].
- Because u_t·h_{L*} = rms(h_{L*})·logit_t, the lens targets this pre-norm proxy, not logit_t itself.

### Per-context Hessian and the H-lens
- **Objective.** For an output covector c ∈ R^d, F_c(h_ℓ) = Σ_{p'∈V} c·h_{L*}[p']. This is a linear readout before the final norm (the J-lens proxy).
- **Blocks.** H_x[c] = ∇²_{h_ℓ}F_c in context x, with d×d blocks H_pq = ∂²F_c/∂h_ℓ[p]∂h_ℓ[q] for p, q ∈ V. Note H_qp = H_pqᵀ, and H_pp is symmetric.
- **H-lens (this work).** T_ℓ[c] = E_x mean_{p∈V} H_x[c]_pp. This is the J-lens reduction one derivative higher. As a tensor, T_ℓ ∈ R^d ⊗ Sym(d), linear in c.
  - The token slice T_ℓ[u_t] is the averaged Hessian of the proxy u_t·h_{L*}, not of logit_t; the final-norm curvature is omitted by construction (T1).
  - The perturbation sits at a single source position p. The cross-position blocks H_pq (p ≠ q) are not part of T_ℓ.
- **Whitening.** H̃ = Σ^{1/2} H Σ^{1/2}, where Σ is the layer-ℓ activation covariance, shrunk by 0.05 toward (tr Σ/d)·I. This measures curvature per unit of natural activation scale. It is a choice, not a consequence.
- **Covectors.** c ~ N(0, Σ_U), with Σ_U = (1/|vocab|) Σ_t u_t u_tᵀ.

### Energy operators and H-space
Three different d×d PSD operators appear in this project:
- **M^lens_ℓ = E_c[ T̃_ℓ[c]² ]**: the energy of the averaged H-lens. This is the draft's §0 "energy operator".
- **M^diag_ℓ = E_{x,c} mean_{p∈V} H̃_pp²**: the per-position diagonal-block energy, a mean of squares over contexts and positions.
- **M^row_ℓ = E_{x,c} mean_{p∈V} Σ_{q∈V} H̃_pq H̃_pqᵀ**: the per-position row energy. It includes the cross-position blocks.

**Proposition 0 (ordering) [proved; toy (v2)].**
- M^row − M^diag = E_{x,c} mean_p Σ_{q≠p} H̃_pq H̃_pqᵀ ⪰ 0.
- M^diag − M^lens = E_c E_{x,p}[(H̃_pp − T̃[c])²] ⪰ 0.
- Hence M^row ⪰ M^diag ⪰ M^lens, and λ_i(M^row) ≥ λ_i(M^diag) ≥ λ_i(M^lens) for every i (Weyl monotonicity).
- The ordering constrains eigenvalues and traces, not eigenvectors. The three top-k subspaces can differ.

*Proof.* The first identity is the definition. For the second, E[A²] − (EA)² = E[(A−EA)²] for any random symmetric A (here A = H̃_pp, with (x, p) random and c fixed), and (A−EA)² = (A−EA)(A−EA)ᵀ ⪰ 0. ∎

**H-space_k(M)** is the top-k eigenspace of a chosen energy operator, with default k = 25 (matching J-space's "≤ 25 active directions"). Every result must say which M it uses. Run 1 (`hspace.py`) estimated M^row (§6A).

## 1. Lemma 1: block decomposition (the second-order adjoint) [standard; proved] [implementation checked]

This is the standard discrete second-order adjoint, also called Hessian backpropagation: Christianson 1992; Wang, Navon, Le Dimet & Zou 1992; Pearlmutter 1994; Griewank & Walther 2008, ch. 5; Dangel, Harmeling & Hennig 2020.

**Setting.**
- Let y_0 = h_ℓ ∈ R^{T×d}, at all positions. Write blocks ℓ+1 … L* as S = 2(L*−ℓ) residual sublayers, y_{s+1} = y_s + φ_s(y_s).
- **Hypotheses:**
  - each φ_s is C² and depends only on y_s;
  - masks and RoPE tables are constants.

  Both hold for Qwen3.5/3.6. Every norm, including the DeltaNet l2norm, has ε inside the root, and SiLU, sigmoid, softplus, exp and softmax are smooth.
- **Definitions:**
  - A_s := I + Dφ_s(y_s).
  - J_{s←ℓ} := A_{s−1}⋯A_0 = ∂y_s/∂h_ℓ. This is the (Td)×(Td) per-input Jacobian, including cross-position transport. It is not the averaged J_ℓ.
  - Adjoints: g_S = ∂F/∂y_S and g_s = A_sᵀ g_{s+1}.
  - C_s := ∇²_y⟨g_{s+1}, φ_s(y)⟩ at y = y_s, with g_{s+1} held at its base-trajectory value.

**Statement.** For every input x and F = F_c,
  ∇²_{h_ℓ} F = Σ_{s=0}^{S−1} J_{s←ℓ}ᵀ C_s J_{s←ℓ}.
In forward-over-reverse form, for any tangent v:
- forward: ẏ_0 = v, ẏ_{s+1} = A_s ẏ_s;
- backward: ġ_S = 0, ġ_s = A_sᵀ ġ_{s+1} + C_s ẏ_s;
- output: ∇²F·v = ġ_0.

**Proof.**
1. Let G_s(y) be F as a function of y_s.
2. The second-order chain rule applied to G_s = G_{s+1} ∘ (y ↦ y + φ_s(y)) gives ∇²G_s = A_sᵀ ∇²G_{s+1} A_s + C_s, since the identity map has no curvature.
3. Unroll from s = S, where ∇²G_S = 0 because F is linear in h_{L*}.
4. Differentiating g_s = A_sᵀ g_{s+1} along ẏ gives the recursion. ∎

If the readout passes through the final RMSNorm (a true logit), add the terminal term J_{S←ℓ}ᵀ ∇²F(y_S) J_{S←ℓ}.

**Scope.**
- The per-position H-lens at p is the (p,p) d×d block of this matrix.
- The identity is exact for each x. It commutes with E_x and mean_p only as a sum of expectations, never as E[J]ᵀE[C]E[J]. So the transported and decoupled forms in §§4–6 are approximations, not consequences of this lemma.
- **Attribution.** Each term is curvature *generated* at sublayer s, weighted by the downstream adjoint and the upstream transport. The split depends on the chosen sublayer partition.
- **Streaming.** The recursion permits block-streamed HVPs in principle (§6). This is not implemented.

**Implementation check [verified].** This is a unit test of our code, not evidence for the mathematics.
- **Setting:** Qwen3.5-0.8B, bf16 checkpoint upcast to fp32 blocks, eager attention, DeltaNet on a patched torch path (equivalent to the reference: fwd 8e-7, grad 1.7e-6, `tests/test_deltanet_patch.py`), T = 96.
- **Comparison:** the sum of the 2(L*−ℓ) pulled-back sublayer HVPs (`hs_anatomy.py`) against the full double-backward HVP.
- **Metric:** max relative Frobenius error over the whole [B, T, d] output, over 24 batches of 2 contexts. The batches draw on 16 distinct contexts, each reused 3 times, with single-position Σ^{1/2}-shaped tangents.

| layer ℓ | 20 | 16 | 12 | 8 | 4 |
|---|---|---|---|---|---|
| max rel. error | 8.8e-7 | 1.1e-6 | 1.2e-6 | 1.3e-6 | 1.45e-6 |

- Source: `out/anatomy_08b.json`. An earlier run at ℓ = 18 logged 1.2e-6 (runlog 18:05); its raw output was overwritten.
- Both sides use autograd on the same module code. That the double-backward HVP is the true Hessian-vector product is checked separately against central finite differences: rel. err 1.3e-4 at ℓ = 12 and 1.7e-4 at ℓ = 8 (`tests/test_hvp.py`, T = 48; runlog line 8).

## 2. Anatomy of local curvature

### Lemma 2 (curvature bookkeeping) [standard; proved]
This is standard second-order reverse-mode AD: Christianson 1992; Griewank & Walther 2008; Gower & Mello 2012 (edge pushing); Dangel, Harmeling & Hennig 2020. It is Lemma 1 at node rather than sublayer granularity.

**Setting.** Write a sublayer φ as a computational graph with three kinds of nodes:
- linear maps;
- unary elementwise maps σ (SiLU, sigmoid, exp, softplus, rsqrt, reciprocal);
- binary products.

For the contraction g·φ:
- let ḡ_n be the adjoint at node n's output;
- let a_n, b_n be the Jacobian rows, at the current point, of node n's inputs with respect to the sublayer input y.

**Statement.** Exactly,
  ∇²_y[g·φ(y)] = Σ_{unary n} ḡ_n σ_n''(z_n) a_n a_nᵀ + Σ_{product n} ḡ_n sym(a_n b_nᵀ).
Linear nodes contribute nothing.

How the model's operations fit this form:
- **n-ary products** are chains of binary products. Each contributes C(n,2) pairwise terms, weighted by the product of the other factors. Examples are the DeltaNet write β_p(v_p − e^{g_p}S_{p−1}k_p)k_pᵀ and a_qp V_p σ(G x_q).
- **Normalisations** (RMSNorm, q/k RMSNorm, L2 norm, RMSNormGated) and the **softmax denominator** are composites of these primitives. Kept as blocks, they are not "scalar × read covectors". [proved; toy (v2): 1.6e-16 for both]
  - **RMSNorm.** With N(y) = γ⊙y·r, r = (‖y‖²/d+ε)^{−1/2} and z = γ⊙c:
    ∇²[c·N(y)] = −(r³/d)[(z·y)I + z yᵀ + y zᵀ] + 3(r⁵/d²)(z·y) y yᵀ.
    This is full rank, with the isotropic eigenvalue −(r³/d)(z·y) on span{y, z}^⊥ (toy: 14 of 16 eigenvalues).
  - **Softmax.** For a = softmax(l) contracted with w: ∇²_l[w·a] = diag(m) − m aᵀ − a mᵀ, with m = a⊙(w − w·a). Its rank is ≤ T−1 (toy: 6 of 7).

### Remark 2.1 (what is invariant, and what is a convention) [proved; elementary]
**Invariant.** Under any invertible linear change of residual basis (Sylvester's law of inertia), each site's rank and inertia are unchanged:
- A unary ("threshold") site is rank 1 and semidefinite.
- A product site has eigenvalues ḡ(a·b ± ‖a‖‖b‖), so inertia (1,1). It is rank 1 if a ∥ b.

**Not invariant.** The split of curvature into product and threshold terms is a property neither of the function nor of its Hessian:
- Polarisation: sym(abᵀ) = ½[(a+b)(a+b)ᵀ − (a−b)(a−b)ᵀ].
- Refactoring: writing silu(z)·u as z·σ(z)·u splits silu''·u into 2σ'·u (a product) plus zσ''·u (a threshold).

The labels therefore attribute curvature to a chosen computational graph, as the neuron basis does. For SwiGLU the native graph is privileged by the weights (w_g, w_u), so we use it.

**Naming.** We call the cross term a *product atom*. It reads as "a AND b" only when both reads are one-sided. For signed reads (SwiGLU's up read, DeltaNet's v and k) it is a sign-agreement interaction. "AND" also collides with the Harsanyi AND/OR interaction literature.

**Reads are point-dependent.**
- Even in the MLP, the residual-space read is r·Pᵀdiag(γ)w_g, with P = I − (r²/d)yyᵀ.
- In the mixers, reads pass through q/k-norm, RoPE, conv1d + SiLU, the L2 norm and softmax.
- So the rank-1/rank-2 structure holds pointwise. After averaging over contexts, a site's contribution has higher rank.

**Atom split of the MLP sublayer [open: review scratch, not in out/ or runlog].**
- Setting: the local per-token Hessian including the pre-MLP RMSNorm. Qwen3.5-0.8B, source ℓ = 12, blocks 13/16/20/23, 6 tokens per block.
- The closed form matched autograd to 1.2e-6. Frobenius-projection shares:

| piece | block 13 | block 16 | block 20 | block 23 |
|---|---|---|---|---|
| product (gate×up) | 0.89 | 0.76 | 0.77 | 0.78 |
| SiLU'' (gate×gate) | 0.11 | 0.23 | 0.23 | 0.21 |
| RMSNorm | 0.005 | 0.011 | 0.002 | 0.011 |
| threshold share under the z·σ(z)·u factorisation | −0.015 | −0.046 | −0.039 | −0.040 |

### 2.1 SwiGLU MLP, per-token closed form [proved: elementary] [verified: 1.1e-6 vs autograd, residual-space Hessian with norm terms, Qwen3.5-0.8B blocks 13 and 15, `tests/test_swiglu.py`]
- **Neuron.** Neuron i computes y_i = silu(z_i)·u_i, where z_i = w_{g,i}·x̃, u_i = w_{u,i}·x̃, and x̃ = N(y) = γ⊙y·r with γ = 1 + w. It writes along w_{down,i}.
- **Post-norm coordinates.** For a fixed adjoint g, let ρ_i = w_{down,i}·g. Then
  ∇²_x̃(g·MLP) = Σ_i ρ_i [ silu''(z_i) u_i · w_{g,i}w_{g,i}ᵀ + silu'(z_i) · sym(w_{g,i}w_{u,i}ᵀ) ].
- **Residual coordinates.**
  ∇²_y(g·MLP∘N) = DNᵀ(∇²_x̃)DN + Hess_y[z·(y r)].
  - DN = r·diag(γ)(I − (r²/d)yyᵀ) and z = γ⊙∇_x̃(g·MLP).
  - Hess_y[z·(y r)] = −(r³/d)(z yᵀ + y zᵀ) − (r³/d)(z·y)I + 3(r⁵/d²)(z·y)yyᵀ.
  - For ε > 0, DN only approximately annihilates the radial direction: (I − (r²/d)yyᵀ)y = εr²y.
- **Reading.** Each neuron is a single product site silu(a·x̃)(b·x̃), with a = w_{g,i} and b = w_{u,i}. It has two blocks:
  - gate×up: indefinite, coefficient ρ_i silu'(z_i);
  - gate×gate: rank 1, coefficient ρ_i silu''(z_i) u_i. This coefficient depends on the up read, so it is not a single-read threshold atom.

  Both coefficients depend on the input. Only the directions are fixed by the weights.

**Relation to prior work.**
- **Bilinear limit.** With silu replaced by the identity, ∇²_x̃ = 2Q. Here Q = Σ_i ρ_i·½ sym(w_{g,i}w_{u,i}ᵀ) is the symmetric interaction matrix of Pearce et al. 2024 (arXiv 2410.08417), for which g·MLP(x̃) = x̃ᵀQx̃. The eigenvectors are the same and the eigenvalues double (toy (v2): 3.4e-17 vs 2Q).
- The residual-space Hessian does not reduce to Q, even in the bilinear case, because of DN and the norm-curvature term.
- A data-averaged quadratic description of one GLU block is in Belrose & Rigg 2025 (arXiv 2502.01032).
- The formula itself is routine. What this work adds is its use:
  1. the exact per-token norm terms;
  2. corpus averaging of curvature transported from downstream layers;
  3. decoding through the J-lens (§5, an assumption);
  4. the mixer analogues.

### 2.2 Gated softmax attention (1 block in 4)
Status: q > p block [proved; toy (v2) 3e-16; real model: open]; q = p and cross-position blocks [inventory only]; averaged form [assumption; not implemented].

**Head.** Let x = RMSNorm(y) be the normalised mixer input, η a query head and k(η) its KV head.
- Q_q = R_q N_q(W_Q x_q), K_j = R_j N_k(W_K x_j), V_j = W_V x_j.
- s_qj = Q_q·K_j/√d_h and a_qj = softmax_{j≤q}(s_qj).
- o_q = W_O^{(η)}[(Σ_j a_qj V_j) ⊙ σ(G x_q)].
- N_q and N_k are per-head RMSNorms with gain (1 + w). R_j is (partial) RoPE.

**Adjoint quantities and reads.** For a fixed output adjoint g_q:
- ω_q = (W_O^{(η)ᵀ}g_q) ⊙ σ(G x_q);
- c_qj = ω_q·V_j and c̄_q = Σ_j a_qj c_qj;
- key read κ_qp = J_K(x_p)ᵀQ_q/√d_h, where J_K = R_p D_k(W_K x_p) W_K and D_k(k) = diag(1+w_k)·r·(I − r²kkᵀ/d_h), r = 1/rms(k);
- value read ν_q = W_Vᵀω_q.

**Lemma 2.2 (later queries) [proved; routine calculus].** For q > p, x_p enters g_q·o_q only through K_p and V_p, and exactly:
  ∇²_{x_p}(g_q·o_q) = a_qp(1−a_qp) sym(κ_qp ν_qᵀ)          [key×value product atom]
    + a_qp(1−2a_qp)(c_qp − c̄_q) κ_qp κ_qpᵀ                 [softmax threshold atom on the key read]
    + a_qp(c_qp − c̄_q) ∇²_{x_p} s_qp                        [k_norm curvature].

*Proof.* Write F = Σ_j a_qj c_qj. Then:
- ∂F/∂s_qp = a_qp(c_qp − c̄_q) and ∂F/∂c_qp = a_qp;
- ∂²F/∂c_qp² = 0;
- ∂²F/∂s_qp∂c_qp = a_qp(1−a_qp);
- ∂²F/∂s_qp² = a_qp(1−2a_qp)(c_qp − c̄_q).

Apply the chain rule through s_qp(x_p) and c_qp = ν_q·x_p. ∎

Toy (v2): one head with sigmoid gate, q/k RMSNorm and partial RoPE, fp64. The three-term formula matches autograd to 3.0e-16; the key×value term alone leaves 65% (with k_norm) and 74% (without) of ‖H‖_F unexplained.

Remarks:
- **Without k_norm,** κ_qp = W_KᵀQ̂_q^{(p)}/√d_h with Q̂^{(p)} = R_pᵀQ_q, and the third term vanishes.
- **With k_norm** (Qwen3.5/3.6 apply it between W_K and RoPE), the key read is rescaled by 1/rms(W_K x_p) and projected off W_K x_p, so it depends on x_p. The draft's read W_KᵀQ̂_q is wrong for these models.
- **Where other positions enter.** Other positions' values enter only through c̄_q, in the threshold term, which is of the same order as the product term.
- **What the atom marks.** a(1−a) vanishes both when q ignores p and when q attends only to p. The key×value atom marks p *competing* for q's attention while carrying content. A saturated retrieval head has near-zero key×value curvature.
- **Context-dependent geometry.** The reads depend on x_q, g_q and (through D_k) x_p. So, unlike SwiGLU atoms, attention atoms do not have a weight-fixed geometry.
- **Real-model check [open: review scratch].** Qwen3.5-0.8B, fp32, layers 3/11/19, 7 source positions:
  - the true-read formula reconstructs Σ_{q>p} of the exact Hessian to ≤ 8.7e-7;
  - the threshold term carries a signed share of 0.03–0.58;
  - the literal W_KᵀQ̂ read has relative error 0.62–0.89;
  - the q > p part is 0.11–0.91 of the local diagonal block.

  To be committed as `tests/test_attn_atoms.py`.

**Same position, q = p [inventory; no closed form].** x_p also enters through Q_p and σ(G x_p). Besides key×value (coefficient a_pp(1−a_pp)), the block contains:
- query×key self-score (coefficient a_pp(c_pp − c̄_p));
- query×value;
- gate×{value, query, key};
- the σ'' threshold atom on the gate read;
- normalisation curvature (q_norm, k_norm, input RMSNorm);
- the query×query softmax term (1/d_h) J_Qᵀ[Σ_j a_pj(c_pj − c̄_p)(K_j − K̄_p)(K_j − K̄_p)ᵀ]J_Q, with K̄_p = Σ_j a_pj K_j and J_Q = R_p D_q W_Q. This is a third central moment of keys and centred values: indefinite, and a softmax nonlinearity of one read, not a product of two reads.

**Cross-position blocks [not covered above].** Lemma 1 applies each downstream sublayer's Hessian to J_{s←ℓ}v, and after the first mixer that tangent is spread over positions. So every later attention block also needs the off-diagonal blocks ∂²/∂x_q∂x_{p'}:
- query(q)×key(p'), the canonical QK interaction;
- query(q)×value(p');
- gate(q)×key/value(p');
- the softmax cross terms ∂²F/∂s_qp∂s_qp' = −a_qp a_qp'(c_qp + c_qp' − 2c̄_q) and ∂²F/∂c_qp∂s_qp' = −a_qp a_qp'.

**Averaged form [assumption: decoupling, §5; not implemented].** Replace g_q by a fixed transported covector ḡ, and treat the coefficients as independent of the transports. The key×value family of head η then averages to
  W_Kᵀ M_η diag(W_O^{(η)ᵀ}ḡ) W_V + transpose,   M_η = E Σ_{q>p} a_qp(1−a_qp) D_k(W_K x_p)ᵀ R_pᵀ Q_q σ(G x_q)ᵀ/√d_h.
- M_η is a d_h×d_h statistic that does not depend on ḡ. Heads sharing a KV head (GQA) are summed after the diag contraction.
- The threshold family needs a separate order-d_h³ statistic per head, contracted with W_O^{(η)ᵀ}ḡ.
- So the full attention curvature does not reduce to one d_h×d_h statistic per head.

**Prior work** (the derivatives are routine):
- QK/OV circuits: Elhage et al. 2021;
- QK feature interactions: Kamath et al. 2025;
- the self-attention Hessian in parameter space: Ormaniec, Dangel & Singh 2024 (arXiv 2410.10986);
- the sigmoid output gate: Qiu et al. 2025 (arXiv 2505.06708).

### 2.3 Gated DeltaNet (3 blocks in 4) [partial inventory; not verified; computed by exact local autograd]
- **Mechanism.**
  - State update: S_p = e^{g_p}S_{p−1} + β_p(v_p − e^{g_p}S_{p−1}k_p)k_pᵀ.
  - Read at q ≥ p: S_q q_q.
  - Output: RMSNormGated(core) ⊙ silu(z_q).
  - q and k are L2-normalised. A depthwise conv1d (kernel 4) + SiLU precedes q, k and v.
- **Sites:**
  - product atoms among {β, v, k, q, e^{g}} at the source position: key×value, β×value, β×key, key×key (from −β e^{g}S_{p−1}k kᵀ), query×key, and decay×{key, β, query} (through e^{g_p}S_{p−1});
  - later-query reads, with coefficients set by the decays;
  - the output gate silu(z) × RMSNormGated(core), and the RMSNormGated block;
  - the conv1d, which spreads a source perturbation over p … p+3 before the SiLU, giving cross-position threshold atoms;
  - the L2-norm blocks on q and k.
- **How we compute it.** No closed form is implemented. `hs_anatomy.py` differentiates each whole sublayer.

### 2.4 Measured anatomy [measured: `out/anatomy_08b.json`; ℓ = 18 from runlog 18:05, earlier code]
**Setup.**
- Qwen3.5-0.8B, T = 96.
- 24 batches × 2 = 48 HVPs per layer, drawn from 16 distinct contexts. Each HVP has a fresh source position p, a tangent v = Σ^{1/2}ξ and a covector c.

**Share definition.**
- Share = ⟨piece, H_pp v⟩/‖H_pp v‖², in the raw Euclidean output metric, at the source position of the *per-context* Hessian.
- It is a signed projection. Shares sum to 1, but a piece can be negative or exceed 1. Per-sample shares were not stored, so cancellation is not excluded.
- These are not energy fractions, and they do not describe the corpus-averaged lens or the whitened operators of §0.

| layer ℓ | MLP sublayer | full attention | DeltaNet | all sublayers, nearest third | MLP, nearest third | ‖H_pp v‖²/‖Hv‖² |
|---|---|---|---|---|---|---|
| 4 | 0.57 | 0.14 | 0.29 | 0.57 | 0.35 | 0.62 |
| 8 | 0.52 | 0.20 | 0.27 | 0.36 | 0.21 | 0.67 |
| 12 | 0.52 | 0.16 | 0.32 | 0.62 | 0.36 | 0.68 |
| 16 | 0.51 | 0.24 | 0.26 | 0.62 | 0.34 | 0.77 |
| 20 | 0.50 | 0.19 | 0.31 | 0.44 | 0.24 | 0.75 |
| 18 (earlier code) | 0.40 | 0.36 | 0.24 | – | – | 0.81 |

**Column notes.**
- "MLP sublayer" is the whole branch MLP(RMSNorm(y)): product, SiLU'' and norm terms together. Mixer shares likewise include their input norms.
- "Nearest third" means the downstream blocks i with (i−ℓ−1)/(24−ℓ−1) < 1/3. That is 33–43% of the downstream blocks, and a single block at ℓ = 20.

**Reading (signed-share sense only).**
- MLP sublayers carry 0.40–0.57 of the share, and the token mixers carry the rest.
- DeltaNet/attention share ratios are 1.1–2.1 (0.65 at ℓ = 18), below the downstream block-count ratios of 2.0–2.8 (1.5 at ℓ = 18). So per block, an attention layer carries 1.2–2.3× the share of a DeltaNet layer. The draft's "≈2×, matching 3:1" is withdrawn.
- The nearest third of downstream blocks carries 36–62%, an enrichment of 1.1–1.7× over its block fraction. The other 38–64% is generated farther away.
- For a single-position tangent, 19–38% of ‖Hv‖² appears at other positions (23–38% excluding ℓ = 18).
- So neither an MLP-only nor a near-blocks-only H-lens captures the full per-context response.
- The draft's comparison with v1's "~55%" mixed a projection share with a subspace overlap, so it is dropped.

## 3. Lemma 3: the mixed-difference (rectangle) identity [standard; proved]

**Statement.** Let f be C² on a neighbourhood of P = {h − a − b + sa + tb : s, t ∈ [0,1]}. P is a parallelogram; it is a rectangle in whitened coordinates when the whitened moves are orthogonal. Then
  I(a,b) := f(h) − f(h−a) − f(h−b) + f(h−a−b) = ∫₀¹∫₀¹ aᵀ∇²f(h − a − b + sa + tb) b ds dt =: aᵀ H̄_P(h; a, b) b.

**Proof.** Let φ(s,t) = f(h − a − b + sa + tb). Then I = φ(1,1) − φ(0,1) − φ(1,0) + φ(0,0) = ∫∫ ∂_s∂_t φ, and ∂_s∂_t φ = aᵀ∇²f b. ∎

**Status.** This is the integral form of the classical mixed-difference mean-value theorem (Rudin, *PMA*, Thm 9.40), and the second-order analogue of Integrated Gradients completeness (Sundararajan et al. 2017). The same four-point contrast is ArchDetect's discrete mixed partial (Tsang et al. 2020) and the Shapley–Taylor pairwise discrete derivative (Sundararajan et al. 2020). It is a tool here, not a contribution.

**Remarks [proved].**
- H̄_P depends on (h, a, b). So I is not a bilinear form in (a, b); it is a pair-specific average of the Hessian over P.
- **Corner expansion:** I = aᵀ∇²f(h)b − ½(D³f(h)[a,a,b] + D³f(h)[a,b,b]) + O(‖(a,b)‖⁴).
- **Centre expansion:** at c = h − (a+b)/2 the odd orders cancel, so I = aᵀ∇²f(c)b + O(‖(a,b)‖⁴).
- So the gap between ∇²f(h) and H̄_P is exactly the third- and higher-order content.
- **Precondition.** The joint move at layer ℓ must equal a + b exactly. For moves at several positions it holds with the full, cross-position Hessian. For input-level interventions, whose layer-ℓ effects are not additive, I gains the term f(h−a−b) − f(h−Δ_joint).

**Consistency check of code and quadrature [verified; `hs_rectangle.py`, `out/rectangle_08b_L12.json`].**
- Qwen3.5-0.8B, fp32 blocks, ℓ = 12, objective F_c.
- Single-position moves a, b are complementary random whitened halves of h_p − μ, so a + b = h_p − μ (λ = 1).
- 8 (h, a, b) draws × 4 covectors. The covectors share each draw, so the effective n is about 8. CIs are from a bootstrap over draws.

| predictor | r (95% CI) | rel. L1 error | explained var. |
|---|---|---|---|
| rectangle average, 5×5 Gauss–Legendre (25 HVPs) | 0.98 (0.97 to 1.00) | 0.16 (per draw 0.005–0.36) | 0.95 |
| uniform diagonal average, 5 nodes | 0.13 (−0.55 to 0.75) | 1.62 | −3.3 |
| Hessian at the corner h | −0.09 (−0.29 to 0.54) | 1.42 | −3.2 |

- **The rectangle row tests code, not the model.** The identity is exact, so this row checks the HVP code and the quadrature.
- **The 5×5 rule looks under-resolved.** The two worst draws carry 74% of the absolute error. A 9×9 rule on those two draws reduces their error from 0.35 to 0.04 (r 0.970 → 0.999), and the centre Hessian gives r = 0.89 against −0.58 at the corner [open: review scratch]. A node sweep (n_q = 1…9) on all draws has not been run [open].
- **A larger run agrees on sign.** In `hs_integrated.py` (32 draws at the same ℓ and λ), the corner Hessian gave r = 0.29 and a uniform 4-node diagonal r = 0.57. Both have negative explained variance (−1.82, −1.88).

So at feature scale the Hessian varies too much over the rectangle for any single point or single segment to predict the interaction. Only the full 2-D average does, at ≥ 25 HVPs per pair and context.

**Implication 1: what an interaction is [proved].** A finite 2×2 interaction of moves that are additive at layer ℓ equals a pair-specific, rectangle-averaged second derivative. No Taylor truncation is involved, but no fixed bilinear form or single "interaction operator" follows.

**Implication 2: v1's failure.** The failure of the quadratic model at h at effective move sizes is equivalent to large third- and higher-order terms over the rectangle. The lemma restates this; it does not explain it. v1's own setting (final-norm logit at the last position, lens-coordinate swaps, layers 16–20) was not re-tested.

**Implication 3: the integrated lens [definition; measured negative].**
- **Definition.** T_ℓ^int = E_x ∫₀¹ w(s) ∇²f(μ + s(h_x − μ)) ds. It averages along one segment, so the lemma does not imply it.
- **When it matches the lemma.** The segment μ → h is the rectangle's diagonal only when a + b = h − μ (λ = 1, complementary parts). Even if ∇²f were constant along a − b, the matching weight would be triangular, w(s) = 4 min(s, 1−s); the code uses uniform weights.
- **Prior art.** With w(s) = −s ln s it is the corpus-averaged form of Integrated Hessians (Janizek, Sturmfels & Lee 2021).
- **Measured** (`hs_integrated.py`, `out/integrated_08b_L12.json`). 32 moves (16 test contexts × 2 positions) × 4 covectors; the lenses are averaged over 16 fit contexts × 1 position. Pearson r over 128 correlated points:

| λ | median \|I\| | local, per context | diagonal (4 nodes), per context | local lens | integrated lens |
|---|---|---|---|---|---|
| 0.1 | 0.002 | 1.00 | 1.00 | 0.26 | 0.31 |
| 0.3 | 0.028 | 0.78 | 0.94 | 0.27 | 0.37 |
| 1.0 | 0.44 | 0.29 (expl. −1.82) | 0.57 (expl. −1.88, slope 0.28) | 0.25 (expl. 0.07) | 0.16 (expl. −0.45) |

- At λ = 1 the integrated lens is *worse* than the local lens. A slope of 0.28 means the diagonal over-predicts by about 3.6×.
- **File caveat.** The stored per-context diagonal slope and explained variance at λ < 1 (0.0099 / −9979 and 0.079 / −119) were scored before the λ² factor was added: `hs_integrated.py` was edited 29 s after the JSON was written. Correlations are unaffected. Rescaling the stored values gives explained variance ≈ 0.998 (λ = 0.1) and ≈ 0.87 (λ = 0.3).

**Implication 4: the dissociation test is not licensed by the lemma [open].**
- The lemma is exact only for the pair-specific H̄_P(h; a₁, a₂). H-space is built from energies of pointwise or rectangle-averaged *diagonal blocks* (§6A), not from these forms.
- Projecting out an energy operator's top subspace bounds no individual a₁ᵀH̄a₂.
- Ablation changes the moves (hence the rectangle) and also removes first-order effects.
- The implemented test (`hspace.py` S7) uses 8-token span removals at other positions, in bf16. Its interactions are cross-position and partly formed before ℓ.

Whether ablating H-space removes interaction energy specifically is an empirical question; the 0.8B result is uninformative (§6A).

## 4. Proposition 4: one atom dictionary, different coefficients; the potential MLP lens
Status: (a) proved, elementary, MLP only; (b) assumption; (c) no scalar analogue at other sites; potential lens: definition, untested.

**(a) Per-block identity [proved].** Fix the SwiGLU sublayer of block s and an adjoint g, which may depend on context. At any point, in the block's post-norm coordinates x̃, the §2.1 formula holds. So any linear average of these Hessians (over data, along a path or over a rectangle) lies in the fixed span of the 2·d_ff forms {w_g w_gᵀ, sym(w_g w_uᵀ)}. Only the coefficients change:
- κ_i^prod = avg[ρ_i silu'(z_i)];
- κ_i^thr = avg[ρ_i silu''(z_i) u_i].

A shared dictionary does not imply a shared top-k subspace, because reweighting the atoms can rotate the eigenspace.

**(b) Lens-level form [assumption: §5 decoupling].** In h_ℓ coordinates each atom picks up three data-dependent pieces:
- the token-dependent RMSNorm Jacobian DNᵀ;
- a norm-curvature term with data-dependent geometry;
- the context-dependent transports J_{s←ℓ}(x) and J_{L*←s}(x).

A single geometry with scalar per-atom coefficients appears only after four approximations:
1. the transports are replaced by their corpus averages J̄;
2. the adjoint is replaced by the J-lens adjoint;
3. the coefficients are treated as independent of the transports;
4. the norm projector and the norm-curvature term are dropped.

The coefficients are then:
- local: κ_i^prod = E_x[silu'(z_i) r²] and κ_i^thr = E_x[silu''(z_i) u_i r²];
- integrated: κ_i^prod = E_x ∫₀¹ w(s) silu'(z_i(h^{(s)})) r(h^{(s)})² ds, along h^{(s)} = μ + s(h − μ). This is a 1-D path average, not Lemma 3's 2-D average.

*Measured* (v1: broadcast object, MLP atoms only, ℓ = 12; §5).
- This rung (L3) keeps a mean top-10 overlap of 0.55 (min 0.46) and cosF 0.74 against the exact Hessian.
- On the same four tokens, the exact per-context MLP-only rung L1 reaches 0.51–0.62, against L3's 0.49–0.59.

This suggests that most of the gap is missing mixer curvature rather than decoupling, but the decoupling error itself was not measured. Nothing here has been measured for the per-position object of §0.

**(c) Other sites: no scalar analogue.** At attention, DeltaNet and norm sites, the reads depend on other positions (e.g. κ_qp), on RoPE offsets, on the recurrent state and decays, or on the token's own direction (§2.2–2.3). Their averaged curvature has data-dependent coefficient *matrices* (d_h×d_h or larger per head), not per-atom scalars. These sites carry about 0.43–0.60 of the signed per-position share (§2.4).

**Definition 4.1 (potential MLP lens).** Use form (b) with κ_i^prod = sup_z silu'(z)·E[r_s²] = 1.0998·E[r_s²] and κ_i^thr = 0.
- The sup is attained at z* = 2.3994, where silu''(z*) = 0, so the zero threshold coefficient is exact for this choice (toy (v2): 1.09984, 2.3994).
- It is a weight-based surrogate, not the Hessian at any input. The gates z = W_g x̃ cannot all sit at z* at once, and opening a gate changes ρ, r and J.
- Within a block it equals, up to a positive scale, the weight-only bilinear interaction matrix of Pearce et al. 2024 applied to the SwiGLU weights. What it adds is transport and decoding through J̄ and Σ.
- The variant κ = E[silu'(z) | z > 0] is not equivalent:
  - its threshold term is nonzero: E[silu''|z>0] = 0.351 / 0.212 / 0.143 for z ~ N(0, σ²), σ = 1 / 2 / 3 (toy (v2));
  - it is undefined for gates that never fire on the data.
- The lens still depends on data through Σ and J̄.

**Hypotheses.**
- **H1 (dormant gates) [open; no implementation or test].** The potential lens ranks MLP gates that are closed on honest data higher than data-weighted lenses do. Caveats:
  - Only deeply closed gates have silu' ≈ 0. silu' < 0 for z < −1.2785, with minimum −0.0998 at z = −2.3994, and is still −0.053 at z = −4 and −0.012 at z = −6. So moderately closed gates enter the local lens with flipped sign, at up to about 9% of peak.
  - A uniform coefficient stops suppressing dormant gates but does not make them stand out among ~10⁶ atoms.
  - The v2 sleeper's saturation was measured as a logit margin (−9.9 honest vs +6.0 lie; `RESULTS-v2.md`), not as silu' ≈ 0 for identified neurons. Its gate is cross-position (last-position edits have no effect) and was planted by all-linear LoRA r16. It may lie outside any per-position MLP-atom lens.
- **H2 (natural-scale interactions): not supported.** At ℓ = 12 and λ = 1:
  - the integrated lens gives r = 0.16 against 0.25 for the local lens;
  - per context, the diagonal gives r = 0.13–0.57 against 0.98 for the full rectangle (§3).

**Related work.**
- Integrated Hessians (Janizek, Sturmfels & Lee 2021).
- Backdoor neurons that stay dormant on clean data (Fine-Pruning, Liu, Dolan-Gavitt & Garg 2018; Adversarial Neuron Pruning, Wu & Wang 2021).
- Saturation blindness of local attribution, the original motivation for Integrated Gradients.

## 5. Approximation 5: a decoupled atom form for the MLP part of the H-lens
Status: exact expansion proved; factorisation assumed; tested only on the broadcast object, where it failed its bar.

**Exact expansion [proved; from Lemma 1 and §2.1].** Fix a source p and a block m > ℓ. Let:
- y_q = h_mid,m[q], the residual entering the MLP of block m at position q;
- J_m(x; q, p) = ∂y_q/∂h_ℓ[p], which is nonzero only for q ≥ p;
- A_{m,i}(x, q) = [∂Σ_{p'∈V}h_{L*}[p']/∂h_m[q]] w_{down,i}, the exact downstream transport of neuron i's write. The MLP writes into h_m, the block-m output.

The MLP is position-wise, so in context x its contribution to ∂²/∂h_ℓ[p]² Σ_{p'}h_{L*}[p'] is
  Σ_{q≥p} Σ_i A_{m,i}(x,q) ⊗ J_m(q,p)ᵀ[ DN_qᵀ( silu'(z_i) sym(w_{g,i}w_{u,i}ᵀ) + silu''(z_i) u_i w_{g,i}w_{g,i}ᵀ ) DN_q + N_{m,i}(q) ] J_m(q,p).
Here N_{m,i} is neuron i's share of the RMSNorm curvature (§2.1).

**Decoupling [assumption]:**
- (A1) A_{m,i} → ω_{m,i} = J̄_m w_{down,i}, the J-lens at the block-m output. A mixer write lands at h_mid, where there is no fitted lens.
- (A2) The coefficients are replaced by separate means, β̄_{m,i} = E[r² silu'(z_i)] and ᾱ_{m,i} = E[r² silu''(z_i) u_i].
- (A3) The radial projector in DN and the term N are dropped.
- (A4) J_m(x; q, p) → the lag-resolved mean J̄^{(Δ)}_{m←ℓ} = E_x mean_p J_m(x; p+Δ, p), for Δ ≥ 0.

Then
  T_ℓ^MLP ≈ Σ_{m>ℓ} Σ_i ω_{m,i} ⊗ Σ_{Δ≥0} [ β̄_{m,i} sym(ã_i^Δ b̃_i^Δᵀ) + ᾱ_{m,i} ã_i^Δ ã_i^Δᵀ ],
with ã_i^Δ = J̄^{(Δ)ᵀ}_{m←ℓ}(γ⊙w_{g,i}) and b̃_i^Δ = J̄^{(Δ)ᵀ}_{m←ℓ}(γ⊙w_{u,i}).

- Keeping only Δ = 0 is a further truncation.
- A single target-summed transport (the J-lens convention) adds spurious q ≠ q' cross terms.
- This is a mean-field, K-FAC-style product of means (Martens & Grosse 2015). The J-lens is the exact mean of a quantity linear in the Jacobian. Here the transport enters quadratically, so averaging is itself an approximation.

**Error.**
- Per atom, the exact term is E[A κ BᵀSB], and the approximation is Ā κ̄ B̄ᵀSB̄, where S is the fixed read pair.
- The difference is the full set of joint central moments: Var(B) contracted with S; Cov(A,κ), Cov(A,B) and Cov(κ,B); and third- and fourth-order terms. Add the A3 terms and the lag truncation.
- In particular, E[(Bᵀa)(Bᵀb)ᵀ] = ã b̃ᵀ + Cov(Bᵀa, Bᵀb). So the rank-2 atom form is produced by the assumption; it is not a property of the exact averaged lens.
- No bound on any error term is proved.

**Scope.** SwiGLU sites only. They carry 0.40–0.57 of the signed share (§2.4). Norm, attention and DeltaNet sites do not have this form (§4c).

**Cost.**
- Only the write side uses the J-lens of record.
- The read-side transports are separately fitted averaged inter-layer Jacobians: d JVPs × n contexts per source layer. That took 1129 s for 11 target layers at one ℓ on 0.8B with 16 contexts (runlog 21:17).
- Storage is O(L·d²) per ℓ.

**Display [convention].**
- A transported read ã is a covector at layer ℓ. It can be shown as the pattern Σ_ℓã (Haufe et al. 2014) and decoded as u_tᵀJ̄_ℓΣ_ℓã. Decoding at the site instead gives a different vector.
- Reading an atom as "IF a AND b THEN w" is an untested hypothesis. Threshold atoms are not conjunctions, signed products are not ANDs, and no decode has been rated.

**Evidence [measured; broadcast object only; `RESULTS.md` T4, `out/ladder_08b.json`, `out/l1_08b.json`].**
- **Setup.** v1: Qwen3.5-0.8B, ℓ = 12, 16 wikitext contexts, T = 96, 10 output tokens, raw coordinates.
- **The object tested** is the *broadcast* MLP Hessian: perturbation at all valid positions, with cross-position blocks included. It is not the per-position T_ℓ.
- **L3** (A1–A3 plus input averaging; position T−1 also gets a constant adjoint): mean top-10 overlap with the exact Hessian 0.55 (range 0.46–0.73), cosF ≈ 0.74. This **failed the preregistered 0.7 bar**.
- **L2** (input averaging only): overlap 0.57, cosF 0.76.
- **L1** (exact per-context, MLP only; 4 tokens): cosF 0.74–0.82, against 0.72–0.83 for L2. ‖H − L1‖² is 0.90–1.02 × ‖H − L2‖² (recomputed from the logged norms and cosines). So the mixer curvature is about as large as L2's whole residual.
- **Not measured:** the decoupling error ‖L1 − L2‖ itself; the per-position whitened T_ℓ; mixer and norm atoms; the Σ-decode.
- **Overlap caveat.** The top-10 subspace is weakly separated (|λ10|, |λ11| = 3.77, 3.73 for ' the'), so overlaps should be read with cosF.

**Related work.**
- Hessian backpropagation (Bishop 1992; Pearlmutter 1994; Dangel et al. 2020).
- K-FAC (Martens & Grosse 2015).
- Bilinear interaction matrices (Sharkey 2023; Pearce et al. 2024).
- Data-averaged first-order global weights (Ameisen et al. 2025).
- The literature sweep found no identical construction. The combination is plausibly new, but it is an assembly of known parts.

## 6. Cost accounting and estimator properties

**Notation.**
- One "backward-equivalent" is one sequence-level VJP through the full depth.
- c_HVP is the cost of one HVP in backward-equivalents. It is assumed, not measured.
- For the 27B model, L* = 63.

### Lemma 7 (Hutchinson over positions) [standard; proved]
This is the Hutchinson estimator (Hutchinson 1990) in its block-probe form (Bekas, Kokiopoulou & Saad 2007).
- **Statement.** For one context and i.i.d. Rademacher s on V, Z = mean_p s_p (H(s⊗v))_p is **unbiased**, not exact, for mean_p H_pp v.
- **Covariance.** (1/|V|²) Σ_{p<q} S_pq v vᵀ S_pqᵀ, with S_pq = H_pq + H_qp. Its trace is ≤ (2/|V|) mean_p Σ_{q≠p} ‖H_pq v‖².
- **Variance decay [assumption].** The variance falls as 1/(#positions × #contexts) only if the per-row cross-position energy stays bounded as T grows. For a corpus-mean target, add Var_x(mean_p H_pp v)/n_ctx.
- **Scope.** First moments only. It does not make the second-moment operator diagonal-only; see Proposition 6.

### Cost (Qwen3.6-27B, d = 5120) [arithmetic; c_HVP assumed]
- **J-lens fit.** n·d backward-equivalents, and one fit yields J_ℓ for all 63 source layers.
  - At n = 25 (the draft's figure) that is 1.28×10⁵.
  - The lens of record (`_n1000` file) has an undocumented prompt count. Sibling configs we inspected report 233 (0.8B) and 417 prompts (`flagship/notes`), so n ≥ 100 (≥ 5.1×10⁵) is the more realistic comparison.
- **H-space at one layer ℓ.** N_probe × B sequence HVPs through blocks ℓ+1 … 63 cost ≈ N_probe·B·c_HVP·(63−ℓ)/64 backward-equivalents. The pass count is independent of d.
  - With 256 × 4 probes and c_HVP ≈ 4, that is ≤ 4.1×10³.
  - With c_HVP ≈ 6–7.5 (PyTorch benchmarks of an HVP at 4–5× a gradient), it is ≈ 6.1–7.7×10³.

| scope | vs J-lens, n = 25 | vs J-lens, n ≥ 100 |
|---|---|---|
| one layer | ≈ 17–31× cheaper | ≥ 67–125× cheaper |
| the 4 preregistered layers (7.4×10³–1.6×10⁴) | ≈ 8–17× cheaper | ≥ 32–69× cheaper |
| all 63 layers (≈ 1.29×10⁵ at c_HVP = 4, with depth truncation) | about break-even | ≈ 4× cheaper |

**Caveats.**
- These are FLOP counts, not wall-clock. The HVPs ran in fp32 without TF32; the lens fits run in bf16.
- The table sets a full d×d lens against a top-k subspace. A top-k J-space could itself be estimated with O(k) context-averaged JVPs/VJPs.
- Seeding several layers in one probe adds cross-layer blocks to the second moment. Per-layer operators need per-layer probes.

### Number of probes [open; measured]
No theorem fixes the probe count.
- M is a Monte Carlo second moment, not an operator we apply, so randomized range-finder (k + o) bounds do not apply. The draft's "(k + o) probes, independent of d" is withdrawn.
- The count depends on M's effective rank, its eigengap at k and the tails of y.

| model, layers | budget | split-half top-25 overlap | participation ratio | λ25/λ26 |
|---|---|---|---|---|
| Qwen3.5-0.8B, ℓ = 6/12/18 | 64 × 2, T = 96 | 0.74 / 0.73 / 0.69 | 23 / 19 / 28 | 1.05 / 1.01 / 1.07 |
| Qwen3.6-27B, ℓ = 16/28/40/52 | 256 × 4, T = 128 | 0.33 / 0.37 / 0.33 / 0.41 | 2.3 / 4.8 / 8.5 / 1.3 | 1.05 / 1.01 / 1.15 / 1.11 |

Sources: `out/hspace_small.json` and `out/pod/hspace_full.json`. At 27B the top-25 subspace is not determined at this budget.

### Averaged H-lens in atom form (§5)
- Cost: one forward pass over n prompts for the per-site statistics, plus the transports J̄_{L*←s} and J̄_{s←ℓ} for every downstream site layer s.
- The fixed-target lens of record supplies only J̄_{L*←s}, so the read transports cost up to (63 − ℓ) more fits.
- Storage: O(#sites) scalars (MLP: 2 × 1.1M), against d·d(d+1)/2 ≈ 6.7×10¹⁰ for the full symmetric tensor.

### Streaming [design; not implemented or tested]
- The Lemma 1 recursion needs only block s resident: about 1.5 GB of fp32 weights per 27B block, excluding embeddings.
- It also needs the boundary states y_s and the tangents ẏ_s for every block stored, and two full weight sweeps (about 2 × 54 GB in bf16) per probe batch.
- The 27B run did not stream; it held the whole bf16 model on a 96 GB GPU.

### Construction 6.1: the J-adjoint local H-lens
Status: error identity proved; fidelity measured only on Qwen3.5-0.8B (`hs_local.py`, `out/local_hlens_08b.json`).

**Construction.**
- Let Ψ_k: h_ℓ ↦ h_{ℓ+k} be blocks ℓ+1 … ℓ+k, at all positions. The exact adjoint is g_{ℓ+k}[p] = ∂F/∂h_{ℓ+k}[p].
- Let ĝ[p] = 1[p∈V] · J̄_{ℓ+k}ᵀc, the published lens at the output of block ℓ+k. Under the jlens convention (sum over later valid targets, mean over valid sources), ĝ is the corpus-and-position mean of g_{ℓ+k} on the lens's fitting distribution.
- The objective is F_k = ĝ·Ψ_k(h_ℓ). Its Hessian needs double backward through only k blocks.

**Error identity [proved; second-order chain rule].**
  ∇²F − ∇²F_k = DΨ_kᵀ(∇²_{h_{ℓ+k}}F)DΨ_k + ∇²_{h_ℓ}[δg·Ψ_k(h_ℓ)],   with δg = g_{ℓ+k} − ĝ held fixed.
1. The first term is curvature generated after block ℓ+k, pulled back through the exact k-block Jacobian.
2. The second term is the local curvature of the kept blocks, contracted with δg. It is linear in δg: a covariance term in the averaged Hessian, and also a quadratic term in M.
   - δg is not only context variance. It has a systematic position component (sources near T−2 have fewer later targets).
   - It also absorbs any mismatch between the lens's fitting corpus and length and those of the probes.

The construction is exact at k = L*−ℓ. For smaller k it is a heuristic, and no subspace-error bound follows.

**Size of term 1.** In §2.4 the nearest third of downstream blocks generates 36–62% of the signed share. So 38–64% lies beyond k ≈ (L*−ℓ)/3, and more beyond k = 1.

**Cost [counted, not timed].**
- k of L*−ℓ downstream blocks are differentiated: 1/17 at ℓ = 6 and 1/11 at ℓ = 12 for k = 1.
- This count omits the forward pass to ℓ (paid per probe here, because each probe uses fresh contexts), the unequal block costs, and the need for a fitted lens at ℓ+k.
- At ~5 forwards per block per HVP, k = 1 costs roughly 13–26% of the exact estimator [assumption].
- The saving is relative to exact H-space, given a published lens.

**Measured fidelity.**
- Setup: Qwen3.5-0.8B, ℓ = 6 and 12, one seed, 96 probes × 2 contexts × T = 96, identical probes for every k, in-sample scoring.
- The reference is the `hspace.py` raw estimator, i.e. M^row (§6A).

| k | blocks differentiated (ℓ = 6, ℓ = 12) | top-25 overlap with exact | exact-M energy in local top-25, as share of exact top-25 energy |
|---|---|---|---|
| 1 | 1/17, 1/11 | 0.60, 0.56 | 83%, 83% |
| 2 | 2/17, 2/11 | 0.66, 0.68 | 87%, 91% |
| 4 | 4/17, 4/11 | 0.70, 0.74 | 90%, 94% |
| 8 | 8/17, 8/11 | 0.77, 0.88 | 93%, 98% |

**Baselines [cross-run: `out/hspace_small.{json,pt}`, 64 probes, different data seed; energy bracketed from the top-64 eigenpairs].** Subspaces that need no HVPs capture the following shares of the exact top-25 energy:
- the global J-lens Gram top-25: 77–79% (overlap 0.57 / 0.57);
- the active J25: 71–75% (overlap 0.54 / 0.52);
- a random 25-dim subspace: ≈ 4% in expectation.

For scale, the exact H25 split-half overlap is 0.74 / 0.73.

**Reading.**
- At k = 1 the local lens is not clearly better than J-space: it gains about 5 energy points and has a similar overlap. From k ≈ 4 it is better.
- **Untested:** whether truncation preserves the J-orthogonal part of H-space; held-out scoring (U_k fit on probe set A, scored against M from set B); CIs and other seeds; the normalised estimators and the 27B model, where `hspace2.py` applies k = 1.

## 6A. What the 27B run measures

**Probe design (run 1, `hspace.py full`; frozen prereg `paper/SCOPE-hspace.md`).**
- Model and layers: Qwen3.6-27B, ℓ = 16/28/40/52, T = 128, |V| = 111, 256 probes × B = 4 windows per layer.
- Each probe draws:
  - one covector c ~ N(0, Σ_U);
  - one whitened direction v = Σ^{1/2}ξ, with ξ ~ N(0, I);
  - Rademacher signs s on every (window, valid position).
- Each probe computes one HVP h = H(s⊗v), where H is block-diagonal over windows. The samples are
  y_p = Σ^{1/2} s_p h_p = s_p Σ_{q∈V} s_q H̃_pq ξ,   for every valid p of every window.
- Everything below is conditional on the plug-in μ and Σ.

**Proposition 6 (what the sign-probe estimators converge to) [proved; toy (v2), see below].**
- **(a) One sign vector** (run 1's M; `hspace2` `loc_raw`): E[y_p y_pᵀ] = Σ_{q∈V} H̃_pq H̃_pqᵀ. The estimator converges to **M^row**, the per-position row energy, cross-position blocks included. It is neither M^lens (the draft's §0 operator) nor M^diag.
- **(b) Two independent sign vectors** s¹, s² with the same ξ (`loc_x`): E[y¹_p | ξ] = H̃_pp ξ, so E[½ sym(y¹_p y²_pᵀ)] = H̃_pp². This converges to **M^diag**.
  - It costs 2 HVPs per probe and is unbiased.
  - Each sample is indefinite, so finite-sample estimates carry negative eigen-mass.
- **(c) Two independent rectangle points** (`int_x`).
  - `hspace2.py` picks moved positions P (every 8th valid position, random offset per window). It splits each moved token's deviation, along a random d/2-dim whitened subspace shared by the probe, into a_p + b_p = h_p − μ (λ = 1).
  - It evaluates the Hessian at X(s,t) = X − (1−s)a − (1−t)b, at moved positions only, with (s, t) ~ U[0,1]² drawn independently for each of the two HVPs.
  - Then E[y^j_p | ξ] = H̄̃_pp ξ, where H̄ = ∫∫ H(X(s,t)) ds dt is the Lemma 3 average for the *joint* moves (a, b) on all of P. So E[½ sym(y¹_p y²_pᵀ)] = H̄̃_pp², and `int_x` converges to
    **M^rect = E_{x,c,split} mean_{p∈P} (H̄̃_pp)²**,
    the diagonal-block energy of the rectangle-averaged Hessian.
  - It is still an energy over random pairs, not a pair-specific form a₁ᵀH̄a₂. Each rectangle also moves the other positions of P.
- **(d) One rectangle point** (`int_raw`): E[y_p y_pᵀ] = E_{s,t} Σ_q H̃(X(s,t))_pq H̃(X(s,t))_pqᵀ. This is the mean row energy at a uniformly random rectangle point, which is neither the averaged Hessian nor its energy. The SCOPE-v2 description of all `int_*` variants as "a Monte Carlo estimate of the rectangle average" is correct only for `int_x`.
- **(e) Normalised variants** (`norm`, `xnorm`; the run-2 primary).
  - M_norm = E[y yᵀ/‖y‖²] has no closed form in H. Its eigenvectors equal those of E[y yᵀ] only for elliptically distributed y, which a mixture over contexts, positions and covectors need not be.
  - `xnorm` converges to E[m mᵀ], where m = E[y/‖y‖ | x, c, ξ, split] averages over signs (and rectangle points).
- **(f) Run 1's "M_lens"** (y averaged over a probe's B windows and |V| positions before squaring) converges to
    M^lens + (1/B) E_c Cov_x(mean_p H̃_pp) + (1/(B|V|²)) E Σ_{p<q} S_pq S_pqᵀ,   S_pq = H̃_pq + H̃_qp,
  where Cov_x(A) := E[(A − EA)²]. Neither bias term shrinks with more probes.

*Proof.* y_p = s_p Σ_q s_q H̃_pq ξ.
- (a) Use s_p² = 1, E[s_q s_q'] = δ_qq' and E[ξξᵀ] = I.
- (b) E_s[s_p s_q] = δ_pq gives E[y_p | ξ]. Independence of s¹ and s² factorises the cross moment, and E_ξ gives H̃_pp H̃_ppᵀ = H̃_pp².
- (c) As (b), with the expectation over each independent point taken inside its own factor. The uniform average over (s,t) is exactly Lemma 3's average.
- (d) As (a), at a random point.
- (f) Summing y_p over p gives |V|·mean_p H̃_pp ξ plus mean-zero sign cross terms, which are uncorrelated across pairs. Averaging over B i.i.d. windows then gives the 1/B terms. ∎

**Toy check (v2).** fp64; T = 5, d = 3; F with an affine Hessian (cubic F), so the rectangle average is available in closed form; 4×10⁵ probes. Relative Frobenius error of each estimator against its stated target, and against the nearest wrong target:

| estimator | vs its target | vs nearest wrong target |
|---|---|---|
| (a) | 0.16% | 184% vs M^diag |
| (b) | 0.44% | 66% vs M^row |
| (c) | 0.62% | 45% vs local M^diag |
| (d) | 0.43% | 310% vs (c)'s target |

**What run 1 reports [measured: `out/pod/hspace_full.json`; C-labels from the frozen prereg; verdict from `paper/verdict.py`].**

| layer | 16 | 28 | 40 | 52 |
|---|---|---|---|---|
| top-25 share of tr M^row (C2: ≥ 0.5) | 0.97 | 0.85 | 0.86 | 0.99 |
| participation ratio | 2.3 | 4.8 | 8.5 | 1.3 |
| λ1/λ2 | 4.9 | 4.9 | 1.2 | 13.1 |
| split-half top-25 overlap (C1: ≥ 0.7) | 0.33 | 0.37 | 0.33 | 0.41 |
| H25–J25 overlap (C3: < 0.5; random 0.005) | 0.15 | 0.23 | 0.21 | 0.29 |
| H25 vs "M_lens" top-25 overlap | 0.44 | 0.58 | 0.56 | 0.54 |
| share of tr M^row in J25 / Jglob / random-25 | 0.34 / 0.35 / 0.007 | 0.24 / 0.29 / 0.006 | 0.27 / 0.29 / 0.005 | 0.30 / 0.34 / 0.004 |
| 1σ interaction ratio, top-6 H pairs vs random (C4: ≥ 10) | 79.6 | 102.5 | 78.7 | 97.2 |

- **C0** (planted bilinear gate at ℓ = 40, calibrated to the 25th eigenvalue): span recovery 0.997. The instrument is valid.
- **C5 and C6** (ablations) are not yet in the file. S7 is being re-run on the saved tensors (`hs_s7.py`, a logged deviation; runlog 20:54) [open].

**Reading.**
- **The verdict is FAIL** by the frozen rule. C1 fails at all four layers, and every non-FAIL verdict needs C1 at ≥ 2 layers, so C5 and C6 cannot change it. This settles the preregistered question for M^row on this model.
- **C2's pass reflects a few dominant eigenvalues** (participation ratio 1.3–8.5, λ1/λ2 up to 13) in an operator whose top-25 subspace is not reproducible. That is consistent with heavy-tailed per-sample magnitudes (run-2 hypothesis H-R), not with a stable low-dimensional M^row. This is an interpretation; the run-2 test is pending.
- **C3's pass is uninformative.** Two halves of the same estimate overlap only 0.33–0.41, so an overlap of 0.15–0.29 with J25 does not separate "H ≠ J" from estimation noise.
- **C4 has no variance-matched control.** At 0.8B, the pair-norm product alone accounted for 23–29× (below).
- **The "M_lens" overlaps compare two biased operators** (Prop. 6f).

**The same estimator at 0.8B (dev run, `hspace.py small`, ℓ = 6/12/18, pre-freeze code; `out/hspace_small.{json,pt}`) [measured].** C-labels are applied for comparison only; this run is not part of the 27B verdict.
- **Spectrum.** The top-25 share of tr M^row is 0.59 / 0.63 / 0.61, with participation ratio 23 / 19 / 28. Under the same Σ weighting:
  - a structureless GOE-type Hessian gives 0.35 / 0.38 / 0.36 (participation ratio 116 / 92 / 109);
  - an identity Hessian gives 0.86 / 0.89 / 0.87 (participation ratio 8.5 / 7.3 / 9.4).

  So the concentration is real but moderate (nulls recomputed for v2 from `Sh`).
- **J-space and PCA.** The H25–J25 overlap is 0.54 / 0.52 / 0.53. Both subspaces lean toward the top activation PCs: H25–PCA25 is 0.78 / 0.73 / 0.69 and J25–PCA25 is 0.48 / 0.50 / 0.64. A null that keeps each subspace's PCA25 energy predicts an H–J overlap of 0.38 / 0.36 / 0.45.
- **Ablation KL.** H25 costs 2.28 / 2.44 / 1.83 nats. That is:
  - 58 / 170 / 69× the mean of three random whitened subspaces;
  - 1.74 / 1.75 / 1.37× J25;
  - only 1.21 / 1.14 / 1.03× raw PCA25 (1.88 / 2.14 / 1.78).

  The raw variance removed is 31–34% for H25, 34–38% for PCA25, 23.5–29% for J25 and 2.4% for random subspaces. So the multiple over random is mostly a variance effect.
- **1σ interactions.** The ratio is 40 / 40 / 69×, but the pair-norm product alone is 23–29× (28–31× for top-6 PCA pairs) [review scratch].
- **Dissociation.** H25 gives 1.00 / 1.32 / 1.39, with CIs [0.54, 1.82], [0.69, 2.95] and [0.58, 3.49]; random subspaces give −0.80 to 3.84. The test is uninformative: it neither shows nor rules out interaction-specific damage.
- **Against the 27B criteria.** It fails C3 and C6 at all three layers and C1 at ℓ = 18 (0.69). It passes C0 (planted span 0.96), C2, C4 and C5.
- **Implication.** At 0.8B, H25 under M^row is largely the high-variance subspace. The variance-matched PCA25 comparison is the relevant baseline at 27B.

**Run 2 (`hspace2.py`; exploratory, designed after seeing run 1).**
- It adds `loc_x` (→ M^diag), `int_x` (→ M^rect) and the spatial-sign variants.
- With these variants, only a 0.8B smoke run exists (4 probes, T = 64; `out/hspace2_smoke.json`). At that size the cross-moment estimates are dominated by noise: `int_x` has negative eigen-mass 0.70 of its trace and split-half 0.21.
- A pre-v2 0.8B run (raw/norm/trim variants) gave split-half 0.70 (raw) vs 0.87 (spatial sign) at ℓ = 6 before an OOM at ℓ = 12 (runlog 20:34; `out/hspace2_small.log`).
- No 27B run-2 numbers exist yet [open].

## 7. White-box signals in control protocols (goal 3)

**Setup.**
- At step k, the untrusted agent emits action a_k. The protocol chooses execute, replace (defer to or edit by the trusted model) or audit, under an audit budget α and a replace budget β.
- The black-box score B_k sees the action text.
- The white-box score W_k sees residual states at positions *before* the first token of a_k.

### G1. LR fusion is optimal at a fixed honest FPR; fixed-weight fusion is not [standard result: Neyman–Pearson lemma]
- **Statement (oracle).** Let p_h, p_a be fixed, known joint laws of (B, W), and Λ = p_a/p_h.
  - At honest FPR α, the test φ* = 1{Λ > c} + γ·1{Λ = c}, with E_h[φ*] = α, maximises TPR over all tests φ(B, W).
  - The randomisation γ is needed whenever Λ has atoms, e.g. integer 0–9 suspicion scores. Counterexample without it: P_h(B=1) = 0.02, P_a(B=1) = 0.5, α = 1%, W ~ U(0,1) independent. Every non-randomised Λ-threshold has TPR 0, while B + W thresholded at 1.5 has FPR 0.01 and TPR 0.25.
  - Tests that use only B, only W, or any fixed fusion g(B, W) are among the competitors. Hence TPR*_(B,W)(α) ≥ max{TPR*_B(α), TPR*_W(α), TPR_g(α)}.
  - LR level sets are nested. So one ordering (audit the top α, replace the next β) is optimal for audit/replace whenever an audit is worth at least a replace.
  - Neyman–Pearson 1933; LR score fusion: Nandakumar, Chen, Dass & Jain, TPAMI 2008. No novelty is claimed.
- **What it does not guarantee:**
  - (i) **Net, not item-wise.** At matched α an informative fusion generally still loses some attacks that B alone catches. NP gives only E[rescued − lost] ≥ 0.
  - (ii) **True Λ only.** An estimated Λ̂ (e.g. a cross-fitted classifier) can do worse than B alone, through tail error at α = 1–2%, miscalibration, or attack shift. Hua et al. note the same limitation.
  - (iii) **Fixed attack law only.** LR fusion lowers suspicion when W looks honest. Example: H0 (B, W) ~ N(0, I); attacks fit as B ~ N(1,1), W ~ N(2,1), so the statistic is B + 2W. An attacker who keeps W at its honest law is caught at 1% FPR with TPR 0.030, against 0.092 for B alone (0.464 if non-adaptive). A safety claim needs a minimax version, and B-only must always be reported.
- **Fixed-weight fusion has no guarantee.**
  - If W ⫫ (B, Y) and c > 0 is fixed, thresholding B + cW is a randomised test based on B, so its TPR ≤ TPR*_B(α). If B is continuous with a monotone LR, it is also ≤ the TPR of thresholding B. Gaussian, d' = 1, α = 1%: B 0.092; B + W 0.053; B + 0.5·max(0, W) 0.082.
  - Independence from B matters, not just from Y. With B = Y + N and W = N, B − W is perfect while B + W drops to 0.034.
  - The failure comes from a fixed a-priori weight, not from additivity. Under B ⫫ W | Y, summing calibrated log-LRs *is* LR fusion.
- **Empirical illustration (exploratory; different setting; `interpcontrol/em/results/final/`).**
  - **Setup.** Untrusted model Qwen-Coder-Insecure-32B; trusted monitor Qwen3.5-2B. The positives were 216 bandit-flagged *vulnerable code outputs*, not attacks. The fusion was the preregistered add-only z_BB + 0.5·max(0, z_WB), with a near-chance WB (AUROC 0.483 [0.429, 0.536]). Thresholds were refit on 117 honest items to 2% audit / 10% flag.
  - **Result.** Rescued 2, lost 41: net −39, bootstrap net-rate CI [−0.278, 0.023]. All 41 crossed at the replace threshold, and audits fell from 62 to 35.
  - **Not a clean pure-noise instance.** The permutation placebo gives mean net −10.75 (95th percentile +2.0), and the observed −39 has p_perm = 0.984. A review re-analysis (not in RESULT.md) attributes the excess to WB correlating with BB on honest items (Spearman 0.25, p = 0.008) but not on vulnerable ones, which inflates the refit threshold.
  - **Caveats.** Held-out honest flag rates were 17.9% (BB) vs 14.5% (fused), against a nominal 10%. With fixed thresholds the same scores give rescued 5, lost 0, but honest audits rise from 3 to 13 of 117, and BB alone at that budget audits more (102 vs 82 of 216). The run's verdict was **Inconclusive**: the lens sanity gate failed (0.705 < 0.8).
- **Prior work.** Hua et al. 2025 (arXiv 2507.15886) already allocate interventions by NP with KDE-estimated LRs. They combine two black-box monitors under a cost budget, and their companion post applies it to a probe plus a black-box monitor. G1 is the zero-cost, always-call special case.
- **Our LR-fusion test.** Only a 12-trajectory smoke run exists (`out/ctrl_score_small.json`: 7 attacks, BB AUROC 0.51, BB TPR 0 at every budget). Its "LR" arm is a cross-fitted two-feature logistic regression, an estimate Λ̂, not Λ.

### G2. Hidden goals as low-rank signals; the workspace as dimension reduction
Status: standard detection-theory and random-matrix results applied to a stylised model; the workspace benefit depends on an unmeasured signal-capture fraction; empirical test pending.

**Model (honest-calibrated, no attack labels).** r_k is the state at the start of action k, projected by P onto a p-dimensional subspace, then residualised and whitened with honest statistics.
- **H0 (assumption A0):** the r_k are i.i.d. N(0, I_p) across steps and trajectories. This needs:
  - (i) oracle context-conditional residualisation, r_k = Pz_k − E_honest[Pz_k | context];
  - (ii) a known honest covariance;
  - (iii) no persistent honest per-trajectory component and no within-trajectory correlation.

  A0 is strong. Any persistent honest component (task topic, a trajectory random effect) is itself a mean or spike under H0. That gives a false-alarm floor that does not shrink with K, and it may concentrate in the same subspace as the signal. `ctrl_score.py` approximates (i) by one pooled honest mean, so A0 does not hold there.
- **H1:** r_k = θ s_k u + g_k, with g_k ~ N(0, I_p), u an unknown persistent unit direction, and s_k = 0 except on a fraction f of the K steps.
  - **(a) Fixed sign,** s_k ∈ {0, 1}: the mean is fθu, a mean shift.
  - **(b) Symmetric sign,** s_k ∈ {−1, 0, +1}: the mean is zero and the second moment is I + fθ²uuᵀ, a rank-one spike.
  - In both cases r_k is a Gaussian mixture, so S = (1/K)Σ r_k r_kᵀ is not Wishart. Given s, it is a rank-one information-plus-noise matrix with uncentred spike ℓ = fθ². Centring by the trajectory mean gives f(1−f)θ² in case (a).

**Results (asymptotic, p/K → c) [standard].**
- **Spectral test.**
  - λ_max(S) leaves the Marchenko–Pastur edge (1 + √c)² iff fθ² > √(p/K), and then sits at (1+ℓ)(1+c/ℓ). Sources: BBP (Baik, Ben Arous & Péché 2005; Baik & Silverstein 2006; Paul 2007); for additive perturbations, Benaych-Georges & Nadakuditi 2012.
  - So the λ_max test is consistent iff **K > p/(fθ²)²**.
  - This threshold belongs to λ_max, not to every detector. Below it, LR and linear-spectral tests keep power (Onatski, Moreira & Hallin 2013). The trace test scales the same way with a worse constant. A per-step scan max_k‖r_k‖² can win when f is small and θ large.
- **Mean test.** ‖K^{−1/2}Σ_k r_k‖² − p is χ²_p − p under H0 and noncentral χ²_p(Kf²θ²) − p under (a).
  - It detects once Kf²θ² ≳ (z_α + z_β)√(2p), i.e. K ≳ √(2p)/(fθ)² up to that constant. This is the classical √p rate (Ingster & Suslina 2003; Baraud 2002).
  - Use exact χ²_p critical values at small p (χ²_25 has skew 0.57).
  - Under (b) the mean test has no power.
- **Which test.** In case (a), K_spike/K_mean = √(p/2)/θ², so the mean test wins whenever θ² < √(p/2): 3.5 at p = 25 and 51 at p = 5120. The spectral test is needed only in case (b) or at large θ².

**Dimension reduction.**
- **Signal-capture fraction.** Let ρ = θ_P²/θ² ∈ [0,1] be the fraction of whitened (Mahalanobis) per-step goal energy kept after projecting and re-whitening.
  - ρ = 1 iff C⁻¹δ ∈ span(P), where δ is the goal displacement and C the honest noise covariance. δ ∈ span(P) is not enough when the noise couples the subspace and its complement.
  - A random p_P-dimensional subspace has E[ρ] ≈ p_P/p.
- **Gains.** Watching p_P dimensions instead of p divides the required K by (p/p_P)ρ² for the spectral test and by √(p/p_P)ρ for the mean test.
- **For p = 5120 and p_P = 25:**
  - the gains are at most **204.8ρ² and 14.3ρ**;
  - both break even at ρ = √(25/5120) ≈ 0.07, so the workspace helps only if it holds ≳ 14× the whitened goal energy of a random 25-dim subspace;
  - under model (a) with ρ = 1, taking the better test in each space gives a gain of ≈ 4.05θ² for 3.5 < θ² < 51, between the two limits.
- **Finite sizes.** These factors are asymptotic, best-case (ρ = 1) values.
  - For the mean test, exact noncentral χ² at α = 5% gives 12.5×, 11.3× and 10.3× at 50%, 80% and 95% power, against the asymptotic 14.3× (verified, scipy).
  - For the spectral test in case (b), a review Monte Carlo gives roughly 50–100× [open: not in repo].
- **Covariance estimation.** The full-dimensional comparison assumes a known honest covariance. Whitening from n honest steps inflates out-of-sample noise by about n/(n−p−1): 3.1× at n = 7500, p = 5120. The fair controls are random-25, and full d only with n ≫ p. The smoke run (≤ 56 honest rows whitening d = 1024) does not test G2.

**What G2 shows.** Within the stylised model, a detector that watches fewer dimensions needs fewer steps, if the subspace captures a large ρ and does not also concentrate persistent honest variance. It does not show that tens of steps suffice; that depends on θ, f and ρ, none of which has been measured.
- **The workspace hypothesis.** The global-workspace hypothesis motivates looking for a high ρ in J25/H25; it does not establish it.
- **Prior work.** The gain is the classical matched-subspace detection gain (Scharf & Friedlander 1994). Spectral detection of hidden subpopulations in activations is established (Tran, Li & Madry 2018; Hayase et al. 2021).
- **The open question:** is ρ(J25 or H25) ≫ 0.07 for agent intent, net of honest task variance?

### G3. Known threats: matched filters on Concept-Lens directions [standard]
- Some side tasks are enumerable. BashArena has four: exfiltration, malware download, privilege escalation and disabling the firewall.
- For these, the matched filter max_i⟨φ(τ_i), r⟩ is the natural statistic, where φ(τ_i) is the Concept-Lens direction of side task i's multi-token description. Taking the max over m candidates costs about a √(2 log m) shift in threshold.
- Gong et al. (arXiv 2608.31084) report 80–91% scoring accuracy on Qwen3.6-27B when the candidate is supplied, which is the known-threat case.
- The same H0 caveats as G2 apply.

### G4. Conditional intent as a product site [open: hypothesis]
- A policy "act IF unobserved AND opportunity" is a conjunction. At honest-looking steps its gate is closed, so first-order reads miss it (the v2 saturation result).
- The potential lens (Definition 4.1) might rank such gates. Three caveats apply:
  - it is untested;
  - it covers per-position MLP atoms only, while the one trained gate we studied is cross-position;
  - a closed gate is also nearly invisible to every *local* second-order lens (v2: the local quadratic explained ~7% of a +5.0-logit interaction).

**Planned tests.** Dataset: `adityaasinha28/control_arena_bash`, 750 attack and 750 honest BashArena trajectories, read by Qwen3.6-27B as a *surrogate* white-box reader. The control read hit an OOM at 16k tokens and will be rerun at 8192 (runlog 20:07).
1. **G2.** Do the honest-calibrated mean and spike statistics in J25/H25 beat the same statistics in random-25? Compare with full d only under a well-estimated covariance, and report estimates of ρ.
2. **G3.** Does the known-threat matched filter work?
3. **G1.** Does LR-fused B + W have **net (rescued − lost) > 0 at matched α** at 1/2/5% audit budgets? Report B-only alongside, and add-only fusion as the contrast. "No lost attacks" is not the criterion (G1(i)).
4. **Pre-emption.** How many steps before the black-box alarm does the white-box alarm fire?

## 8. Open questions
1. **Does H-space exist at 27B?**
   - For M^row (run 1, preregistered): **FAIL**. C1 fails at all four layers, with split-half 0.33–0.41. C5 and C6 are pending but cannot change the verdict.
   - Open: is M^diag, M^rect or the spatial-sign operator reproducible at 27B (run 2, exploratory)?
2. **Does the integrated H-lens fix v1's effective-scale failure?** Not at 0.8B, ℓ = 12, λ = 1 (r 0.16 vs 0.25 for the local lens). Open: other layers, the triangular weight, and v1's own setting.
3. **Does the potential lens find the planted pirate × animal sleeper from honest data?** Open; nothing is implemented. The gate is cross-position, which a per-position MLP lens may not see.
4. **Is H-space distinct from J-space?**
   - At 0.8B, H25 under M^row is largely the high-variance subspace (H25–PCA25 0.69–0.78; ablation KL only 1.03–1.21× PCA25).
   - At 27B, H25–J25 is 0.15–0.29, which is uninformative given split-half 0.33–0.41. Open.
5. **Real-model verification** of Lemma 2.2 and of the MLP atom split. Open until `tests/test_attn_atoms.py` and the split script are committed.
6. **Quadrature convergence.** An n_q sweep for the rectangle average on all draws. Open.
7. **Does Construction 6.1 preserve the J-orthogonal part of H-space?** Also open: held-out fidelity, and fidelity at 27B with normalised estimators.
8. **Is ρ(J25/H25) ≫ 0.07 for agent intent (G2)?** Open.
