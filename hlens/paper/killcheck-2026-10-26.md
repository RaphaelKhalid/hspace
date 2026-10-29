# Kill-check: cross-position H-lens and the J x J interaction map (2026-10-26)

This extends `lit-2026-10-05.md` and the Oct 4 literature sweep (private notes, not in this repo). Items covered there are not repeated unless something new turned up. Nothing was built and nothing was spent.

Evidence tags:
- **[full]**: I read the full text or the relevant section myself.
- **[section]**: I read the section that matters, via a summarizer over the full HTML.
- **[abstract]**: abstract page only.
- **[TODO-verify]**: seen in a search result only. Do not cite until opened.

## Verdicts

| Idea | Verdict | Deciding works |
|---|---|---|
| 1. Cross-position readout-anchored Hessian (d²f / dx_q dx_p, averaged, whitened, top singular directions) | **PARTIALLY SCOOPED** | Kamath et al. 2025 (QK attributions); Gurnee, Sofroniew et al. 2026 (J-lens readouts of W_Q/W_K/W_V/W_O, broadcast heads, frozen-QK variant). Lee et al. 2026 adds dataset-level low-rank QK subspaces for binding. |
| 2. J x J interaction map plus J x J / J x ⊥ / ⊥ x ⊥ energy split | **OPEN, but thin** | Gurnee, Sofroniew et al. 2026 list "beyond a bag of concepts" as future work, and read individual transcoder AND features and heads through the J-lens. Pearce et al. 2024 already do pairwise interaction matrices in a feature basis. |

Nobody I found computes a corpus-averaged mixed cross-position second derivative of a readout at a mid-layer residual and takes its subspace. Nobody I found measures the J x J block of any Hessian. But the main pieces of idea 1 are known, and idea 2 is a basis change on an operator this project already has.

## Differentiators that survive

**Idea 1.** QK analyses explain the pre-softmax attention *score*. The J-lens is first order across positions. What neither covers:
- the readout-anchored mixed derivative through *all* downstream blocks, not one head's score;
- the "OV transport then downstream MLP" term: x_q is moved to p by a value path, then combines with x_p in a later SwiGLU at p;
- gated-attention output-gate x value terms (gate read at p, value read at q). Qwen3.6-27B has these. QK attribution does not see them;
- Gated DeltaNet cross-position terms (query at p against a key-value state written at q). Qwen3.6-27B is mostly DeltaNet layers. I found no interpretability work on DeltaNet states;
- softmax competition terms, which Kamath et al. list as unsolved ("we do not yet have a way of automatically surfacing the important inhibitory effects").

State the claim as: "a corpus-averaged, readout-anchored cross-position curvature operator, split into QK-routed, OV-then-MLP, gate x value and DeltaNet parts, with decoy controls". Do not claim "first cross-token second-order analysis". Integrated Hessians (2021) and HETA (2026) already define cross-token Hessians, and QK work already does cross-position bilinear decompositions.

**Idea 2.** The surviving claim is a *measurement*: how much readout curvature sits in the J x J block once first-order gain is controlled. The method alone is not novel. It is Pearce-style interaction matrices with J-lens vectors as the basis, and this project's own Cor 5 already writes SwiGLU atoms as "IF [J-read a] AND [J-read b] THEN [write]".

## Reasons the ideas may be wrong on their face

**Idea 1 partly reduces to known objects.** Only attention and DeltaNet mix positions. So the cross-position block is a sum over downstream mixing layers. For one softmax head with readout gradient g at p, the mixed derivative is, to leading order:
- a scalar times W_QK, where the scalar is a_pq · gᵀ(V_q − mean value at p). Averaged over data, this is weight-based QK analysis (Elhage et al. 2021; Franco & Crovella 2024; HeadVis 2026) reweighted per head;
- plus a rank-1 term (W_Vᵀ g) ⊗ (W_Qᵀ(K_q − mean key)) · a_pq.

The key-side factor W_Vᵀ g is "which value direction moves the readout". With a J-lens adjoint, that is close to what the J-lens paper's broadcast-head analysis measures. The OV-then-MLP term is the same-position Hessian with one leg moved through the value path. The same-position H is already J-dominated (6-13x), and the J-lens paper shows a subset of heads selectively relays J-space content. So expect the key-side singular space to be J-aligned **by construction**. The query side reduces to QK structure. Before building, predict both from this decomposition. If the prediction holds, the result is a known object reweighted.

**Idea 1 heavy tails will be worse than same-position.** The mixed derivative scales with a_pq. Attention is sparse, and sinks (position 0, massive-activation dims) take a lot of mass. Run 2 already had 0.1% of rows carrying 97.5% of the energy. Exclude sinks and condition on lag, or the top direction will be the sink.

**Idea 1 is lag-dependent.** Rotary dimensions make the effective QK form depend on p − q. Averaging over all lags mixes rotations. Use a lag-conditioned or lag-banded operator. Yan et al. (2608.25347) already show first-order cross-position Jacobian energy is mostly diagonal ("short horizons") plus a few "broadcast origin" positions. Expect the cross block to be thin and concentrated on those same positions.

**Idea 1 was already half-measured here.** The run-1 operator was per-position row energy, so it included cross-position blocks (paper §4, theory-v2 §6A). The corrected anatomy gives an off-position share of 19-38%. Cross-position curvature is a minority of the total. Whatever subspace the off-diagonal block has, it competes against that baseline.

**Idea 2 J-dominance may be gain, not structure.** The readout Hessian is Σ_s J_sᵀ ∇²[·] J_s, pulled back through the same downstream Jacobians the J-lens averages. Directions with large J-gain get gain-squared weight in any derivative of the readout. Ostrow et al. (2610.31955) frame J-lens directions as amplified, Lyapunov-like modes, though they say the weighting is "not necessarily a strong weighting". A large J x J share is the default expectation, not a finding. It only means something against **first-order-gain-matched** decoys, or as interaction per unit main effect. HANDOFF notes the v10 J-matched twins were loose (82-93% deviation in J-response energy). Fix that first.

**Idea 2 depends on whitening and on k.** A random k-dim subspace in d = 5120 holds about (k/d)² of the energy. J-space is overcomplete and sparse-nonnegative, not a subspace. The energy split changes with whitening (Anthropic's J-lens does not whiten) and with the choice of J25 vs top-5. Fix both in a prereg.

**Idea 2 has a scoop risk.** The J-lens paper names this as future work (quoted below). Anthropic has the attribution-graph and QK tooling to do it.

## Close works

### A. Cross-position bilinear / QK analyses (idea 1)

| Work | Authors | Year, venue | URL opened | Tag | How close |
|---|---|---|---|---|---|
| Tracing Attention Computation Through Feature Interactions | Kamath, Ameisen, Kauvar, Luger, Gurnee, Pearce, Zimmerman, Batson, Conerly, Olah, Lindsey | 2025, Transformer Circuits | https://transformer-circuits.pub/2025/attention-qk/index.html | [full] | **Closest to idea 1.** It writes each head's pre-softmax score as a sum of (query-position feature) x (key-position feature) terms through W_QK, per prompt, then attaches head loadings to attribution-graph edges. It notes these matrices are "typically approximately low-rank". It explains scores, not readout curvature, and leaves softmax competition and OV-side combination open. |
| HeadVis: An Interactive Tool For Investigating Attention Heads | Luger, Kamath, Finkbeiner, Goel, Jermyn, Zimmerman, Batson, Conerly | 2026-05-04, Transformer Circuits | https://transformer-circuits.pub/2026/headvis/index.html | [full] | Distribution-level head analysis with low-rank QK/OV component projections and QK attributions. It argues attention units are rank > 1 (perhaps rank 2-6). Per head, score-level. No readout Hessian. |
| Decomposing Query-Key Feature Interactions Using Contrastive Covariances | Lee, Belinkov, Viégas, Wattenberg | 2026-02-04, arXiv | https://arxiv.org/abs/2602.04752 (HTML read) | [section] | Dataset-averaged contrastive q·kᵀ covariances per head, SVD into low-rank query/key subspaces. Finds an order-ID binding subspace (rank 2-3) and a lexical one (rank 9-10) on Llama-3.1-8B and Qwen3-4B. This is a low-rank cross-position subspace for binding, found without Hessians. It stops at attention, not output. |
| Verbalizable Representations Form a Global Workspace in Language Models (J-lens) | Gurnee, Sofroniew et al. | 2026, Transformer Circuits / arXiv 2607.15495 | https://transformer-circuits.pub/2026/workspace/index.html | [full] | Three relevant parts. (1) The J-lens averages over target positions t' ≥ t, so it is already a first-order cross-position object, and it has a frozen-QK variant. (2) It decodes each head's W_Q, W_K, W_V, W_O through the J-lens (Fig. 93-94). That is a J-space reading of the QK bilinear form. (3) "Broadcast heads" selectively relay J-space content between positions. |
| Decomposing the QK circuit with Bilinear Sparse Dictionary Learning | Wynroe, Sharkey | 2024-07-02, LessWrong/AF | https://www.greaterwrong.com/posts/2ep6FGjTQoGDRnhrq/decomposing-the-qk-circuit-with-bilinear-sparse-dictionary | [full] | Sparse query/key transcoders plus a sparse learned mask over query-feature x key-feature x head. Score-level only; OV left as future work. |
| Sparse Attention Decomposition Applied to Circuit Tracing | Franco, Crovella | 2024, arXiv 2410.00340 | https://arxiv.org/abs/2410.00340 | [abstract] | Inter-head communication features are sparse in attention-head singular vectors (GPT-2 small, IOI). Weight-SVD precedent for the W_QK part of idea 1. |
| Short Horizons and Sparse Concepts: a Mathematical View of the Readout in the J-lens | Yan, Ding, Tan et al. | 2026-08-26, arXiv 2608.25347 | https://arxiv.org/html/2608.25347 | [section] | Maps first-order cross-position Jacobian energy E(t, t'): mostly diagonal, plus sparse "broadcast origin" positions. Defines a Hessian only to measure the affine readout's "intrinsic nonlinear error". No mixed cross-position term. |
| A Mathematical Theory of the Jacobian Lens | Ostrow, Qian, Liu, Fiete | 2026-10-22, arXiv 2610.31955 | https://arxiv.org/html/2610.31955 | [section] | First-order only. Defines lagged cross-position Jacobians and frames J-lens directions as coherent, Lyapunov-like amplified modes. Relevant to the gain confound in idea 2. |
| Representational Analysis of Binding in Language Models | Dai, Heinzerling, Inui | 2024, arXiv 2409.05448 (venue TODO-verify) | https://arxiv.org/abs/2409.05448 | [abstract] | Low-rank ordering-ID subspace, found by PCA, causal under patching. The binding intuition behind idea 1 already has a first-order low-rank answer. |
| Cell-Based Representation of Relational Binding in Language Models | Dai, Heinzerling, Inui | 2026, ACL (2604.19052) | not opened | [TODO-verify] | Search snippet says a low-dim linear binding subspace found by PLS. Same caveat as above. |

### B. Cross-token second-order attribution (idea 1, input level)

| Work | Authors | Year, venue | URL opened | Tag | How close |
|---|---|---|---|---|---|
| Explaining Explanations: Axiomatic Feature Interactions for Deep Networks (Integrated Hessians) | Janizek, Sturmfels, Lee | 2021, JMLR 22 | https://jmlr.csail.mit.edu/papers/volume22/20-1223/20-1223.pdf | [full] | Cross-token Hessian interactions on DistilBERT sentiment, including the "not bad" negation interaction. That is idea 1's negation intuition, at the input embeddings, per sentence, summed to scalars. No subspace. |
| Hessian-Enhanced Token Attribution (HETA) | Pramanik, Maliha, Bastian, Jha | 2026-04, ICLR 2026 | https://arxiv.org/html/2604.13258 | [section] | The Hessian is over all positions' input embeddings, so off-diagonal token-pair blocks exist. The estimator only probes each token's own block, so cross-position blocks are not estimated. Per-token scalars. |
| SPEX: Scaling Feature Interaction Explanations for LLMs | Kang, Butler, Agarwal, Erginbas, Pedarsani, Ramchandran, Yu | 2025, arXiv 2502.13870 (ICML 2025 per search) | https://arxiv.org/abs/2502.13870 | [abstract] | Black-box sparse-Fourier interactions among about 1,000 input features (tokens/sentences). Input-level, no internals. |
| Explaining Caption-Image Interactions in CLIP Models with Second-Order Attributions | Möller, Tilli, Vu, Padó | 2024-2025, TMLR | https://arxiv.org/abs/2408.14153 | [abstract] | Second-order attribution onto interactions between two inputs of a dual encoder. Method precedent for mixed derivatives across two input streams. |
| Feature Interactions Reveal Linguistic Structure in Language Models | Jumelet, Zuidema | 2023, ACL Findings | https://arxiv.org/abs/2306.12181 | [abstract] | Token-pair interaction attributions can recover grammar rules in PCFG-trained models. Input-level. |
| The Metagame of Interpretability and Meta-Attributions (Meta-AttnLRP) | not recorded | 2026, arXiv 2605.06295 | https://arxiv.org/abs/2605.06295 | [abstract] | Directional token-token interaction scores in instruction-tuned LMs via a Shapley metagame over AttnLRP. Not a Hessian, not a subspace. Authors TODO-verify. |

### C. Feature-pair interaction maps (idea 2)

| Work | Authors | Year, venue | URL opened | Tag | How close |
|---|---|---|---|---|---|
| Verbalizable Representations Form a Global Workspace (limitations; transcoder and head readouts) | Gurnee, Sofroniew et al. | 2026 | https://transformer-circuits.pub/2026/workspace/index.html | [full] | **Closest to idea 2.** It says the J-lens "treats the workspace as a flat collection of independently active concept vectors" and calls identifying the structure on top "an important direction for future work". It reads an addition transcoder feature whose encoder J-lens shows two conditions, and calls it "the AND of those conditions". That is a qualitative, weight-based J x J reading of single features. No Hessian, no energy split. |
| Bilinear MLPs enable weight-based mechanistic interpretability | Pearce, Dooms, Rigg, Oramas, Sharkey | 2024, ICLR 2025 | https://arxiv.org/html/2410.08417v2 | [section] | Interaction matrices for an output feature, expressed in an SAE input-feature basis (§3.1). Swap SAE features for J-lens vectors and you have idea 2's map for one MLP. Same-position only; they do not call it a Hessian. |
| PolySAE: Modeling Feature Interactions in Sparse Autoencoders via Polynomial Decoding | Koromilas, Demou, Oldfield, Panagakis, Nicolaou | 2026, arXiv 2602.01322 | https://arxiv.org/abs/2602.01322 | [abstract] | Learned pairwise/triple feature interaction terms in an SAE decoder. A learned map, not a readout curvature, not J-space. |
| Pairwise matrices for sparse autoencoders: single-feature inspection mislabels causal axes | Riegler, Torpmann-Hagen | 2026, arXiv 2605.03160 | https://arxiv.org/abs/2605.03160 | [abstract] | Joint steering of feature sets on Qwen3-1.7B/Gemma-2-2B finds features that are jointly necessary but individually substitutable. Empirical non-additivity, no Hessian, no J-space. |
| Extracting Steering Vectors from the J Space | not recorded | 2026, NeurIPS | https://neurips.cc/virtual/2026/163009 | [abstract] | Single-behaviour steering from J-space; no combinations. Shows nobody has published J x J steering. Authors TODO-verify. |
| J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses | Ayonrinde, Lindsey | 2026, LessWrong | https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace-lenses | [TODO-verify] | Page body did not load. Search snippet suggests first-order lens variants for writing to the workspace. |
| Compositionality Unlocks Deep Interpretable Models | Dooms, Gauderis, Wiggins, Oramas | 2025, arXiv 2504.02667 | https://arxiv.org/abs/2504.02667 | [abstract] | Tensor-network architecture (χ-nets) with weight-level global structure. New architecture, not pretrained LLMs. Background only. |

### D. Searched, nothing found
- "Hessian lens", "curvature lens", "second-order lens" for LMs: no hits.
- Activation-space Hessian low-rank structure in LMs: only parameter-space work (e.g. 2405.10927) and bilinear-MLP eigenstructure.
- Second-order edge-level circuit discovery on q/k/v edges: none. HVP-corrected attribution patching (2606.09899, NeurIPS 2026) is component-level.
- Gated DeltaNet interpretability: none found.
- J-lens follow-ups Oct 6-26: Ostrow et al. (first-order theory) and J-Comm (2610.32871, agent communication, not opened). Neither is second-order.
- LessWrong/AF posts on Hessian or second-order residual interactions in 2026: none found.

## What to do before building
1. Derive the predicted cross-position operator from the per-head decomposition above (W_QK part, (W_Vᵀg)(W_Qᵀ ΔK) part, OV-then-MLP part). If its top singular spaces already match J-space on the key side and W_QK SVD on the query side, idea 1 is a reweighted known object. Kill it there.
2. Report the cross block split by route: frozen-QK (OV-then-MLP), QK-routed, gate x value, DeltaNet. Only the non-QK routes are clearly uncovered by prior work.
3. Exclude sink positions and condition on lag.
4. For idea 2, build first-order-gain-matched decoys before computing the energy split. Without them, a large J x J share is expected and says nothing.
5. Cite Kamath et al. 2025, Lee et al. 2026, the J-lens paper's head readouts and limitations, Integrated Hessians and HETA up front.
