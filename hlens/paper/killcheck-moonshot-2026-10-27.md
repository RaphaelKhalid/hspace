# Kill-check: J-state-conditional H-lens and Jacobian switch modes (2026-10-27)

This extends `killcheck-2026-10-26.md` and `killcheck-v12-2026-10-26.md`. Works covered there (Kamath 2025, HeadVis, Lee 2026, Pearce bilinear MLPs, Yan 2608.25347, Ostrow 2610.31955 on gain) are not repeated except where they decide something. Nothing was built. Nothing was spent. About 22 tool calls.

Evidence tags, same as before:
- **[full]** / **[section]**: read the full text or the relevant section, via a summarizer over the page.
- **[abstract]**: abstract page only.
- **[TODO-verify]**: seen in a search result only. Do not cite until opened.

## Verdicts

| Idea | Verdict | Deciding works |
|---|---|---|
| 1. J-state-conditional H-lens: E[H \| J25 cluster], φ_H vs token / random / PCA clusters | **PARTIALLY SCOOPED** | Ayonrinde & Lindsey 2026 (J++ Lens) already fit per-cluster "Expert Jacobians" on k-means clusters of activation PCA. They name "consider the second-order term" as the alternative they did not take. |
| 2. Jacobian switch modes: J_x ≈ J̄ + Σ_k α_k(x) Δ_k, decoded Δ_k, α_k as gates, H-space as control plane | **PARTIALLY SCOOPED, and at risk of being a relabel of the old H-space** | Ostrow et al. 2026 (shared plus context-specific Jacobian modes, keyed toy model); J++ Lens (piecewise-linear expert Jacobians); Schug et al. 2025 (attention as a hypernetwork with a low-dim latent code); Ameisen et al. 2025 (frozen attention and norms give a per-prompt linear map); Smith, Linderman, Sussillo 2021 (Jacobian switching linear systems). |

Nobody I found measures a cluster-conditional **Hessian** mean, or φ_H under any partition. Nobody I found estimates the **rank of the context-variation of the J-lens Jacobian** in a real LLM and decodes its modes. Those two measurements are open. The framings are not.

## Differentiators that survive

**Idea 1.** J++ is first order and is built to improve lens recall, not to test invariance. The open claim is narrow: "conditioning on J-space state makes second-order structure context-invariant, more than conditioning on token, position or generic activation PCA at matched cluster count and size." Do not claim "first conditional lens". J++ already is one.

Useful prior from J++ itself. Per-input k-means routing to its experts scored 44.1% recall@10. Random routing scored 47.7%. A three-class token-type router scored 50.2%. So at first order, activation-cluster conditioning was worse than random routing and worse than token conditioning. That matches tonight's result that token conditioning raises φ_dir 2-3x. Expect J-state clusters to lose to token clusters unless the workspace story is real.

**Idea 2.** The parts already published: J = shared + context-specific (Ostrow, assumed, not estimated, no rank bound on the context part); piecewise expert Jacobians (J++, 8 k-means experts, combined into one map); attention as a context-generated linear map (Schug, Ameisen). The open measurement is the effective rank of Cov_x(J_x) for the J-lens readout at scale, with decoded Δ_k and gates α_k that are not reducible to token, position or residual PCs. The "control plane" reinterpretation is not a result by itself. See the flaw below.

## Reasons each idea may be wrong or trivial on its face

### Idea 1: any informative partition raises φ_H. The test is a variance decomposition.

Write the conditional mean energy as
Σ_k p_k ‖E[H | k]‖² = ‖E H‖² + Σ_k p_k ‖E[H | k] − E H‖².
- The first term is tonight's result, about 0.
- The second term is the between-cluster variance of H. It is nonnegative for every partition. It grows with cluster count and with how homogeneous x is within clusters. At singleton clusters φ_H = 1.
- Random clusters are a valid null for estimator bias only. Their expected between-cluster term is about 0. They are not a null for "this partition is special". Any partition correlated with x will beat random.

Specific confounds:
1. **Token identity.** J-lens coordinates at a layer read the current and next token heavily. J25 clusters will partly be token clusters. Token conditioning alone already gives 2-3x.
2. **Residual homogeneity.** PCA clusters on the top residual PCs capture more variance of x than J25 clusters. If φ_H just tracks "x is homogeneous, so H is homogeneous", PCA clusters win and the gating story dies. Only a J-state win over PCA at matched dimension, K and within-cluster dispersion means anything.
3. **Readout-side curvature.** J-lens directions are amplified modes (Ostrow). Clustering on them may select regimes of the final norm and softmax. That is the curvature that already rises with depth (φ_dir 0.085 at L52). The conditional H-lens may just rediscover readout saturation.
4. **Heavy tails and sinks.** At L16, 0.1% of rows carry 97.5% of energy. A cluster that isolates sink or BOS rows will show high φ_H for a trivial reason.
5. **Small-cluster noise and pairing.** The unbiased cross-window product needs at least two independent windows per cluster. Same-document positions are not independent contexts. Pair across documents only.
6. **Wrong target.** φ_J also rises under conditioning. "J-like" must mean φ_H | J-state approaches unconditional φ_J, or φ_H | k / φ_J | k approaches 1 for the same partition. Beating 0 is not the bar. Unconditional φ_J at 0.8B is still PENDING (moonshot §4). Get it first.

**Proposed control.** A nested, matched design.
- Partitions compared at the same K and the same cluster-size distribution: J25 k-means, top-25-PC k-means (this is J++'s clustering), token identity, position bins, random.
- Match on token entropy H(token | cluster), on I(cluster; position), and on within-cluster trace of Cov(x). Use resampling to hit the match, or report φ_H against each matching variable as a curve.
- The deciding test: J-state clusters **within a fixed token**. Stratify by token first, then split by J25 state versus by PC state of equal dimension. If J-state does not add φ_H beyond token, and beyond PC state, the claim is dead.
- Run with sinks excluded, and once with the final norm frozen. That removes the readout-saturation route.

### Idea 2: to first order, switch modes are the Hessian. The control plane is the old H-space.

Taylor-expand around the mean state: J_x − J̄ ≈ H_x̄[x − x̄]. Then:
- Cov(vec J_x) ≈ H Σ_x Hᵀ. Its top modes Δ_k are the Hessian 3-tensor contracted with the top principal directions of x.
- α_k(x) ≈ PC scores of x, reweighted by curvature.
- The "input directions that drive α_k" are the top eigenvectors of an E_x[Hᵀ H]-type operator. That is the second-moment operator (M^row) the project already measured in run 1, and that moonshot §1 called "large in some context", not J-like.

So the "control plane" reading is a rename of the existing H-space estimator, not a new object. It only becomes new if finite-size gate switching (SwiGLU sign flips, attention reassignments) carries structure the Taylor term misses. That is testable.

Further reasons it is trivial by default:
1. **Low rank is the default.** Any smooth matrix function of x inherits the fast-decaying spectrum of the residual stream. Low effective rank of Cov(vec J) is expected, not a finding. It needs a rank null from a model with the same Σ_x.
2. **Attention gives the decomposition by construction.** With attention frozen, J_x is Σ_heads a_h(x) times OV paths plus MLP gate terms (Ameisen; Schug). So some Δ_k will just be OV maps and α_k attention weights. That is a known object.
3. **Token switching gives it by construction.** An input-switched affine model (Foerster et al. 2017, TODO-verify) has one linear map per token. The top modes will likely be token-identity and position rewirings.
4. **J++ evidence against sharp modes.** Its expert Jacobians largely agree in later layers by cosine. Cluster routing did not beat random routing.
5. **E[H] = 0 is not a contradiction.** Var J can be large while E H is zero. So the decomposition is not ruled out. It is just not implied by tonight's result either.

**Proposed control.**
- Fit α_k from data. Then regress α_k on (a) token identity and position, and (b) the top-m residual PC scores, with m equal to the number of modes. Report the variance of α_k left over. If CCA between α and the PC scores is near 1, the modes are the Taylor term. Kill.
- Compare Δ_k with the Taylor prediction from per-context Hessians already on disk. Report the share of Cov(vec J) the first-order Taylor term explains. Only the residual share is a candidate for "switching".
- Run once with attention patterns and norm denominators frozen. Modes that vanish were attention or norm reweighting.
- Rank null: same estimator on a model or surrogate with matched Σ_x (for example a Gaussian input with the same mean and covariance, as Boix-Adsera 2025 uses for MLPs).
- Decode test: a shuffled-gate null. Permute α across contexts within token. If Δ_k decode equally well, the decode reflects token content, not gating.

## Close works

### A. Conditional or piecewise Jacobian lenses

| Work | Authors | Year, venue | URL opened | Tag | Overlap |
|---|---|---|---|---|---|
| J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses | Ayonrinde, Lindsey | 2026-10-08, LessWrong | https://www.greaterwrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace-lenses | [section] | **Closest to both ideas.** Fits one averaged "Expert Jacobian" per k-means cluster (8 per layer, on 64-dim PCA of normalised activations). It is the first-order version of idea 1 and a discrete version of idea 2. It lists the second-order term as an option it did not take. No random-cluster or size-matched control for the expert fit. |
| A Mathematical Theory of the Jacobian Lens | Ostrow, Qian, Liu, Fiete | 2026-10-22, arXiv 2610.31955 | https://arxiv.org/html/2610.31955 | [section] | Writes per-context Jacobians as shared modes plus context-specific modes. The averaged lens keeps the shared part. A keyed toy model has a key state that changes read-in and read-out directions. That is a gate. The context part is only norm-bounded, with no rank, no estimated gates, no Hessian. |
| Verbalizable Representations Form a Global Workspace in Language Models | Gurnee, Sofroniew, Pearce et al. | 2026-07-16, arXiv 2607.15495 | https://arxiv.org/html/2607.15495v1 | [section], first 100k chars | States that a single-prompt Jacobian "conflates two kinds of structure" and averages to keep the shared one. Does not quantify per-context dispersion in the part read. Explicitly first order. |
| Short Horizons and Sparse Concepts | Yan, Ding, Tan et al. | 2026-08-26, arXiv 2608.25347 | https://arxiv.org/html/2608.25347 | [section] | No variance across contexts, no clustering, no conditioning. Defines a Hessian only as the affine readout's nonlinear error. |

### B. Context-dependent linear maps and gating

| Work | Authors | Year, venue | URL opened | Tag | Overlap |
|---|---|---|---|---|---|
| Circuit Tracing: Revealing Computational Graphs in Language Models | Ameisen, Lindsey, Pearce, Gurnee et al. | 2025-03-27, Transformer Circuits | https://transformer-circuits.pub/2025/attribution-graphs/methods.html | [section] | Freezes attention patterns and norm denominators, so each prompt gets a linear graph. Contrasts prompt-specific graphs with global virtual weights and reweights by co-activation (TWERA). This is the standard "context gates a linear map" frame. |
| Tracing Attention Computation Through Feature Interactions | Kamath et al. | 2025, Transformer Circuits | covered in killcheck-2026-10-26 | [full] | Explains the QK half: how attention, the context gate, is computed from feature pairs. |
| Attention as a Hypernetwork | Schug, Kobayashi, Akram, Sacramento, Pascanu | 2024 arXiv 2406.05816, ICLR 2025 | https://arxiv.org/abs/2406.05816 | [abstract] | Multi-head attention emits a "composable, low-dimensional latent code" that picks the linear operation per key-query pair. That is J_x = Σ α_k(x) Δ_k for the attention part, by architecture. |
| Large Language Models are Locally Linear Mappings (later "Equivalent Linear Mappings of Large Language Models") | Golden | 2025-05-30, arXiv 2505.24293 (TMLR 2025 per search, TODO-verify) | https://arxiv.org/html/2505.24293v1 | [section] | Exact per-input linear map via a detached Jacobian. Low-rank SVD decodes to tokens. Per input only. No shared-plus-modes split, no clustering. |
| Reverse engineering RNNs with Jacobian switching linear dynamical systems | Smith, Linderman, Sussillo | 2021, arXiv 2111.01256 (NeurIPS 2021 per search, TODO-verify) | https://arxiv.org/abs/2111.01256 | [abstract] | Switching linear regimes built from first-order Taylor expansions, generalised to a continuous manifold of switching points. Dynamical-systems precedent for "J_x as switch modes". RNNs, synthetic tasks. |
| Secret mixtures of experts inside your LLM | Boix-Adsera | 2025-12-20, arXiv 2512.18452 | https://arxiv.org/html/2512.18452v1 | [section] | Dense MLPs distil better into sparse MoE students with low-rank linear routers. The advantage largely disappears on mean-and-covariance-matched Gaussian inputs. Supports a gating view. No Jacobians. The Gaussian-match trick is a good null for idea 2. |
| Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations | Farnik, Lawson, Houghton, Aitchison | 2025, ICML (PMLR 267) | https://arxiv.org/abs/2502.18147 | [abstract] | Sparse per-input Jacobians between input and output SAE latents. Says "MLPs are approximately linear when rewritten in the JSAE basis." Sparsity of J, not low-rank variation of J. No 2026 follow-up found. |
| Bilinear MLPs enable weight-based mechanistic interpretability | Pearce, Dooms, Rigg, Oramas, Sharkey | 2024, ICLR 2025 | covered in killcheck-2026-10-26 | [section] | In a bilinear layer J_x is exactly affine in x. Idea 2 is then exact and weight-based: Δ_k are tensor slices, α_k are input coordinates. |
| Input Switched Affine Networks | Foerster et al. | 2017, ICML | not opened (https://proceedings.mlr.press/v70/foerster17a.html seen in search) | [TODO-verify] | One affine map per input token in an RNN. Token-indexed switch modes by design. |
| Neural Path Features and Neural Path Kernel: role of gates | Lakshminarayanan, Singh | 2020, arXiv 2006.10529 (NeurIPS 2020 per search) | not opened | [TODO-verify] | ReLU nets as gated linear networks. The active gate set selects the linear sub-network per input. |

### C. Searched, nothing found
- Cluster-conditional or state-conditional Hessians, or feature interactions conditioned on activation clusters, in LMs: none.
- Rank or PCA of input-output Jacobians across contexts in LMs: none. Golden is per input. J++ uses 8 fixed clusters.
- Jacobian SAE follow-ups in 2026: none found.
- J-lens follow-ups since July other than the ones above: Looped Transformers under the Jacobian Lens (2609.01924, not opened, TODO-verify) is first order per its search snippet.

## What to do before building
1. Get unconditional φ_J at 0.8B (moonshot §4). It is the bar for "J-like". Without it, idea 1 has no target.
2. Idea 1: run the nested within-token design with matched K, cluster sizes, token entropy and within-cluster dispersion. Include J++-style PCA k-means as a named baseline. Exclude sinks. Repeat with the final norm frozen.
3. Idea 2: before fitting modes, compute the Taylor prediction H Σ_x Hᵀ from per-context Hessians already on disk. If it explains most of Cov(vec J), stop. The idea is the old H-space.
4. Cite J++ Lens up front for both ideas. Cite Ostrow for the shared-plus-context split. Cite Schug and Ameisen for attention as a context gate.
