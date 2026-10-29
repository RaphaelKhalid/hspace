# Authoring guide for the explainer

This folder is a static site: plain HTML, no build step except two small scripts. Read this whole file before writing a level. Use `00.html` as the reference page: copy its structure exactly.

## The reader

- Raphael, CS + Political Science, full-stack and ML. Capable, but **hates symbols that appear without definition**, and hates explanations that are not tied to a concrete example showing the mechanics.
- Wants extreme depth. Do not skip steps in algebra. Show every intermediate line of a worked example.
- Wants every technical word to have a hover definition (the glossary does this automatically, see below).

## Hard rules

1. **Symbols table first.** Every level starts with `<h2>Symbols on this page</h2>` and a `table.symbols` with columns Symbol | Read as | Meaning | Unit | Example. Every symbol used anywhere on the page must be in it. If a symbol has no unit, write "none (pure number)".
2. **Define before use.** In the text too, introduce each symbol in a sentence ("Let \(W\) be the water used, in buckets") before it appears in an equation.
3. **Units everywhere.** Every quantity in a worked example carries its unit. Every derivative's unit is stated ("bushels per bucket per bag").
4. **Every concept gets: a real-world example, an intuition pump, the mechanics.** The mechanics means: what happens to the numbers, step by step, when one variable changes and the others do not.
5. **Simulations are simple.** One idea per figure. White canvas, thin black lines, blue/red accents. Axes labelled with units. A live numeric readout that matches an equation in the text. A caption that says what to do and what to look for.
6. **House style for prose.** Very short plain sentences. **No em dashes** (the checker fails on them; use a colon, a comma, parentheses or a new sentence). No hype, no "simply", no "obviously". British spelling (metre, normalise, fertiliser).
7. **No invented facts or numbers.** Toy simulations may use made-up functions; say they are toys. Numbers about the actual H-space runs must come from `assets/hspace_numbers.js` (`window.HS`, generated from `../results/*.json`) or be quoted from `../hlens/paper/theory-v2.md` / `hspace-paper.md` with the file named on the page. Literature claims: only those already cited in theory-v2.md; otherwise leave them out.
8. **Do only your levels.** Do not edit `assets/site.js`, `assets/sim.js`, `assets/style.css`, `assets/glossary/core.json`, `index.html`, or other levels. If one of those has a bug or a wrong definition, write it in your final report.

## Page template

Copy `00.html`. Keep the `<head>`, the `<body data-level="N">`, the `<div class="page">`, and the five script tags in this order at the end:

```html
<script src="assets/katex/katex.min.js"></script>
<script src="assets/katex/auto-render.min.js"></script>
<script src="assets/glossary.js"></script>
<script src="assets/sim.js"></script>
<script src="assets/site.js"></script>
<script> S.ready(function () { /* your figures */ }); </script>
```

The prev/next navigation, the top bar and the footer are added automatically. Do not write them.

Sections, in order:
1. `<h1>Level N: Title</h1>` (title exactly as in `assets/site.js` LEVELS) and a one-line `p.subtitle`.
2. `div.box.goals` "By the end of this level you can", plus a "Prerequisites" line linking earlier levels.
3. Symbols table.
4. Numbered sections `<h2>N.1 ...</h2>`, `<h2>N.2 ...</h2>` ... with worked examples, figures and boxes.
5. A section "Why this matters for H-space" that points concretely at where the idea is used at Level 14/15.
6. `<h2>Exercises</h2>` with 6 to 10 exercises, easy to hard.

Boxes: `div.box.key` (a rule or definition to remember), `div.box.plain` (grey aside), `div.box.warn` (dashed: a common mistake). Give each a `span.boxtitle`.

## Maths

- KaTeX with delimiters `\( ... \)` inline and `$$ ... $$` display. **Never single `$`** for maths (a plain `$` is a dollar sign).
- In HTML, write `<` inside maths as `\lt` and `>` as `\gt`, or use `&lt;`/`&gt;`.
- Conventions (keep them across all levels):
  - \(\Delta x\) a finite change; never use \(h\) for a step size (\(h\) is reserved for the residual state from Level 14).
  - \(f'(x)\), \(f''(x)\) for one input; \(\partial f/\partial x\) for partials; \(\partial^2 f/\partial x\,\partial y\) mixed.
  - Vectors bold lower case \(\mathbf{v}\) in Levels 5 to 13; plain \(v\) is allowed in Levels 14 to 15 to match theory-v2 (say so on the page).
  - Matrices upper case \(A\), Jacobian \(J\), Hessian \(H\) or \(\nabla^2 f\), gradient \(\nabla f\), transpose \(A^\top\), norm \(\lVert \mathbf{v} \rVert\), dot product \(\mathbf{a}\cdot\mathbf{b}\) or \(\mathbf{a}^\top\mathbf{b}\).
  - Statistics (Level 13+): mean \(\mathbb{E}[\cdot]\), covariance \(\Sigma\), standard deviation \(\sigma\).

## Glossary (hover definitions)

- `assets/site.js` underlines every glossary word automatically (first use in each paragraph, list item, table cell or caption). You do not mark terms by hand.
- `assets/glossary/core.json` already has ~220 terms. Check it before adding.
- **If your page uses a technical word that is not in the glossary, add it** to `assets/glossary/LNN.json` (NN = your level, two digits), same format:
  `{"key": {"t": "shown term", "a": ["aliases", "plurals"], "l": N, "d": "Plain definition, 1 to 3 sentences, with an everyday example if possible. No maths delimiters; use plain unicode like x², ∂, Σ."}}`
- Then run `python tools/build_glossary.py` (from `explainer/`). It reports collisions; resolve any for your keys.
- Avoid aliases that are ordinary English words (like "map", "draw", "order"), since every occurrence would be linked.

## Simulation helpers (`assets/sim.js`, global `S`)

Read the header comment of `assets/sim.js`. The pattern from `00.html`:

```js
var host = document.getElementById("sim-x");          // <figure class="sim" id="sim-x"><figcaption>...</figcaption></figure>
var cap = host.querySelector("figcaption");
var wrap = document.createElement("div"); host.insertBefore(wrap, cap);
var c = S.canvas(wrap, { aspect: 0.55 });
var v = S.view(c, { xmin: 0, xmax: 10, ymin: 0, ymax: 40, padLeft: 40, padBottom: 30 });
var a = S.slider(wrap, { label: "a", min: 0, max: 5, step: 0.1, value: 1, unit: "m", digits: 1, oninput: function () { c.redraw(); } });
var out = S.readout(wrap);
c.draw = function () { v.axes({ xlabel: "t (s)", ylabel: "s (m)" }); v.fn(function (t) { return a.get() * t * t; }); out.textContent = "..."; };
c.redraw();
```

Available: `S.canvas`, `S.view` (`axes`, `fn`, `line`, `arrow`, `dot`, `text`, `poly`, `rect`, `heat`, `contour`), `S.slider`, `S.button`, `S.checkbox`, `S.readout`, `S.drag` (draggable points), `S.rng(seed)` (seeded uniform/normal), `S.linalg` (`det2`, `inv2`, `mul2`, `eigSym2`, `eigSym` for any n, `matVec`, `dot`, `norm`), `S.v3` (simple 3-D wireframe: `p`, `surface`, `line`, `dot`, `text`), `S.diverge(z, lo, hi)` (blue-white-red colour), `S.fmt(x, digits)`, `S.col` colours.

Wrap each figure in its own `(function () { ... })();` inside `S.ready`. You may add small helpers inside your own page script. Do not load other libraries.

## Exercises

```html
<div class="ex" data-answer="0.5" data-tol="0.01" data-unit="bushels per bucket">
<p><b>Exercise 6.3.</b> Question text.</p>
<details><summary>Show answer</summary><p>Full worked answer, every step.</p></details>
</div>
```

`data-answer` adds a number box with a Check button (tolerance is relative, default 1%). Leave it out for conceptual questions.

## Running examples (use these exact definitions; do not invent different ones)

**The car (Levels 1 to 4).** Position \(s(t) = t^2\) metres at time \(t\) seconds, for \(0 \le t \le 10\). Formally \(s(t) = \tfrac12 a t^2\) with constant acceleration \(a = 2\ \text{m/s}^2\). So speed \(s'(t) = 2t\) m/s and acceleration \(s''(t) = 2\) m/s². Other one-input functions are fine as additional examples.

**The farm (Levels 5, 6, 9, and referred back to later).** Crop yield
\(Y(W, F) = 2W + 3F + 0.5\,WF - 0.25\,W^2 - 0.5\,F^2\) bushels,
with \(W\) = water in buckets (1 bucket = 10 litres) and \(F\) = fertiliser in bags, \(W, F \ge 0\).
- \(\partial Y/\partial W = 2 + 0.5F - 0.5W\) bushels per bucket; \(\partial Y/\partial F = 3 + 0.5W - F\) bushels per bag.
- \(\partial^2 Y/\partial W\,\partial F = 0.5\) bushels per bucket per bag (the interaction: each extra bag makes each bucket worth 0.5 bushels more).
- \(\partial^2 Y/\partial W^2 = -0.5\) bushels per bucket², \(\partial^2 Y/\partial F^2 = -1\) bushels per bag².
- Hessian \(\begin{pmatrix} -0.5 & 0.5 \\ 0.5 & -1 \end{pmatrix}\): determinant 0.25, trace −1.5, negative definite. Maximum yield 29 bushels at \(W = 14\), \(F = 10\).
- The 2×2 test from \((W, F) = (4, 2)\) with moves of 1 bucket and 1 bag: \(Y(5,3) - Y(5,2) - Y(4,3) + Y(4,2) = 0.5\) bushels exactly (the function is quadratic, so the finite interaction equals the mixed partial).

**The rubber sheet / map projection (Levels 7 and 8).** Free choice, but use one consistent nonlinear map across Level 8, for example \(\mathbf{F}(x, y) = (x + 0.3\,y^2,\ y + 0.2\,x\,y)\).

**The tiny network (Levels 10 to 12).** One shared toy, defined on Level 10 and reused on 11 and 12: two inputs, one hidden layer of two SiLU neurons, one output. Write its weights out as numbers on Level 10.

## What each level covers

Keep to your level's scope. Earlier levels may be referenced; later levels may be previewed in one sentence with a link, never used.

- **01 Variables, functions and graphs.** Variable, constant, function as machine (input → output, same input same output), notation f(x), graphs and axes, coordinates, linear functions (slope-intercept in words, but save "slope" as a rate for Level 2), quadratics and the parabola, exponents and square roots, exp and log briefly (they appear in softmax later), functions with several inputs as a teaser. The car position table and graph.
- **02 Slope and the derivative.** Average rate over an interval (difference quotient, secant), shrinking Δt, the limit, tangent line, derivative as instantaneous rate, units of a derivative, derivatives of tⁿ, eˣ, sums and constant multiples (power rule derived for t² and t³ by expanding), when derivatives fail (corners), the car's speed. Numerical derivative vs exact.
- **03 The second derivative: curvature.** Derivative of the derivative, units (m/s²), acceleration of the car, concave up/down, inflection, critical points and the second-derivative test, curvature as "how wrong a straight line is", a second difference (f(x+Δ) − 2f(x) + f(x−Δ))/Δ² as a measurable estimate.
- **04 Taylor approximation.** Rebuilding f near a from f(a), f′(a), f″(a); linear vs quadratic approximation; error shrinking like Δ², Δ³; big-O; the third derivative; where it fails far away (sin or eˣ examples, and a function whose curvature changes fast). Seed the idea that "the Hessian at one point can fail to predict what happens at natural move sizes" (this is the v1 failure in theory-v2 §3, said in plain words).
- **05 Two inputs: partial derivatives and the gradient.** Functions of two inputs, the surface and contour map, partial derivatives as "hold the other fixed" (slices), the gradient vector, directional derivative, steepest ascent, the gradient perpendicular to contours. The farm.
- **06 Interaction: the mixed partial derivative.** Main effects vs interaction, the additive function has zero interaction, the 2×2 mixed difference with all four corners computed, the mixed partial as its limit, symmetry of mixed partials (with the farm and a non-quadratic example where the finite 2×2 interaction depends on the move size), positive vs negative interaction (complements vs substitutes), interaction depends on where you stand for non-quadratic functions. Introduce that I(a,b) = f(both) − f(a only) − f(b only) + f(none) is exactly the quantity v9 measures (in nats of log-likelihood) with 1σ moves.
- **07 Vectors, matrices and the determinant.** Vectors as arrows and lists, adding and scaling, length, dot product and angle, unit vectors, orthogonality, matrices as machines, matrix-vector product row-by-row and column-by-column views, linear maps keep grids straight, matrix multiplication as composition, identity, transpose, determinant as signed area scaling, inverse, basis and coordinates, span, subspace, rank, projection onto a line and a subspace, the outer product as "read along b, write along a" (rank-1). High-dimensional vectors (d = 5,120) as "the same thing with more components".
- **08 The Jacobian.** Vector-valued functions, the Jacobian as the matrix of partials (rows = outputs, columns = inputs, units of each entry), local linearity by zooming in on a warped grid, Jacobian-vector product as "how outputs move when inputs move along v", Jacobian determinant as local area scaling and orientation, inverse function theorem (local only), the 2026 Jacobian conjecture counterexample as an example of "invertible everywhere locally, not globally" (Alpöge, July 2026, ℂ³, determinant −2, three points collide: keep claims to exactly these reported facts and say it is pending peer review), the averaged Jacobian idea as a preview of the J-lens.
- **09 The Hessian.** The Hessian as the Jacobian of the gradient, entries and units, symmetry, the quadratic form vᵀHv as curvature along v, eigenvectors and eigenvalues of a symmetric matrix as principal curvatures, bowl / dome / saddle by eigenvalue signs, definiteness, second-derivative test in 2-D, trace and determinant, off-diagonal = interaction (and rotating the basis turns interaction into pure curvature: eigenvectors are the no-interaction axes), rank-1 Hessians from products (sym(abᵀ)), Frobenius norm, Sylvester's inertia. The farm Hessian.
- **10 The chain rule and backpropagation.** Composition, chain rule in 1-D with units, chain rule for vectors as Jacobian multiplication, neural networks as compositions (neuron, weight, bias, activation, ReLU, sigmoid, SiLU), the tiny network, forward-mode JVP vs reverse-mode VJP and why reverse mode wins for one output and many inputs, backprop as multiplying transposed Jacobians from the output end, computational graph, adjoints, cost in passes. Loss function and gradient descent briefly.
- **11 Hessian-vector products.** Why the full Hessian is impossible at d = 5,120 per position (count the entries), H·v as "how the gradient changes along v", finite-difference HVP and its rounding trade-off, the exact HVP by double backward (Pearlmutter), cost a few gradients, checking an HVP against central finite differences (relative error), floating point and bf16/fp32/fp64 (the bf16-stored, fp32-computed trick in `hlens/bf16w.py` explained in words), the tiny network's HVP computed live.
- **12 The second-order chain rule and the rectangle identity.** Second-order chain rule in 1-D (f(g(x)))″ = f″(g)g′² + f′(g)g″ with units and a worked example; vector form Hess(g∘f) = JᵀHJ + Σ ∂g/∂yᵢ Hess(fᵢ); the second-order adjoint (theory-v2 Lemma 1) as "curvature created at each sublayer, carried back through the Jacobians", the residual sublayer y ↦ y + φ(y) and why the identity adds no curvature; product sites sym(abᵀ) and threshold sites σ''·aaᵀ (Lemma 2, in plain words); the rectangle identity (Lemma 3) with its proof via ∂s∂t of φ(s,t), shown on a 2-D toy with an interactive rectangle; corner vs centre expansions; the 0.8B check (corner Hessian r = −0.09, rectangle average r = 0.98, both from theory-v2 §3) explained as "the Hessian at one point is a bad predictor of interactions at natural move sizes".
- **13 Statistics of many directions.** Random variables, samples, mean/expectation, variance, standard deviation, covariance and the covariance matrix Σ (with a 2-D scatter you can stretch), Gaussian, PCA as eigenvectors of Σ, spectrum, participation ratio, whitening (Σ^{-1/2} and Σ^{1/2}: measuring in units of natural spread), top-k subspace, subspace overlap (mean squared cosine; random baseline k/d), split-half reproducibility, noise floor, heavy tails (a few rows dominate; a live demo where trimming the top rows restores split-half), estimators, unbiasedness, Monte Carlo, Rademacher signs and why random signs make cross terms vanish on average (Hutchinson's idea, at a simple level), null hypotheses, p-values, randomization tests, confidence intervals, bootstrap, preregistration, importance sampling and effective sample size.
- **14 Transformers, the residual stream and the J-lens.** Tokens, vocabulary, embeddings, the residual stream as a shared notepad, residual sublayers y ↦ y + φ(y), RMSNorm, attention (weights over earlier positions, softmax), the MLP and SwiGLU (silu(gate read) × up read: a product of two reads, hence curvature), unembedding and logits, softmax, log-likelihood and nats, KL divergence, Qwen3.6-27B sizes (64 blocks, d = 5,120) and Qwen3.5-0.8B (24 blocks, d = 1,024) from theory-v2 §0, massive activations. The J-lens: J_ℓ = average Jacobian of the summed final residual with respect to layer ℓ (theory-v2 §0), decoding u_tᵀ J_ℓ h, J-space as its active top subspace, what "verbalisable first-order content" means, and the question H-space asks one derivative up.
- **15 The H-lens and what the runs found.** Follow theory-v2 §0, §6A, §6B and paper §4–§4d. The per-context Hessian blocks H_pq, the H-lens T_ℓ, whitening H̃ = Σ^{1/2}HΣ^{1/2}, the three energy operators M^row ⪰ M^diag ⪰ M^lens (Proposition 0 with its proof), H-space_k, sign probes and what each estimator converges to (Proposition 6, simplified), run 1 (the C0–C6 table: FAIL on C1), heavy tails (Observation 8.2 numbers from `HS.heavy_tails`), the Σ prior (Proposition 8 with its proof, and a live toy showing a function-of-Σ curvature returning PCA axes reproducibly), the sign-flip test (Corollary 8.1, live toy), v5 (PASS: numbers from `HS.flip5`), the Σ-orbit twin causal test (FAIL), v8 (replication, `HS.v8`), v9 (PASS, `HS.v9`: 5.8× at L16, 2.5× at L40, as exp of the stored log ratios), and the exact list of what is and is not established. End with the open questions from theory-v2 §8.

## Before you finish

From `explainer/`:

```
python tools/build_glossary.py
node tools/check_page.js NN.html
```

Fix every error. Treat warnings seriously. Then report: files written, glossary terms added, anything in the shared files you think is wrong, and any claim you were unsure about.
