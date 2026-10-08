# v11a: position-general H-space (laptop, $0), decision rule

*Written and pushed before any all-position or non-punctuation-pooled H-lens estimate is computed. 14th test in the H-space sequence. Code: `hs_v11a_allpos.py`; hashes in `results/runlog_laptop.md`.*

## Why

J-space is defined over all token positions. My H-space so far is punctuation-conditional: v8's robust estimator replicated on punctuation rows, and v10a found no other single token class that replicates at both layers.

v10a never pooled positions. Meanwhile v10's W stage showed that the frozen punctuation H-space also carries more interaction than word-position twins: 3.4× at L16 and 2.6× at L40. The question here is whether a **position-general** H-space exists under the same estimator and the same replication rule. It is asked before any causal test of such an object.

## Data (already on disk, no new compute)

- **Fresh rows:** the v8 HVP rows, all 28,416 sampled positions per layer: `hs_v8_full_L{16,40}.pt`, with two Hutchinson probes per row.
- **Replication rows:** the independent run-2 rows: `hspace2_full_dump_L{16,40}.pt`.
- **Whitening frame:** run 2 (`hspace2_full.pt`), with the same `drop_mask` as v8 and v10a.

## Classes (decided now)

- **all**: every row. *Primary.*
- **nonpunct**: every row whose token is not punctuation, i.e. function, content, digit and whitespace pooled. *Secondary.* It checks that "all" is not carried by the punctuation rows.
- **punctuation**: the reference. It must reproduce v8 and v10a.

## Rule (identical to v10a's, applied per class and layer)

The estimator is v10a's: `hs_flip5.run_layer`, variant d, 200 sign flips, 50 planted signals, with the norm and xnorm estimators.

A class **replicates at layer l** iff all of the following hold:
- **For both norm and xnorm:**
  - λ₁ is above the flip-null maximum.
  - The split-half top-3 overlap is above the flip-null maximum and is at least 0.7.
  - The planted signal is detected.
- **Across runs:** the overlap between the top-5 norm patterns from the v8 rows and from the run-2 rows is at least 0.5. Chance is about 0.006.

**Verdict:**
- **"position-general H-space candidate"** iff **all** replicates at both L16 and L40.
- **"position-general at L16 only"** (or at L40 only) iff it replicates at exactly one layer.
- Otherwise **"no position-general H-space under this estimator"**.

**nonpunct is reported** with the same rule. If **all** replicates but **nonpunct** does not, the candidate gets the qualifier "punctuation-carried".

## Outputs

- `out/hs_v11a.json`.
- `out/v11a_subspaces.pt`: the top-5 patterns per class and layer, in the whitened run-2 frame. These are the frozen candidate objects for any later causal test, and they are never re-estimated after this run.

## What this does not test

Causal use, distinctness from J-space, other layers, and other models. Those belong to the v11 registered report, which is frozen separately before its pod run. Unlike v10's EX/H-space, the all-position candidate is not cherry-picked: the class is fixed here, before estimation.

## Deviations

None yet.
