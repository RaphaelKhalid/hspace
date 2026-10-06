# H-space project: artifact index (Oct 5–6 2026)

## Documents
| file | what it is | status |
|---|---|---|
| `hspace-paper.md` | paper draft | living |
| `theory-v2.md` | theory after a 48-agent red-team (Lemmas 1/2/3/7, Prop 0/4/6, Approx 5, Construction 6.1, §6A estimator targets, §7 control) | current |
| `theory.md` | pre-review theory draft | superseded by v2 |
| `redteam-theory.json` | per-claim adjudicated verdicts and replacement texts | evidence for v2 |
| `SCOPE-hspace.md` | **run-1 preregistration** (C0–C6), frozen 19:13 PDT Oct 5 (sha bc9248fa…) | frozen |
| `SCOPE-hspace-v2.md` | run-2 addendum (exploratory), frozen 04:20 UTC Oct 6 (sha 56dc269d…) | frozen |
| `SCOPE-hspace-v3.md` | **v3 preregistration** (C1′ reliability at the noise-floor dimension; C6′ interaction ablation), frozen 04:51 UTC, before any run-2 data (sha f954fc20…) | frozen |
| `verdict.py` / `verdict_v23.py` | mechanical verdicts for run 1 / run 2 + v3 | code |
| `decodes_run1.md` | J-lens decodes of the 27B H-directions and their natural-scale interactions | descriptive |
| `lit-2026-10-05.md` | live literature sweep | reference |
| `ctrl-score-v2-spec.md` | judge-panel design for the goal-3 white-box monitor analysis | spec |

## Code (`interpcontrol/hlens/`)
- `bf16w.py`: bf16-stored, fp32-computed linears that are differentiable to any order. HVP matches all-fp32 to 7e-7 (`tests/test_bf16w.py`).
- `hspace.py`: run 1 (S2–S8).
- `hs_s7.py`: run-1 S7, verbatim, on 455 windows.
- `hspace2.py`: run 2 (cross-moment / spatial-sign / rectangle estimators, causal tests, k = 1 local sweep).
- `hs_c6prime.py`: C6′.
- `hs_anatomy.py` / `hs_integrated.py` / `hs_rectangle.py` / `hs_local.py`: 0.8B development checks.
- `hs_diag_top.py`: massive-activation diagnostic.
- `hs_dump_analysis.py`: heavy-tail diagnostics on run-2 dumps.
- `ctrl_read.py` (sanitized render) / `ctrl_score_v2.py`: goal 3.
- `pod/`: `hs_run.sh`, `chain2.sh`, `chain3.sh`, `watchdog11.sh`, `hf_sync.py` (pod → private HF dataset relay).
- `dash_hs/`: the OLED live monitor.

## Results
- **Laptop:** `out/` (0.8B dev), `out/pod/` (27B JSON + slim tensors), `out/pod/hf/` (downloaded from HF).
- **Private HF dataset:** `RaphaelRaphaelRaphael/hspace-27b-results` (all pod artifacts; synced every 5 min).
- **`runlog.md`:** every logged line, with freeze hashes and deviations.

## Verdicts so far
- **Run 1 (preregistered): FAIL.**
  - C1 fails 4/4: split-half 0.33–0.41.
  - C6 fails 4/4.
  - C0 passes (0.997), and C2–C5 pass at all 4 layers.
- **Run 2 / v3:** pending.
