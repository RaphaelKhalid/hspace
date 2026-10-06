#!/usr/bin/env bash
# If (and only if) the frozen v5 flip rule PASSES: Sigma-orbit twin ablation on L40/L52 (v5 patterns), after C6prime.
cd /workspace/mats/interpcontrol/hlens
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "DECISION v5" out/hs_flip5_full.log 2>/dev/null || grep -q "v5 flip test exit [1-9]" runlog.md; do sleep 20; done
if ! grep -q "DECISION v5.*PASS" out/hs_flip5_full.log; then
  echo "[pod $(date +%T)] v5 flip rule did not pass: no twin ablation (per frozen rule)" >> runlog.md; exit 0
fi
until grep -q "C6P DONE" runlog.md; do sleep 15; done
echo "[pod $(date +%T)] v5 PASSED: starting Sigma-orbit twin ablation (L40, L52)" >> runlog.md
python hs_c6twin.py full 40 52 --late --v5 > out/hs_c6twin_full.log 2>&1; echo "[pod $(date +%T)] c6twin exit $?" >> runlog.md
python pod/hf_sync.py once >> out/hf_sync.log 2>&1
echo "[pod $(date +%T)] C6T DONE" >> runlog.md
