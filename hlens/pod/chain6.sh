#!/usr/bin/env bash
# Reordered (logged): after run 2, the frozen twin causal test (v5 passed) runs first, then the preregistered C6prime
# (v3 verdict already fixed by C1prime failing at 4/4 layers, so C6prime is descriptive; its frozen 10:30 deadline applies).
cd /workspace/mats/interpcontrol/hlens
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "FINAL DONE" runlog.md; do sleep 15; done
echo "[pod $(date +%T)] v5 PASSED: starting Sigma-orbit twin ablation (L40, L52), deadline 10:50" >> runlog.md
python hs_c6twin.py full 40 52 --v5 > out/hs_c6twin_full.log 2>&1; echo "[pod $(date +%T)] c6twin exit $?" >> runlog.md
python pod/hf_sync.py once >> out/hf_sync.log 2>&1
echo "[pod $(date +%T)] C6T DONE" >> runlog.md
echo "[pod $(date +%T)] starting C6prime (hs_c6prime.py full)" >> runlog.md
python hs_c6prime.py full > out/hs_c6prime_full.log 2>&1; echo "[pod $(date +%T)] c6prime exit $?" >> runlog.md
python pod/hf_sync.py once >> out/hf_sync.log 2>&1
echo "[pod $(date +%T)] C6P DONE (all results synced to HF)" >> runlog.md
echo "[pod $(date +%T)] ALL DONE" >> runlog.md
