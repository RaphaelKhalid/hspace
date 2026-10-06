#!/usr/bin/env bash
# After chain2 (run 2) finishes: C6' interaction-ablation test (prereg SCOPE-hspace-v3), then a final HF sync.
cd /workspace/mats/interpcontrol/hlens
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "FINAL DONE" runlog.md; do sleep 15; done
echo "[pod $(date +%T)] starting C6' (hs_c6prime.py full)" >> runlog.md
python hs_c6prime.py full > out/hs_c6prime_full.log 2>&1; echo "[pod $(date +%T)] c6prime exit $?" >> runlog.md
python pod/hf_sync.py once >> out/hf_sync.log 2>&1
echo "[pod $(date +%T)] C6P DONE (all results synced to HF)" >> runlog.md
