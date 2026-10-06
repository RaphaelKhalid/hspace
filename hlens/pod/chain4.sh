#!/usr/bin/env bash
# Post-hoc (NOT preregistered) massive-activation confound checks, after the preregistered C6' (chain3) finishes.
cd /workspace/mats/interpcontrol/hlens
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "C6P DONE" runlog.md; do sleep 15; done
echo "[pod $(date +%T)] starting post-hoc C6'-DEFL (hs_c6defl.py full)" >> runlog.md
python hs_c6defl.py full > out/hs_c6defl_full.log 2>&1; echo "[pod $(date +%T)] c6defl exit $?" >> runlog.md
python pod/hf_sync.py once >> out/hf_sync.log 2>&1
echo "[pod $(date +%T)] C6D DONE (all results synced to HF)" >> runlog.md
