#!/usr/bin/env bash
# post-hoc: relative-curvature + gain diagnostics per layer as soon as each run-2 dump exists
cd /workspace/mats/interpcontrol/hlens
export HF_HOME=/workspace/hf HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=8
for l in 28 40 52; do
  until grep -q "L$l split-half" out/hspace2_full.log && test -s out/hspace2_full_dump_L$l.pt; do sleep 20; done
  sleep 20
  nice -n 10 python hs_relcurv2.py out/hspace2_full.pt out $l > out/hs_relcurv2_L$l.log 2>&1
  nice -n 10 python hs_relcurv_gain.py out/hspace2_full.pt out/relcurv2_subs.pt 16 28 40 52 > out/hs_relcurv_gain.log 2>&1
  echo "[pod $(date +%T)] relcurv L$l done" >> runlog.md
done
