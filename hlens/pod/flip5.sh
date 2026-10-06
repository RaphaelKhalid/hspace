#!/usr/bin/env bash
cd /workspace/mats/interpcontrol/hlens
until grep -q "split-half" <(grep "L52 split-half" out/hspace2_full.log) && test -s out/hspace2_full_dump_L52.pt; do sleep 20; done
while pgrep -f "[h]s_flip[.]py" > /dev/null; do sleep 15; done
sleep 10
echo "[pod $(date +%T)] starting v5 flip test (L40, L52)" >> runlog.md
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=12 nice -n 10 python hs_flip5.py out/hspace2_full.pt out 40 52 --n 200 --nplant 50 --device cpu > out/hs_flip5_full.log 2>&1
echo "[pod $(date +%T)] v5 flip test exit $?" >> runlog.md
