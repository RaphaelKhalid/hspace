#!/usr/bin/env bash
# Pod job chain after run-1 S4-S8: standalone S7 -> control read -> run 2 (only after HSPACE2_READY flag).
cd /workspace/mats/interpcontrol/hlens
export HF_HOME=/workspace/hf PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
# new hard-stop watchdog (10:00 UTC); old watchdogs are disarmed separately (parent subshell killed first)
END=$(date -d "2026-10-06 10:00:00 UTC" +%s); NOW=$(date +%s)
( sleep $(( END - NOW )); echo "[watchdog $(date +%T)] hard stop 10:00 UTC, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1 ) > /dev/null 2>&1 &
echo "[pod $(date +%T)] watchdog re-armed: hard stop 10:00 UTC" >> runlog.md
until grep -q "\[hspace\] S7 labels 1/" runlog.md || ! pgrep -f "python hspace.py full" > /dev/null; do sleep 5; done
pkill -f "python hspace.py full"; sleep 3
echo "[pod $(date +%T)] run-1 S4-S8 saved; stopped run-1 S7 (would crash at window 456); starting standalone S7" >> runlog.md
python hs_s7.py full > out/hs_s7_full.log 2>&1; echo "[pod $(date +%T)] S7-standalone exit $?" >> runlog.md
python ctrl_read.py full > out/ctrl_full.log 2>&1; echo "[pod $(date +%T)] ctrl exit $?" >> runlog.md
until [ -f out/HSPACE2_READY ]; do sleep 10; done
echo "[pod $(date +%T)] starting hspace2 (run 2)" >> runlog.md
python hspace2.py full > out/hspace2_full.log 2>&1; echo "[pod $(date +%T)] hspace2 exit $?" >> runlog.md
sha256sum out/hspace_full.json out/hspace_full.pt out/ctrl_full.pt out/hspace2_full.json out/hspace2_full.pt > out/SHA256SUMS 2>/dev/null
echo "[pod $(date +%T)] FINAL DONE" >> runlog.md
