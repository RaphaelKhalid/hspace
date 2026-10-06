#!/usr/bin/env bash
# Hard stop at 11:00 UTC (user: use the full RunPod budget; 8.2 h x $1.69 = $13.8 < $16.99). TERMINATES the pod.
cd /workspace/mats/interpcontrol/hlens
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
END=$(date -d "2026-10-06 11:00:00 UTC" +%s); NOW=$(date +%s)
echo "[pod $(date +%T)] watchdog moved: hard stop 11:00 UTC ($(( (END - NOW) / 60 )) min from now)" >> runlog.md
sleep $(( END - NOW ))
echo "[watchdog $(date +%T)] hard stop 11:00 UTC, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log
curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1
