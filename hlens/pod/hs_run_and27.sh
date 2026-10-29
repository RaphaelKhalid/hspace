#!/usr/bin/env bash
# AND battery on Qwen3.6-27B (prereg paper/PREREG-and-battery-2026-10-28.md). Money rules:
#  - watchdog TERMINATES the pod after HS_MAX_MIN minutes no matter what;
#  - a 2-class cost check runs first; the pod terminates if the projected total exceeds HS_MAX_USD at HS_USD_PER_HR;
#  - after a verified upload to the private HF dataset the pod terminates itself.
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out ../results/and_battery
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8 HF_XET_HIGH_PERFORMANCE=1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
terminate() {
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1
}
( sleep $(( ${HS_MAX_MIN:-170} * 60 ));
  echo "[watchdog $(date +%T)] max time reached, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log; terminate ) > /dev/null 2>&1 &
START=$(date +%s)
echo "[pod $(date +%T)] AND27 watchdog armed (${HS_MAX_MIN:-170} min) for pod ${RUNPOD_POD_ID:-UNKNOWN}" | tee -a runlog.md
LAYERS="16 28 40 52"
python - <<'PY' || { echo "[pod] download FAILED"; terminate; exit 1; }
import time
t = time.time()
from huggingface_hub import snapshot_download
snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/train-*"])
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
print(f"downloads done in {time.time()-t:.0f}s", flush=True)
PY
echo "[pod $(date +%T)] downloads done (0.8B battery runs on the laptop, not here)" | tee -a runlog.md
# ---- cost check: 2 classes, all layers; projection includes elapsed time so far
T0=$(date +%s)
python hs_and_battery.py --model Qwen/Qwen3.6-27B --layers $LAYERS --bf16 --tag 27b_ccheck --classes negation_polarity > out/and27_ccheck.log 2>&1 \
  || { echo "[pod] cost check FAILED"; tail -5 out/and27_ccheck.log; terminate; exit 1; }
T1=$(date +%s)
python - "$START" "$T0" "$T1" <<'PY' || { echo "[pod] projected cost over cap: terminating" | tee -a runlog.md; terminate; exit 1; }
import json, os, sys
import hs_and_items as I
start, t0, t1 = (int(x) for x in sys.argv[1:4])
import hs_and_items_t2 as I2
n_cc = len(I.negation_polarity())
n_all = sum(len(f()) for f in I.CLASSES.values()) + sum(len(f()) for f in I2.CLASSES_T2.values())
load = 120                                            # model load + Sigma, paid once per process
per_item = max((t1 - t0 - load) / n_cc, 1.0)
proj_min = ((t1 - start) + load + per_item * n_all + 300) / 60      # + 5 min upload slack
usd = proj_min / 60 * float(os.environ["HS_USD_PER_HR"])
print(f"COST CHECK: {t1-t0}s for {n_cc} items -> {per_item:.1f}s/item, projected total {proj_min:.0f} min, ${usd:.2f}", flush=True)
open("out/and27_cost_check.txt", "w").write(f"{per_item:.2f} {proj_min:.1f} {usd:.2f}\n")
sys.exit(0 if usd <= float(os.environ.get("HS_MAX_USD", "4.5")) and proj_min < float(os.environ.get("HS_MAX_MIN", "170")) - 5 else 1)
PY
cat out/and27_cost_check.txt | tee -a runlog.md
sync_once() {
python - <<'PY'
from huggingface_hub import HfApi
import glob, os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
for f in glob.glob("../results/and_battery/*.json") + glob.glob("out/and27*") + glob.glob("out/and08*") + ["runlog.md"]:
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"and_battery/{os.path.basename(f)}", repo_id=R, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print("sync retry later", f, repr(e)[:100], flush=True)
PY
}
( while sleep 300; do sync_once >> out/hf_sync.log 2>&1; done ) &
SYNC_PID=$!
python hs_and_battery.py --model Qwen/Qwen3.6-27B --layers $LAYERS --bf16 --transfer --tag 27b > out/and27_main.log 2>&1; RC=$?
echo "[pod $(date +%T)] AND27 exit $RC" | tee -a runlog.md
kill $SYNC_PID 2>/dev/null
python - <<'PY'
from huggingface_hub import HfApi
import glob, hashlib, os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
fs = [f for f in glob.glob("../results/and_battery/*.json") + glob.glob("out/and27*") + glob.glob("out/and08*") +
      ["runlog.md", "hs_and_battery.py", "hs_and_items.py", "hs_and_items_t2.py"] if os.path.exists(f)]
for f in fs:
    for _ in range(3):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"and_battery/{os.path.basename(f)}", repo_id=R, repo_type="dataset"); break
        except Exception as e:  # noqa: BLE001
            print("retry", f, repr(e)[:120], flush=True)
rm = {it.path: it for it in api.list_repo_tree(R, repo_type="dataset", path_in_repo="and_battery", recursive=True, expand=True) if hasattr(it, "size")}
bad = 0
for f in fs:
    it = rm.get(f"and_battery/{os.path.basename(f)}")
    ok = it is not None and it.size == os.path.getsize(f)
    bad += not ok
    print("verified" if ok else "PROBLEM", f, flush=True)
print("UPLOAD-VERIFY problems:", bad, flush=True)
open("out/verify_ok", "w").write(str(bad))
PY
echo "[pod $(date +%T)] AND27 ALL DONE" | tee -a runlog.md
if [ "$(cat out/verify_ok 2>/dev/null)" = "0" ] && grep -q 'VERDICT' out/and27_main.log 2>/dev/null; then
  echo "[pod $(date +%T)] verdict written and results verified on HF; self-terminating" | tee -a runlog.md
  terminate
else
  echo "[pod $(date +%T)] verify problem or no verdict: still terminating to protect budget (partial results synced)" | tee -a runlog.md
  sync_once > /dev/null 2>&1; terminate
fi
