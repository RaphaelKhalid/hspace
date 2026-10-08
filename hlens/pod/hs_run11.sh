#!/usr/bin/env bash
# v11 one-shot (prereg paper/SCOPE-hspace-v11.md). Watchdog TERMINATES the pod after HS_MAX_MIN (default 190) minutes.
# Results relay to the private HF dataset every 10 min (partial results survive a crash) and once at the end; after a
# verified final upload the pod terminates itself.
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
terminate() {
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1
}
( sleep $(( ${HS_MAX_MIN:-190} * 60 ));
  echo "[watchdog $(date +%T)] max time reached, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log; terminate ) > /dev/null 2>&1 &
echo "[pod $(date +%T)] v11 watchdog armed (${HS_MAX_MIN:-190} min) for pod ${RUNPOD_POD_ID:-UNKNOWN}" | tee -a runlog.md
python - <<'PY'
import time
t = time.time()
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/validation-*", "wikitext-103-raw-v1/train-*"])
R = "RaphaelRaphaelRaphael/hspace-27b-results"
hf_hub_download(R, "out/hspace2_full.pt", repo_type="dataset", local_dir=".")
p = hf_hub_download(R, "v10/v10_inputs_full.pt", repo_type="dataset", local_dir=".")
import shutil; shutil.copy(p, "out/v10_inputs_full.pt")
print(f"downloads done in {time.time()-t:.0f}s", flush=True)
PY
echo "[pod $(date +%T)] downloads done" | tee -a runlog.md
if ! sha256sum out/v11a_subspaces.pt 2>/dev/null | grep -q "^29e0b7cb95991011"; then
  echo "[pod $(date +%T)] v11a_subspaces.pt missing or hash mismatch: terminating" | tee -a runlog.md; terminate; exit 1
fi
sync_once() {
python - <<'PY'
from huggingface_hub import HfApi
import os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
for f in ("out/hs_v11_full.json", "out/hs_v11_full_raw.pt", "out/hs_v11.log", "runlog.md"):
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"v11/{f}", repo_id=R, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print("sync retry later", f, repr(e)[:100], flush=True)
PY
}
( while sleep 600; do sync_once >> out/hf_sync.log 2>&1; done ) &
SYNC_PID=$!
python hs_v11.py full --minutes ${HS_RUN_MIN:-160} > out/hs_v11.log 2>&1; RC=$?; echo "[pod $(date +%T)] v11 exit $RC" | tee -a runlog.md
kill $SYNC_PID 2>/dev/null
python - <<'PY'
from huggingface_hub import HfApi
import hashlib, os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
fs = [f for f in ("out/hs_v11_full.json", "out/hs_v11_full_raw.pt", "out/hs_v11.log", "runlog.md", "hs_v11.py") if os.path.exists(f)]
for f in fs:
    for _ in range(3):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"v11/{f}", repo_id=R, repo_type="dataset"); break
        except Exception as e:  # noqa: BLE001
            print("retry", f, repr(e)[:120], flush=True)
rm = {it.path: it for it in api.list_repo_tree(R, repo_type="dataset", path_in_repo="v11", recursive=True, expand=True) if hasattr(it, "size")}
bad = 0
for f in fs:
    it = rm.get(f"v11/{f}"); h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    lfs = getattr(getattr(it, "lfs", None), "sha256", None) if it else None
    ok = it is not None and it.size == os.path.getsize(f) and (lfs is None or lfs == h); bad += not ok
    print("verified" if ok else "PROBLEM", f, flush=True)
print("UPLOAD-VERIFY problems:", bad, flush=True)
open("out/verify_ok", "w").write(str(bad))
PY
echo "[pod $(date +%T)] V11 ALL DONE" | tee -a runlog.md
if [ "$(cat out/verify_ok 2>/dev/null)" = "0" ] && [ "$RC" = "0" ] && grep -q '"FINAL"' out/hs_v11_full.json 2>/dev/null; then
  echo "[pod $(date +%T)] verdict written and results verified on HF; self-terminating" | tee -a runlog.md
  sync_once > /dev/null 2>&1; terminate
else
  echo "[pod $(date +%T)] NOT self-terminating (rc=$RC or no verdict): left up for inspection; watchdog stays armed" | tee -a runlog.md
fi
