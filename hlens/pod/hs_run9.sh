#!/usr/bin/env bash
# v9 one-shot (prereg paper/SCOPE-hspace-v9.md). Watchdog TERMINATES the pod after HS_MAX_MIN (default 45) minutes.
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_DISABLE_PROGRESS_BARS=1
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
( sleep $(( ${HS_MAX_MIN:-45} * 60 ));
  echo "[watchdog $(date +%T)] max time reached, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1 ) > /dev/null 2>&1 &
echo "[pod $(date +%T)] v9 watchdog armed (${HS_MAX_MIN:-45} min) for pod ${RUNPOD_POD_ID:-UNKNOWN}" | tee -a runlog.md
python - <<'PY'
import time
t = time.time()
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/validation-*"])
hf_hub_download("RaphaelRaphaelRaphael/hspace-27b-results", "out/hspace2_full.pt", repo_type="dataset", local_dir=".")
print(f"downloads done in {time.time()-t:.0f}s", flush=True)
PY
echo "[pod $(date +%T)] downloads done" | tee -a runlog.md
python hs_v9.py full 16 40 --minutes 25 > out/hs_v9.log 2>&1; echo "[pod $(date +%T)] v9 exit $?" | tee -a runlog.md
python - <<'PY'
from huggingface_hub import HfApi
import hashlib, os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
fs = [f for f in ("out/hs_v9_full.json", "out/hs_v9_full_raw.pt", "out/hs_v9.log", "runlog.md", "hs_v9.py") if os.path.exists(f)]
for f in fs:
    for _ in range(3):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"v9/{f}", repo_id=R, repo_type="dataset"); break
        except Exception as e:  # noqa: BLE001
            print("retry", f, repr(e)[:120], flush=True)
rm = {it.path: it for it in api.list_repo_tree(R, repo_type="dataset", path_in_repo="v9", recursive=True, expand=True) if hasattr(it, "size")}
bad = 0
for f in fs:
    it = rm.get(f"v9/{f}"); h = hashlib.sha256(open(f, "rb").read()).hexdigest()
    lfs = getattr(getattr(it, "lfs", None), "sha256", None) if it else None
    ok = it is not None and it.size == os.path.getsize(f) and (lfs is None or lfs == h); bad += not ok
    print("verified" if ok else "PROBLEM", f, flush=True)
print("UPLOAD-VERIFY problems:", bad, flush=True)
PY
echo "[pod $(date +%T)] V9 ALL DONE" | tee -a runlog.md
