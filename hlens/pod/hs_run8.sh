#!/usr/bin/env bash
# v8 one-shot (prereg paper/SCOPE-hspace-v8.md). Watchdog TERMINATES the pod after HS_MAX_MIN (default 50) minutes.
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
( sleep $(( ${HS_MAX_MIN:-50} * 60 ));
  echo "[watchdog $(date +%T)] max time reached, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1 ) > /dev/null 2>&1 &
echo "[pod $(date +%T)] v8 watchdog armed (${HS_MAX_MIN:-50} min) for pod ${RUNPOD_POD_ID:-UNKNOWN}" | tee -a runlog.md
pip install --break-system-packages -q "transformers==5.5.4" scipy pyarrow pandas accelerate hf_transfer 2>&1 | tail -1
pip install --break-system-packages -q flash-linear-attention 2>&1 | tail -1 || true
python - <<'PY'
import time
t = time.time()
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
for sp in ("train",):
    snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=[f"wikitext-103-raw-v1/{sp}-*"])
for f in ("out/hspace2_full.pt", "out/hspace2_full_dump_L16.pt", "out/hspace2_full_dump_L40.pt"):
    hf_hub_download("RaphaelRaphaelRaphael/hspace-27b-results", f, repo_type="dataset", local_dir=".")
print(f"downloads done in {time.time()-t:.0f}s", flush=True)
PY
echo "[pod $(date +%T)] downloads done" | tee -a runlog.md
python hs_v8.py full 16 40 > out/hs_v8.log 2>&1; echo "[pod $(date +%T)] v8 exit $?" | tee -a runlog.md
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
for f in ("out/hs_v8_full.json", "out/hs_v8.log", "runlog.md", "out/hs_v8_full_L16.pt", "out/hs_v8_full_L40.pt"):
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo=f"v8/{f}", repo_id="RaphaelRaphaelRaphael/hspace-27b-results", repo_type="dataset")
        print("uploaded", f, flush=True)
    except Exception as e:  # noqa: BLE001
        print("upload failed", f, repr(e)[:200], flush=True)
PY
echo "[pod $(date +%T)] V8 ALL DONE" | tee -a runlog.md
