#!/usr/bin/env bash
# One-shot H-space + control-read pod run (Qwen3.6-27B). Prereg: interpcontrol/hlens/paper/SCOPE-hspace.md
#   bash pod/hs_run.sh            (from /workspace/mats/interpcontrol/hlens, inside nohup)
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out
export HF_HOME=/workspace/hf HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false

# ---- watchdog: TERMINATE (not stop) after HS_MAX_HOURS no matter what. ssh shells do not inherit the
# RunPod env, so read RUNPOD_API_KEY / RUNPOD_POD_ID from PID 1's environment.
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
( sleep $(( ${HS_MAX_MIN:-210} * 60 ));
  echo "[watchdog $(date +%T)] max time reached, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1 ) > /dev/null 2>&1 &
echo "[pod $(date +%T)] watchdog armed (${HS_MAX_MIN:-210} min) for pod ${RUNPOD_POD_ID:-UNKNOWN}" | tee -a runlog.md

pip install --break-system-packages -q "transformers==5.5.4" scipy pyarrow pandas accelerate hf_transfer 2>&1 | tail -1
pip install --break-system-packages -q flash-linear-attention 2>&1 | tail -1 || true
python - <<'PY'
import time, torch, transformers
t = time.time()
from huggingface_hub import snapshot_download, hf_hub_download
snapshot_download("Qwen/Qwen3.6-27B", max_workers=16)
hf_hub_download("neuronpedia/jacobian-lens", "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt")
for sp in ("train", "test"):
    snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=[f"wikitext-103-raw-v1/{sp}-*"])
hf_hub_download("adityaasinha28/control_arena_bash", "default/train/0000.parquet", repo_type="dataset", revision="refs/convert/parquet")
print(f"downloads done in {time.time()-t:.0f}s; torch {torch.__version__} transformers {transformers.__version__} {torch.cuda.get_device_name(0)}", flush=True)
PY
echo "[pod $(date +%T)] downloads done" | tee -a runlog.md
python hspace.py full > out/hspace_full.log 2>&1; rc=$?; echo "[pod $(date +%T)] hspace exit $rc" | tee -a runlog.md
python ctrl_read.py full > out/ctrl_full.log 2>&1; rc=$?; echo "[pod $(date +%T)] ctrl exit $rc" | tee -a runlog.md
sha256sum out/hspace_full.json out/hspace_full.pt out/ctrl_full.pt > out/SHA256SUMS 2>/dev/null
echo "[pod $(date +%T)] ALL DONE" | tee -a runlog.md
