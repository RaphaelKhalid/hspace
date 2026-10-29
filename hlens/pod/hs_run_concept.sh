#!/usr/bin/env bash
# Concept full-Hessian run (exploratory; prereg in hs_fullh_concept.py docstring). Qwen3.5-0.8B L18, 8 tokens x 64 contexts.
# Money rules: watchdog TERMINATES the pod after HS_MAX_MIN (default 70) minutes no matter what; a 4-context cost check
# runs first and the pod terminates if the projected full run exceeds HS_MAX_USD at HS_USD_PER_HR; after a verified
# upload to the private HF dataset the pod terminates itself.
set -uo pipefail
cd /workspace/mats/interpcontrol/hlens
mkdir -p out ../results/fullh
export HF_HUB_ENABLE_HF_TRANSFER=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONIOENCODING=utf-8
eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^RUNPOD_(API_KEY|POD_ID)=' | sed 's/^/export /')"
terminate() {
  curl -s -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" -H "Authorization: Bearer $RUNPOD_API_KEY" >> out/pod_watchdog.log 2>&1
  runpodctl remove pod "$RUNPOD_POD_ID" >> out/pod_watchdog.log 2>&1
}
( sleep $(( ${HS_MAX_MIN:-70} * 60 ));
  echo "[watchdog $(date +%T)] max time reached, terminating $RUNPOD_POD_ID" >> out/pod_watchdog.log; terminate ) > /dev/null 2>&1 &
echo "[pod $(date +%T)] concept watchdog armed (${HS_MAX_MIN:-70} min) for pod ${RUNPOD_POD_ID:-UNKNOWN}" | tee -a runlog.md
TOKS=" simulated| fake| virtual| artificial| real| genuine| river| music"
C=${HS_C:-64}; G=${HS_G:-8}
python - <<'PY'
import time
t = time.time()
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-0.8B", max_workers=16)
snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/train-*"])
print(f"downloads done in {time.time()-t:.0f}s", flush=True)
PY
echo "[pod $(date +%T)] downloads done" | tee -a runlog.md
# ---- cost check: 4 contexts, all 8 tokens, same batch layout as the full run
T0=$(date +%s)
python hs_fullh.py --toklist "$TOKS" --center --n 4 --flips 2 --C $C --G $G --check --tag ccheck > out/ccheck.log 2>&1 || { echo "[pod] cost check FAILED"; tail -5 out/ccheck.log; terminate; exit 1; }
T1=$(date +%s)
python - "$T0" "$T1" <<'PY' || { echo "[pod] projected cost over cap: terminating" | tee -a runlog.md; terminate; exit 1; }
import os, re, sys
log = open("out/ccheck.log").read()
assert "check: fast vs full-sequence" in log, "exactness check missing"
err = float(re.search(r"rel err ([0-9.e+-]+)", log).group(1)); assert err < 1e-3, err
sec = int(sys.argv[2]) - int(sys.argv[1])
per_ctx = sec / 4.0                                   # upper bound: includes model load and the tiny stats stage
proj_min = (64 * per_ctx) / 60 + 20                   # + stats, concept tests, deflation (3 kinds x 200 flips), upload
usd = proj_min / 60 * float(os.environ.get("HS_USD_PER_HR", "2.69"))
print(f"COST CHECK: exactness rel err {err:.1e} | {sec}s for 4 contexts -> projected {proj_min:.0f} min, ${usd:.2f}", flush=True)
open("out/cost_check.txt", "w").write(f"{err} {sec} {proj_min:.1f} {usd:.2f}\n")
sys.exit(0 if usd <= float(os.environ.get("HS_MAX_USD", "5")) else 1)
PY
cat out/cost_check.txt | tee -a runlog.md
sync_once() {
python - <<'PY'
from huggingface_hub import HfApi
import glob, os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
fs = glob.glob("../results/fullh/*concept*.json") + glob.glob("out/fullh_concept*.log") + ["runlog.md", "out/cost_check.txt"]
for f in fs:
    if os.path.exists(f):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"concept/{os.path.basename(f)}", repo_id=R, repo_type="dataset")
        except Exception as e:  # noqa: BLE001
            print("sync retry later", f, repr(e)[:100], flush=True)
PY
}
( while sleep 300; do sync_once >> out/hf_sync.log 2>&1; done ) &
SYNC_PID=$!
python hs_fullh.py --toklist "$TOKS" --center --n 64 --C $C --G $G --tag concept > out/fullh_concept_main.log 2>&1; RC1=$?
python hs_fullh_concept.py --tag concept > out/fullh_concept_tests.log 2>&1; RC2=$?
python hs_fullh_defl.py --toklist "$TOKS" --center --kinds own,all8,allvar --tag concept > out/fullh_concept_defl.log 2>&1; RC3=$?
RC=$(( RC1 + RC2 + RC3 )); echo "[pod $(date +%T)] concept exit $RC1/$RC2/$RC3" | tee -a runlog.md
kill $SYNC_PID 2>/dev/null
python - <<'PY'
from huggingface_hub import HfApi
import glob, hashlib, os
api = HfApi(); R = "RaphaelRaphaelRaphael/hspace-27b-results"
fs = (glob.glob("../results/fullh/*concept*.json") + glob.glob("out/fullh_concept*.log") + glob.glob("out/fullh_concept_L18_*")
      + ["runlog.md", "out/cost_check.txt", "out/ccheck.log", "hs_fullh.py", "hs_fullh_concept.py", "hs_fullh_defl.py"])
fs = [f for f in fs if os.path.exists(f)]
for f in fs:
    for _ in range(3):
        try:
            api.upload_file(path_or_fileobj=f, path_in_repo=f"concept/{os.path.basename(f)}", repo_id=R, repo_type="dataset"); break
        except Exception as e:  # noqa: BLE001
            print("retry", f, repr(e)[:120], flush=True)
rm = {it.path: it for it in api.list_repo_tree(R, repo_type="dataset", path_in_repo="concept", recursive=True, expand=True) if hasattr(it, "size")}
bad = 0
for f in fs:
    it = rm.get(f"concept/{os.path.basename(f)}")
    ok = it is not None and it.size == os.path.getsize(f)
    if ok and os.path.getsize(f) < 2**31:
        lfs = getattr(getattr(it, "lfs", None), "sha256", None)
        ok = lfs is None or lfs == hashlib.sha256(open(f, "rb").read()).hexdigest()
    bad += not ok
    print("verified" if ok else "PROBLEM", f, flush=True)
print("UPLOAD-VERIFY problems:", bad, flush=True)
open("out/verify_ok", "w").write(str(bad))
PY
echo "[pod $(date +%T)] CONCEPT ALL DONE" | tee -a runlog.md
if [ "$(cat out/verify_ok 2>/dev/null)" = "0" ] && [ "$RC" = "0" ] && grep -q 'VERDICT' out/fullh_concept_tests.log 2>/dev/null; then
  echo "[pod $(date +%T)] verdict written and results verified on HF; self-terminating" | tee -a runlog.md
  terminate
else
  echo "[pod $(date +%T)] NOT self-terminating (rc=$RC or no verdict): watchdog still armed" | tee -a runlog.md
fi
