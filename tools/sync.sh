#!/usr/bin/env bash
# Copy the working H-space files from the research workspace into this repo (code, docs, small results; never data,
# tensors, tokens). Usage: bash tools/sync.sh   (then review `git status` and commit)
set -euo pipefail
SRC="${HS_SRC:-/c/Users/rapha/OneDrive/Desktop/Claude/matsempirical2027}"
R="$(cd "$(dirname "$0")/.." && pwd)"
H="$SRC/interpcontrol/hlens"
cp "$H"/{bf16w,common,hl2,hvp,ladder,hspace,hs_s7,hspace2,hs_c6prime,hs_anatomy,hs_integrated,hs_rectangle,hs_local,hs_diag_top,hs_dump_analysis,hs_confound,hs_c6defl,hs_relcurv,hs_relcurv_gain,ctrl_read,ctrl_score,ctrl_score_v2}.py "$R/hlens/" 2>/dev/null || true
cp "$H"/tests/{conftest,test_bf16w,test_swiglu,test_hvp,test_deltanet_patch}.py "$R/hlens/tests/" 2>/dev/null || true
cp "$H"/pod/{hs_run.sh,hs_bundle.sh,chain2.sh,chain3.sh,chain4.sh,relcurv_layers.sh,watchdog11.sh,hf_sync.py} "$R/hlens/pod/" 2>/dev/null || true
cp "$H"/dash_hs/{server.py,index.html} "$R/hlens/dash_hs/"
cp "$H"/paper/*.md "$H"/paper/*.py "$H"/paper/redteam-theory.json "$R/hlens/paper/" 2>/dev/null || true
cp "$SRC"/flagship/src/wor/{__init__,_paths,modeling}.py "$R/flagship/src/wor/"
mkdir -p "$R/flagship/third_party/jlens/data"
cp "$SRC"/flagship/third_party/jlens/{__init__,_logging,fitting,hf,hooks,lens,protocol,vis,examples}.py "$SRC"/flagship/third_party/jlens/{LICENSE,PROVENANCE.md} "$R/flagship/third_party/jlens/"
O="$H/out"
for f in anatomy_08b.json integrated_08b_L12.json rectangle_08b_L12.json local_hlens_08b.json hspace_small.json hspace2_small.json; do [ -f "$O/$f" ] && cp "$O/$f" "$R/results/"; done
for f in hspace_full.json; do [ -f "$O/pod/$f" ] && cp "$O/pod/$f" "$R/results/"; done
for f in "$O"/pod/hf/out/*.json; do [ -f "$f" ] && cp "$f" "$R/results/"; done
cp "$H/runlog.md" "$R/results/runlog.md"
echo "synced into $R"
# public copy: pod address / key / start time come from the environment, never hardcoded
sed -i -e 's|^HOST, PORT = .*|HOST, PORT = os.environ.get("HS_POD_HOST", ""), os.environ.get("HS_POD_PORT", "22")|' \
       -e 's|^POD_ID = .*|POD_ID = os.environ.get("HS_POD_ID", "")|' \
       -e 's|^KEY = .*|KEY = os.environ.get("HS_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))|' \
       -e 's|^STARTED = .*|STARTED = float(os.environ.get("HS_POD_STARTED_EPOCH", "0"))|' \
       -e 's|^import json$|import json\nimport os|' "$R/hlens/dash_hs/server.py"
if grep -rnE "hf_[A-Za-z0-9]{20,}|gh[ops]_[A-Za-z0-9]{20,}|sk-[A-Za-z0-9]{20,}|BEGIN (RSA|OPENSSH|EC) PRIVATE" "$R" --exclude-dir=.git \
   || grep -rnE "[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}:[0-9]{2,5}" "$R" --exclude-dir=.git | grep -v "127\.0\.0\.1"; then
  echo "SECRET-LIKE STRING FOUND, aborting"; exit 1
fi
echo "secret scan clean"
