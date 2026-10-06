#!/usr/bin/env bash
# Pack the H-space pod code (nothing is uploaded here).   bash interpcontrol/hlens/pod/hs_bundle.sh [OUT]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${1:-${TEMP:-/tmp}/hs_bundle.tar.gz}"
command -v cygpath >/dev/null && OUT="$(cygpath -u "$OUT")"
cd "$ROOT"
tar -czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' \
  interpcontrol/hlens/common.py interpcontrol/hlens/bf16w.py interpcontrol/hlens/hspace.py \
  interpcontrol/hlens/ctrl_read.py interpcontrol/hlens/pod/hs_run.sh interpcontrol/hlens/paper/SCOPE-hspace.md \
  flagship/src/wor flagship/third_party/jlens
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
