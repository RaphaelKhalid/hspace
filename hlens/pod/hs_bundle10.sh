#!/usr/bin/env bash
# Pack the v10 pod code (nothing is uploaded here).   bash interpcontrol/hlens/pod/hs_bundle10.sh [OUT]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="${1:-${TEMP:-/tmp}/hs_bundle10.tar.gz}"
command -v cygpath >/dev/null && OUT="$(cygpath -u "$OUT")"
cd "$ROOT"
tar -czf "$OUT" --exclude='__pycache__' --exclude='*.pyc' \
  interpcontrol/hlens/common.py interpcontrol/hlens/bf16w.py interpcontrol/hlens/hspace.py interpcontrol/hlens/hs_v9.py \
  interpcontrol/hlens/hs_v10.py interpcontrol/hlens/hs_v10a_classes.py interpcontrol/hlens/hs_v8.py interpcontrol/hlens/hs_flip5.py \
  interpcontrol/hlens/v10_banks interpcontrol/hlens/pod/hs_run10.sh interpcontrol/hlens/paper/SCOPE-hspace-v10.md \
  flagship/src/wor flagship/third_party/jlens
echo "wrote $OUT ($(du -h "$OUT" | cut -f1))"
