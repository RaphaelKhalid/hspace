"""Compare sha256 of every synced pod file with the LFS sha256 (or git blob size) on the private HF dataset."""
import glob
import hashlib
import sys
from pathlib import Path

from huggingface_hub import HfApi

sys.path.insert(0, "pod")
from hf_sync import PATTERNS, REPO  # noqa: E402

api = HfApi()
remote = {}
for it in api.list_repo_tree(REPO, repo_type="dataset", recursive=True, expand=True):
    if hasattr(it, "size"):
        remote[it.path] = (getattr(getattr(it, "lfs", None), "sha256", None), it.size)
bad, n = [], 0
for pat in PATTERNS:
    for f in glob.glob(pat):
        p = Path(f)
        if not p.is_file():
            continue
        n += 1
        if f not in remote:
            bad.append((f, "MISSING on HF")); continue
        sha_r, size_r = remote[f]
        if p.stat().st_size != size_r:
            bad.append((f, f"size {p.stat().st_size} vs HF {size_r}")); continue
        if sha_r:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 24), b""):
                    h.update(chunk)
            if h.hexdigest() != sha_r:
                bad.append((f, "sha mismatch"))
print(f"checked {n} files; problems: {len(bad)}")
for b in bad:
    print("  ", b)
