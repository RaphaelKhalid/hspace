"""Pod -> private Hugging Face dataset relay (the direct pod->laptop link is ~40 KB/s). Uses the HF login cached on the
pod (huggingface_hub.login, done once out-of-band; the token is never written to project files or logs).

    python pod/hf_sync.py once        # upload changed result files now
    python pod/hf_sync.py loop        # every 5 min until FINAL DONE appears in runlog.md
"""
import hashlib
import sys
import time
from pathlib import Path

from huggingface_hub import HfApi

REPO = "RaphaelRaphaelRaphael/hspace-27b-results"
D = Path("/workspace/mats/interpcontrol/hlens")
PATTERNS = ["out/hspace_full.json", "out/hspace_full.pt", "out/hspace_full_slim.pt", "out/hs_s7_full.log",
            "out/hs_diag_top_full.json", "out/ctrl_full.pt", "out/ctrl_full.log", "out/ctrl_score*_full.json",
            "out/hspace2_full.json", "out/hspace2_full.pt", "out/hspace2_full_dump_L*.pt", "out/hspace2_full.log",
            "out/hs_dump_analysis_full.json", "out/hspace_full.log", "out/SHA256SUMS", "runlog.md",
            "out/c6prime_full.json", "out/hs_c6prime_full.log", "out/hf_sync.log",
            "out/c6defl_full.json", "out/hs_c6defl_full.log", "out/hs_confound_*.json", "out/hs_confound_*.log",
            "out/ctrl_score_v2_full.log",
            "out/hs_flip*_full.json", "out/hs_flip*_full.log", "out/hs_flip5_misspec*", "out/hs_flip5_tokens.*", "out/c6twin5_full.json",
            "out/hs_c6twin_full.log", "out/hs_dump_analysis_full.log", "out/relcurv2_full.json", "out/hs_relcurv*.log", "out/hs_confound*.json",
            "pod/*.sh", "pod/*.py", "*.py"]
api = HfApi()
api.create_repo(REPO, repo_type="dataset", private=True, exist_ok=True)
seen = {}


def sync():
    n = 0
    for pat in PATTERNS:
        for f in sorted(D.glob(pat)):
            if f.name.endswith(".tmp"):
                continue
            st = f.stat()
            key = (st.st_size, int(st.st_mtime))
            if seen.get(f) == key or time.time() - st.st_mtime < 20:      # skip unchanged / still being written
                continue
            api.upload_file(path_or_fileobj=str(f), path_in_repo=str(f.relative_to(D)), repo_id=REPO, repo_type="dataset",
                            commit_message=f"sync {f.name}")
            seen[f] = key
            n += 1
            print(f"[hf_sync {time.strftime('%H:%M:%S')}] uploaded {f.relative_to(D)} ({st.st_size/1e6:.1f} MB)", flush=True)
    return n


if __name__ == "__main__":
    if sys.argv[1:] == ["once"]:
        sync()
    else:
        while True:
            try:
                sync()
            except Exception as e:  # noqa: BLE001
                print(f"[hf_sync] error {type(e).__name__}: {e}", flush=True)
            if "FINAL DONE" in (D / "runlog.md").read_text(errors="ignore"):
                sync(); print("[hf_sync] final sync done", flush=True); break
            time.sleep(300)
