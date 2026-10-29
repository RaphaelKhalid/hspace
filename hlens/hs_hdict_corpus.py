"""Corpus trigram counts for the H-lens dictionary ground truth (PREREG-hlens-dictionary-2026-10-28.md). CPU, $0.

First 20M tokens of WikiText-103 train (Qwen3.5 tokenizer, article text concatenated in file order). Counts, over the
1000 most frequent tokens K (prev r, cur s) and the 16 targets t (next):
  N_rst[t, r, s], N_rs[r, s], N_st[s, t] (cur s -> next t), N_r2t[r, t] (r two before t), N_s[s], N_t[t], total
    python hs_hdict_corpus.py
"""
import glob
import json
import os
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = [" the", " .", " ,", " of", " and", " in", " to", " a",
           " simulated", " fake", " virtual", " artificial", " real", " genuine", " river", " music"]


def main():
    import pyarrow.parquet as pq
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B")
    root = snapshot_download("Salesforce/wikitext", repo_type="dataset", allow_patterns=["wikitext-103-raw-v1/train-*"])
    ids, n = [], 0
    for f in sorted(glob.glob(os.path.join(root, "wikitext-103-raw-v1", "train-*.parquet"))):
        lines = pq.read_table(f).column("text").to_pylist()
        for i in range(0, len(lines), 2000):
            chunk = "".join(lines[i:i + 2000])
            x = tok(chunk, add_special_tokens=False)["input_ids"]
            ids.append(np.asarray(x, dtype=np.int32)); n += len(x)
            if n >= 20_000_000:
                break
        if n >= 20_000_000:
            break
    a = np.concatenate(ids)[:20_000_000]
    print(f"tokenized {len(a)} tokens in {time.time()-t0:.0f}s", flush=True)
    V = int(a.max()) + 1
    cnt = np.bincount(a, minlength=V)
    K = np.argsort(-cnt)[:1000]
    L = np.full(V, -1, dtype=np.int64); L[K] = np.arange(1000)
    tid = [tok(t, add_special_tokens=False)["input_ids"] for t in TARGETS]
    assert all(len(x) == 1 for x in tid), tid
    tid = [x[0] for x in tid]
    Tm = np.full(V, -1, dtype=np.int64); Tm[tid] = np.arange(len(tid))
    nk, nt = 1000, len(tid)
    r, s, t = L[a[:-2]], L[a[1:-1]], Tm[a[2:]]
    m2 = (r >= 0) & (s >= 0)
    N_rs = np.bincount(r[m2] * nk + s[m2], minlength=nk * nk).reshape(nk, nk)
    m3 = m2 & (t >= 0)
    N_rst = np.bincount((t[m3] * nk + r[m3]) * nk + s[m3], minlength=nt * nk * nk).reshape(nt, nk, nk)
    s1, t1 = L[a[:-1]], Tm[a[1:]]
    mm = (s1 >= 0) & (t1 >= 0)
    N_st = np.bincount(s1[mm] * nt + t1[mm], minlength=nk * nt).reshape(nk, nt)
    r2, t2 = L[a[:-2]], Tm[a[2:]]
    mm = (r2 >= 0) & (t2 >= 0)
    N_r2t = np.bincount(r2[mm] * nt + t2[mm], minlength=nk * nt).reshape(nk, nt)
    N_s = cnt[K]
    N_t = cnt[tid]
    out = HERE / "out" / "hdict_corpus_counts.npz"
    np.savez_compressed(out, K=K, tid=np.asarray(tid), N_rs=N_rs, N_rst=N_rst, N_st=N_st, N_r2t=N_r2t, N_s=N_s, N_t=N_t,
                        total=len(a))
    json.dump({"tokens": len(a), "targets": TARGETS, "target_counts": {t: int(c) for t, c in zip(TARGETS, N_t)},
               "pairs_with_N_rs_ge_30": int((N_rs >= 30).sum()), "seconds": time.time() - t0},
              open(HERE.parent / "results" / "fullh" / "hdict_corpus_summary.json", "w"), indent=1)
    print(f"saved {out}; pairs with N(r,s) >= 30: {(N_rs >= 30).sum()}; target counts {dict(zip(TARGETS, N_t.tolist()))} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
