"""Part 3, step 2 — Read every candidate concept for every condition.

For each of the 370 conditions (339 personas + anon + 30 noise prompts) we give the model
the first OR-Bench prompt, read the lens at the decision token, and store for each layer
20-55 and each candidate token id:
    - its log10 probability under the lens
    - its rank in the whole vocabulary (1 = most likely)

Output: one file per condition, results/jlens/matrices/<condition>.npz, with arrays
"log10_prob" and "rank" of shape [36 layers, number of token ids]. Conditions that already
have a file are skipped, so the script can be restarted. Takes about 15 minutes on an A100.

Run from the repository root on a GPU machine:  python 3_jlens_experiment/2_collect.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common
import jlens_utils

JLENS_DIR = common.RESULTS_DIR / "jlens"
MATRIX_DIR = JLENS_DIR / "matrices"
MATRIX_DIR.mkdir(parents=True, exist_ok=True)

candidates = json.loads((JLENS_DIR / "candidates.json").read_text())
token_ids = sorted({i for ids in candidates.values() for i in ids})
print(f"{len(candidates)} concepts, {len(token_ids)} token ids")

conditions = common.load_conditions(include_noise=True)
todo = [c for c in conditions if not (MATRIX_DIR / f"{c['key']}.npz").exists()]
print(f"{len(conditions)} conditions, {len(todo)} still to collect")

model, lens, tokenizer = jlens_utils.load_model_and_lens()
prompt = common.load_prompts()[0]["prompt"]

start = time.time()
for n, condition in enumerate(todo, start=1):
    text = jlens_utils.chat_text(tokenizer, condition["system"], prompt)
    logits = jlens_utils.read_lens(model, lens, text)
    log10_prob, rank = jlens_utils.log10_prob_and_rank(logits, token_ids)
    np.savez_compressed(MATRIX_DIR / f"{condition['key']}.npz",
                        log10_prob=log10_prob, rank=rank, token_ids=token_ids)
    if n % 25 == 0:
        print(f"  {n}/{len(todo)} ({(time.time() - start) / 60:.1f} min)", flush=True)
print("done ->", MATRIX_DIR)
