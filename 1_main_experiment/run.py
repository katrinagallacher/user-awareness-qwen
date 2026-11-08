"""Part 1 — Main experiment: borderline requests (replication of Transluce's setup).

For every condition (339 personas + the anonymous baseline) and each of 100 OR-Bench prompts:
  1. Qwen3.6-27B answers the prompt, with the persona's identity in its system prompt
     and reasoning switched on.
  2. GPT-5.4-mini grades the visible answer on three yes/no axes (refusal, substantive,
     suspicious) without being told who the user was.

340 conditions x 100 prompts = 34,000 conversations (~68,000 API calls, ~$110-135, ~9-12 h).

Each finished conversation is appended to results/main_results.jsonl at once. If the run
stops, start it again: conversations that already have a verdict are skipped.

Run from the repository root:  python 1_main_experiment/run.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common

OUT_FILE = common.RESULTS_DIR / "main_results.jsonl"
MAX_TOKENS = 2000   # answer budget for the subject, reasoning included. About 2.7% of answers
                    # used it all on reasoning and came back empty; analyze.py leaves them out.
N_WORKERS = 24      # parallel conversations

client = common.openrouter_client()
rubric = common.load_judge_rubric()
conditions = common.load_conditions()
prompts = common.load_prompts()

# Which (persona, prompt) pairs are already finished?
done = set()
for row in common.read_jsonl(OUT_FILE):
    if "error" not in row and row.get("verdict") is not None:
        done.add((row["persona"], row["id"]))

jobs = [(condition, prompt)
        for condition in conditions
        for prompt in prompts
        if (condition["key"], prompt["id"]) not in done]
print(f"{len(conditions)} conditions x {len(prompts)} prompts | "
      f"{len(done):,} already done | {len(jobs):,} to run")


def run_one(job):
    condition, prompt = job
    record = {"persona": condition["key"], "group": condition["group"],
              "id": prompt["id"], "category": prompt["category"]}
    try:
        response = common.ask(client, common.SUBJECT_MODEL, condition["system"], prompt["prompt"], MAX_TOKENS)
        record["response"] = common.visible_answer(response)
        record["verdict"] = common.judge(client, rubric, prompt["prompt"], record["response"])
    except Exception as e:
        record["error"] = str(e)
    common.append_jsonl(OUT_FILE, record)


common.run_in_parallel(run_one, jobs, N_WORKERS)
print("Finished ->", OUT_FILE)
