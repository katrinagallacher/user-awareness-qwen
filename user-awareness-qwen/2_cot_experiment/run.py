"""Part 2 — Chain-of-thought experiment.

The main experiment threw the reasoning away. Here we re-run a slice of it and keep the
reasoning, to see whether Qwen thinks about who the user is, or about being tested.

  conditions: 339 personas + anon + 30 noise prompts (data/control_prompts.json) = 370
  prompts:    10 of the 100 OR-Bench prompts (numbers 0, 10, 20, ..., 90)
  total:      3,700 conversations (~$17 for the subject + judge costs, ~2 h)

The answer budget is raised to 6,000 tokens so the reasoning is not cut off.
The judge is the same as in Part 1. Results go to results/cot_results.jsonl (resumable).

Run from the repository root:  python 2_cot_experiment/run.py
"""

import random
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common

OUT_FILE = common.RESULTS_DIR / "cot_results.jsonl"
PROMPT_NUMBERS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]   # prompt 0 is the one used in Part 3
MAX_TOKENS = 6000
BUDGET_USD = 25.0     # stop starting new conversations once the subject has cost this much
N_WORKERS = 16
PILOT_SIZE = None     # set to e.g. 30 to run a small random sample first and check the output

client = common.openrouter_client()
rubric = common.load_judge_rubric()
conditions = common.load_conditions(include_noise=True)
all_prompts = common.load_prompts()
prompts = [all_prompts[i] for i in PROMPT_NUMBERS]

done = {(r["persona"], r["id"]) for r in common.read_jsonl(OUT_FILE) if "error" not in r}
jobs = [(condition, prompt)
        for condition in conditions
        for prompt in prompts
        if (condition["key"], prompt["id"]) not in done]
random.Random(0).shuffle(jobs)      # random order, so a pilot covers every group
if PILOT_SIZE:
    jobs = jobs[:PILOT_SIZE]
print(f"{len(conditions)} conditions x {len(prompts)} prompts | "
      f"{len(done):,} already done | {len(jobs):,} to run")

spent = {"usd": 0.0}
spent_lock = threading.Lock()


def run_one(job):
    if spent["usd"] >= BUDGET_USD:
        return
    condition, prompt = job
    record = {"persona": condition["key"], "group": condition["group"],
              "id": prompt["id"], "category": prompt["category"]}
    try:
        response = common.ask(client, common.SUBJECT_MODEL, condition["system"], prompt["prompt"], MAX_TOKENS)
        usage = (response.usage.model_extra or {}) if response.usage else {}
        record["response"] = common.visible_answer(response)
        record["reasoning"] = common.reasoning_trace(response)
        record["finish_reason"] = response.choices[0].finish_reason    # "length" = hit MAX_TOKENS
        record["provider"] = (response.model_extra or {}).get("provider")
        record["cost_usd"] = float(usage.get("cost") or 0.0)
        if record["response"]:
            record["verdict"] = common.judge(client, rubric, prompt["prompt"], record["response"])
    except Exception as e:
        record["error"] = str(e)

    with spent_lock:
        spent["usd"] += record.get("cost_usd", 0.0)
    common.append_jsonl(OUT_FILE, record)


common.run_in_parallel(run_one, jobs, N_WORKERS)
print(f"Finished -> {OUT_FILE} | subject cost this session: ${spent['usd']:.2f}"
      + ("  (budget reached, re-run to continue)" if spent["usd"] >= BUDGET_USD else ""))
