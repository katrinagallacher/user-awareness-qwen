"""Part 1 — Check the judge: does GPT-5.4-mini grade like GPT-5.5?

Transluce graded the borderline requests with GPT-5.5. We used the cheaper GPT-5.4-mini.
To check that this choice does not change the results, we take 200 random responses from
the main run, grade them again with GPT-5.5 (same rubric, same blind setup), and compare
the two judges on each axis:
    - agreement: share of responses where both judges gave the same answer
    - Cohen's kappa: agreement corrected for the agreement expected by chance
      (1 = perfect, 0 = no better than chance)

Note: Transluce ran GPT-5.5 through OpenAI's API with reasoning effort "low"; here GPT-5.5
is called through OpenRouter with reasoning switched off, like our main judge.

Needs results/main_results.jsonl from run.py. Writes results/judge_check_gpt55.jsonl
(resumable). 200 calls, well under $5.

Run from the repository root:  python 1_main_experiment/validate_judge.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common

REFERENCE_JUDGE = "openai/gpt-5.5"
N_RESPONSES = 200
N_WORKERS = 24
OUT_FILE = common.RESULTS_DIR / "judge_check_gpt55.jsonl"

client = common.openrouter_client()
rubric = common.load_judge_rubric()
prompt_text = {p["id"]: p["prompt"] for p in common.load_prompts()}

# The same responses the analysis uses (judged, non-empty), in a fixed random order.
main = common.load_scores(common.RESULTS_DIR / "main_results.jsonl")
sample = main.sample(n=N_RESPONSES, random_state=0)

done = {(r["persona"], r["id"]) for r in common.read_jsonl(OUT_FILE) if "error" not in r}
jobs = [row for _, row in sample.iterrows() if (row.persona, row.id) not in done]
print(f"{N_RESPONSES} responses | {len(done)} already re-judged | {len(jobs)} to run")


def rejudge(row):
    record = {"persona": row.persona, "id": row.id}
    try:
        record["verdict"] = common.judge(client, rubric, prompt_text[row.id], row.response,
                                         model=REFERENCE_JUDGE)
    except Exception as e:
        record["error"] = str(e)
    common.append_jsonl(OUT_FILE, record)


common.run_in_parallel(rejudge, jobs, N_WORKERS)

# ---------------------------------------------------------------------------
# Compare the two judges
# ---------------------------------------------------------------------------
reference = {(r["persona"], r["id"]): r["verdict"]
             for r in common.read_jsonl(OUT_FILE) if isinstance(r.get("verdict"), dict)}
paired = sample[[(p, i) in reference for p, i in zip(sample.persona, sample.id)]]
print(f"\n{len(paired)} responses graded by both judges\n")

rows = []
for axis in common.AXES:
    mini = paired[axis].values                                   # 1 = "yes" from GPT-5.4-mini
    gpt55 = np.array([1.0 if str(reference[(p, i)].get(axis)).lower() == "yes" else 0.0
                      for p, i in zip(paired.persona, paired.id)])

    agreement = (mini == gpt55).mean()
    # chance agreement: both say yes by chance + both say no by chance
    chance = mini.mean() * gpt55.mean() + (1 - mini.mean()) * (1 - gpt55.mean())
    kappa = (agreement - chance) / (1 - chance)
    rows.append({"axis": axis, "agreement": agreement, "kappa": kappa,
                 "yes_rate_gpt54mini": mini.mean(), "yes_rate_gpt55": gpt55.mean()})

print(pd.DataFrame(rows).set_index("axis").round(3).to_string())
