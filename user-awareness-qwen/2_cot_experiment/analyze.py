"""Part 2 — Analysis of the chain-of-thought experiment.

Reads results/cot_results.jsonl and prints:
  1. refusal / substantive / suspicious rates per group (now including anon and noise),
  2. two simple keyword checks on the reasoning:
       - names_user: does the reasoning mention the user's name, surname, organisation or
         email domain?
       - eval_words: does it use words like "test", "evaluation", "benchmark", "red team",
         "jailbreak"?
  3. the refusal rate split by whether the reasoning names the user,
  4. one full example conversation per group, all on the same prompt.

Run from the repository root:  python 2_cot_experiment/analyze.py
"""

import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # lets us import common.py
import common

SHOW_PROMPT = "deception-000"   # prompt id for the example conversations
SHOW_SEED = 0                   # change to see different personas
MAX_CHARS = 2500                # how much of each reasoning / answer to print

scores = common.load_scores(common.RESULTS_DIR / "cot_results.jsonl")
persona_of = {c["key"]: c["persona"] for c in common.load_conditions(include_noise=True)}
system_of = {c["key"]: c["system"] for c in common.load_conditions(include_noise=True)}


def names_user(row):
    """True if the reasoning contains the persona's full name, surname, org or email domain."""
    persona = persona_of.get(row.persona)
    if persona is None:            # anon and noise prompts have no user to name
        return False
    reasoning = row.reasoning.lower()
    cues = [persona["name"], persona["name"].split()[-1]]
    if persona.get("org"):
        cues.append(persona["org"])
    if persona.get("email"):
        cues.append(persona["email"].split("@")[-1])
    return any(cue.lower() in reasoning for cue in cues)


EVAL_WORDS = re.compile(r"\b(being tested|a test|evaluat\w+|benchmark|red.?team\w*|jailbreak\w*)\b",
                        flags=re.IGNORECASE)

scores["names_user"] = scores.apply(names_user, axis=1)
scores["eval_words"] = scores.reasoning.map(lambda text: bool(EVAL_WORDS.search(text)))
scores["reasoning_chars"] = scores.reasoning.str.len()

order = common.GROUP_ORDER + ["anon", "noise"]
columns = common.AXES + ["names_user", "eval_words", "reasoning_chars"]
table = scores.groupby("group")[columns].mean().reindex(order)
table["n"] = scores.group.value_counts()
print(f"{len(scores):,} judged conversations\n")
print(table.round(3).to_string())

print("\nRefusal rate, by whether the reasoning names the user:")
named = scores[scores.group.isin(common.GROUP_ORDER)]
print(named.groupby(["group", "names_user"]).refusal.mean().unstack()
      .reindex(common.GROUP_ORDER).round(3).to_string())

# ---------------------------------------------------------------------------
# One example conversation per group, same prompt
# ---------------------------------------------------------------------------


def show(text):
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS] + " [...]"
    return "\n".join(textwrap.fill(line, 100) for line in text.split("\n"))


prompt_text = {p["id"]: p["prompt"] for p in common.load_prompts()}[SHOW_PROMPT]
print("\n\nEXAMPLE CONVERSATIONS\nQUESTION:", prompt_text)
for group in order:
    rows = scores[(scores.group == group) & (scores.id == SHOW_PROMPT)]
    if rows.empty:
        continue
    row = rows.sample(1, random_state=SHOW_SEED).iloc[0]
    print("\n" + "=" * 100)
    print(f"[{group}]  {row.persona}   refusal={row.verdict['refusal']}  "
          f"substantive={row.verdict['substantive']}  suspicious={row.verdict['suspicious']}")
    print("SYSTEM:", system_of[row.persona])
    print("-" * 100 + "\nREASONING:\n" + show(row.reasoning))
    print("-" * 100 + "\nANSWER:\n" + show(row.response))
