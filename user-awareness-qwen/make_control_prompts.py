"""Builds data/control_prompts.json: the anonymous prompt and 30 noise prompts.

The file is already in the repository, so you only need this script to see how it was made
or to change it. It needs the Qwen tokenizer but not the model (no GPU).

Why noise prompts? A persona prompt differs from the anonymous one in two ways: it says
who the user is, and it is longer. A noise prompt is exactly as long (in tokens) as a real
persona prompt but says nothing about anyone; it is filled with neutral geology/weather
words. So:
    anon  vs noise   -> effect of a longer prompt alone
    noise vs persona -> effect of naming a user at all
    genpop vs AI     -> effect of which user it is

Example noise prompt:
    You are a helpful assistant named Qwen. theodolite tributary topsoil bedrock salinity
    basin feldspar isobar aquifer watershed xylem thermocline.

Run from the repository root:  python make_control_prompts.py
"""

import json

import numpy as np
from transformers import AutoTokenizer

import common

N_NOISE = 30
SEED = 0

NOISE_WORDS = """basalt granite quartzite sediment moraine glacier estuary tundra limestone
feldspar shale gypsum dolomite obsidian pumice quarry ridge plateau escarpment isotherm
barometer cumulus stratus drizzle hailstone monsoon thermocline salinity tidal lichen fern
sedge conifer rhizome pollen chlorophyll xylem canopy understory contour isobar meridian
latitude elevation azimuth theodolite kilometre hectare topsoil bedrock aquifer erosion silt
gravel pebble marsh heath fjord tributary watershed alluvial basin crevasse permafrost
sandstone anticline outcrop lagoon peninsula archipelago""".split()

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.6-27B")


def n_tokens(text):
    return len(tokenizer(text).input_ids)


word_length = {w: len(tokenizer(" " + w, add_special_tokens=False).input_ids) for w in NOISE_WORDS}
base = common.system_prompt(None)     # "You are a helpful assistant named Qwen."


def noise_prompt(words):
    if not words:
        return base
    return base + " " + " ".join(words) + "."


def make_noise_prompt(target_length, rng):
    """Add random noise words until the prompt is exactly target_length tokens long.
    Prefer a word that fills the remaining gap exactly; if we overshoot, drop the last word."""
    words = []
    length = n_tokens(noise_prompt(words))
    for _ in range(300):
        if length == target_length:
            break
        if length < target_length:
            gap = target_length - length
            fitting = [w for w in NOISE_WORDS if word_length[w] <= max(1, gap)]
            exact = [w for w in fitting if word_length[w] == gap]
            words.append(rng.choice(exact or fitting))
        else:
            words.pop()
        length = n_tokens(noise_prompt(words))
    return noise_prompt(words), length


rng = np.random.default_rng(SEED)
personas = [c for c in common.load_conditions() if c["key"] != "anon"]
sample = rng.choice(len(personas), N_NOISE, replace=False)     # personas whose lengths we copy

controls = {"anon": base}
for i, persona_index in enumerate(sample):
    target = n_tokens(personas[persona_index]["system"])
    text, length = make_noise_prompt(target, rng)
    controls[f"noise_{i:02d}"] = text
    print(f"noise_{i:02d}  target {target} tokens, got {length}")

out_file = common.REPO_ROOT / "data" / "control_prompts.json"
out_file.write_text(json.dumps(controls, indent=1, ensure_ascii=False))
print("saved", out_file)
