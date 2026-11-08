"""Helpers for Part 3: loading Qwen3.6-27B with its Jacobian lens, and reading the lens.

The Jacobian lens (J-lens) turns the model's hidden state at any layer into a probability
distribution over the whole vocabulary: "what is this layer disposed to make the model say?"
We read it at one position only, the decision token, which is the last token of the input.
With Qwen3.6's chat template the input ends with

    <|im_start|>assistant\n<think>\n

so the decision token is the newline right after the opening <think> tag: the position
whose next-token prediction is the first token of the model's reasoning.
"""

import numpy as np
from scipy.special import logsumexp

MODEL_ID = "Qwen/Qwen3.6-27B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_FILE = "qwen3.6-27b/jlens/Salesforce-wikitext/Qwen3.6-27B_jacobian_lens_n1000.pt"
LENS_REVISION = "qwen-n1000"

# Layers 20-55 of 64: the "workspace" layers (Gurnee et al. 2026). Earlier layers are not yet
# interpretable, and the last layers only compose the output.
LAYERS = list(range(20, 56))

# A token counts as "present" in the workspace if its rank is 25 or better.
TOP_K = 25


def load_model_and_lens():
    """Load the model on one GPU (needs ~60 GB, e.g. an A100 80GB) and the pre-fitted lens."""
    import jlens
    import torch
    import transformers

    hf_model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).cuda()
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION)
    return model, lens, tokenizer


def chat_text(tokenizer, system, user):
    """The exact text the model sees: system and user message in Qwen's chat template,
    ending where the assistant's turn begins."""
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def read_lens(model, lens, text):
    """Lens logits at the decision token (last input position) for every layer in LAYERS.
    Returns a numpy array of shape [len(LAYERS), vocabulary size]."""
    lens_logits, _, _ = lens.apply(model, text, layers=LAYERS, positions=[-1])
    return np.stack([lens_logits[layer][0].numpy() for layer in LAYERS])


def log10_prob_and_rank(logits, token_ids):
    """For each layer, the log10 probability and the rank (1 = most likely) of each token id.
    Returns two arrays of shape [len(LAYERS), len(token_ids)]."""
    log10_prob = np.zeros((len(logits), len(token_ids)), dtype=np.float32)
    rank = np.zeros((len(logits), len(token_ids)), dtype=np.int32)
    for i, layer_logits in enumerate(logits):
        values = layer_logits[token_ids]
        log10_prob[i] = (values - logsumexp(layer_logits)) / np.log(10)
        # rank = 1 + number of vocabulary tokens with a higher logit
        sorted_logits = np.sort(layer_logits)
        rank[i] = len(layer_logits) - np.searchsorted(sorted_logits, values, side="right") + 1
    return log10_prob, rank
