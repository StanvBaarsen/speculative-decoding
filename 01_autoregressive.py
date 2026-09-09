"""Step 1: ordinary autoregressive sampling, written by hand.

This is the baseline that speculative decoding has to beat, and also the
reference distribution it has to match. One forward pass per generated token.

The KV cache is the only non-obvious piece. Without it, every step would
re-process the whole sequence. With it, after the first (prefill) pass we feed
the model just ONE new token per step and it attends to the cached keys and
values of everything before it. So the logits shape per step is (1, 1, vocab).
"""

import time

import torch
from transformers import DynamicCache

from common import DRAFT, chat_prompt, load, pick_device, probs_from_logits, sample


@torch.no_grad()
def generate_autoregressive(model, input_ids, max_new_tokens, temperature=1.0, eos_id=None, verbose=False):
    """input_ids: (1, prompt_len). Returns (generated token ids as a list, number of forward passes)."""
    cache = DynamicCache(config=model.config)
    n_forward = 0

    # Prefill: process the whole prompt at once. Logits: (1, prompt_len, vocab).
    out = model(input_ids, past_key_values=cache, use_cache=True)
    n_forward += 1
    if verbose:
        print(f"prefill: fed {tuple(input_ids.shape)}, logits {tuple(out.logits.shape)}, cache len {cache.get_seq_length()}")

    generated = []
    for step in range(max_new_tokens):
        # Only the LAST position matters: it predicts the next token.
        p = probs_from_logits(out.logits[0, -1], temperature)   # (vocab,)
        x = sample(p)                                            # scalar token id
        generated.append(x.item())
        if eos_id is not None and x.item() == eos_id:
            break

        # Decode step: feed exactly one token. The cache supplies the rest.
        out = model(x.view(1, 1), past_key_values=cache, use_cache=True)
        n_forward += 1
        if verbose and step < 2:
            print(f"step {step}: fed (1, 1), logits {tuple(out.logits.shape)}, cache len {cache.get_seq_length()}")

    return generated, n_forward


if __name__ == "__main__":
    device = pick_device()
    tok, model = load(DRAFT, device)
    eos_id = tok.convert_tokens_to_ids("<|im_end|>")

    input_ids = chat_prompt(tok, "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?").to(device)

    torch.manual_seed(0)
    t0 = time.perf_counter()
    generated, n_forward = generate_autoregressive(model, input_ids, max_new_tokens=128, temperature=0.7, eos_id=eos_id, verbose=True)
    dt = time.perf_counter() - t0

    print()
    print(tok.decode(generated))
    print()
    print(f"{len(generated)} tokens, {n_forward} forward passes, {dt:.2f}s, {len(generated)/dt:.1f} tok/s")
