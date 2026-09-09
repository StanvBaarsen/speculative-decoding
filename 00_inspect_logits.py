"""Step 0: what does the model actually give us?

Run a single forward pass and look at the shape of the output, then show that
the logits at position t are the model's prediction for the token at t + 1.
Everything later in this project depends on getting that indexing right.
"""

import torch
from common import DRAFT, chat_prompt, load, pick_device, probs_from_logits

device = pick_device()
tok, model = load(DRAFT, device)

input_ids = chat_prompt(tok, "What is 7 times 8?").to(device)
print("input_ids shape:", tuple(input_ids.shape), "  # (batch, prompt_len)")
print("vocab size:     ", model.config.vocab_size)
print()

with torch.no_grad():
    out = model(input_ids)

logits = out.logits
print("logits shape:   ", tuple(logits.shape), "  # (batch, prompt_len, vocab)")
print("logits dtype:   ", logits.dtype)
print()

# One distribution per position. Position t predicts token t+1.
probs = probs_from_logits(logits, temperature=1.0)
print("probs shape:    ", tuple(probs.shape))
print("each row sums to 1:", torch.allclose(probs.sum(-1), torch.ones_like(probs.sum(-1))))
print()

# Show the off-by-one: for each position t, does the model's argmax at t equal
# the actual token at t+1? For a prompt it often will not (the model did not
# write the prompt), but the alignment is what matters.
print("position | token at t        | model's top guess for t+1 | actual t+1")
ids = input_ids[0].tolist()
for t in range(len(ids) - 1):
    guess = probs[0, t].argmax().item()
    print(f"{t:8d} | {tok.decode([ids[t]])!r:17} | {tok.decode([guess])!r:25} | {tok.decode([ids[t+1]])!r}")
print()

# The distribution we actually sample the next token from is the LAST position.
next_dist = probs[0, -1]                     # shape (vocab,)
print("next-token distribution shape:", tuple(next_dist.shape))
top = torch.topk(next_dist, 8)
print("top 8 candidates for the first generated token:")
for pr, idx in zip(top.values.tolist(), top.indices.tolist()):
    print(f"  {pr:7.4f}  id={idx:6d}  {tok.decode([idx])!r}")
print(f"  mass in top 8: {top.values.sum().item():.4f}, rest spread over {len(next_dist) - 8} tokens")
