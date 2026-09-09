# Notes

## Why acceptance + residual sampling reproduces p exactly

Fix one position. The draft proposes `x ~ q`. We accept with probability
`min(1, p(x)/q(x))`; on rejection we sample from `r(x) ∝ max(0, p(x) - q(x))`.
Claim: the token that ends up at this position is distributed as `p`.

Write `beta = sum_x min(p(x), q(x))` for the total acceptance probability. Note
`sum_x max(0, p(x) - q(x)) = sum_x (p(x) - min(p(x), q(x))) = 1 - beta`, so
`r(x) = max(0, p(x) - q(x)) / (1 - beta)`.

For any token `x`:

```
P(output = x) = P(draft proposed x and it was accepted)
              + P(some draft was rejected) * r(x)
              = q(x) * min(1, p(x)/q(x))  +  (1 - beta) * r(x)
              = min(q(x), p(x))           +  max(0, p(x) - q(x))
              = p(x)
```

The last step: if `p(x) >= q(x)` the terms are `q(x) + (p(x) - q(x)) = p(x)`;
if `p(x) < q(x)` they are `p(x) + 0 = p(x)`. Done.

Note that `x` here is a single token id throughout, and `p`, `q`, `r` are
vectors over the vocabulary. `min(p(x), q(x))` is a min of two scalars, but
`beta` sums that scalar over every token, so `beta` is a property of the two
whole vectors.

The expected number of accepted drafts per round (out of K) is
`(1 - beta^(K+1)) / (1 - beta)` when the per-position acceptance is roughly
constant at `beta`, plus the bonus token. That is the formula the benchmark
curves should approximately follow.

## Transformers 5.x API details that matter here (checked 9 September 2026, v5.17.0)

- `from_pretrained(..., dtype=...)`. The old `torch_dtype=` name is gone.
- KV caches are `DynamicCache` objects, not tuples. Pass one as
  `past_key_values=cache` to `model(...)`; the model appends to it in place.
- `cache.get_seq_length()` gives how many positions are cached.
- `cache.crop(-n)` removes the last `n` positions. Calling it with a positive
  number (meaning "truncate to this absolute length") is deprecated and will be
  removed in 5.18, so use the negative form.
- Qwen3 with `enable_thinking=False` in the chat template appends an empty
  `<think>\n\n</think>\n\n` block to the prompt. That is expected.
