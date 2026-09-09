# Speculative decoding from scratch

A minimal, educational implementation of speculative decoding in plain PyTorch and
Hugging Face Transformers. No vLLM, no `assistant_model=`, no library hiding the
interesting parts. Draft model: Qwen3-0.6B. Target model: Qwen3-0.6B while
debugging (draft == target, so every token should be accepted), then Qwen3-1.7B.
Test prompts come from GSM8K.

## Setup

```sh
uv sync                                # installs torch, transformers, datasets into .venv
uv run python 00_inspect_logits.py     # step 0: what does a model actually output?
```

Models download from the Hugging Face Hub on first use (about 1.5 GB for 0.6B,
3.4 GB for 1.7B in bf16) into `~/.cache/huggingface`.

## Plan

Each step is one small script. Later steps import from earlier ones.

- [x] 1. Explain the algorithm (this README)
- [x] 2. Environment (uv, torch with MPS, transformers 5.x)
- [x] 3. `00_inspect_logits.py`: look at the tensor the model returns and the distribution slices
- [x] 4. `01_autoregressive.py`: ordinary sampling loop with a KV cache, as the baseline
- [ ] 5. `02_draft.py`: draft model proposes K tokens, keeps q for each
- [ ] 6. `03_verify.py`: target runs once over prompt + K drafts, gives p for each position
- [ ] 7. `04_speculative.py`: accept/reject, residual sampling, sequence handling after rejection
- [ ] 8. `05_test_distribution.py`: empirical check that output distribution matches plain target sampling
- [ ] 9. `06_benchmark.py`: acceptance rate, target forward passes, tokens per target pass, tokens/s, speedup, sweep K in {1, 2, 4, 8, 16}
- [ ] 10. Make it faster and cleaner

## The algorithm in one page

### Why it can work at all

Generating one token with a big model costs one forward pass. On a laptop that
pass is dominated by moving the weights through memory, not by arithmetic. So a
forward pass over 1 new token and a forward pass over 8 new tokens cost almost the
same. Speculative decoding exploits that: a cheap draft model guesses several
tokens, and the expensive target model checks all of them in a single pass.

The clever part is that the result is not an approximation. The tokens that come
out are distributed exactly as if you had sampled them one at a time from the
target model. Speed changes, the distribution does not.

### Notation

Fix a prefix (the prompt plus everything generated so far).

- `q` is the draft model's next-token distribution given the prefix. It is a
  vector of length `vocab_size` (151,936 for Qwen3) that sums to 1.
- `p` is the target model's next-token distribution given the same prefix. Also a
  vector of length `vocab_size`.
- `x` is one token id, an integer in `[0, vocab_size)`. `q(x)` and `p(x)` are the
  scalars `q[x]` and `p[x]`.

That last line is the thing to keep straight: `p` and `q` are whole vectors,
`x` is one index into them. When the paper writes `p(x)` it means "the entry of
the vector `p` at position `x`".

### One round

1. **Draft.** Starting from the prefix, sample `K` tokens autoregressively from
   the draft model: `x_1 ~ q_1`, `x_2 ~ q_2`, ..., `x_K ~ q_K`. Each `q_i` is the
   draft distribution given the prefix plus `x_1..x_{i-1}`. Keep all `K` vectors
   `q_i`. Cost: `K` small forward passes.

2. **Verify.** Run the target model **once** on `prefix + [x_1, ..., x_K]`. The
   output logits have shape `(1, prefix_len + K, vocab_size)`. The last `K + 1`
   positions give us `K + 1` distributions:
   - the logits at the position of the last prefix token give `p_1`, the target's
     distribution for what should come *after* the prefix, i.e. the position where
     the draft put `x_1`;
   - the logits at the position of `x_1` give `p_2`, the distribution for the
     position where the draft put `x_2`;
   - ...
   - the logits at the position of `x_K` give `p_{K+1}`, the distribution for one
     token *beyond* the drafts. We get this for free.

   Remember that a causal LM's logits at position `t` are its prediction for
   token `t + 1`. That "off by one" is the source of most bugs here, and
   `00_inspect_logits.py` shows it concretely.

3. **Accept or reject, left to right.** For `i = 1..K`:
   - Draw `u ~ Uniform(0, 1)`.
   - Accept `x_i` if `u < min(1, p_i(x_i) / q_i(x_i))`. Both numerator and
     denominator are scalars: the probability each model assigned to the specific
     token the draft picked.
   - If accepted, move on to `i + 1`.
   - If rejected, stop. Do not look at `x_{i+1}, ..., x_K`; they were sampled
     conditioned on `x_i`, which we just threw out, so they are meaningless now.

4. **Fill in the rejected position.** If `x_i` was rejected, we still need a token
   at position `i`. We sample it from the *residual* distribution

   ```
   r_i(x) = max(0, p_i(x) - q_i(x)) / sum_x' max(0, p_i(x') - q_i(x'))
   ```

   Here `x` ranges over the whole vocabulary: `r_i` is a new vector of length
   `vocab_size`. In code it is one line on vectors:
   `r = torch.clamp(p - q, min=0); r = r / r.sum()`. Then `x_i ~ r_i`.

   Why this exact vector? Intuitively: the draft over-proposed tokens where
   `q > p` (that is where rejections come from), so when we reject we should
   compensate by sampling only tokens the target wanted *more* than the draft did.
   The `max(0, ...)` zeroes out every token the draft already over-covers. The
   proof that acceptance + residual sampling gives exactly `p` is four lines and
   lives in [`NOTES.md`](NOTES.md).

5. **Bonus token.** If all `K` drafts were accepted, sample one extra token from
   `p_{K+1}`, the free distribution from step 2. So one round produces between
   1 token (first draft rejected, residual sample) and `K + 1` tokens (all
   accepted plus bonus). Never zero, so the loop always makes progress.

6. **Fix the caches.** Both models have KV caches that now contain entries for
   tokens that were rejected. Crop them back to the accepted length before the
   next round.

### What we measure

- **Acceptance rate**: accepted drafts / proposed drafts. Depends on how well `q`
  matches `p`, and on the temperature.
- **Target forward passes**: one per round. The whole point is to make this
  smaller than the number of tokens generated.
- **Tokens per target pass**: `tokens generated / target passes`. Upper bound
  `K + 1`.
- **Wall-clock tokens/s** and **speedup** over plain autoregressive sampling of
  the target. Draft passes are not free, so the best `K` is a trade-off:
  larger `K` means more tokens per target pass but more wasted draft work after
  the first rejection.

### Simplifications in this repo

- Batch size 1.
- Sampling with temperature only, no top-k or top-p. Those can be added later
  by applying them to both `p` and `q` before the acceptance test; the theory
  still holds as long as `p` and `q` are the distributions actually sampled from.
- Draft and target share a tokenizer (all Qwen3 sizes do), so token ids line up.

## Chat logs

The `chat/` folder holds transcripts of the teaching sessions that produced
this code, one file per session, so the reasoning behind each step is kept
with the code.

## References

- Leviathan, Kalman, Matias. *Fast Inference from Transformers via Speculative
  Decoding.* ICML 2023. https://arxiv.org/abs/2211.17192
- Chen et al. *Accelerating Large Language Model Decoding with Speculative
  Sampling.* 2023. https://arxiv.org/abs/2302.01318
