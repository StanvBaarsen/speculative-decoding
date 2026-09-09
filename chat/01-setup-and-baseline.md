# Chat log 1: setup, algorithm, baseline

Transcript of the working session on 9 September 2026 with Claude Code. Tool calls
and command output are left out; the file contents they produced are in the repo.

---

**Stan:**

I want to learn speculative decoding by implementing it myself from scratch.

I'm on an M5 MacBook Air and want to run small open-weight models locally. I'm a
reasonably technical programmer, but I want to understand the algorithm rather
than hide it behind an inference library.

Please help me build a minimal educational implementation in Python/PyTorch using
Hugging Face Transformers.

Start with:

- Qwen3-0.6B as the draft model
- Qwen3-0.6B as the target model initially, purely so I can debug the algorithm
- Then we can later upgrade to Qwen3-1.7B as the target once the implementation works
- Use GSM8K as a small test set

My goals are:

- Understand exactly what probability distributions the draft and target models produce.
- Implement the speculative decoding algorithm myself, including:
  - generating K draft tokens
  - running the target model over the proposed sequence
  - calculating the acceptance probability
  - accepting/rejecting each proposed token
  - sampling from the residual distribution when a token is rejected
  - correctly handling the sequence after the first rejection
- Verify empirically that my implementation produces the same distribution as
  ordinary sampling from the target model.
- Measure: acceptance rate, number of target-model forward passes, tokens
  generated per target forward pass, wall-clock tokens/sec, speedup vs ordinary
  autoregressive decoding.
- Experiment with K = 1, 2, 4, 8, 16 and see how acceptance rate and speed change.

Important: teach me as we go. Don't just dump a finished implementation on me.

I particularly want to understand the tensor shapes at every step. For example,
explicitly show me that the model produces something like
`(batch, sequence_length, vocabulary_size)` and explain which slice corresponds
to the distribution for each proposed token.

Also explain the subtle part of the algorithm very carefully:
`r(x) ∝ max(0, p(x) - q(x))`. I previously got confused about whether x
represents the whole vocabulary. Make clear that x is one candidate token and p
and q are distributions/vectors over the vocabulary.

Please structure this as a small coding project:

1. Explain the algorithm conceptually.
2. Set up the environment.
3. Implement ordinary autoregressive sampling first.
4. Implement the draft generation.
5. Implement target verification.
6. Implement rejection/residual sampling.
7. Test against ordinary sampling.
8. Benchmark it.
9. Then make it faster/cleaner.

Prefer simple, readable code over production-quality abstractions. Don't use
vLLM, SGLang, or another speculative-decoding implementation to hide the
interesting parts. Whenever there is a choice between giving me code and
explaining what the code is doing, explain first and then give me the smallest
useful code. If something about the exact Qwen3 architecture or Hugging Face
API has changed, check the current documentation rather than assuming. Please
start with making a github repo right away so I can continue this work on
another pc pretty much instantly later.

---

**Claude:**

Repo is live at https://github.com/StanvBaarsen/speculative-decoding and cloned
locally to `~/Developer/speculative-decoding`. On another machine: `git clone`,
`uv sync`, `uv run python 00_inspect_logits.py`. Models download on first use.

Here is lesson one. It covers steps 1 to 4 of your list: the concept, the
environment, what the model outputs, and the autoregressive baseline. The
speculative parts come next, once you have read this.

### The algorithm, and the part that confused you

Generating one token costs one target forward pass. On your M5 that pass is
memory-bound: the time goes into streaming the weights, not the arithmetic. So
checking 8 tokens in one pass costs about the same as checking 1. Speculative
decoding has a cheap draft model guess K tokens, then the target checks all K in
one pass, and a rejection rule makes the output exactly distributed as target
sampling.

The notation, fixed once and for all:

- `p` and `q` are **vectors of length vocab_size** (151,936 for Qwen3). `p` is
  the target's next-token distribution, `q` the draft's, both for the same prefix.
- `x` is **one token id**, an integer. `p(x)` means `p[x]`, a scalar.
- Accept draft token `x` with probability `min(1, p[x] / q[x])`. Scalar divided
  by scalar.
- On rejection, build the residual **vector** `r = clamp(p - q, min=0)`,
  normalise it, and sample the replacement from `r`. Here "x ranges over the
  vocabulary" only in the sense that `r` has one entry per token. The
  `max(0, ·)` zeroes every token the draft already over-covers, so the
  replacement is drawn only from tokens the target wanted more than the draft did.

The four-line proof that this sums to exactly `p` is in NOTES.md. It is worth
reading once. After a rejection you throw away all later drafts too, because
they were conditioned on the token you just rejected. If all K are accepted you
get a free bonus token from the target's extra distribution. So each round
yields between 1 and K+1 tokens.

### What the model actually outputs

Script `00_inspect_logits.py` ran one forward pass on a 20-token chat prompt.
The shapes:

```
input_ids  (1, 20)            (batch, prompt_len)
logits     (1, 20, 151936)    (batch, prompt_len, vocab), float32
probs      (1, 20, 151936)    softmax over the last axis, every row sums to 1
```

The crucial fact: **the logits at position t are the prediction for token t+1.**
The script prints a table showing the model's top guess at each position against
the real next token. At position 3 (`'What'`) the top guess is `' is'`, which is
indeed the token at position 4. The distribution you sample the first new token
from is `probs[0, -1]`, the last row, shape `(151936,)`. For this prompt it puts
0.99 on `'7'`.

This off-by-one is exactly what step 6 (verification) hinges on. When the target
processes `prefix + [x_1..x_K]`, the row at the last prefix position is `p_1`,
the row at `x_1` is `p_2`, and so on. The row at `x_K` is the bonus `p_{K+1}`.

One thing to notice in the table: Qwen3's chat template with thinking disabled
appends an empty `<think>\n\n</think>\n\n` block to the prompt. That is normal,
and it keeps GSM8K answers short.

### The baseline

Script `01_autoregressive.py` is the hand-written sampling loop. Read
`generate_autoregressive` in that file; it is 20 lines. The non-obvious piece is
the KV cache. After the prefill pass over the whole prompt, each step feeds
**one** token, and the logits per step are `(1, 1, vocab)`. The cache grows by
one each step, which the script prints. Result on the Natalia GSM8K question
with the 0.6B model in float32:

| tokens | forward passes | tok/s |
|---|---|---|
| 98 | 98 | 29.3 |

One pass per token, by construction. That ratio is what speculative decoding
attacks.

Two decisions I made, both easy to change:

- **float32** while debugging, so `p` and `q` are as exact as possible. When
  draft equals target we want to see near-100% acceptance. We switch to bfloat16
  for the benchmarks.
- **Temperature only, no top-k or top-p.** The theory holds for any `p` and `q`
  as long as they are the distributions you actually sample from. Adding
  truncation later means applying it to both vectors before the acceptance test.

Two API facts I verified against transformers 5.17 rather than assuming: the
load argument is `dtype=` (the old name is gone), and `cache.crop(-n)` removes
the last n positions. The positive form is deprecated in 5.18. Both are in
NOTES.md.

### Next

Step 5 is `02_draft.py`: the draft loop that proposes K tokens and keeps the K
vectors `q_i`, shape `(K, vocab)`. Before I write it, a check for you: after the
draft has produced K tokens with a KV cache, how many positions does its cache
hold, and how many of those may turn out to be garbage? Run the two scripts,
read NOTES.md, and tell me when to continue.

---

**Stan:**

where did you do all this? and what do you mean with these questions? "Before I
write it, a check for you: after the draft has produced K tokens with a KV
cache, how many positions does its cache hold, and how many of those may turn
out to be garbage?"

---

**Claude:**

All the files are in one folder on this Mac: `~/Developer/speculative-decoding`.
It is a git repo pushed to https://github.com/StanvBaarsen/speculative-decoding.
Open it with `cd ~/Developer/speculative-decoding` and run
`uv run python 00_inspect_logits.py`. The folder holds README.md (the algorithm
explanation), NOTES.md (the proof and the API notes), common.py (shared
helpers), and the two numbered scripts.

The question was a comprehension check before the next step, not a task. Let me
answer it, since it is the setup for what we build next.

Say the prefix is 50 tokens and K is 4. The draft model runs 4 forward passes
with a KV cache. Its cache then holds 54 positions: the 50 prefix tokens plus
the 4 drafts it proposed. Now suppose the target rejects draft 2. Only draft 1
survives. Drafts 2, 3 and 4 are gone, and the replacement token at position 2
comes from the residual distribution, not from the draft. So of the 54 cached
positions, the last 3 describe tokens that are no longer in the sequence. Before
the next round we must crop the draft cache back to 51, then feed it the
replacement token so it catches up. The same applies to the target cache, which
after verification holds 54 positions too, and must also be cut back to 51.

Forgetting this crop is the classic bug: the model keeps attending to phantom
tokens and generates fluent-looking garbage. That is why I asked. In the worst
case, when draft 1 is rejected, all K cached draft positions are garbage.

Say when you want me to continue with step 5, the draft loop.
