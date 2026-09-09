"""Shared helpers: device, model loading, prompts, sampling."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DRAFT = "Qwen/Qwen3-0.6B"
TARGET_SMALL = "Qwen/Qwen3-0.6B"   # for debugging: draft == target
TARGET = "Qwen/Qwen3-1.7B"


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load(name: str, device: torch.device, dtype=torch.float32):
    """Load a causal LM in eval mode. float32 by default: while we are checking
    correctness we want the two models' probabilities to be as exact as
    possible. Switch to bfloat16 later for speed."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype).to(device).eval()
    return tok, model


def chat_prompt(tok, question: str) -> torch.Tensor:
    """Wrap a question in Qwen3's chat template, thinking disabled (so answers
    stay short), and return input_ids of shape (1, prompt_len)."""
    messages = [{"role": "user", "content": question}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok(text, return_tensors="pt").input_ids


def gsm8k_questions(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [ds[i]["question"] for i in range(n)]


def probs_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """logits: (..., vocab) -> probabilities (..., vocab). Softmax in float32
    so that tiny probabilities are not flushed to zero."""
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample(probs: torch.Tensor) -> torch.Tensor:
    """probs: (vocab,) -> one token id as a 0-d tensor."""
    return torch.multinomial(probs, num_samples=1)[0]
