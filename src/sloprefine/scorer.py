"""Optional language-model backend for perplexity-based signals.

Everything else in sloprefine is model-free on purpose. This module is the
one place a language model is allowed, it is an optional extra, and it is
off unless asked for.

What the literature does here
-----------------------------
  [MIT23] Mitchell et al., DetectGPT (ICML 2023). Machine text sits in a
          region of negative log-probability curvature: perturb it and the
          likelihood drops more than it does for human text. Needs many
          perturbations, so it is slow.
  [BAO24] Bao et al., Fast-DetectGPT (ICLR 2024). Same idea via conditional
          probability curvature, one forward pass instead of hundreds.
  [HAN24] Hans et al., Binoculars (ICML 2024). Ratio of perplexity under an
          observer model to cross-perplexity under a performer model. Zero
          shot, no training data from the target model, and it defuses the
          "Capybara problem" where an unusual prompt yields high-perplexity
          output that raw-perplexity detectors misread. Reported >90% TPR at
          0.01% FPR.

What this module does NOT do
----------------------------
It returns numbers, never a verdict. sloprefine has no "AI-written" output and
no threshold that would produce one. Binoculars' own authors ship a fixed
global threshold and still caution against unsupervised use; the same caution,
applied honestly, means a writing tool should surface the signal and stop.

The scorer is a protocol so the heavy dependency stays optional and so the
pipeline is testable without downloading a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    """Anything that can return mean token log-probability for a string."""

    name: str

    def logprobs(self, text: str) -> list[float]:
        """Per-token log probabilities under the model."""
        ...


@dataclass
class PerplexitySignal:
    scorer: str
    mean_logprob: float
    perplexity: float
    token_count: int
    window_perplexities: list[float]
    window_cv: float

    def as_dict(self) -> dict:
        return {
            "scorer": self.scorer,
            "mean_logprob": round(self.mean_logprob, 4),
            "perplexity": round(self.perplexity, 3),
            "token_count": self.token_count,
            "window_cv": round(self.window_cv, 4),
        }


def measure(scorer: Scorer, text: str, window: int = 32) -> PerplexitySignal:
    """Mean log-probability, perplexity, and how much perplexity varies across
    windows. The variation matters more than the level: uniform predictability
    across a whole document is the flat-rhythm finding again, one layer down.
    """
    import math
    import statistics

    lps = scorer.logprobs(text)
    if not lps:
        raise ValueError("scorer returned no token log-probabilities")
    mean_lp = statistics.mean(lps)

    windows = [lps[i:i + window] for i in range(0, len(lps), window)]
    windows = [w for w in windows if len(w) >= max(4, window // 4)]
    ppls = [math.exp(-statistics.mean(w)) for w in windows] or [math.exp(-mean_lp)]
    mean_ppl = statistics.mean(ppls)
    cv = (statistics.pstdev(ppls) / mean_ppl) if len(ppls) > 1 and mean_ppl else 0.0

    return PerplexitySignal(
        scorer=getattr(scorer, "name", scorer.__class__.__name__),
        mean_logprob=mean_lp,
        perplexity=math.exp(-mean_lp),
        token_count=len(lps),
        window_perplexities=[round(p, 2) for p in ppls],
        window_cv=cv,
    )


class TransformersScorer:
    """Causal-LM scorer. Requires the optional 'lm' extra.

        pip install "sloprefine[lm]"
        sloprefine check draft.md --lm gpt2

    Small models are fine here. We report perplexity structure, not a verdict,
    so the observer/performer pair that [HAN24] needs is not required. If you
    want Binoculars proper, use their implementation; this is a signal, not a
    reimplementation.
    """

    def __init__(self, model_name: str = "gpt2", device: str = "cpu"):
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional path
            raise ImportError(
                "the --lm option needs the optional extra: "
                'pip install "sloprefine[lm]"'
            ) from exc
        self._torch = torch
        self.name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

    def logprobs(self, text: str) -> list[float]:  # pragma: no cover - optional
        torch = self._torch
        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=1024).input_ids.to(self.device)
        if ids.shape[1] < 2:
            return []
        with torch.no_grad():
            logits = self.model(ids).logits
        log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
        targets = ids[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return gathered[0].tolist()
