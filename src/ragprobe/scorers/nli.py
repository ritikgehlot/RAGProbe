"""NLI entailment scorer -- the semantic upgrade over lexical overlap.

For each answer sentence (hypothesis), the premise is the top-k most
lexically similar passage sentences (keeps within the 512-token limit of
the cross-encoder). Support = P(entailment). Negation flips that fool
token overlap collapse entailment probability -- this is the scorer's
whole reason to exist.

Model downloads from Hugging Face on first use (~140 MB for the default
deberta-v3-small cross-encoder); runs comfortably on an 8 GB GPU or CPU.
"""
from __future__ import annotations

import logging

from ..dataset import split_sentences
from .base import Scorer
from .lexical import _tokens

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "roberta-large-mnli"


class NLIScorer(Scorer):
    name = "nli"

    def __init__(self, threshold: float = 0.5,
                 model_name: str = DEFAULT_MODEL,
                 top_k_premise: int = 3, device: str | None = None) -> None:
        self.threshold = threshold
        self.model_name = model_name
        self.top_k = top_k_premise
        self._device = device
        self._model = None  # lazy: importing torch/transformers is slow

    def _load(self):
        if self._model is None:
            import torch
            from transformers import (AutoModelForSequenceClassification,
                                      AutoTokenizer)
            device = self._device or ("cuda" if torch.cuda.is_available()
                                      else "cpu")
            logger.info("Loading %s on %s ...", self.model_name, device)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name).to(device).eval()
            self._devname = device
            # label order differs across NLI checkpoints -- resolve it
            id2label = {i: l.lower() for i, l in
                        self._model.config.id2label.items()}
            self._entail_idx = next(i for i, l in id2label.items()
                                    if "entail" in l)
        return self._model

    def _premise_for(self, passage: str, sentence: str) -> str:
        sents = split_sentences(passage)
        stoks = {t for t, _ in _tokens(sentence)}
        ranked = sorted(sents, key=lambda p: -len(
            stoks & {t for t, _ in _tokens(p)}))
        return " ".join(ranked[:self.top_k])

    def score_sentences(self, passage: str,
                        sentences: list[str]) -> list[float]:
        import torch
        model = self._load()
        pairs = [(self._premise_for(passage, s), s) for s in sentences]
        enc = self._tok([p for p, _ in pairs], [h for _, h in pairs],
                        padding=True, truncation=True, max_length=512,
                        return_tensors="pt").to(self._devname)
        with torch.no_grad():
            probs = torch.softmax(model(**enc).logits, dim=-1)
        return [float(p[self._entail_idx]) for p in probs.cpu()]
