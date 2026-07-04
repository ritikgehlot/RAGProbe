"""Scorer interface: every scorer maps (passage, answer sentences)
-> per-sentence support scores in [0,1], plus a threshold that turns
scores into flagged sentences. An answer is flagged unfaithful if ANY
sentence falls below the scorer's threshold."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class SentenceVerdict:
    sentence: str
    support: float          # 1.0 = fully supported by the passage
    flagged: bool


@dataclass
class AnswerResult:
    sample_id: str
    scorer: str
    verdicts: list[SentenceVerdict] = field(default_factory=list)

    @property
    def flagged(self) -> bool:
        return any(v.flagged for v in self.verdicts)

    @property
    def flagged_indices(self) -> list[int]:
        return [i for i, v in enumerate(self.verdicts) if v.flagged]


class Scorer(ABC):
    name: str = "base"
    threshold: float = 0.5

    @abstractmethod
    def score_sentences(self, passage: str,
                        sentences: list[str]) -> list[float]:
        """Return support in [0,1] for each answer sentence."""

    def evaluate(self, sample_id: str, passage: str,
                 sentences: list[str]) -> AnswerResult:
        scores = self.score_sentences(passage, sentences)
        return AnswerResult(
            sample_id=sample_id, scorer=self.name,
            verdicts=[SentenceVerdict(s, round(sc, 4), sc < self.threshold)
                      for s, sc in zip(sentences, scores)],
        )
