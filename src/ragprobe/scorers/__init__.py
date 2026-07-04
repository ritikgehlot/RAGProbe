from .base import Scorer, SentenceVerdict, AnswerResult
from .lexical import LexicalScorer

__all__ = ["Scorer", "SentenceVerdict", "AnswerResult", "LexicalScorer",
           "get_scorer"]


def get_scorer(name: str, **kwargs):
    """Factory so the CLI can construct scorers by name; heavy deps lazy."""
    if name == "lexical":
        return LexicalScorer(**kwargs)
    if name == "nli":
        from .nli import NLIScorer
        return NLIScorer(**kwargs)
    if name == "judge":
        from .judge import JudgeScorer
        return JudgeScorer(**kwargs)
    raise ValueError(f"unknown scorer: {name!r} (choose lexical, nli, judge)")
