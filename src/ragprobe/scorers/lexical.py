"""Lexical support scorer -- the honest cheap baseline.

Support(sentence) = best content-token coverage against any window of the
passage. Numbers and capitalized entities weigh triple: swapping "Tesla"
for "Nokia" or "1912" for "4402" should crater the score even in a long
sentence. Known blind spot BY DESIGN: negation ("did not win" vs "did
win") barely moves token overlap -- the meta-evaluation exists to expose
exactly this kind of failure.
"""
from __future__ import annotations

import re

from ..dataset import split_sentences
from .base import Scorer

_TOKEN = re.compile(r"[A-Za-z]+|\d[\d,.]*")
_STOP = frozenset("""a an the of in on at to for from by with as is was are
were be been being and or but not no this that these those it its his her
their which who whom whose when where has have had does did do can could
will would""".split())


def _tokens(text: str) -> list[tuple[str, float]]:
    out = []
    for m in _TOKEN.finditer(text):
        tok = m.group(0)
        low = tok.lower().strip(",.")
        if low in _STOP or len(low) < 2:
            continue
        weight = 3.0 if (tok[0].isupper() or any(c.isdigit() for c in tok)) else 1.0
        out.append((low, weight))
    return out


class LexicalScorer(Scorer):
    name = "lexical"

    def __init__(self, threshold: float = 0.62) -> None:
        self.threshold = threshold

    def score_sentences(self, passage: str,
                        sentences: list[str]) -> list[float]:
        pass_sents = split_sentences(passage)
        # windows of 1-2 consecutive passage sentences as support candidates
        windows = pass_sents + [" ".join(pass_sents[i:i + 2])
                                for i in range(len(pass_sents) - 1)]
        window_sets = [{t for t, _ in _tokens(w)} for w in windows]
        scores = []
        for sent in sentences:
            toks = _tokens(sent)
            if not toks:
                scores.append(1.0)
                continue
            total = sum(w for _, w in toks)
            best = max((sum(w for t, w in toks if t in ws) / total
                        for ws in window_sets), default=0.0)
            scores.append(best)
        return scores
