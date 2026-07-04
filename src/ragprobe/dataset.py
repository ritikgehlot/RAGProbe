"""Build a faithfulness eval set from SQuAD v2 passages.

Design: contamination-by-construction.
Each eval sample starts as a *grounded* answer -- sentences taken from the
source passage around the true answer span, so support is guaranteed by
construction. A seeded perturbation then corrupts a copy of half the
samples. Because we injected the corruption ourselves, ground truth for
"is this answer faithful to the passage?" is provable, and every detection
metric downstream is defensible.

Honest limitation (stated in the README too): constructed answers are
near-verbatim, while real RAG answers paraphrase. Detection here is an
upper bound for extractive-style answers, not a universal hallucination
benchmark.
"""
from __future__ import annotations

import json
import logging
import random
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SQUAD_URL = ("https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/"
             "master/dataset/dev-v2.0.json")

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(])")


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter (no heavy NLP dependency needed)."""
    parts = [s.strip() for s in _SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


@dataclass
class EvalSample:
    """One (passage, question, answer) faithfulness test case."""
    sample_id: str
    question: str
    passage: str
    answer_sentences: list[str]
    is_contaminated: bool
    contamination_type: str | None = None      # set by perturb step
    corrupted_sentence_idx: int | None = None  # ground-truth localization
    meta: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> "EvalSample":
        return cls(**json.loads(line))


def download_squad(dest: Path) -> Path:
    if dest.exists():
        logger.info("SQuAD already at %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading SQuAD v2 dev set ...")
    urllib.request.urlretrieve(SQUAD_URL, dest)  # noqa: S310 (fixed https URL)
    return dest


def _grounded_answer(context: str, answer_text: str, answer_start: int
                     ) -> list[str] | None:
    """Build a 2-3 sentence grounded answer around the true answer span."""
    sents = split_sentences(context)
    # locate the sentence containing the answer span
    pos, idx = 0, None
    for i, s in enumerate(sents):
        start = context.find(s, pos)
        if start <= answer_start < start + len(s):
            idx = i
            break
        pos = start + len(s)
    if idx is None:
        return None
    picked = [sents[idx]]
    if idx + 1 < len(sents):
        picked.append(sents[idx + 1])
    if len(picked[0].split()) < 8 and idx > 0:  # too thin, add left context
        picked.insert(0, sents[idx - 1])
    if not (2 <= len(picked) <= 3):
        return None
    if any(len(s.split()) < 4 for s in picked):
        return None
    return picked


def build_clean_samples(squad_path: Path, n: int, seed: int) -> list[EvalSample]:
    """Sample n grounded (passage, question, answer) triples, seeded."""
    data = json.loads(squad_path.read_text(encoding="utf-8"))["data"]
    pool: list[EvalSample] = []
    for article in data:
        for para in article["paragraphs"]:
            ctx = para["context"]
            if len(ctx.split()) < 60:
                continue
            for qa in para["qas"]:
                if qa.get("is_impossible") or not qa["answers"]:
                    continue
                ans = qa["answers"][0]
                sents = _grounded_answer(ctx, ans["text"], ans["answer_start"])
                if sents is None:
                    continue
                pool.append(EvalSample(
                    sample_id=qa["id"], question=qa["question"], passage=ctx,
                    answer_sentences=sents, is_contaminated=False,
                    meta={"title": article["title"], "span": ans["text"]},
                ))
    rng = random.Random(seed)
    rng.shuffle(pool)
    if len(pool) < n:
        raise RuntimeError(f"only {len(pool)} usable samples, wanted {n}")
    logger.info("Sampled %d clean grounded samples from pool of %d", n, len(pool))
    return pool[:n]


def write_jsonl(samples: list[EvalSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(s.to_json() + "\n")
    logger.info("Wrote %d samples -> %s", len(samples), path)


def read_jsonl(path: Path) -> list[EvalSample]:
    with path.open(encoding="utf-8") as f:
        return [EvalSample.from_json(line) for line in f if line.strip()]
