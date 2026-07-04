"""Run scorers over an eval set and META-EVALUATE them.

The meta-evaluation is the honest core of RAGProbe: because contamination
was injected by us, we can measure each scorer's actual detection quality --
precision, recall, F1 at answer level, false-alarm rate on clean answers,
per-contamination-type recall, and sentence-level localization accuracy.
A faithfulness scorer that has never been meta-evaluated is itself an
unverified claim.
"""
from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

from .dataset import EvalSample
from .perturb import PERTURBATION_TYPES
from .scorers.base import AnswerResult, Scorer

logger = logging.getLogger(__name__)


@dataclass
class ScorerReport:
    scorer: str
    threshold: float
    n_clean: int
    n_contaminated: int
    true_positives: int
    false_negatives: int
    false_positives: int
    true_negatives: int
    precision: float
    recall: float
    f1: float
    false_alarm_rate: float
    recall_by_type: dict[str, float]
    localization_accuracy: float   # among detected: flagged the right sentence?
    seconds: float
    results: list[AnswerResult] = field(default_factory=list)

    def summary_dict(self) -> dict:
        d = asdict(self)
        d.pop("results")
        return d


def _safe_div(a: float, b: float) -> float:
    return round(a / b, 4) if b else 0.0


def run_scorer(scorer: Scorer, samples: list[EvalSample]) -> ScorerReport:
    t0 = time.time()
    results = [scorer.evaluate(s.sample_id, s.passage, s.answer_sentences)
               for s in samples]
    seconds = round(time.time() - t0, 2)

    by_id = {r.sample_id: r for r in results}
    tp = fn = fp = tn = 0
    type_hits: dict[str, list[int]] = {t: [] for t in PERTURBATION_TYPES}
    loc_hits, loc_total = 0, 0
    for s in samples:
        r = by_id[s.sample_id]
        if s.is_contaminated:
            hit = int(r.flagged)
            tp += hit
            fn += 1 - hit
            if s.contamination_type in type_hits:
                type_hits[s.contamination_type].append(hit)
            if hit:
                loc_total += 1
                loc_hits += int(s.corrupted_sentence_idx in r.flagged_indices)
        else:
            fp += int(r.flagged)
            tn += int(not r.flagged)

    report = ScorerReport(
        scorer=scorer.name, threshold=scorer.threshold,
        n_clean=fp + tn, n_contaminated=tp + fn,
        true_positives=tp, false_negatives=fn,
        false_positives=fp, true_negatives=tn,
        precision=_safe_div(tp, tp + fp),
        recall=_safe_div(tp, tp + fn),
        f1=_safe_div(2 * tp, 2 * tp + fp + fn),
        false_alarm_rate=_safe_div(fp, fp + tn),
        recall_by_type={t: _safe_div(sum(h), len(h))
                        for t, h in type_hits.items() if h},
        localization_accuracy=_safe_div(loc_hits, loc_total),
        seconds=seconds, results=results,
    )
    logger.info("%s: P=%.3f R=%.3f F1=%.3f FAR=%.3f loc=%.3f (%.1fs)",
                scorer.name, report.precision, report.recall, report.f1,
                report.false_alarm_rate, report.localization_accuracy, seconds)
    return report
