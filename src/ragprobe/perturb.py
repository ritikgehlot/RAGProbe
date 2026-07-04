"""Seeded contamination injection -- the ground-truth generator.

Four perturbation families, each corrupting exactly ONE sentence of a
grounded answer and recording which one (sentence-level ground truth):

* entity_swap      -- replace a capitalized entity with one from another passage
* number_perturb   -- change a numeric value materially
* negation_flip    -- insert or remove a negation after an auxiliary verb
* foreign_sentence -- append a fluent sentence lifted from an unrelated passage

Why these four: they map to real RAG failure modes (wrong entity, wrong
figure, inverted claim, injected unsupported content) AND they stress
scorers differently -- lexical overlap barely changes under negation_flip,
while NLI entailment collapses. That asymmetry is what the meta-evaluation
is designed to expose.
"""
from __future__ import annotations

import copy
import logging
import random
import re

from .dataset import EvalSample, split_sentences

logger = logging.getLogger(__name__)

PERTURBATION_TYPES = ("entity_swap", "number_perturb", "negation_flip",
                      "foreign_sentence")

_ENTITY = re.compile(r"\b(?!(?:The|A|An|In|On|At|It|He|She|They|We|But|And|Or|"
                     r"However|This|That|These|Those|As|By|For|From|With|"
                     r"After|Before|During|Its|His|Her|Their|When|While|Since)\b)"
                     r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")
_NUMBER = re.compile(r"\b\d{1,4}(?:,\d{3})*(?:\.\d+)?\b")
_AUX = re.compile(r"\b(is|was|are|were|has|have|had|can|could|will|would|"
                  r"does|did|do)\b(?!\s+not\b)", re.IGNORECASE)
_AUX_NOT = re.compile(r"\b(is|was|are|were|has|have|had|can|could|will|would|"
                      r"does|did|do)\s+not\b", re.IGNORECASE)


def _entities_in(text: str) -> list[str]:
    # skip sentence-initial word (capitalized by grammar, not entity-hood)
    body = text.split(" ", 1)[1] if " " in text else ""
    return [m.group(1) for m in _ENTITY.finditer(body)]


def perturb_entity_swap(sentence: str, donor_text: str,
                        rng: random.Random) -> str | None:
    targets = _entities_in(sentence)
    donors = [e for e in _entities_in(donor_text) if e not in targets]
    if not targets or not donors:
        return None
    victim = rng.choice(targets)
    replacement = rng.choice(donors)
    out = sentence.replace(victim, replacement, 1)
    return out if out != sentence else None


def perturb_number(sentence: str, rng: random.Random) -> str | None:
    matches = list(_NUMBER.finditer(sentence))
    if not matches:
        return None
    m = rng.choice(matches)
    raw = m.group(0).replace(",", "")
    val = float(raw)
    factor = rng.choice([0.31, 0.47, 2.3, 3.7, 7.5])
    new = val * factor
    new_str = str(int(new)) if raw.isdigit() else f"{new:.1f}"
    if new_str == m.group(0):
        return None
    return sentence[:m.start()] + new_str + sentence[m.end():]


def perturb_negation(sentence: str, rng: random.Random) -> str | None:
    if (m := _AUX_NOT.search(sentence)):        # remove existing negation
        return sentence[:m.start()] + m.group(1) + sentence[m.end():]
    matches = list(_AUX.finditer(sentence))
    if not matches:
        return None
    m = rng.choice(matches)
    return sentence[:m.end()] + " not" + sentence[m.end():]


def perturb_foreign_sentence(donor_text: str, rng: random.Random) -> str | None:
    candidates = [s for s in split_sentences(donor_text)
                  if 8 <= len(s.split()) <= 30]
    return rng.choice(candidates) if candidates else None


def contaminate(samples: list[EvalSample], seed: int) -> list[EvalSample]:
    """Return contaminated twins (deep copies) -- one perturbation each.

    Perturbation types are assigned round-robin over a seeded shuffle so
    the four types are balanced; a sample that can't support its assigned
    type (e.g. no number present) falls through to the next type.
    """
    rng = random.Random(seed)
    donors = samples[:]  # unrelated-passage donors for swaps/injections
    out: list[EvalSample] = []
    for i, src in enumerate(samples):
        order = list(PERTURBATION_TYPES)
        rng.shuffle(order)
        preferred = PERTURBATION_TYPES[i % len(PERTURBATION_TYPES)]
        order.remove(preferred)
        order.insert(0, preferred)

        twin = copy.deepcopy(src)
        twin.sample_id = f"{src.sample_id}-contam"
        twin.is_contaminated = True
        donor = donors[(i + len(donors) // 2) % len(donors)]
        applied = False
        for ptype in order:
            sent_idx = rng.randrange(len(twin.answer_sentences))
            target = twin.answer_sentences[sent_idx]
            if ptype == "entity_swap":
                new = perturb_entity_swap(target, donor.passage, rng)
            elif ptype == "number_perturb":
                new = perturb_number(target, rng)
            elif ptype == "negation_flip":
                new = perturb_negation(target, rng)
            else:  # foreign_sentence appends a NEW sentence
                injected = perturb_foreign_sentence(donor.passage, rng)
                if injected is None:
                    continue
                twin.answer_sentences = twin.answer_sentences + [injected]
                twin.contamination_type = ptype
                twin.corrupted_sentence_idx = len(twin.answer_sentences) - 1
                applied = True
                break
            if new is not None and new != target:
                twin.answer_sentences = (twin.answer_sentences[:sent_idx]
                                         + [new]
                                         + twin.answer_sentences[sent_idx + 1:])
                twin.contamination_type = ptype
                twin.corrupted_sentence_idx = sent_idx
                applied = True
                break
        if applied:
            out.append(twin)
        else:
            logger.debug("no perturbation applicable for %s", src.sample_id)
    counts: dict[str, int] = {}
    for t in out:
        counts[t.contamination_type or "?"] = counts.get(t.contamination_type or "?", 0) + 1
    logger.info("Contaminated %d/%d samples: %s", len(out), len(samples), counts)
    return out
