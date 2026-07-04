"""LLM-judge scorer via the Anthropic API (optional -- needs
ANTHROPIC_API_KEY in the environment).

The judge sees the passage and the numbered answer sentences and must
return STRICT JSON verdicts per sentence. Structured output + a low-drama
rubric keeps it comparable to the other scorers. Every call logs token
usage; failures raise clearly instead of silently scoring 0.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request

from .base import Scorer

logger = logging.getLogger(__name__)

API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_MODEL = "claude-sonnet-4-6"

PROMPT = """You are verifying whether each sentence of an answer is fully \
supported by the source passage.

<passage>
{passage}
</passage>

<answer_sentences>
{numbered}
</answer_sentences>

For each sentence decide: "supported" (every claim in it is stated in the \
passage) or "unsupported" (any entity, number, negation, or claim differs \
from or is absent from the passage).

Respond with ONLY this JSON, no other text:
{{"verdicts": [{{"index": 0, "verdict": "supported"}}, ...]}}"""


class JudgeScorer(Scorer):
    name = "judge"

    def __init__(self, threshold: float = 0.5,
                 model: str = DEFAULT_MODEL, max_retries: int = 3,
                 api_key: str | None = None) -> None:
        self.threshold = threshold
        self.model = model
        self.max_retries = max_retries
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "JudgeScorer needs ANTHROPIC_API_KEY set in the environment.")
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def _call_api(self, prompt: str) -> str:
        body = json.dumps({
            "model": self.model, "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(API_URL, data=body, headers={
            "content-type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
        })
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read())
                usage = data.get("usage", {})
                self.total_input_tokens += usage.get("input_tokens", 0)
                self.total_output_tokens += usage.get("output_tokens", 0)
                return "".join(b.get("text", "") for b in data["content"]
                               if b.get("type") == "text")
            except Exception as e:  # noqa: BLE001 -- retry then surface
                last_err = e
                wait = 2 ** attempt
                logger.warning("judge API attempt %d failed (%s); retry in %ds",
                               attempt + 1, e, wait)
                time.sleep(wait)
        raise RuntimeError(f"judge API failed after retries: {last_err}")

    @staticmethod
    def parse_verdicts(raw: str, n_sentences: int) -> list[float]:
        """Parse strict-JSON verdicts; tolerate accidental code fences."""
        cleaned = raw.strip().removeprefix("```json").removeprefix("```")
        cleaned = cleaned.removesuffix("```").strip()
        payload = json.loads(cleaned)
        scores = [0.5] * n_sentences
        for v in payload.get("verdicts", []):
            i = int(v["index"])
            if 0 <= i < n_sentences:
                scores[i] = 1.0 if str(v["verdict"]).lower() == "supported" else 0.0
        return scores

    def score_sentences(self, passage: str,
                        sentences: list[str]) -> list[float]:
        numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences))
        raw = self._call_api(PROMPT.format(passage=passage, numbered=numbered))
        return self.parse_verdicts(raw, len(sentences))
