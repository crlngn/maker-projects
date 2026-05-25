"""Decider — turns the user's natural-language observation into a yes/no
question and parses the model's reply into a structured Decision.

Prompt template lives here (vs spread across callers) so we can change it
in one place. See PROMPTS.md for the design notes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .vision_client import VisionClient


PROMPT_TEMPLATE = (
    "Look at this image. {question} "
    "Respond with exactly one word ('yes' or 'no'), then a colon, then one "
    "short sentence describing what you see that justifies your answer. "
    "Example: 'yes: I see a person sitting in an office chair facing a laptop.'"
)


@dataclass
class Decision:
    """One frame's worth of model output, normalized."""
    condition: bool        # what the trigger evaluator consumes
    raw_verdict: str       # the model's "yes"/"no" before negation
    reason: str
    raw_response: str
    latency_sec: float
    parsed: bool           # False if we couldn't extract a yes/no


class Decider:
    """Wraps a VisionClient with the prompt template + response parsing.

    The ``negate`` flag is what lets us write a positively-phrased question
    ("Is the bottle on the desk?") and still trigger on the negative
    ("...fire when it is *not* on the desk", i.e. user is drinking).
    """

    def __init__(
        self,
        client: VisionClient,
        question: str,
        *,
        negate: bool = False,
    ):
        self.client = client
        self.question = question.strip()
        self.negate = negate

    @property
    def prompt(self) -> str:
        return PROMPT_TEMPLATE.format(question=self.question)

    def decide(self, frame_bgr: np.ndarray) -> Decision:
        result = self.client.query(frame_bgr=frame_bgr, prompt=self.prompt)
        verdict, reason, parsed = _parse(result.raw_response)
        condition = (verdict == "yes")
        if self.negate:
            condition = not condition
        return Decision(
            condition=condition,
            raw_verdict=verdict,
            reason=reason,
            raw_response=result.raw_response,
            latency_sec=result.latency_sec,
            parsed=parsed,
        )


def _parse(response: str) -> tuple[str, str, bool]:
    """Extract (verdict, reason, parsed_ok) from a 'yes: ...' / 'no: ...' reply.

    Tolerates leading whitespace and extra prose before the verdict.
    """
    s = response.strip()
    if not s:
        return ("", "", False)
    # First non-empty word, normalized
    first = s.split(":", 1)[0].strip().lower()
    # Strip common prefixes the model sometimes emits ("Answer: yes")
    for prefix in ("answer", "verdict", "response"):
        if first.startswith(prefix):
            first = first[len(prefix):].lstrip(" :").lower()
    if first.startswith("yes"):
        verdict = "yes"
    elif first.startswith("no"):
        verdict = "no"
    else:
        return ("", s, False)
    reason = s.split(":", 1)[1].strip() if ":" in s else ""
    return (verdict, reason, True)
