"""Atomic Fact Generation (AFG)."""

import re
from typing import Callable, List, Union

from .schemas import AtomicFact, TextGenerator


Generator = Union[TextGenerator, Callable[[str], Union[str, tuple]]]


class AtomicFactGenerator:
    """Extract independently verifiable claims from an answer."""

    SYSTEM_INSTRUCTION = (
        "Break the answer into independent atomic facts. Each fact must contain "
        "exactly one verifiable claim, preserve the original meaning, and add no "
        "new information. Return one fact per line, prefixed with '-'."
    )

    def __init__(self, generator: Generator):
        self.generator = generator

    def extract(self, answer: str) -> List[AtomicFact]:
        if not answer or not answer.strip():
            return []
        prompt = f"{self.SYSTEM_INSTRUCTION}\n\nAnswer:\n{answer.strip()}\n\nAtomic facts:"
        output = self._generate(prompt)
        facts = []
        seen = set()
        for line in output.splitlines():
            match = re.match(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)(.+?)\s*$", line)
            if not match:
                continue
            text = match.group(1).strip()
            normalized = re.sub(r"\s+", " ", text).casefold()
            if not normalized or normalized in seen:
                continue
            facts.append(AtomicFact(text=text, fact_id=f"fact-{len(facts) + 1:04d}"))
            seen.add(normalized)
        return facts

    def _generate(self, prompt: str) -> str:
        target = self.generator.generate if hasattr(self.generator, "generate") else self.generator
        output = target(prompt)
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, str):
            raise TypeError("AFG generator must return a string or a tuple whose first item is a string")
        return output
