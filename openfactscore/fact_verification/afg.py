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
            text = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip()
            if not text or text.lower().startswith(("atomic facts:", "answer:")):
                continue
            if text not in seen:
                facts.append(AtomicFact(text=text))
                seen.add(text)
        return facts

    def _generate(self, prompt: str) -> str:
        target = self.generator.generate if hasattr(self.generator, "generate") else self.generator
        output = target(prompt)
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, str):
            raise TypeError("AFG generator must return a string or a tuple whose first item is a string")
        return output
