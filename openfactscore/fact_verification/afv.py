"""Atomic Fact Validation (AFV)."""

import re
from typing import Callable, List, Optional, Union

from .schemas import AtomicFact, EvidencePassage, FactValidation, TextGenerator


Generator = Union[TextGenerator, Callable[[str], Union[str, tuple]]]


class AtomicFactValidator:
    """Decide whether supplied evidence supports each atomic fact."""

    def __init__(self, generator: Generator):
        self.generator = generator

    def verify(
        self,
        fact: AtomicFact,
        evidence: List[EvidencePassage],
    ) -> FactValidation:
        if not evidence:
            return FactValidation(fact=fact, supported=False, evidence=[])

        context = "\n\n".join(
            f"[{index}] Title: {passage.title or 'Untitled'}\n{passage.text}"
            for index, passage in enumerate(evidence, start=1)
        )
        prompt = (
            "Determine whether the context supports the atomic fact. Use only the "
            "context. Reply exactly with 'SUPPORTED' or 'UNSUPPORTED'.\n\n"
            f"Context:\n{context}\n\nAtomic fact: {fact.text}\nVerdict:"
        )
        raw_output = self._generate(prompt).strip()
        supported = self._parse_verdict(raw_output)
        return FactValidation(
            fact=fact,
            supported=supported,
            evidence=evidence,
            confidence=self._parse_confidence(raw_output),
            raw_output=raw_output,
        )

    def verify_batch(
        self,
        facts: List[AtomicFact],
        evidence_by_fact: List[List[EvidencePassage]],
    ) -> List[FactValidation]:
        if len(facts) != len(evidence_by_fact):
            raise ValueError("facts and evidence_by_fact must have equal lengths")
        return [self.verify(fact, evidence) for fact, evidence in zip(facts, evidence_by_fact)]

    def _generate(self, prompt: str) -> str:
        target = self.generator.generate if hasattr(self.generator, "generate") else self.generator
        output = target(prompt)
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, str):
            raise TypeError("AFV generator must return a string or a tuple whose first item is a string")
        return output

    @staticmethod
    def _parse_verdict(output: str) -> bool:
        normalized = output.upper()
        unsupported = re.search(r"\b(?:UNSUPPORTED|FALSE|NOT SUPPORTED)\b", normalized)
        supported = re.search(r"\b(?:SUPPORTED|TRUE)\b", normalized)
        if unsupported:
            return False
        if supported:
            return True
        raise ValueError(f"AFV returned an unrecognized verdict: {output!r}")

    @staticmethod
    def _parse_confidence(output: str) -> Optional[float]:
        match = re.search(r"(?:confidence\s*[:=]\s*)?(0(?:\.\d+)?|1(?:\.0+)?)", output.lower())
        return float(match.group(1)) if match else None
