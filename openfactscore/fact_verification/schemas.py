"""Shared data structures for fact verification."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Union


class TextGenerator(Protocol):
    """A model adapter that returns generated text for a prompt."""

    def generate(self, prompt: str) -> Union[str, tuple]:
        ...


@dataclass(frozen=True)
class AtomicFact:
    text: str
    source_sentence: Optional[str] = None


@dataclass(frozen=True)
class EvidencePassage:
    text: str
    title: str = ""
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactValidation:
    fact: AtomicFact
    supported: bool
    evidence: List[EvidencePassage]
    confidence: Optional[float] = None
    raw_output: str = ""


@dataclass(frozen=True)
class VerificationReport:
    answer: str
    validations: List[FactValidation]
    route: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if not self.validations:
            return 0.0
        return sum(item.supported for item in self.validations) / len(self.validations)

    @property
    def unsupported_facts(self) -> List[AtomicFact]:
        return [item.fact for item in self.validations if not item.supported]
