"""Shared data structures for fact verification."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Union


class TextGenerator(Protocol):
    """A model adapter that returns generated text for a prompt."""

    def generate(self, prompt: str) -> Union[str, tuple]:
        ...


@dataclass(frozen=True)
class AtomicFact:
    text: str
    source_sentence: Optional[str] = None
    fact_id: Optional[str] = None


@dataclass(frozen=True)
class EvidencePassage:
    text: str
    title: str = ""
    score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FactValidation:
    fact: AtomicFact
    label: str
    evidence: List[EvidencePassage]
    confidence: Optional[float] = None
    raw_output: str = ""

    @property
    def supported(self) -> bool:
        """Backward-compatible binary view of the validation label."""
        return self.label.upper() == "SUPPORTED"


@dataclass(frozen=True)
class VerificationReport:
    answer: str
    validations: List[FactValidation]
    route: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    question: Optional[str] = None

    @property
    def score(self) -> Optional[float]:
        """Backward-compatible alias for evidence_support_score."""
        return self.evidence_support_score

    @property
    def evidence_support_score(self) -> Optional[float]:
        """Fraction of atomic facts labelled SUPPORTED."""
        if not self.validations:
            return None
        return sum(item.supported for item in self.validations) / len(self.validations)

    @property
    def pe_evid(self) -> Optional[float]:
        """Evidence penalty: one minus the supported-fact fraction."""
        support_score = self.evidence_support_score
        return None if support_score is None else 1.0 - support_score

    @property
    def unsupported_facts(self) -> List[AtomicFact]:
        return [item.fact for item in self.validations if not item.supported]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the score and every intermediate pipeline result."""
        atomic_facts = [
            {
                "fact_id": item.fact.fact_id,
                "text": item.fact.text,
                "source_sentence": item.fact.source_sentence,
            }
            for item in self.validations
        ]
        retrieved_evidence = [
            {
                "fact_id": item.fact.fact_id,
                "passages": [
                    {
                        "title": passage.title,
                        "text": passage.text,
                        "score": passage.score,
                        "metadata": passage.metadata,
                    }
                    for passage in item.evidence
                ],
            }
            for item in self.validations
        ]
        afv_results = [
            {
                "fact_id": item.fact.fact_id,
                "label": item.label,
                "confidence": item.confidence,
                "raw_output": item.raw_output,
            }
            for item in self.validations
        ]
        return {
            "question": self.question,
            "answer": self.answer,
            "atomic_facts": atomic_facts,
            "retrieved_evidence": retrieved_evidence,
            "afv_results": afv_results,
            "evidence_support_score": self.evidence_support_score,
            "PE_evid": self.pe_evid,
            "route": self.route,
            "metadata": self.metadata,
        }

    def save_json(self, output_path: Union[str, os.PathLike]) -> Path:
        """Atomically save a complete report as UTF-8 JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = path.with_name(f".{path.name}.tmp")
        temporary_path.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(path)
        return path
