"""End-to-end AFG, retrieval, AFV, and persistence orchestration."""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

from .afg import AtomicFactGenerator
from .afv import AtomicFactValidator
from .retriever import Retriever
from .schemas import VerificationReport


class FactVerificationPipeline:
    def __init__(
        self,
        afg: AtomicFactGenerator,
        afv: AtomicFactValidator,
        retriever: Retriever,
        top_k: int = 5,
    ):
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        self.afg = afg
        self.afv = afv
        self.retriever = retriever
        self.top_k = top_k

    def evaluate(
        self,
        answer: str,
        route: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> VerificationReport:
        facts = self.afg.extract(answer)
        evidence = [self.retriever.retrieve(fact, self.top_k) for fact in facts]
        validations = self.afv.verify_batch(facts, evidence)
        return VerificationReport(
            answer=answer,
            validations=validations,
            route=route,
            metadata=metadata or {},
            question=question,
        )

    def evaluate_and_save(
        self,
        answer: str,
        output_path: Union[str, os.PathLike],
        route: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        question: Optional[str] = None,
    ) -> VerificationReport:
        """Run the complete pipeline and persist all intermediate results."""
        report = self.evaluate(
            answer=answer,
            route=route,
            metadata=metadata,
            question=question,
        )
        report.save_json(Path(output_path))
        return report
