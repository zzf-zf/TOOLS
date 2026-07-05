"""End-to-end AFG, retrieval, and AFV orchestration."""

from typing import Any, Dict, Optional

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
    ) -> VerificationReport:
        facts = self.afg.extract(answer)
        evidence = [self.retriever.retrieve(fact, self.top_k) for fact in facts]
        validations = self.afv.verify_batch(facts, evidence)
        return VerificationReport(
            answer=answer,
            validations=validations,
            route=route,
            metadata=metadata or {},
        )
