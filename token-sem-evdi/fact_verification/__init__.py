"""Reusable atomic-fact generation and verification components."""

from .afg import AtomicFactGenerator
from .afv import AtomicFactValidator
from .pipeline import FactVerificationPipeline
from .retriever import CallableRetriever, InMemoryRetriever, Retriever
from .schemas import (
    AtomicFact,
    EvidencePassage,
    FactValidation,
    VerificationReport,
)

__all__ = [
    "AtomicFact",
    "AtomicFactGenerator",
    "AtomicFactValidator",
    "CallableRetriever",
    "EvidencePassage",
    "FactValidation",
    "FactVerificationPipeline",
    "InMemoryRetriever",
    "Retriever",
    "VerificationReport",
]
