"""Evidence retrieval interfaces and lightweight adapters."""

import math
import re
from abc import ABC, abstractmethod
from collections import Counter
from typing import Callable, Iterable, List, Mapping, Sequence, Union

from .schemas import AtomicFact, EvidencePassage


RawPassage = Union[EvidencePassage, str, Mapping[str, object]]


def _coerce_passage(item: RawPassage) -> EvidencePassage:
    if isinstance(item, EvidencePassage):
        return item
    if isinstance(item, str):
        return EvidencePassage(text=item)
    return EvidencePassage(
        text=str(item["text"]),
        title=str(item.get("title", "")),
        score=float(item["score"]) if item.get("score") is not None else None,
        metadata=dict(item.get("metadata", {})),
    )


class Retriever(ABC):
    @abstractmethod
    def retrieve(self, fact: AtomicFact, top_k: int = 5) -> List[EvidencePassage]:
        """Return evidence ranked by relevance to one atomic fact."""


class CallableRetriever(Retriever):
    """Adapt an existing search function to the Retriever interface."""

    def __init__(self, search: Callable[[str, int], Sequence[RawPassage]]):
        self.search = search

    def retrieve(self, fact: AtomicFact, top_k: int = 5) -> List[EvidencePassage]:
        return [_coerce_passage(item) for item in self.search(fact.text, top_k)]


class InMemoryRetriever(Retriever):
    """Small dependency-free TF-IDF retriever for tests and prototypes."""

    def __init__(self, passages: Iterable[RawPassage]):
        self.passages = [_coerce_passage(item) for item in passages]
        self.tokens = [self._tokenize(f"{p.title} {p.text}") for p in self.passages]
        document_frequency = Counter(token for doc in self.tokens for token in set(doc))
        count = max(len(self.passages), 1)
        self.idf = {
            token: math.log((count + 1) / (frequency + 1)) + 1
            for token, frequency in document_frequency.items()
        }

    def retrieve(self, fact: AtomicFact, top_k: int = 5) -> List[EvidencePassage]:
        query = Counter(self._tokenize(fact.text))
        ranked = []
        for passage, tokens in zip(self.passages, self.tokens):
            terms = Counter(tokens)
            score = sum(query[token] * terms[token] * self.idf.get(token, 0.0) for token in query)
            if score > 0:
                ranked.append((score, passage))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [
            EvidencePassage(
                text=passage.text,
                title=passage.title,
                score=score,
                metadata=passage.metadata,
            )
            for score, passage in ranked[:top_k]
        ]

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())
