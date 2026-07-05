"""Parse evidence blocks from model-generated text.

This module deliberately knows nothing about prompts, model providers, RL
frameworks, or reward functions. Pass only the model-generated response to
``EvidenceExtractor.extract``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple


@dataclass(frozen=True)
class EvidenceBlock:
    """One evidence block and its character span in the model response."""

    text: str
    start: int
    end: int


@dataclass(frozen=True)
class EvidenceExtractionResult:
    """Structured extraction result; malformed output does not raise."""

    blocks: Tuple[EvidenceBlock, ...]
    valid: bool
    error: Optional[str] = None

    @property
    def texts(self) -> Tuple[str, ...]:
        return tuple(block.text for block in self.blocks)

    @property
    def last_text(self) -> Optional[str]:
        return self.blocks[-1].text if self.blocks else None


@dataclass(frozen=True)
class GroundingResult:
    """Whether each evidence block can be found in the supplied sources."""

    grounded: bool
    source_indices: Tuple[Optional[int], ...]


class EvidenceExtractor:
    """Extract XML-like evidence blocks from a model response.

    Args:
        tag: Tag name, for example ``original_evidence`` or ``evidence``.
        allow_multiple: Whether more than one evidence block is accepted.
        case_sensitive_tags: Whether tag matching is case-sensitive.
        reject_empty: Whether empty evidence blocks make the result invalid.

    The parser uses a non-greedy match with DOTALL, so evidence may span
    multiple lines. It is intended for simple model-output tags, not arbitrary
    or nested XML.
    """

    _VALID_TAG = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*\Z")

    def __init__(
        self,
        tag: str = "original_evidence",
        *,
        allow_multiple: bool = False,
        case_sensitive_tags: bool = False,
        reject_empty: bool = True,
    ) -> None:
        if not self._VALID_TAG.fullmatch(tag):
            raise ValueError(f"Invalid evidence tag: {tag!r}")

        self.tag = tag
        self.allow_multiple = allow_multiple
        self.reject_empty = reject_empty

        flags = re.DOTALL
        if not case_sensitive_tags:
            flags |= re.IGNORECASE

        escaped_tag = re.escape(tag)
        self._pattern = re.compile(
            rf"<{escaped_tag}\s*>(.*?)</{escaped_tag}\s*>",
            flags,
        )
        tag_flags = 0 if case_sensitive_tags else re.IGNORECASE
        self._opening_tag = re.compile(rf"<{escaped_tag}\s*>", tag_flags)
        self._closing_tag = re.compile(rf"</{escaped_tag}\s*>", tag_flags)

    def extract(self, model_response: str) -> EvidenceExtractionResult:
        """Extract evidence from model-generated response text only."""

        if not isinstance(model_response, str):
            raise TypeError("model_response must be a string")

        matches = list(self._pattern.finditer(model_response))
        blocks = tuple(
            EvidenceBlock(
                text=match.group(1).strip(),
                start=match.start(1),
                end=match.end(1),
            )
            for match in matches
        )

        opening_count = len(self._opening_tag.findall(model_response))
        closing_count = len(self._closing_tag.findall(model_response))
        if opening_count != closing_count or len(matches) != opening_count:
            return EvidenceExtractionResult(
                blocks=blocks,
                valid=False,
                error=(
                    f"Malformed <{self.tag}> tags: "
                    f"{opening_count} opening, {closing_count} closing"
                ),
            )

        if not blocks:
            return EvidenceExtractionResult(
                blocks=(),
                valid=False,
                error=f"No <{self.tag}> block found",
            )

        if self.reject_empty and any(not block.text for block in blocks):
            return EvidenceExtractionResult(
                blocks=blocks,
                valid=False,
                error="Evidence block is empty",
            )

        if not self.allow_multiple and len(blocks) > 1:
            return EvidenceExtractionResult(
                blocks=blocks,
                valid=False,
                error=f"Expected one evidence block, found {len(blocks)}",
            )

        return EvidenceExtractionResult(blocks=blocks, valid=True)

    @staticmethod
    def check_grounding(
        evidence: Iterable[str],
        sources: Sequence[str],
        *,
        normalize_whitespace: bool = True,
        case_sensitive: bool = False,
    ) -> GroundingResult:
        """Check whether each evidence string is a substring of any source.

        This is a conservative extractive-grounding check. Semantic paraphrases
        will not pass; use a separate evaluator if paraphrased evidence is
        allowed by the project.
        """

        evidence_list = list(evidence)

        def normalize(text: str) -> str:
            if normalize_whitespace:
                text = " ".join(text.split())
            return text if case_sensitive else text.casefold()

        normalized_sources = [normalize(source) for source in sources]
        source_indices = []
        for item in evidence_list:
            needle = normalize(item)
            found = next(
                (
                    index
                    for index, source in enumerate(normalized_sources)
                    if needle and needle in source
                ),
                None,
            )
            source_indices.append(found)

        return GroundingResult(
            grounded=bool(evidence_list)
            and all(index is not None for index in source_indices),
            source_indices=tuple(source_indices),
        )
