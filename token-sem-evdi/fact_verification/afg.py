"""Atomic Fact Generation (AFG)."""

import re
from typing import Callable, List, Optional, Union

from .schemas import AtomicFact, TextGenerator


Generator = Union[TextGenerator, Callable[[str], Union[str, tuple]]]


class AtomicFactGenerator:
    """Extract independently verifiable claims from an answer."""

    VERIFIABLE = "VERIFIABLE"
    NOT_VERIFIABLE = "NOT_VERIFIABLE"

    SYSTEM_INSTRUCTION = (
        "You are an Atomic Fact Generation system.\n"
        "Extract atomic facts from the given answer.\n\n"
        "Rules:\n"
        "- One bullet = one atomic fact.\n"
        "- One fact contains exactly one independently verifiable claim.\n"
        "- Preserve original wording.\n"
        "- Do not infer.\n"
        "- Do not explain.\n"
        "- Do not merge facts.\n"
        "- Do not split entities.\n"
        "- Remove duplicates.\n"
        "- Do not output partial sentences.\n"
        "- If the answer is only an abstract opinion, output one bullet and mark it as NOT_VERIFIABLE.\n"
        "- Output ONLY bullet list.\n\n"
        "Example:\n"
        "Answer:\n"
        "北京交通大学位于北京市海淀区，是教育部直属高校。\n\n"
        "Output:\n"
        "- 北京交通大学位于北京市海淀区。\n"
        "- 北京交通大学是教育部直属高校。"
    )

    def __init__(self, generator: Generator):
        self.generator = generator

    def extract(self, answer: str) -> List[AtomicFact]:
        if not answer or not answer.strip():
            return []
        prompt = f"{self.SYSTEM_INSTRUCTION}\n\nAnswer:\n{answer.strip()}\n\nAtomic facts:"
        output = self._generate(prompt)
        facts: List[AtomicFact] = []
        seen = set()
        for text in self._parse_output(output):
            validity, drop_reason = self._classify_fact(text)
            if validity == "DROP":
                continue
            normalized = self._normalize_fact_key(text)
            if not normalized or self._already_seen(normalized, seen, validity):
                continue
            facts.append(
                AtomicFact(
                    text=text,
                    fact_id=f"fact-{len(facts) + 1:04d}",
                    validity=validity,
                    drop_reason=drop_reason,
                    metadata={"afn_version": "v1.1"},
                )
            )
            seen.add(normalized)
        return facts

    def _parse_output(self, output: str) -> List[str]:
        facts: List[str] = []
        for line in output.splitlines():
            for fragment in self._split_packed_bullets(line):
                cleaned = self._clean_fact_text(fragment)
                if not cleaned or self._is_heading(cleaned):
                    continue
                facts.extend(self._normalize_atomicity(cleaned))
        return facts

    def _split_packed_bullets(self, line: str) -> List[str]:
        text = line.strip()
        if not text:
            return []
        text = text.replace("•", "-")
        split_text = re.sub(
            r"\s+(?=(?:[-*•]\s+|\d+[.)]\s+))",
            "\n",
            text,
        )
        return [part.strip() for part in split_text.splitlines() if part.strip()]

    def _clean_fact_text(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(r"^#+\s*", "", cleaned)
        cleaned = cleaned.strip("`")
        cleaned = re.sub(
            r"^(?:[-*•]\s*|\d+[.)]\s*)+",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"^(?:Output|Atomic\s+Facts?|Answer)\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(
            r"^\(?\s*(?:NOT[_\s-]?VERIFIABLE|NO[_\s-]?EVIDENCE)\s*\)?\s*[:：.-]*\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = re.sub(r"\s*[-*•]\s*$", "", cleaned).strip()
        cleaned = cleaned.strip("\"'“”‘’")
        cleaned = re.sub(
            r"^(?:事实|Fact|Claim)\s*[:：]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = self._normalize_mixed_spacing(cleaned)
        return cleaned

    def _is_heading(self, text: str) -> bool:
        normalized = text.strip().strip(":：").casefold()
        return normalized in {
            "output",
            "atomic facts",
            "atomic fact",
            "answer",
            "not_verifiable",
            "not verifiable",
        }

    def _normalize_atomicity(self, text: str) -> List[str]:
        candidates: List[str] = []
        for sentence in self._split_sentences(text):
            cleaned = self._strip_hedging(sentence)
            for fact in self._split_merged_claim(cleaned):
                normalized = self._ensure_terminal_punctuation(fact.strip())
                if normalized:
                    candidates.append(normalized)
        return candidates

    def _split_sentences(self, text: str) -> List[str]:
        parts = [
            part.strip()
            for part in re.split(r"(?<=[。！？!?])\s+|[\n\r]+", text)
            if part.strip()
        ]
        if len(parts) == 1:
            return parts
        return [self._ensure_terminal_punctuation(part) for part in parts]

    def _split_merged_claim(self, text: str) -> List[str]:
        clauses = [
            clause.strip()
            for clause in re.split(r"[；;]", text)
            if clause.strip()
        ]
        expanded: List[str] = []
        for clause in clauses:
            expanded.extend(self._split_chinese_compound_clause(clause))
        return expanded

    def _split_chinese_compound_clause(self, text: str) -> List[str]:
        raw = text.strip().rstrip("。.!?")
        parts = [part.strip() for part in raw.split("，") if part.strip()]
        if len(parts) <= 1:
            return [text]

        subject = self._infer_subject(parts[0])
        if not subject:
            return [text]

        results: List[str] = []
        current = parts[0]
        for part in parts[1:]:
            if self._starts_new_predicate(part):
                results.append(current)
                current = subject + part
                continue
            current = f"{current}，{part}"
        results.append(current)
        return results

    def _infer_subject(self, text: str) -> Optional[str]:
        match = re.match(
            r"^(.+?)(?:位于|位在|坐落于|是|为|属于|包括|使用|采用|依赖|关注|提供|设有)",
            text,
        )
        if not match:
            return None
        subject = match.group(1).strip()
        if 1 <= len(subject) <= 40:
            return subject
        return None

    def _starts_new_predicate(self, text: str) -> bool:
        return bool(
            re.match(
                r"^(?:是|为|属于|包括|使用|采用|依赖|关注|提供|设有|核心|位于|位在|坐落于)",
                text,
            )
        )

    def _strip_hedging(self, text: str) -> str:
        cleaned = text.strip()
        cleaned = re.sub(
            r"^(?:我认为|我觉得|大概|可能|也许|应该是|应该|似乎|据我所知)[，,\s]*",
            "",
            cleaned,
        )
        cleaned = re.sub(r"(?:应该是|应该|可能|大概|也许)", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        cleaned = cleaned.replace("是位于", "位于")
        return cleaned

    def _normalize_mixed_spacing(self, text: str) -> str:
        text = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fff])", r"\1 \2", text)
        text = re.sub(r"([\u4e00-\u9fff])([A-Za-z0-9])", r"\1 \2", text)
        return text.strip()

    def _ensure_terminal_punctuation(self, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return ""
        if cleaned[-1] in "。.!?！？":
            return cleaned
        if re.search(r"[\u4e00-\u9fff]", cleaned):
            return cleaned + "。"
        return cleaned + "."

    def _classify_fact(self, text: str) -> tuple[str, Optional[str]]:
        if self._is_invalid_fragment(text):
            return "DROP", "invalid_fragment"
        if self._is_not_verifiable(text):
            return self.NOT_VERIFIABLE, "not_verifiable_or_opinion"
        if not self._has_verifiable_relation(text):
            return "DROP", "missing_verifiable_relation"
        return self.VERIFIABLE, None

    def _is_invalid_fragment(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 5:
            return True
        if re.fullmatch(
            r"\(?\s*(?:NOT[_\s-]?VERIFIABLE|NO[_\s-]?EVIDENCE)\s*\)?[.。:：-]*",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        if stripped.endswith(("：", ":", "，", ",", "；", ";")):
            return True
        if re.match(r"^(?:并且|以及|而且|同时|此外|因此|所以|因为)", stripped):
            return True
        if re.match(r"^[（(].+[）)]。?$", stripped):
            return True
        if re.match(
            r"^(?:The provided|Therefore|However|Explanation|The statement|"
            r"The given|This statement|It does not|It is marked|"
            r"Beijing Jiaotong University's location)",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        if re.search(
            r"\b(?:output is|marked as NOT[_\s-]?VERIFIABLE|matter of public record|"
            r"can be verified through official sources|does not contain an independently verifiable)\b",
            stripped,
            flags=re.IGNORECASE,
        ):
            return True
        if stripped.count("。") > 1 or stripped.count(".") > 1:
            return False
        return False

    def _is_not_verifiable(self, text: str) -> bool:
        normalized = text.strip()
        if re.search(r"\b(?:NOT[_\s-]?VERIFIABLE|NO[_\s-]?EVIDENCE)\b", normalized, re.IGNORECASE):
            return True

        subjective_patterns = [
            r"没有唯一结论",
            r"不同.+解释",
            r"值得.+关注",
            r"有助于",
            r"能够帮助.+理解",
            r"帮助研究者.+理解",
            r"可以帮助.+理解",
            r"视角能够帮助",
            r"很重要",
            r"更好地",
        ]
        if any(re.search(pattern, normalized) for pattern in subjective_patterns):
            return True

        hard_fact_patterns = [
            r"位于",
            r"坐落于",
            r"属于",
            r"直属",
            r"包括",
            r"使用",
            r"采用",
            r"依赖",
            r"关注",
            r"提供一种.+视角",
            r"\b(?:is|are|was|were|uses|includes|located)\b",
        ]
        if any(re.search(pattern, normalized, re.IGNORECASE) for pattern in hard_fact_patterns):
            return False

        return False

    def _has_verifiable_relation(self, text: str) -> bool:
        relation_patterns = [
            r"位于",
            r"位在",
            r"坐落于",
            r"是",
            r"为",
            r"属于",
            r"直属",
            r"包括",
            r"包含",
            r"使用",
            r"采用",
            r"依赖",
            r"不依赖",
            r"基于",
            r"关注",
            r"提供",
            r"设有",
            r"\b(?:is|are|was|were|uses|includes|contains|located|belongs)\b",
        ]
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in relation_patterns)

    def _normalize_fact_key(self, text: str) -> str:
        return re.sub(r"[\s。．.，,；;：:]+", "", text).casefold()

    def _already_seen(self, normalized: str, seen: set, validity: str) -> bool:
        if normalized in seen:
            return True
        if validity != self.NOT_VERIFIABLE:
            return False
        return any(normalized in item or item in normalized for item in seen)

    def _generate(self, prompt: str) -> str:
        target = self.generator.generate if hasattr(self.generator, "generate") else self.generator
        output = target(prompt)
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, str):
            raise TypeError("AFG generator must return a string or a tuple whose first item is a string")
        return output
